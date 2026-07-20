// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Software ST 2110 sender/receiver for P5 bench testing — no DPDK, no root, runs on any NIC.
//
// Reuses the engine's PURE framing (spark_st2110_core Packetizer/Depacketizer and spark_audio_core
// AudioPacketizer) over ordinary kernel multicast UDP sockets, so you can feed a real 2110-20 video
// (+ optional 2110-30 audio) multicast to the processor's RX without a camera — and validate the
// stream end-to-end over loopback with the matching --recv mode (the harness self-test).
//
// Not ST 2110-21 hardware-paced (kernel sockets), but a correct on-wire RFC 4175 / RFC 3190 stream:
// enough to exercise discover -> connect -> IGMP join -> depacketize -> process.
//
//   send:  st2110_software_sender --send --group 239.100.0.10 --port 5004 --iface 192.0.2.50 --profile 1080p
//   recv:  st2110_software_sender --recv --group 239.100.0.10 --port 5004 --iface 192.0.2.50
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

#include "operators/audio/audio_st2110.hpp"
#include "operators/st2110_tx/rtp_st2110.hpp"

namespace {
std::atomic<bool> g_stop{false};
void on_sig(int) { g_stop = true; }

uint64_t now_ns() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}
void sleep_until(uint64_t deadline_ns) {
  timespec ts{time_t(deadline_ns / 1000000000ULL), long(deadline_ns % 1000000000ULL)};
  clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
}

struct Args {
  std::string mode = "send";
  std::string group = "239.100.0.10";
  uint16_t port = 5004;
  std::string iface = "0.0.0.0";  // local IP whose interface carries the multicast
  std::string profile = "1080p";
  int64_t frames = 0;  // 0 = run until Ctrl-C
  int ttl = 8;
  bool loopback = false;            // IP_MULTICAST_LOOP (on for same-box --selftest)
  bool audio = false;               // also send ST 2110-30 audio
  std::string audio_group = "239.100.0.20";
  uint16_t audio_port = 5004;
};

bool eq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

int open_tx(const Args& a) {
  int s = socket(AF_INET, SOCK_DGRAM, 0);
  if (s < 0) { perror("socket"); return -1; }
  in_addr ifa{};
  inet_pton(AF_INET, a.iface.c_str(), &ifa);
  setsockopt(s, IPPROTO_IP, IP_MULTICAST_IF, &ifa, sizeof(ifa));
  unsigned char ttl = (unsigned char)a.ttl;
  setsockopt(s, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl));
  unsigned char loop = a.loopback ? 1 : 0;
  setsockopt(s, IPPROTO_IP, IP_MULTICAST_LOOP, &loop, sizeof(loop));
  int tos = 0xb8;  // DSCP EF — ST 2110 media class
  setsockopt(s, IPPROTO_IP, IP_TOS, &tos, sizeof(tos));
  return s;
}

void fill_dst(sockaddr_in& d, const std::string& group, uint16_t port) {
  std::memset(&d, 0, sizeof(d));
  d.sin_family = AF_INET;
  d.sin_port = htons(port);
  inet_pton(AF_INET, group.c_str(), &d.sin_addr);
}

// ---- ST 2110-30 audio sender (own thread): an L24 sine, 1 ms packets ----
void send_audio(const Args& a) {
  auto fmt = spark::st2110::audio_stereo_l24();
  spark::st2110::AudioPacketizer pktz(fmt, /*pt=*/97, /*ssrc=*/0x53504b32);
  const uint32_t spp = fmt.samples_per_packet();
  const uint32_t payload = fmt.packet_payload_bytes();
  std::vector<uint8_t> pcm(payload), wire(spark::st2110::kRtpHeaderBytes + payload);

  int s = open_tx(a);
  if (s < 0) return;
  sockaddr_in dst;
  fill_dst(dst, a.audio_group, a.audio_port);

  const uint64_t pkt_ns = uint64_t(fmt.packet_time_ms * 1e6);
  uint64_t deadline = now_ns();
  uint32_t rtp_ts = 0, phase = 0;
  while (!g_stop.load()) {
    for (uint32_t i = 0; i < spp; ++i) {  // ~1 kHz tone, L24 big-endian, both channels
      const int32_t v = int32_t(8000000.0 * __builtin_sinf(float(phase++) * 0.13f));
      for (uint16_t c = 0; c < fmt.channels; ++c) {
        uint8_t* p = &pcm[(i * fmt.channels + c) * 3];
        p[0] = uint8_t(v >> 16); p[1] = uint8_t(v >> 8); p[2] = uint8_t(v);
      }
    }
    const uint32_t len = pktz.write_packet(rtp_ts, pcm.data(), wire.data());
    sendto(s, wire.data(), len, 0, (sockaddr*)&dst, sizeof(dst));
    rtp_ts += spp;
    deadline += pkt_ns;
    sleep_until(deadline);
  }
  close(s);
}

int do_send(const Args& a) {
  using namespace spark::st2110;
  VideoFormat fmt = (a.profile == "2160p") ? profile_2160p() : profile_1080p();
  Packetizer pktz(fmt, /*max_payload=*/1420, /*pt=*/96, /*ssrc=*/0x53504b31);
  const uint32_t ppf = pktz.packets_per_frame();
  const uint64_t frame_ns = uint64_t(1e9 / fmt.fps);

  std::vector<uint8_t> frame(fmt.octets_per_frame());
  std::vector<uint8_t> pkt(1500);

  int s = open_tx(a);
  if (s < 0) return 1;
  sockaddr_in dst;
  fill_dst(dst, a.group, a.port);

  std::thread audio_thr;
  if (a.audio) audio_thr = std::thread(send_audio, a);

  std::printf("[sw-sender] sending %s %ux%u @%.3f -> %s:%u via %s (%u pkts/frame%s)\n",
              a.profile.c_str(), fmt.width, fmt.height, fmt.fps, a.group.c_str(), a.port,
              a.iface.c_str(), ppf, a.audio ? " + audio" : "");

  // Pace per FRAME: burst the frame's packets then sleep to the frame boundary. Sub-microsecond
  // per-packet sleeps fight the OS timer granularity; a single ~frame-interval sleep is reliable, and
  // the receiver/engine ring drains each ~few-MB frame burst in the gap before the next frame.
  uint64_t deadline = now_ns();
  uint64_t pkts = 0;
  for (int64_t n = 0; (a.frames == 0 || n < a.frames) && !g_stop.load(); ++n) {
    deadline += frame_ns;
    // moving gradient so the receiver can see motion; transport-valid packed octets
    for (size_t i = 0; i < frame.size(); ++i) frame[i] = uint8_t((i * 131u + 7u + n * 17u) & 0xff);
    pktz.start_frame(uint32_t(uint64_t(n) * 90000ULL * 1001ULL / 60000ULL));
    for (PacketPlan p; pktz.next(p);) {
      pktz.write_payload(p, frame.data(), pkt.data());
      sendto(s, pkt.data(), p.payload_len, 0, (sockaddr*)&dst, sizeof(dst));
      ++pkts;
    }
    if ((n % 60) == 0) std::printf("[sw-sender] frame %lld (%llu pkts)\n", (long long)n, (unsigned long long)pkts);
    sleep_until(deadline);
  }
  g_stop = true;
  if (audio_thr.joinable()) audio_thr.join();
  close(s);
  std::printf("[sw-sender] done: %lld frames, %llu packets\n", (long long)(a.frames), (unsigned long long)pkts);
  return 0;
}

int do_recv(const Args& a) {
  using namespace spark::st2110;
  VideoFormat fmt = (a.profile == "2160p") ? profile_2160p() : profile_1080p();
  Depacketizer depkt(fmt);
  std::vector<uint8_t> frame(fmt.octets_per_frame());
  std::vector<uint8_t> buf(2048);

  int s = socket(AF_INET, SOCK_DGRAM, 0);
  if (s < 0) { perror("socket"); return 1; }
  int one = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  int rcvbuf = 32 * 1024 * 1024;  // absorb bursts (no HW pacing) so we don't drop at the socket
  setsockopt(s, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
  sockaddr_in bindaddr{};
  bindaddr.sin_family = AF_INET;
  bindaddr.sin_port = htons(a.port);
  bindaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(s, (sockaddr*)&bindaddr, sizeof(bindaddr)) < 0) { perror("bind"); return 1; }
  ip_mreq mreq{};
  inet_pton(AF_INET, a.group.c_str(), &mreq.imr_multiaddr);
  inet_pton(AF_INET, a.iface.c_str(), &mreq.imr_interface);
  if (setsockopt(s, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
    perror("IP_ADD_MEMBERSHIP");
    return 1;
  }
  std::printf("[sw-recv] joined %s:%u via %s — receiving %s ...\n", a.group.c_str(), a.port,
              a.iface.c_str(), a.profile.c_str());

  uint64_t pkts = 0, frames = 0, lost = 0, bad = 0;
  uint32_t last_seq = 0;
  bool have_last = false;
  uint64_t t_report = now_ns();
  while (!g_stop.load()) {
    const ssize_t n = recv(s, buf.data(), buf.size(), 0);
    if (n <= 0) continue;
    RxPacketInfo info;
    if (!depkt.parse(buf.data(), uint32_t(n), info)) { ++bad; continue; }
    if (have_last) {
      const uint32_t gap = info.sequence - (last_seq + 1);
      if (gap != 0 && gap < 0x80000000u) lost += gap;
    }
    last_seq = info.sequence;
    have_last = true;
    depkt.scatter(info, buf.data(), frame.data());
    ++pkts;
    if (info.marker) ++frames;
    if (now_ns() - t_report > 1000000000ULL) {
      t_report = now_ns();
      std::printf("[sw-recv] frames=%llu packets=%llu lost=%llu bad=%llu\n",
                  (unsigned long long)frames, (unsigned long long)pkts, (unsigned long long)lost,
                  (unsigned long long)bad);
    }
  }
  setsockopt(s, IPPROTO_IP, IP_DROP_MEMBERSHIP, &mreq, sizeof(mreq));
  close(s);
  std::printf("[sw-recv] STOP frames=%llu packets=%llu lost=%llu bad=%llu\n",
              (unsigned long long)frames, (unsigned long long)pkts, (unsigned long long)lost,
              (unsigned long long)bad);
  return frames > 0 ? 0 : 2;  // nonzero if nothing arrived (selftest failure)
}
}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: live logs even when redirected to a file
  Args a;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
    if (eq(argv[i], "--send")) a.mode = "send";
    else if (eq(argv[i], "--recv")) a.mode = "recv";
    else if (eq(argv[i], "--group")) a.group = next();
    else if (eq(argv[i], "--port")) a.port = uint16_t(std::atoi(next()));
    else if (eq(argv[i], "--iface")) a.iface = next();
    else if (eq(argv[i], "--profile")) a.profile = next();
    else if (eq(argv[i], "--frames")) a.frames = std::atoll(next());
    else if (eq(argv[i], "--ttl")) a.ttl = std::atoi(next());
    else if (eq(argv[i], "--loopback")) a.loopback = true;
    else if (eq(argv[i], "--audio")) a.audio = true;
    else if (eq(argv[i], "--audio-group")) a.audio_group = next();
    else if (eq(argv[i], "--audio-port")) a.audio_port = uint16_t(std::atoi(next()));
    else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
  }
  std::signal(SIGINT, on_sig);
  std::signal(SIGTERM, on_sig);
  return a.mode == "recv" ? do_recv(a) : do_send(a);
}
