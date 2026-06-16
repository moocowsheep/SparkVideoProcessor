// Unit test for the ST 2110-30 audio framing + lip-sync delay (audio_st2110.hpp). Pure CPU, no NIC.
// Round-trips PCM through the real Packetizer/Depacketizer (byte-exact), checks the RTP header and
// packet geometry for L24/L16 + packet times, and verifies the delay line's fixed-lag behaviour.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../audio_st2110.hpp"

using namespace spark::st2110;

namespace {
int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
}

void test_roundtrip(const char* name, AudioFormat fmt) {
  std::printf("----- %s (%u ch, %s, %.3f ms) -----\n", name, fmt.channels,
              fmt.encoding == AudioEncoding::L24 ? "L24" : "L16", fmt.packet_time_ms);
  const uint32_t payload = fmt.packet_payload_bytes();
  std::vector<uint8_t> pcm(payload);
  for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = uint8_t((i * 97u + 13u) & 0xff);

  AudioPacketizer pktz(fmt, /*pt=*/97, /*ssrc=*/0xA5A5A5A5);
  AudioDepacketizer depkt(fmt);

  std::vector<uint8_t> wire(kRtpHeaderBytes + payload);
  const uint32_t ts0 = 1234567;
  const uint32_t len = pktz.write_packet(ts0, pcm.data(), wire.data());
  check(len == kRtpHeaderBytes + payload, "packet length = 12 + payload");
  check((wire[0] >> 6) == 2, "RTP version 2");
  check((wire[1] & 0x7f) == 97, "payload type 97");
  check((wire[1] & 0x80) == 0, "audio marker bit clear");

  AudioRxPacketInfo info;
  check(depkt.parse(wire.data(), len, info), "depacketize parses");
  check(info.rtp_timestamp == ts0, "rtp timestamp round-trips");
  check(info.data_offset == kRtpHeaderBytes, "pcm begins after RTP header");
  check(info.pcm_len == payload, "pcm length matches");
  bool exact = true;
  for (uint32_t i = 0; i < payload; ++i) exact &= (wire[info.data_offset + i] == pcm[i]);
  check(exact, "PCM bytes are byte-exact through the round trip");

  // sequence increments across packets
  std::vector<uint8_t> w2(kRtpHeaderBytes + payload);
  pktz.write_packet(ts0 + fmt.samples_per_packet(), pcm.data(), w2.data());
  AudioRxPacketInfo i2;
  depkt.parse(w2.data(), len, i2);
  check(uint16_t(info.sequence + 1) == i2.sequence, "RTP sequence increments");
}
}  // namespace

int main() {
  // 48k stereo L24 @ 1ms -> 48 samples/pkt * 2ch * 3B = 288 payload bytes
  AudioFormat stereo = audio_stereo_l24();
  check(stereo.samples_per_packet() == 48, "48k @1ms -> 48 samples/packet");
  check(stereo.packet_payload_bytes() == 288, "stereo L24 @1ms -> 288 payload bytes");
  test_roundtrip("stereo L24 @1ms", stereo);

  // 8ch L24 @ 125us -> 6 samples/pkt * 8ch * 3B = 144 payload bytes
  AudioFormat ch8{48000, 8, AudioEncoding::L24, 0.125};
  check(ch8.samples_per_packet() == 6, "48k @125us -> 6 samples/packet");
  check(ch8.packet_payload_bytes() == 144, "8ch L24 @125us -> 144 payload bytes");
  test_roundtrip("8ch L24 @125us", ch8);

  // 16-bit variant
  test_roundtrip("stereo L16 @1ms", AudioFormat{48000, 2, AudioEncoding::L16, 1.0});

  std::printf("----- lip-sync delay line -----\n");
  AudioDelayLine line(/*delay_blocks=*/3);
  AudioBlock out;
  auto blk = [](uint32_t ts) { AudioBlock b; b.rtp_timestamp = ts; return b; };
  check(!line.push(blk(10), out), "block 0 buffered (filling)");
  check(!line.push(blk(11), out), "block 1 buffered (filling)");
  check(!line.push(blk(12), out), "block 2 buffered (filling)");
  check(line.push(blk(13), out) && out.rtp_timestamp == 10, "4th push releases the 1st (lag 3)");
  check(line.push(blk(14), out) && out.rtp_timestamp == 11, "5th push releases the 2nd");
  check(line.buffered() == 3, "steady-state buffering == delay_blocks");

  AudioDelayLine zero(0);
  check(zero.push(blk(99), out) && out.rtp_timestamp == 99, "delay 0 passes through immediately");

  std::printf(g_failures ? "\nFAILED (%d)\n" : "\nPASS\n", g_failures);
  return g_failures ? 1 : 0;
}
