// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

#include "st2110_rx.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../audio/audio_bridge.hpp"
#include "../common/rtp_time.hpp"

namespace spark::ops {

// Capture per-packet HW arrival time + line number for pacing analysis. SPARK_RX_CAPTURE=N records the
// first N matched packets (relative ns) and writes a CSV once full — reveals the sender's gapped schedule.
void St2110RxOp::cap_record(const spark::st2110::RxPacketInfo& info, const spark::net::RxPacket& pkt) {
  if (!cap_target_ || cap_done_) return;
  if (cap_.empty()) cap_t0_ = pkt.has_timestamp ? pkt.hw_timestamp_ns : 0;
  cap_.push_back({pkt.has_timestamp ? pkt.hw_timestamp_ns - cap_t0_ : 0,
                  static_cast<uint16_t>(info.nsrd ? info.srd[0].line_no : 0),
                  static_cast<uint8_t>(info.nsrd), static_cast<uint8_t>(info.marker ? 1 : 0),
                  static_cast<uint16_t>(pkt.len)});
  if (cap_.size() >= cap_target_) {
    if (FILE* f = std::fopen("/tmp/spark_rx_capture.csv", "w")) {
      std::fprintf(f, "t_ns,line,nsrd,marker,len\n");
      for (const auto& p : cap_)
        std::fprintf(f, "%llu,%u,%u,%u,%u\n", static_cast<unsigned long long>(p.t_ns), p.line, p.nsrd,
                     p.marker, p.len);
      std::fclose(f);
    }
    HOLOSCAN_LOG_INFO("st2110_rx: captured {} packets -> /tmp/spark_rx_capture.csv", cap_.size());
    cap_done_ = true;
  }
}

void St2110RxOp::setup(holoscan::OperatorSpec& spec) {
  // Output used only in source mode (emit_frames=true); harmlessly unconnected in sink mode.
  spec.output<spark::st2110::VideoFrame>("frame");
  spec.param(pci_addr_, "pci_addr", "RX PCI", "CX-7 RX port BDF", std::string("0002:01:00.1"));
  spec.param(udp_port_, "udp_port", "UDP port", "RTP destination port to accept", uint32_t(20000));
  spec.param(profile_, "profile", "Profile", "1080p|2160p (frame geometry)", std::string("1080p"));
  spec.param(mcast_group_, "mcast_group", "Multicast group",
             "ST 2110 group to IGMP-join + filter (NMOS); empty = legacy promiscuous", std::string(""));
  spec.param(src_ip_, "src_ip", "SSM source", "source-specific filter (sender IP)", std::string(""));
  spec.param(iface_ip_, "iface_ip", "Interface IP", "local media IP (IGMP report source)", std::string(""));
  spec.param(in_width_, "in_width", "Input width", "override profile geometry (0 = profile)", uint32_t(0));
  spec.param(in_height_, "in_height", "Input height", "override profile geometry (0 = profile)", uint32_t(0));
  spec.param(in_fps_, "in_fps", "Input fps", "source frame rate (0 = profile)", 0.0);
  spec.param(rxd_, "rxd", "RX descriptors", "RX ring depth", uint32_t(4096));
  spec.param(eal_cores_, "eal_cores", "EAL cores", "DPDK lcore list", std::string("2,3"));
  spec.param(run_seconds_, "run_seconds", "Run seconds", "poll duration (sink mode)", 12.0);
  spec.param(manage_eal_, "manage_eal", "Manage EAL",
             "true: own rte_eal_init; false: shared DpdkEal already up", true);
  spec.param(emit_frames_, "emit_frames", "Emit frames",
             "true: source mode (one VideoFrame per compute); false: terminal sink", false);
  spec.param(ip10_, "ip10", "IP10 source",
             "source is Blackmagic IP10 (8-bit 4:2:2 pgroups, decoded to 10-bit downstream)", false);
  spec.param(jxs_, "jxs", "JPEG XS source",
             "source is ST 2110-22 JPEG XS (RFC 9134); decoded to 10-bit planar downstream", false);
  spec.param(jxs_max_bpp_, "jxs_max_bpp", "JPEG XS budget",
             "worst-case codestream bits/pixel used to size reassembly buffers", 12.0);
  spec.param(audio_mcast_, "audio_mcast", "Audio group",
             "companion ST 2110-30 multicast group on this port (empty = no audio)", std::string(""));
  spec.param(audio_src_, "audio_src", "Audio SSM source", "audio source-specific filter",
             std::string(""));
  spec.param(audio_port_, "audio_port", "Audio UDP port", "audio RTP destination port", uint32_t(0));
  spec.param(audio_rate_, "audio_rate", "Audio RTP clock", "audio media clock (2110-30: 48000)",
             uint32_t(48000));
  spec.param(frame_q_depth_, "frame_q_depth", "Frame queue depth",
             "completed-frame queue; fixed-latency mode stores its standing frames here", uint32_t(8));
}

void St2110RxOp::start() {
  fmt_ = (profile_.get() == "2160p") ? spark::st2110::profile_2160p()
                                     : spark::st2110::profile_1080p();
  // Adopt the source's actual geometry/rate from the SDP (NMOS) when provided; the depacketizer
  // geometry must match the wire, and the rate flows downstream for correct TX pacing.
  if (in_width_.get() > 0) fmt_.width = in_width_.get();
  if (in_height_.get() > 0) fmt_.height = in_height_.get();
  if (in_fps_.get() > 0.0) fmt_.fps = in_fps_.get();
  // IP10 source: the wire carries 8-bit codeword pgroups (4 octets/pgroup), so the depacketizer
  // geometry + reassembly buffer must be 8-bit; UnpackOp IP10-decodes them back to 10-bit planar.
  if (ip10_.get()) fmt_.sampling = spark::st2110::Sampling::YCbCr422_8;
  // JPEG XS source: no pgroup raster at all — the wire carries a codestream (RFC 9134). It replaces
  // the pixel-level codec choice above rather than layering on it, so refuse the ambiguous config
  // instead of guessing which one the operator meant.
  if (jxs_.get()) {
    if (ip10_.get())
      HOLOSCAN_LOG_WARN(
          "st2110_rx: both ip10 and jxs set for the source — they describe the same wire; using JPEG XS");
    fmt_.codec = spark::st2110::Codec::JpegXS;
    fmt_.sampling = spark::st2110::Sampling::YCbCr422_10;  // the DECODED sample format
    jxs_depkt_ = std::make_unique<spark::st2110::JxsDepacketizer>();
  }
  depkt_ = std::make_unique<spark::st2110::Depacketizer>(fmt_);
  frame_buf_.assign(frame_buffer_bytes(), 0);

  backend_ = spark::net::make_dpdk_rx_backend();
  spark::net::RxBackendConfig cfg;
  cfg.pci_addr = pci_addr_.get();
  cfg.udp_port = static_cast<uint16_t>(udp_port_.get());
  cfg.rxd = static_cast<uint16_t>(rxd_.get());
  cfg.eal_core_list = eal_cores_.get();
  cfg.manage_eal = manage_eal_.get();
  cfg.mcast_group = mcast_group_.get();
  cfg.src_ip = src_ip_.get();
  cfg.iface_ip = iface_ip_.get();
  audio_enabled_ = !audio_mcast_.get().empty() && audio_port_.get() != 0;
  if (audio_enabled_) {
    cfg.audio_mcast_group = audio_mcast_.get();
    cfg.audio_src_ip = audio_src_.get();
    cfg.audio_udp_port = static_cast<uint16_t>(audio_port_.get());
  }
  backend_->init(cfg);
  if (const char* c = std::getenv("SPARK_RX_CAPTURE")) {
    cap_target_ = static_cast<uint32_t>(std::atoll(c));
    if (cap_target_) cap_.reserve(cap_target_);
  }
  if (const char* d = std::getenv("SPARK_RX_DRAIN")) drain_enabled_ = d[0] == '1';
  HOLOSCAN_LOG_INFO("st2110_rx started: RX {} udp:{} group={} profile={} emit_frames={}{}{}", cfg.pci_addr,
                    cfg.udp_port, cfg.mcast_group.empty() ? "(none)" : cfg.mcast_group, profile_.get(),
                    emit_frames_.get(),
                    fmt_.is_jxs() ? " JPEG-XS" : (ip10_.get() ? " IP10" : ""),
                    cap_target_ ? " CAPTURE" : "");
  if (fmt_.is_jxs())
    HOLOSCAN_LOG_INFO("st2110_rx: JPEG XS reassembly buffers {} bytes ({:.1f} bpp cap at {}x{})",
                      frame_buffer_bytes(), jxs_max_bpp_.get(), fmt_.width, fmt_.height);

  // Source mode: a dedicated thread drains the NIC continuously (see poll_loop). compute() only
  // pops finished frames, so NIC polling never stalls while TX paces the previous frame.
  if (emit_frames_.get()) poll_thread_ = std::thread(&St2110RxOp::poll_loop, this);
}

void St2110RxOp::account(uint32_t sequence, const spark::net::RxPacket& pkt, uint64_t now_ns) {
  if (have_last_) {
    const uint32_t gap = sequence - (last_seq_ + 1);  // uint32 wrap-safe
    if (gap != 0 && gap < 0x80000000u) lost_ += gap;  // forward gap = loss; ignore reorder
  }
  last_seq_ = sequence;
  have_last_ = true;
  if (pkt.has_timestamp && now_ns >= pkt.hw_timestamp_ns) {
    const uint64_t lat = now_ns - pkt.hw_timestamp_ns;
    lat_sum_ += lat;
    ++lat_cnt_;
    if (lat < lat_min_) lat_min_ = lat;
    if (lat > lat_max_) lat_max_ = lat;
  }
  ++packets_;
}

// Companion 2110-30 audio packet: validate the RTP header, unwrap its 48 kHz timestamp to the
// absolute capture instant (same PHC reference as video), and hand the packet — verbatim — to the
// TX audio relay. No format assumptions: channels/depth/ptime pass through untouched.
void St2110RxOp::audio_ingest(const spark::net::RxPacket& pkt, uint64_t now_ns) {
  if (pkt.len < 12 || (pkt.payload[0] >> 6) != 2) {  // RTP v2 header minimum
    ++audio_bad_;
    return;
  }
  const uint32_t rtp_ts = (uint32_t(pkt.payload[4]) << 24) | (uint32_t(pkt.payload[5]) << 16) |
                          (uint32_t(pkt.payload[6]) << 8) | uint32_t(pkt.payload[7]);
  const uint64_t ref = pkt.has_timestamp ? pkt.hw_timestamp_ns : now_ns;
  const uint32_t rate = audio_rate_.get();
  uint64_t cap = spark::st2110::rtp_unwrap_ns(rtp_ts + audio_ts_delta_, rate, ref);
  // Sender-stamp sanity: a correct 2110-30 stamp sits within packetization+network delay of its
  // arrival. Anything beyond kSaneNs is a broken sender epoch (observed on a BMD: audio frozen
  // seconds off TAI while video was correct, stepping again mid-run) — without this the relay
  // would drop 100% of audio forever. Latch a constant tick delta that puts the stream back on
  // the arrival instant; prefer delta=0 (bit-transparent) whenever the raw stamp is sane again.
  constexpr int64_t kSaneNs = 500000000;  // generous vs ptime+device pipeline, tiny vs a bad epoch
  if (const int64_t err = int64_t(cap) - int64_t(ref); err > kSaneNs || err < -kSaneNs) {
    const uint64_t raw = spark::st2110::rtp_unwrap_ns(rtp_ts, rate, ref);
    const int64_t raw_err = int64_t(raw) - int64_t(ref);
    if (audio_ts_delta_ != 0 && raw_err <= kSaneNs && raw_err >= -kSaneNs) {
      audio_ts_delta_ = 0;  // sender recovered: back to the verbatim stamp
      cap = raw;
    } else {
      audio_ts_delta_ = spark::st2110::rtp_restamp_delta(rtp_ts, rate, ref);
      cap = spark::st2110::rtp_unwrap_ns(rtp_ts + audio_ts_delta_, rate, ref);
    }
    ++audio_relatch_;
    const double t = std::chrono::duration<double>(
                         std::chrono::steady_clock::now().time_since_epoch()).count();
    if (t - audio_relatch_warn_s_ >= 1.0) {  // rate-limit: a jittering sender relatches per packet
      audio_relatch_warn_s_ = t;
      HOLOSCAN_LOG_WARN(
          "st2110_rx: audio sender stamp {:.3f}s off its arrival — broken sender epoch; "
          "re-latched ts_delta={} (relatch #{}); lip-sync now anchored to arrival time",
          err / 1e9, audio_ts_delta_, audio_relatch_);
    }
  }
  spark::st2110::AudioBridge::Pkt p;
  p.rtp.assign(pkt.payload, pkt.payload + pkt.len);
  p.capture_ns = cap;
  p.ts_delta = audio_ts_delta_;
  if (spark::st2110::AudioBridge::instance().push(std::move(p)))
    ++audio_pkts_;
  else
    ++audio_drop_;
}

// Video frame capture instant, with the same broken-sender-epoch guard as audio_ingest above:
// unwrap the 90 kHz stamp against the frame's first-packet arrival, and if it lands beyond kSaneNs
// of that arrival (observed on a BMD: video epoch stepped +2.0s while PTP was clean — same device
// fault as the audio epochs), latch a constant tick delta re-anchoring the stream to arrival time.
// Cadence-exact (integer add), auto-returns to verbatim when the sender recovers. The fixed-latency
// schedule AND the outgoing wire stamps both derive from capture_ts (wire = capture + L), so this
// one correction keeps the schedule real and the output 2110-10 stamps on the PTP epoch. Runs once
// per frame; each relatch = one step in output timestamps (downstream receivers resync once).
// Threshold is TIGHTER than audio's ±500 ms: a healthy video stamp sits within ~20 ms of its
// first-packet arrival (measured −17.9 ms on a sane BMD), and a sender whose epoch steps in
// sub-threshold hops accumulates each hop straight into wire latency until a hop crosses the
// line (observed live 2026-07-04: +72 ms then +421 ms hops → e2e sawtoothed 65→565 ms under the
// old 500 ms bound). 150 ms caps that excursion while staying 7× above legit stamp lag; a
// genuine >150 ms delivery stall relatches once and self-corrects on the next sane frame.
uint64_t St2110RxOp::video_capture_ns(uint32_t rtp_ts, uint64_t ref) {
  constexpr int64_t kSaneNs = 150000000;
  uint64_t cap = spark::st2110::rtp_unwrap_ns(rtp_ts + video_ts_delta_, 90000, ref);
  if (const int64_t err = int64_t(cap) - int64_t(ref); err > kSaneNs || err < -kSaneNs) {
    const uint64_t raw = spark::st2110::rtp_unwrap_ns(rtp_ts, 90000, ref);
    const int64_t raw_err = int64_t(raw) - int64_t(ref);
    if (video_ts_delta_ != 0 && raw_err <= kSaneNs && raw_err >= -kSaneNs) {
      video_ts_delta_ = 0;  // sender recovered: back to the verbatim stamp
      cap = raw;
    } else {
      video_ts_delta_ = spark::st2110::rtp_restamp_delta(rtp_ts, 90000, ref);
      cap = spark::st2110::rtp_unwrap_ns(rtp_ts + video_ts_delta_, 90000, ref);
    }
    ++video_relatch_;
    const double t = std::chrono::duration<double>(
                         std::chrono::steady_clock::now().time_since_epoch()).count();
    if (t - video_relatch_warn_s_ >= 1.0) {  // rate-limit: a jittering sender relatches per frame
      video_relatch_warn_s_ = t;
      HOLOSCAN_LOG_WARN(
          "st2110_rx: video sender stamp {:.3f}s off its arrival — broken sender epoch; "
          "re-latched ts_delta={} (relatch #{}); capture now anchored to arrival time",
          err / 1e9, video_ts_delta_, video_relatch_);
    }
  }
  return cap;
}

// Reassembly-buffer size. A raw frame is exactly one pixel raster; a JPEG XS frame is a
// variable-length codestream, so the buffer is a CAP derived from the configured worst-case rate and
// VideoFrame::payload_bytes says how much of it a given frame actually filled.
size_t St2110RxOp::frame_buffer_bytes() const {
  if (!fmt_.is_jxs()) return static_cast<size_t>(fmt_.octets_per_frame());
  double bpp = jxs_max_bpp_.get();
  if (!(bpp > 0.0)) bpp = 12.0;
  const double bytes = static_cast<double>(fmt_.width) * fmt_.height * bpp / 8.0;
  return static_cast<size_t>(bytes) + 4096;  // + slack for headers/markers at tiny rasters
}

std::shared_ptr<std::vector<uint8_t>> St2110RxOp::next_buffer() {
  if (buf_ring_.empty()) {
    buf_ring_.resize(8);
    for (auto& b : buf_ring_) b = std::make_shared<std::vector<uint8_t>>(frame_buffer_bytes());
  }
  // Hand back a buffer that ONLY the ring references (use_count == 1). A buffer still held by a queued
  // or in-flight frame must never be reused — overwriting it mid-pipeline corrupts a frame that's about
  // to be sent (visible as glitches under a transient queue backlog). If every ring slot is in flight,
  // grow the ring instead of overwriting.
  for (size_t k = 0; k < buf_ring_.size(); ++k) {
    auto& b = buf_ring_[buf_idx_];
    buf_idx_ = (buf_idx_ + 1) % buf_ring_.size();
    if (b.use_count() == 1) return b;
  }
  auto nb = std::make_shared<std::vector<uint8_t>>(frame_buffer_bytes());
  buf_ring_.push_back(nb);
  return nb;
}

// --- per-packet reassembly ------------------------------------------------------------------------
// Both return true when the packet completed the frame currently in cur_buf_.

// RFC 4175: every packet says where its pixels go, so a lost packet leaves a hole and the frame is
// still worth showing. Scatter and wait for the marker.
bool St2110RxOp::ingest_raw(const spark::net::RxPacket& pkt, uint64_t now_ns) {
  spark::st2110::RxPacketInfo info;
  if (!depkt_->parse(pkt.payload, pkt.len, info)) {
    ++bad_;
    return false;
  }
  account(info.sequence, pkt, now_ns);
  cap_record(info, pkt);
  if (cur_first_) {
    cur_ts_ = info.rtp_timestamp;
    cur_arrival_ns_ = pkt.has_timestamp ? pkt.hw_timestamp_ns : now_ns;
    cur_first_ = false;
  }
  depkt_->scatter(info, pkt.payload, cur_buf_->data());
  return info.marker;
}

// RFC 9134 codestream mode: packets carry no offset, only the {SEP,P} counter, so the codestream is
// rebuilt by concatenating fragments in order. That makes loss unrecoverable in a way raw loss is
// not — a missing fragment shifts every following byte, so the decoder would either error out or
// reconstruct plausible garbage. Any discontinuity therefore latches cur_bad_ and the whole frame is
// dropped at the marker. Frames still arrive at the source cadence; a lost one is a repeat, not a tear.
bool St2110RxOp::ingest_jxs(const spark::net::RxPacket& pkt, uint64_t now_ns) {
  spark::st2110::JxsRxPacketInfo info;
  if (!jxs_depkt_->parse(pkt.payload, pkt.len, info)) {
    ++bad_;
    return false;
  }
  seq_ext_ = spark::st2110::seq16_extend(seq_ext_, info.sequence, have_seq_ext_);
  have_seq_ext_ = true;
  account(seq_ext_, pkt, now_ns);
  if (info.slice_mode) {
    // Slice mode packetizes each slice independently with its own headers; concatenating those as
    // if they were codestream fragments would produce a malformed stream. Report, do not guess.
    ++jxs_slice_mode_;
    if (jxs_slice_mode_ == 1)
      HOLOSCAN_LOG_WARN(
          "st2110_rx: sender uses JPEG XS SLICE packetization (K=1), which this receiver does not "
          "reassemble — ask the sender for codestream mode (packetmode=0)");
    cur_bad_ = true;
    return false;
  }

  if (cur_first_) {
    cur_ts_ = info.rtp_timestamp;
    cur_arrival_ns_ = pkt.has_timestamp ? pkt.hw_timestamp_ns : now_ns;
    cur_first_ = false;
    cur_len_ = 0;
    cur_bad_ = false;
    jxs_next_pkt_ = 0;
  } else if (info.rtp_timestamp != cur_ts_) {
    // A new frame started before the previous one's marker arrived: its tail was lost. Abandon the
    // partial frame and begin this one here, rather than splicing two codestreams together.
    ++jxs_incomplete_;
    if (jxs_incomplete_ == 1 || (jxs_incomplete_ % 60) == 0)
      HOLOSCAN_LOG_WARN("st2110_rx: JPEG XS frame ended without its marker — {} incomplete so far",
                        jxs_incomplete_);
    cur_ts_ = info.rtp_timestamp;
    cur_arrival_ns_ = pkt.has_timestamp ? pkt.hw_timestamp_ns : now_ns;
    cur_len_ = 0;
    cur_bad_ = false;
    jxs_next_pkt_ = 0;
  }

  if (info.packet_counter != jxs_next_pkt_) cur_bad_ = true;  // gap or reorder: position is lost
  jxs_next_pkt_ = info.packet_counter + 1;

  if (cur_len_ + info.data_len > cur_buf_->size()) {
    // The sender's rate exceeds the configured cap (jxs_max_bpp). Truncating would hand the decoder
    // a codestream missing its tail, so drop the frame and say what to change.
    if (!cur_bad_)
      HOLOSCAN_LOG_WARN(
          "st2110_rx: JPEG XS codestream exceeds the {} byte reassembly budget — raise "
          "SPARK_JXS_MAX_BPP (currently {:.1f} bpp)",
          cur_buf_->size(), jxs_max_bpp_.get());
    cur_bad_ = true;
  } else {
    std::memcpy(cur_buf_->data() + cur_len_, pkt.payload + info.data_offset, info.data_len);
    cur_len_ += info.data_len;
  }

  if (!info.marker) return false;
  if (cur_bad_ || cur_len_ == 0) {
    ++jxs_corrupt_;
    if (jxs_corrupt_ == 1 || (jxs_corrupt_ % 60) == 0)
      HOLOSCAN_LOG_WARN("st2110_rx: dropped a JPEG XS frame with a broken codestream — {} so far",
                        jxs_corrupt_);
    cur_first_ = true;  // restart reassembly on the next packet; caller sees no completed frame
    cur_len_ = 0;
    return false;
  }
  return true;
}

void St2110RxOp::compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
                         holoscan::ExecutionContext&) {
  if (emit_frames_.get())
    compute_emit_one(op_output);
  else
    compute_sink();
}

void St2110RxOp::compute_sink() {
  using clock = std::chrono::steady_clock;
  const auto deadline = clock::now() + std::chrono::duration<double>(run_seconds_.get());
  HOLOSCAN_LOG_INFO("st2110_rx: polling for {}s ...", run_seconds_.get());

  spark::net::RxPacket pkts[256];
  while (clock::now() < deadline) {
    emit_live();  // 1 Hz live stats (throttled)
    const uint16_t n = backend_->receive(pkts, 256);
    if (n == 0) continue;
    const uint64_t now = backend_->now_ns();
    for (uint16_t i = 0; i < n; ++i) {
      if (fmt_.is_jxs()) {  // sink mode only counts; no reassembly target beyond the header parse
        spark::st2110::JxsRxPacketInfo info;
        if (!jxs_depkt_->parse(pkts[i].payload, pkts[i].len, info)) {
          ++bad_;
          continue;
        }
        seq_ext_ = spark::st2110::seq16_extend(seq_ext_, info.sequence, have_seq_ext_);
        have_seq_ext_ = true;
        account(seq_ext_, pkts[i], now);
        if (info.marker) ++frames_;
        continue;
      }
      spark::st2110::RxPacketInfo info;
      if (!depkt_->parse(pkts[i].payload, pkts[i].len, info)) {
        ++bad_;
        continue;
      }
      account(info.sequence, pkts[i], now);
      cap_record(info, pkts[i]);
      depkt_->scatter(info, pkts[i].payload, frame_buf_.data());
      if (info.marker) ++frames_;
    }
    backend_->release(pkts, n);
  }
  print_stats();
}

// Dedicated NIC-drain thread (emit mode). Continuously receives bursts and reassembles frames,
// pushing each completed frame onto a bounded queue. Because it never blocks on the Holoscan emit
// path, the NIC ring stays drained even while TX paces the previous frame (the fix for the HW-drop /
// latency seen when polling only inside compute()). On queue-full it drops the frame but keeps
// draining (whole-frame drop beats HW packet loss).
void St2110RxOp::poll_loop() {
  const size_t kMaxQ = std::max<uint32_t>(frame_q_depth_.get(), 2);
  cur_buf_ = next_buffer();
  cur_first_ = true;
  spark::net::RxPacket pkts[256];
  while (!stop_poll_.load(std::memory_order_relaxed)) {
    const uint16_t n = backend_->receive(pkts, 256);
    if (n == 0) continue;
    const uint64_t now = backend_->now_ns();
    for (uint16_t i = 0; i < n; ++i) {
      if (pkts[i].is_audio) {  // companion 2110-30 flow: relay verbatim via the bridge
        audio_ingest(pkts[i], now);
        continue;
      }
      // Reassemble by wire codec (RFC 4175 raster vs RFC 9134 codestream); everything downstream of
      // "the frame is complete" — timing, the queue, the drain governor — is identical for both.
      const bool complete =
          fmt_.is_jxs() ? ingest_jxs(pkts[i], now) : ingest_raw(pkts[i], now);
      if (complete) {  // frame complete (a burst spans < one frame, so at most once)
        spark::st2110::VideoFrame f;
        f.data = cur_buf_;
        f.format = fmt_;
        // JPEG XS: the codestream filled only part of the worst-case buffer. 0 for raw, where the
        // whole buffer is the frame.
        f.payload_bytes = fmt_.is_jxs() ? cur_len_ : 0;
        // ABSOLUTE capture time (PTP epoch): the 32-bit RTP ts unwrapped against the first packet's
        // PHC arrival. Restart-invariant and wrap-free, so a FIXED genlock offset (capture + L) is
        // a true end-to-end latency — and audio, unwrapped the same way, shares the timeline.
        f.capture_ts_ns = video_capture_ns(cur_ts_, cur_arrival_ns_);
        f.frame_number = frames_++;
        {
          std::lock_guard<std::mutex> lk(q_mu_);
          // Standing-queue drain governor (see header): if the queue never dipped below 2 for a
          // whole 32-push window (~1s), that depth is dead latency no steady-state process can
          // remove (production == consumption). Shed the OLDEST frame — bounded to 1/window — and
          // let the TX trim servo reclaim the freed time from the genlock offset.
          if (drain_enabled_) {
            q_win_min_ = std::min(q_win_min_, frame_q_.size());
            if (++q_win_count_ >= 32) {
              if (q_win_min_ >= 2 && !frame_q_.empty()) {
                frame_q_.pop_front();
                ++q_drained_;
                HOLOSCAN_LOG_INFO(
                    "st2110_rx: standing queue (floor {} over 32 frames) — drained oldest ({} total)",
                    q_win_min_, q_drained_);
              }
              q_win_count_ = 0;
              q_win_min_ = SIZE_MAX;
            }
          }
          if (frame_q_.size() < kMaxQ)
            frame_q_.push_back(std::move(f));
          else {
            ++q_dropped_;
            // Surface queue overflow (downstream/TX briefly fell behind) so transient stalls are visible
            // in the log with timestamps — rate-limited to avoid flooding during a burst.
            if (q_dropped_ == 1 || (q_dropped_ % 20) == 0)
              HOLOSCAN_LOG_WARN("st2110_rx: frame queue full (depth {}) — dropped {} frames total",
                                kMaxQ, q_dropped_);
          }
        }
        q_cv_.notify_one();
        cur_buf_ = next_buffer();
        cur_first_ = true;
      }
    }
    backend_->release(pkts, n);
  }
}

// Source mode: pop one finished frame from the poll thread's queue and emit it.
void St2110RxOp::compute_emit_one(holoscan::OutputContext& op_output) {
  emit_live();  // 1 Hz live stats (throttled)
  std::unique_lock<std::mutex> lk(q_mu_);
  if (!q_cv_.wait_for(lk, std::chrono::seconds(2),
                      [&] { return !frame_q_.empty() || stop_poll_.load(); }))
    return;  // timed out (upstream stopped) — emit nothing this tick
  if (frame_q_.empty()) return;
  auto f = std::move(frame_q_.front());
  frame_q_.pop_front();
  lk.unlock();
  op_output.emit(f, "frame");
}

void St2110RxOp::emit_live(bool force) {
  const double t =
      std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if (!force && t - last_live_s_ < 1.0) return;
  last_live_s_ = t;
  const uint64_t avg_us = lat_cnt_ ? (lat_sum_ / lat_cnt_) / 1000 : 0;  // ns -> us
  // Unique tokens so the control daemon can grab each field unambiguously (latest-wins).
  HOLOSCAN_LOG_INFO("spark_live rx_frames={} rx_packets={} rx_lost={} rx_latency_us={} rx_relatch={}",
                    frames_, packets_, lost_, avg_us, video_relatch_);
  if (fmt_.is_jxs())
    HOLOSCAN_LOG_INFO("spark_live jxs_rx_corrupt={} jxs_rx_incomplete={} jxs_rx_slicemode={}",
                      jxs_corrupt_, jxs_incomplete_, jxs_slice_mode_);
  if (audio_enabled_)
    HOLOSCAN_LOG_INFO("spark_live audio_rx_pkts={} audio_rx_bad={} audio_rx_drop={} audio_rx_relatch={}",
                      audio_pkts_, audio_bad_, audio_drop_, audio_relatch_);
}

void St2110RxOp::print_stats() {
  if (stats_printed_) return;
  stats_printed_ = true;
  const auto rs = backend_ ? backend_->stats() : spark::net::RxStats{};
  const uint64_t avg = lat_cnt_ ? lat_sum_ / lat_cnt_ : 0;
  HOLOSCAN_LOG_INFO(
      "st2110_rx stats: frames={} matched_pkts={} lost={} bad={} q_dropped={} | nic_ipackets={} "
      "raw_burst={} hw_missed={} nombuf={} | ingest latency min/avg/max = {}/{}/{} ns ({} samples)",
      frames_, packets_, lost_, bad_, q_dropped_, rs.rx_packets, rs.raw_received, rs.rx_missed,
      rs.rx_nombuf, (lat_min_ == UINT64_MAX ? 0 : lat_min_), avg, lat_max_, lat_cnt_);
  if (fmt_.is_jxs())
    HOLOSCAN_LOG_INFO(
        "st2110_rx JPEG XS: corrupt={} incomplete={} slice_mode_pkts={} (corrupt/incomplete frames "
        "are dropped whole — a JPEG XS codestream cannot be partially decoded)",
        jxs_corrupt_, jxs_incomplete_, jxs_slice_mode_);
}

void St2110RxOp::stop() {
  // Stop + join the poll thread BEFORE shutting the backend down (it owns the NIC the thread polls).
  stop_poll_.store(true);
  q_cv_.notify_all();
  if (poll_thread_.joinable()) poll_thread_.join();
  emit_live(true);  // final live snapshot so the daemon captures end-of-run values
  print_stats();
  if (backend_) {
    backend_->shutdown();
    backend_.reset();
  }
}

}  // namespace spark::ops
