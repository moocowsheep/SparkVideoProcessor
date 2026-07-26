// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// St2110RxOp — Holoscan operator that receives an ST 2110-20 stream and measures it.
//
// Pipeline role (eventual):  [st2110_rx] -> unpack -> ...
// v1 scope: a terminal sink for loopback validation. It polls the NIC for `run_seconds`, depacketizes
// (RFC 4175), reassembles frames, and reports packet/frame counts, loss (RTP sequence gaps), and
// zero-copy ingest latency (NIC HW rx timestamp -> operator processing time, same PHC base). Driven
// by a CountCondition(1): compute() runs once and loops internally for the duration.
//
// Two drive modes:
//   * emit_frames=false (default): terminal sink. CountCondition(1); compute() loops for run_seconds.
//   * emit_frames=true: source. Each compute() assembles exactly one frame (poll until the RTP marker)
//     and emits it on the "frame" output — this is the rx -> tx pass-through path.
// Loss + ingest-latency tracking and the stop() summary work in both modes.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "../st2110_tx/rtp_jxs.hpp"
#include "../st2110_tx/rtp_st2110.hpp"
#include "../st2110_tx/st2110_format.hpp"
#include "rx_backend.hpp"

namespace spark::ops {

class St2110RxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(St2110RxOp)
  St2110RxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;

 private:
  void print_stats();
  void emit_live(bool force = false);  // periodic machine-parseable "spark_live rx_*" line (1 Hz)
  void compute_sink();                                  // emit_frames=false: loop run_seconds
  void compute_emit_one(holoscan::OutputContext& out);  // emit_frames=true: pop one frame, emit
  void poll_loop();  // emit mode: dedicated thread, continuously drains the NIC into frames
  void account(uint32_t sequence, const spark::net::RxPacket& pkt,
               uint64_t now_ns);  // loss + ingest-latency bookkeeping
  // Per-packet reassembly, one function per wire codec. Both return true when the packet completed
  // the frame now sitting in cur_buf_ (raw: the pixel raster; JPEG XS: cur_len_ codestream bytes).
  bool ingest_raw(const spark::net::RxPacket& pkt, uint64_t now_ns);
  bool ingest_jxs(const spark::net::RxPacket& pkt, uint64_t now_ns);
  void audio_ingest(const spark::net::RxPacket& pkt, uint64_t now_ns);  // 2110-30 -> AudioBridge
  uint64_t video_capture_ns(uint32_t rtp_ts, uint64_t ref);  // unwrap + broken-epoch guard
  size_t frame_buffer_bytes() const;  // raw: octets_per_frame(); JPEG XS: the codestream cap
  std::shared_ptr<std::vector<uint8_t>> next_buffer();  // ring buffer for emitted frames

  holoscan::Parameter<std::string> pci_addr_;
  holoscan::Parameter<uint32_t> udp_port_;
  holoscan::Parameter<std::string> profile_;
  holoscan::Parameter<std::string> mcast_group_;  // ST 2110 group to join (NMOS); "" = legacy
  holoscan::Parameter<std::string> src_ip_;       // SSM source filter
  holoscan::Parameter<std::string> iface_ip_;     // local media interface IP (IGMP report source)
  holoscan::Parameter<uint32_t> in_width_;        // override profile geometry from the SDP (0 = profile)
  holoscan::Parameter<uint32_t> in_height_;
  holoscan::Parameter<double> in_fps_;            // source frame rate from the SDP (0 = profile)
  holoscan::Parameter<uint32_t> rxd_;
  holoscan::Parameter<std::string> eal_cores_;
  holoscan::Parameter<double> run_seconds_;
  holoscan::Parameter<bool> manage_eal_;    // false in multi-backend processes (shared DpdkEal)
  holoscan::Parameter<bool> emit_frames_;   // true: source mode (emit one VideoFrame per compute)
  holoscan::Parameter<bool> ip10_;          // source carries Blackmagic IP10 (8-bit pgroups); decoded downstream
  // Source carries ST 2110-22 JPEG XS (RFC 9134). Reassembly becomes ordered concatenation of a
  // variable-length codestream instead of addressed writes into a pixel raster; JxsDecodeOp decodes
  // it downstream. Mutually exclusive with ip10_ (both describe the same wire).
  holoscan::Parameter<bool> jxs_;
  // Worst-case codestream budget, in bits per pixel, used to size the reassembly buffers. The
  // sender's real rate is not knowable in advance (it is not in the SDP), so this is a cap: a frame
  // that overruns it is dropped rather than allowed to overflow. Raw 4:2:2 10-bit is 20 bpp, so the
  // default still saves most of the memory while leaving generous headroom over typical 2110-22 rates.
  holoscan::Parameter<double> jxs_max_bpp_;
  // Companion ST 2110-30 audio flow (M10): received on the same port/queue, classified by the
  // backend, and handed to the TX audio relay via AudioBridge. Empty group = no audio.
  holoscan::Parameter<std::string> audio_mcast_;
  holoscan::Parameter<std::string> audio_src_;
  holoscan::Parameter<uint32_t> audio_port_;
  holoscan::Parameter<uint32_t> audio_rate_;       // RTP clock of the audio flow (2110-30: 48 kHz)
  // Frame-queue depth: in fixed-latency mode the standing store of (L - pipeline floor) worth of
  // source frames lives HERE (the NIC's tx_pp window can only hold ~ms), so the app sizes it from L.
  holoscan::Parameter<uint32_t> frame_q_depth_;

  std::unique_ptr<spark::net::ISt2110RxBackend> backend_;
  std::unique_ptr<spark::st2110::Depacketizer> depkt_;
  std::unique_ptr<spark::st2110::JxsDepacketizer> jxs_depkt_;  // set only when jxs_ is on
  std::vector<uint8_t> frame_buf_;  // reassembly target for sink mode (size == frame_buffer_bytes())
  spark::st2110::VideoFormat fmt_;

  // emit-mode reassembly: a ring of frame buffers + the in-progress one (poll-thread local).
  std::vector<std::shared_ptr<std::vector<uint8_t>>> buf_ring_;
  size_t buf_idx_ = 0;
  std::shared_ptr<std::vector<uint8_t>> cur_buf_;
  bool cur_first_ = true;
  uint32_t cur_ts_ = 0;
  uint64_t cur_arrival_ns_ = 0;  // first-packet NIC HW arrival — the RTP-unwrap reference
  // --- JPEG XS reassembly (poll-thread only) ---
  // A JPEG XS packet is position-IMPLICIT: it carries no offset, only a counter, so fragments are
  // concatenated in order and a single gap invalidates the whole codestream. cur_bad_ latches that,
  // and the frame is dropped instead of handed to the decoder — a torn codestream is not a torn
  // picture, it is a decoder error (or worse, plausible garbage).
  size_t cur_len_ = 0;            // codestream bytes accumulated into cur_buf_
  bool cur_bad_ = false;          // this frame lost/reordered a fragment or overran the buffer
  uint32_t jxs_next_pkt_ = 0;     // expected {SEP,P} counter of the next fragment
  uint32_t seq_ext_ = 0;          // 16-bit RTP sequence extended to 32 bits (no RFC 4175 ESN here)
  bool have_seq_ext_ = false;
  uint64_t jxs_corrupt_ = 0;      // frames dropped: missing/reordered fragment or buffer overrun
  uint64_t jxs_incomplete_ = 0;   // frames abandoned: a new RTP timestamp arrived before the marker
  uint64_t jxs_slice_mode_ = 0;   // packets rejected: slice packetization mode (K=1), unsupported
  uint64_t jxs_interlaced_ = 0;   // packets rejected: interlaced (I != 00) — progressive-only pipeline
  uint32_t jxs_done_ts_ = 0;      // RTP ts of the last frame delivered or dropped WHOLE: stragglers
  bool jxs_have_done_ = false;    //   and duplicates of it are ignored, not treated as a new frame
  // Broken-sender-epoch guard for the VIDEO stamps (poll-thread only), mirroring the audio one
  // below: same device fault observed on both essences (video epoch seen +2.0s off TAI).
  uint32_t video_ts_delta_ = 0;      // latched correction (0 = verbatim stamps)
  uint64_t video_relatch_ = 0;       // times the sanity check re-latched (>0 = sender epoch broken)
  double video_relatch_warn_s_ = 0;  // WARN rate-limit

  // emit-mode: dedicated NIC-drain thread feeding a bounded frame queue (decouples NIC polling from
  // the emit cadence, so the ring never overflows while TX paces the previous frame).
  std::thread poll_thread_;
  std::atomic<bool> stop_poll_{false};
  std::mutex q_mu_;
  std::condition_variable q_cv_;
  std::deque<spark::st2110::VideoFrame> frame_q_;
  uint64_t q_dropped_ = 0;  // frames dropped at the queue when TX can't keep up (poll-thread only)
  // Standing-queue drain governor (poll-thread only, under q_mu_): drops the OLDEST frame when the
  // queue floor stays >=2 over a window. DEFAULT OFF (SPARK_RX_DRAIN=1 opts in): tried on the rig
  // 2026-07-02 and it fired continuously (402 drains/7min) — a backpressured chain legitimately
  // holds ~2 frames here as its working set (the consumer gulps only when the whole downstream has
  // room), so the "standing depth" regenerates instantly and the drops just discard 1 source
  // frame/s (visible brackets) without lowering latency. Standing pipeline latency is set by the
  // inter-op queue CAPACITIES (see pipeline_caps.hpp: depth x frame period == latency); shrinking
  // those per-hop is the real lever, not draining here.
  bool drain_enabled_ = false;
  uint32_t q_win_count_ = 0;    // pushes seen in the current window
  size_t q_win_min_ = SIZE_MAX;  // min queue depth (before push) seen in the window
  uint64_t q_drained_ = 0;       // standing-backlog frames dropped by the governor

  // --- packet-schedule capture (debug): SPARK_RX_CAPTURE=N dumps N packets' HW timestamps + line
  // numbers to /tmp/spark_rx_capture.csv, to measure a sender's ST 2110-21 pacing (e.g. a BMD IP10 ref).
  struct CapPkt { uint64_t t_ns; uint16_t line; uint8_t nsrd; uint8_t marker; uint16_t len; };
  void cap_record(const spark::st2110::RxPacketInfo& info, const spark::net::RxPacket& pkt);
  uint32_t cap_target_ = 0;
  uint64_t cap_t0_ = 0;
  bool cap_done_ = false;
  std::vector<CapPkt> cap_;

  uint64_t packets_ = 0, frames_ = 0, lost_ = 0, bad_ = 0;
  uint32_t last_seq_ = 0;
  bool have_last_ = false;
  uint64_t lat_sum_ = 0, lat_cnt_ = 0, lat_min_ = UINT64_MAX, lat_max_ = 0;
  bool stats_printed_ = false;
  double last_live_s_ = 0;

  // audio ingest (poll-thread only): RTP packets relayed verbatim into AudioBridge with their
  // absolute capture time (RTP ts unwrapped against the PHC arrival timestamp).
  bool audio_enabled_ = false;
  uint64_t audio_pkts_ = 0, audio_bad_ = 0, audio_drop_ = 0;
  uint32_t audio_ts_delta_ = 0;      // latched broken-sender-epoch correction (0 = verbatim stamps)
  uint64_t audio_relatch_ = 0;       // times the sanity check re-latched (>0 = sender epoch broken)
  double audio_relatch_warn_s_ = 0;  // WARN rate-limit
};

}  // namespace spark::ops
