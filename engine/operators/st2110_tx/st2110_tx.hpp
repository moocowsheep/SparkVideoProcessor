// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// St2110TxOp — Holoscan sink operator that transmits VideoFrames as ST 2110-20, tx_pp-paced.
//
// Pipeline role:  ... -> pack -> [st2110_tx]
// It packetizes each frame (RFC 4175, see rtp_st2110.hpp) and emits the packets through a transport
// backend (raw DPDK/mlx5 tx_pp v1; advanced_network later) at HW-scheduled send times.
//
// The pacing insight from gate 4 (docs/M1-gate4-pacing.md): the NIC's tx_pp scheduling window is
// small, so we must NOT hand the NIC a whole frame of future timestamps at once (that floods with
// future_errors, exactly the testpmd failure). Instead compute() self-throttles its submission to
// keep the in-flight schedule within `pacing_horizon_ns` of the NIC clock — the closed-loop feeding a
// real media generator can do and testpmd's open-loop txonly could not. compute() therefore runs for
// ~one frame interval per frame (correct for a real-time sender), backpressuring the upstream graph.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

#include <holoscan/holoscan.hpp>

#include "rtp_jxs.hpp"
#include "rtp_st2110.hpp"
#include "st2110_format.hpp"
#include "tx_backend.hpp"

namespace spark::ops {

class St2110TxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(St2110TxOp)
  St2110TxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;

 private:
  void ensure_pacer(const spark::st2110::VideoFormat& fmt);
  // JPEG XS egress: fragment + submit one codestream on the schedule starting at `base`. Split out
  // because it shares nothing with the raster walk — no lines, no SRDs, and a per-frame packet count.
  void send_jxs_frame(const spark::st2110::VideoFrame& frame, uint64_t base, uint64_t now, bool pace);
  void emit_live(bool force = false);  // periodic "spark_live tx_*" line (1 Hz) for the daemon
  void audio_loop();  // relay thread: AudioBridge -> capture+L -> tx_pp queue 1 (fixed mode only)

  // --- parameters ---
  holoscan::Parameter<std::string> pci_addr_;
  holoscan::Parameter<std::string> dst_mac_;   // "aa:bb:cc:dd:ee:ff" (gate-4: the RX port MAC)
  holoscan::Parameter<std::string> src_ip_;
  holoscan::Parameter<std::string> dst_ip_;
  holoscan::Parameter<uint32_t> udp_port_;  // uint32 for YAML/from_config friendliness
  holoscan::Parameter<uint32_t> payload_size_;     // UDP payload budget per packet (octets)
  holoscan::Parameter<uint32_t> tx_pp_ns_;
  holoscan::Parameter<uint32_t> txd_;
  holoscan::Parameter<uint32_t> pacing_horizon_ns_;  // keep schedule within this of the NIC clock
  // Lead the schedule base over the NIC clock on (re)anchor. 0 = use pacing_horizon_ns_ (legacy). A
  // SMALL lead with a LARGE horizon lets compute() submit a whole frame to the NIC at once and return
  // in ~ms — the NIC tx_pp HW-paces it, leaving ~a frame of slack so jitter never lags the grid.
  holoscan::Parameter<uint32_t> reanchor_lead_ns_;
  holoscan::Parameter<uint32_t> ssrc_;
  holoscan::Parameter<std::string> eal_cores_;
  holoscan::Parameter<bool> pacing_;
  holoscan::Parameter<double> pacing_fill_;  // spread a frame over fill×interval (<1 finishes early)
  // ST 2110-21 sender compliance profile. false = Narrow (2110TPN): pace over the ACTIVE period with
  // per-line gapped bursts (matches the Blackmagic reference; validated for 2160p59.94 IP10). true =
  // Wide (2110TPW): pace EVENLY over the full frame interval — lower peak rate, no gaps — for receivers
  // (e.g. BiDirect-2) whose larger wide buffer absorbs tx_pp/pipeline jitter that narrow's tight buffer
  // couldn't (the 29.97 dips). Keep this in sync with the SDP's TP= (NMOS node reads the same SPARK_TX_TP).
  holoscan::Parameter<bool> tx_wide_;
  holoscan::Parameter<bool> manage_eal_;   // false in multi-backend processes (shared DpdkEal)
  holoscan::Parameter<uint32_t> warmup_ms_;  // one-time delay before first send (let RX start first)
  // Latency trim: max ns/frame the genlock base (and the RTP media clock with it) may slew toward the
  // target lead. Converges end-to-end latency to (steady pipeline delay + reanchor_lead) after startup
  // transients instead of freezing whatever the cold first frame baked in. 0 = off (legacy behavior).
  holoscan::Parameter<uint32_t> trim_ns_;
  // FIXED end-to-end latency (M10): when > 0, wire time = absolute capture_ts + latency_ns — a
  // CONFIGURED constant. No calibration, no trim servo, no re-anchor: latency is deterministic
  // across restarts, and frames that miss the schedule skip (freeze) rather than shift it. Requires
  // the absolute capture timestamps from the RX RTP-unwrap (i.e. a PTP-locked source + synced PHC).
  // 0 = the adaptive servo above (legacy). The audio relay uses the SAME constant -> exact lip-sync.
  holoscan::Parameter<uint64_t> latency_ns_;
  holoscan::Parameter<std::string> audio_dst_ip_;  // companion 2110-30 egress group ("" = no audio)
  holoscan::Parameter<uint32_t> audio_port_;
  holoscan::Parameter<uint32_t> audio_rate_;       // audio RTP clock (2110-30: 48000)
  holoscan::Parameter<int64_t> av_offset_ns_;      // extra audio delay (+) / advance (-) vs video

  // --- runtime state ---
  std::unique_ptr<spark::net::ISt2110TxBackend> backend_;
  std::unique_ptr<spark::st2110::Packetizer> pktz_;
  // ST 2110-22 egress. Exactly one of pktz_/jxs_pktz_ is set, chosen from the incoming frame's
  // VideoFormat::codec at ensure_pacer() time.
  std::unique_ptr<spark::st2110::JxsPacketizer> jxs_pktz_;
  double jxs_pace_span_ = 0.0;      // fraction of the frame interval a codestream is spread over
  uint64_t gap_ns_ = 0;             // per-packet pacing interval (even/linear)
  // Gapped (2110TPN) pacing for IP10: each line's packets burst early in its line slot then idle,
  // matching the Blackmagic reference. pace_gapped_ enables it; else even gap_ns_ is used.
  bool pace_gapped_ = false;
  // Per-line-burst pacing (wide 2160p): HW-timestamp only the first packet of each line; the rest send
  // back-to-back (send_ts=0). Cuts mlx5 send-on-timestamp WQEs ~7x vs per-packet (which cost ~25ms/frame
  // and capped compute() at the frame budget). The wide receiver buffer absorbs the intra-line burst.
  bool pace_line_burst_ = false;
  uint64_t t_line_ns_ = 0;          // duration of one raster line slot (T_active / active_lines)
  uint64_t intra_gap_ns_ = 0;       // spacing between a line's packets (bunched within the slot)
  uint64_t frame_interval_ns_ = 0;  // 1e9 / fps
  uint64_t eff_horizon_ns_ = 0;     // throttle horizon, scaled up at low frame rates so compute() keeps up
  uint64_t schedule_base_ns_ = 0;   // send time of the current frame's first packet
  uint64_t media_ts_ns_ = 0;        // monotonic RTP media clock (anchored once; +interval per frame)
  uint64_t frames_sent_ = 0;
  uint64_t packets_sent_ = 0;
  uint64_t reanchors_ = 0;  // grid resyncs (should be near-zero; high = a real upstream stall)
  uint64_t skipped_late_ = 0;  // late frames dropped unsent to purge queue backlog (genlock catch-up)
  uint64_t codec_mismatch_ = 0;  // frames refused: wire codec != the latched packetizer (wiring bug)
  uint32_t skip_streak_ = 0;   // consecutive late-skips; caps at ~32 then re-anchors (new latency floor)
  // Skip-rate window (two 64-frame buckets) for the chronic-clipping guard, and the AIMD trim
  // state. Lead-based trim servos were tried and removed: with standing upstream queues the
  // measured lead is offset-INVARIANT (frames arrive when TX frees up), so min-lead gates read
  // noise and reverse trim ratchets. Trim is skip-gated instead: shave after a long clean run,
  // back off 4ms on the skip that probes the floor.
  uint32_t win_count_ = 0;
  uint32_t win_skips_ = 0;         // shallow-late skips this window (chronic-clipping guard input)
  uint32_t win_skips_prev_ = 0;    // previous bucket; >25% over both windows -> re-anchor once
  uint32_t frames_since_skip_ = 0;   // consecutive clean sends; > trim_arm_frames_ arms the shave
  uint32_t trim_arm_frames_ = 512;   // AIMD re-arm delay; doubles per floor probe (cap 8192)
  int64_t shaved_since_probe_ = 0;   // shave total since last probe; half is given back on probe
  uint64_t genlock_offset_ = 0;  // send_base = source capture_ts + this (calibrated once; GM-locked => const)
  bool throttle_enabled_ = true;  // self-disables if the spin caps out (likely now_ns() unit issue)
  bool warmed_ = false;
  double last_live_s_ = 0;
  // Latency telemetry (1 Hz spark_live line + a trim servo): lead = schedule base over the NIC clock
  // at compute time (the TX-side jitter margin AND queue+GPU slack — end-to-end latency rides on it);
  // trim_total = how much the servo has pulled out of the baked-in genlock offset so far.
  int64_t last_lead_ns_ = 0;
  int64_t trim_total_ns_ = 0;  // net latency trimmed (signed: reverse trim subtracts)
  bool genlocked_ = false;  // last frame used the capture_ts genlock path (media clock may slew)

  // --- fixed-latency mode state ---
  bool fixed_active_ = false;      // this frame is on the fixed schedule (per-frame)
  bool fixed_checked_ = false;     // one-time clock-consistency check ran
  uint32_t fixed_late_streak_ = 0; // consecutive frames under the skip floor (L below the pipe floor)
  int64_t min_lead_win_ = INT64_MAX;  // min lead since the last live line -> tx_margin_us telemetry

  // --- async NIC stats (emit_live) ---
  // backend_->stats() sweeps the mlx5 xstats via firmware mailbox round-trips (~10-20 ms measured)
  // — far more than the pacing slack, so it must NEVER run on the compute/pacing thread: sampled
  // there it debited the 1 Hz frame's lead and fixed mode skipped exactly that frame every second.
  // A poller thread refreshes the cache at 1 Hz; emit_live() only copies it under the mutex.
  std::thread stats_thread_;
  std::atomic<bool> stop_stats_{false};
  std::mutex stats_mtx_;
  spark::net::TxStats stats_cache_{};

  // --- audio relay (fixed mode only) ---
  std::thread audio_thread_;
  std::atomic<bool> stop_audio_{false};
  std::atomic<uint64_t> audio_tx_pkts_{0};
  std::atomic<uint64_t> audio_late_{0};   // sent past due (immediate, unpaced)
  std::atomic<uint64_t> audio_drop_{0};   // >50 ms late — dropped
  bool audio_on_ = false;
};

}  // namespace spark::ops
