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

#include <cstdint>
#include <memory>

#include <holoscan/holoscan.hpp>

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
  void emit_live(bool force = false);  // periodic "spark_live tx_*" line (1 Hz) for the daemon

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
  holoscan::Parameter<bool> manage_eal_;   // false in multi-backend processes (shared DpdkEal)
  holoscan::Parameter<uint32_t> warmup_ms_;  // one-time delay before first send (let RX start first)

  // --- runtime state ---
  std::unique_ptr<spark::net::ISt2110TxBackend> backend_;
  std::unique_ptr<spark::st2110::Packetizer> pktz_;
  uint64_t gap_ns_ = 0;             // per-packet pacing interval (even/linear)
  // Gapped (2110TPN) pacing for IP10: each line's packets burst early in its line slot then idle,
  // matching the Blackmagic reference. pace_gapped_ enables it; else even gap_ns_ is used.
  bool pace_gapped_ = false;
  uint64_t t_line_ns_ = 0;          // duration of one raster line slot (T_active / active_lines)
  uint64_t intra_gap_ns_ = 0;       // spacing between a line's packets (bunched within the slot)
  uint64_t frame_interval_ns_ = 0;  // 1e9 / fps
  uint64_t schedule_base_ns_ = 0;   // send time of the current frame's first packet
  uint64_t media_ts_ns_ = 0;        // monotonic RTP media clock (anchored once; +interval per frame)
  uint64_t frames_sent_ = 0;
  uint64_t packets_sent_ = 0;
  uint64_t reanchors_ = 0;  // grid resyncs (should be near-zero; high = a real upstream stall)
  uint64_t genlock_offset_ = 0;  // send_base = source capture_ts + this (calibrated once; GM-locked => const)
  bool throttle_enabled_ = true;  // self-disables if the spin caps out (likely now_ns() unit issue)
  bool warmed_ = false;
  double last_live_s_ = 0;
};

}  // namespace spark::ops
