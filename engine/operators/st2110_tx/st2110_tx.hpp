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
  holoscan::Parameter<uint32_t> ssrc_;
  holoscan::Parameter<std::string> eal_cores_;
  holoscan::Parameter<bool> pacing_;

  // --- runtime state ---
  std::unique_ptr<spark::net::ISt2110TxBackend> backend_;
  std::unique_ptr<spark::st2110::Packetizer> pktz_;
  uint64_t gap_ns_ = 0;             // per-packet pacing interval
  uint64_t frame_interval_ns_ = 0;  // 1e9 / fps
  uint64_t schedule_base_ns_ = 0;   // send time of the current frame's first packet
  uint64_t frames_sent_ = 0;
  uint64_t packets_sent_ = 0;
};

}  // namespace spark::ops
