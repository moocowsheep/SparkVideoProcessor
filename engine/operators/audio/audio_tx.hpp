// AudioTxOp — packetize each AudioBlock into one ST 2110-30 RTP packet and transmit it, tx_pp-paced
// (packets evenly spaced at the packet time). Mirrors St2110TxOp but one packet per block.
#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/holoscan.hpp>

#include "../st2110_tx/tx_backend.hpp"
#include "audio_st2110.hpp"

namespace spark::ops {

class AudioTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AudioTxOp)
  AudioTxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override;
  void stop() override;

 private:
  holoscan::Parameter<std::string> pci_addr_;
  holoscan::Parameter<std::string> dst_mac_;
  holoscan::Parameter<std::string> src_ip_;
  holoscan::Parameter<std::string> dst_ip_;
  holoscan::Parameter<uint32_t> udp_port_;
  holoscan::Parameter<uint32_t> ssrc_;
  holoscan::Parameter<uint32_t> tx_pp_ns_;
  holoscan::Parameter<uint32_t> txd_;
  holoscan::Parameter<uint32_t> pacing_horizon_ns_;
  holoscan::Parameter<bool> pacing_;
  holoscan::Parameter<bool> manage_eal_;

  std::unique_ptr<spark::net::ISt2110TxBackend> backend_;
  std::unique_ptr<spark::st2110::AudioPacketizer> pktz_;
  uint64_t packet_time_ns_ = 1000000;  // set from the first block's packet time
  uint64_t schedule_base_ns_ = 0;
  uint64_t packets_sent_ = 0;
};

}  // namespace spark::ops
