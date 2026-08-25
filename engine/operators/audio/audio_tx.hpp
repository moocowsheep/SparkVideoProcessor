// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// AudioTxOp — packetize each AudioBlock into one ST 2110-30 RTP packet and transmit it, tx_pp-paced
// (packets evenly spaced at the packet time). Mirrors St2110TxOp but one packet per block.
#pragma once

#include <cstdint>
#include <memory>

#include "runtime/runtime.hpp"

#include "../st2110_tx/tx_backend.hpp"
#include "audio_st2110.hpp"

namespace spark::ops {

class AudioTxOp : public spark::rt::Operator {
 public:
  SPARK_OPERATOR_FORWARD_ARGS(AudioTxOp)
  AudioTxOp() = default;

  void setup(spark::rt::OperatorSpec& spec) override;
  void start() override;
  void compute(spark::rt::InputContext& op_input, spark::rt::OutputContext&,
               spark::rt::ExecutionContext&) override;
  void stop() override;

 private:
  spark::rt::Parameter<std::string> pci_addr_;
  spark::rt::Parameter<std::string> dst_mac_;
  spark::rt::Parameter<std::string> src_ip_;
  spark::rt::Parameter<std::string> dst_ip_;
  spark::rt::Parameter<uint32_t> udp_port_;
  spark::rt::Parameter<uint32_t> ssrc_;
  spark::rt::Parameter<uint32_t> tx_pp_ns_;
  spark::rt::Parameter<uint32_t> txd_;
  spark::rt::Parameter<uint32_t> pacing_horizon_ns_;
  spark::rt::Parameter<bool> pacing_;
  spark::rt::Parameter<bool> manage_eal_;

  std::unique_ptr<spark::net::ISt2110TxBackend> backend_;
  std::unique_ptr<spark::st2110::AudioPacketizer> pktz_;
  uint64_t packet_time_ns_ = 1000000;  // set from the first block's packet time
  uint64_t schedule_base_ns_ = 0;
  uint64_t packets_sent_ = 0;
};

}  // namespace spark::ops
