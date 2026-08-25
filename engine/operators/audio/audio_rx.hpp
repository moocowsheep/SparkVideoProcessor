// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// AudioRxOp — receive an ST 2110-30 stream and emit one AudioBlock per RTP packet (source mode).
// Mirrors St2110RxOp's emit path: a dedicated thread drains the NIC into a bounded queue so polling
// never stalls while the downstream paces; compute() pops one block and emits it.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "runtime/runtime.hpp"

#include "../st2110_rx/rx_backend.hpp"
#include "audio_st2110.hpp"

namespace spark::ops {

class AudioRxOp : public spark::rt::Operator {
 public:
  SPARK_OPERATOR_FORWARD_ARGS(AudioRxOp)
  AudioRxOp() = default;

  void setup(spark::rt::OperatorSpec& spec) override;
  void start() override;
  void compute(spark::rt::InputContext&, spark::rt::OutputContext& op_output,
               spark::rt::ExecutionContext&) override;
  void stop() override;

 private:
  void poll_loop();
  std::shared_ptr<std::vector<uint8_t>> next_buffer();

  spark::rt::Parameter<std::string> pci_addr_;
  spark::rt::Parameter<uint32_t> udp_port_;
  spark::rt::Parameter<std::string> mcast_group_;
  spark::rt::Parameter<std::string> src_ip_;
  spark::rt::Parameter<std::string> iface_ip_;
  spark::rt::Parameter<uint32_t> channels_;
  spark::rt::Parameter<uint32_t> bit_depth_;      // 16 or 24
  spark::rt::Parameter<double> packet_time_ms_;
  spark::rt::Parameter<bool> manage_eal_;

  std::unique_ptr<spark::net::ISt2110RxBackend> backend_;
  std::unique_ptr<spark::st2110::AudioDepacketizer> depkt_;
  spark::st2110::AudioFormat fmt_;

  std::vector<std::shared_ptr<std::vector<uint8_t>>> buf_ring_;
  size_t buf_idx_ = 0;

  std::thread poll_thread_;
  std::atomic<bool> stop_poll_{false};
  std::mutex q_mu_;
  std::condition_variable q_cv_;
  std::deque<spark::st2110::AudioBlock> q_;
  uint64_t packets_ = 0, dropped_ = 0;
};

}  // namespace spark::ops
