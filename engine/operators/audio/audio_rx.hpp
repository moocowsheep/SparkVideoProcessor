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

#include <holoscan/holoscan.hpp>

#include "../st2110_rx/rx_backend.hpp"
#include "audio_st2110.hpp"

namespace spark::ops {

class AudioRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AudioRxOp)
  AudioRxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override;
  void stop() override;

 private:
  void poll_loop();
  std::shared_ptr<std::vector<uint8_t>> next_buffer();

  holoscan::Parameter<std::string> pci_addr_;
  holoscan::Parameter<uint32_t> udp_port_;
  holoscan::Parameter<std::string> mcast_group_;
  holoscan::Parameter<std::string> src_ip_;
  holoscan::Parameter<std::string> iface_ip_;
  holoscan::Parameter<uint32_t> channels_;
  holoscan::Parameter<uint32_t> bit_depth_;      // 16 or 24
  holoscan::Parameter<double> packet_time_ms_;
  holoscan::Parameter<bool> manage_eal_;

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
