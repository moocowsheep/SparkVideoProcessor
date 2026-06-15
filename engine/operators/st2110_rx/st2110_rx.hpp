// St2110RxOp — Holoscan operator that receives an ST 2110-20 stream and measures it.
//
// Pipeline role (eventual):  [st2110_rx] -> unpack -> ...
// v1 scope: a terminal sink for loopback validation. It polls the NIC for `run_seconds`, depacketizes
// (RFC 4175), reassembles frames, and reports packet/frame counts, loss (RTP sequence gaps), and
// zero-copy ingest latency (NIC HW rx timestamp -> operator processing time, same PHC base). Driven
// by a CountCondition(1): compute() runs once and loops internally for the duration.
//
// Follow-on for the real pass-through: switch to one-frame-per-compute with an async/periodic
// condition and emit a VideoFrame on a "frame" output (then st2110_rx -> ... -> st2110_tx).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <holoscan/holoscan.hpp>

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

  holoscan::Parameter<std::string> pci_addr_;
  holoscan::Parameter<uint32_t> udp_port_;
  holoscan::Parameter<std::string> profile_;
  holoscan::Parameter<uint32_t> rxd_;
  holoscan::Parameter<std::string> eal_cores_;
  holoscan::Parameter<double> run_seconds_;

  std::unique_ptr<spark::net::ISt2110RxBackend> backend_;
  std::unique_ptr<spark::st2110::Depacketizer> depkt_;
  std::vector<uint8_t> frame_buf_;  // reassembly target (size == fmt.octets_per_frame())
  spark::st2110::VideoFormat fmt_;

  uint64_t packets_ = 0, frames_ = 0, lost_ = 0, bad_ = 0;
  uint32_t last_seq_ = 0;
  bool have_last_ = false;
  uint64_t lat_sum_ = 0, lat_cnt_ = 0, lat_min_ = UINT64_MAX, lat_max_ = 0;
  bool stats_printed_ = false;
};

}  // namespace spark::ops
