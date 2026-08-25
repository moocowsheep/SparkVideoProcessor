// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// TestPatternOp — synthetic VideoFrame source for ST 2110 TX bring-up (no RX path needed).
//
// Emits a fixed packed ST 2110-20 frame buffer each compute(), stamping a synthetic capture time that
// advances by one frame interval (so RTP timestamps are well-formed). The actual transmit rate is set
// downstream by St2110TxOp's pacing, which backpressures this source — so a CountCondition bounds the
// run length and the TX operator governs cadence. Swap this out for st2110_rx once that exists.
#pragma once

#include <chrono>
#include <cstdint>

#include "runtime/runtime.hpp"

#include "../st2110_tx/st2110_format.hpp"

namespace spark::ops {

class TestPatternOp : public spark::rt::Operator {
 public:
  SPARK_OPERATOR_FORWARD_ARGS(TestPatternOp)
  TestPatternOp() = default;

  void setup(spark::rt::OperatorSpec& spec) override;
  void start() override;
  void compute(spark::rt::InputContext& op_input, spark::rt::OutputContext& op_output,
               spark::rt::ExecutionContext& context) override;

 private:
  spark::rt::Parameter<std::string> profile_;  // "1080p" | "2160p"

  spark::st2110::VideoFormat fmt_;
  std::shared_ptr<std::vector<uint8_t>> buffer_;
  uint64_t frame_number_ = 0;
  uint64_t base_ts_ns_ = 0;
  uint64_t frame_interval_ns_ = 0;
};

}  // namespace spark::ops
