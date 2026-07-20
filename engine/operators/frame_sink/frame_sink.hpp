// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// FrameSinkOp — trivial VideoFrame sink (drops frames, counts them). Header-only.
//
// st2110_rx always declares a "frame" output (it can't know its mode until after setup()), and
// Holoscan only ticks an operator whose output is connected. In sink-mode apps (loopback, rx_smoke)
// st2110_rx never actually emits, but its output must be wired somewhere — that's this. In emit mode
// st2110_rx feeds a real consumer (st2110_tx) instead.
#pragma once

#include <cstdint>

#include <holoscan/holoscan.hpp>

#include "../st2110_tx/st2110_format.hpp"

namespace spark::ops {

class FrameSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FrameSinkOp)
  FrameSinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<spark::st2110::VideoFrame>("frame");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    if (op_input.receive<spark::st2110::VideoFrame>("frame")) ++count_;
  }

 private:
  uint64_t count_ = 0;
};

}  // namespace spark::ops
