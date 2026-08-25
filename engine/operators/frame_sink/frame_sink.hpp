// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// FrameSinkOp — trivial VideoFrame sink (drops frames, counts them). Header-only.
//
// st2110_rx always declares a "frame" output (it can't know its mode until after setup()). Holoscan
// only ticked an operator whose output was connected, so sink-mode apps (loopback, rx_smoke) had to
// wire that output somewhere — this. spark::rt ticks every operator regardless and discards emits on
// an unconnected port, so this is no longer load-bearing; it is kept because it also counts frames,
// which those apps report. In emit mode st2110_rx feeds a real consumer (st2110_tx) instead.
#pragma once

#include <cstdint>

#include "runtime/runtime.hpp"

#include "../st2110_tx/st2110_format.hpp"

namespace spark::ops {

class FrameSinkOp : public spark::rt::Operator {
 public:
  SPARK_OPERATOR_FORWARD_ARGS(FrameSinkOp)
  FrameSinkOp() = default;

  void setup(spark::rt::OperatorSpec& spec) override {
    spec.input<spark::st2110::VideoFrame>("frame");
  }

  void compute(spark::rt::InputContext& op_input, spark::rt::OutputContext&,
               spark::rt::ExecutionContext&) override {
    if (op_input.receive<spark::st2110::VideoFrame>("frame")) ++count_;
  }

 private:
  uint64_t count_ = 0;
};

}  // namespace spark::ops
