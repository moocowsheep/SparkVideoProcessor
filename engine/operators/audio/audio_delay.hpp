// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// AudioDelayOp — holds audio by a fixed lip-sync delay so it stays aligned with the video path's
// added latency (resize + FRC). One AudioBlock in, zero-or-one (the delayed block) out.
#pragma once

#include <cstdint>
#include <memory>

#include "runtime/runtime.hpp"

#include "audio_st2110.hpp"

namespace spark::ops {

class AudioDelayOp : public spark::rt::Operator {
 public:
  SPARK_OPERATOR_FORWARD_ARGS(AudioDelayOp)
  AudioDelayOp() = default;

  void setup(spark::rt::OperatorSpec& spec) override;
  void compute(spark::rt::InputContext& op_input, spark::rt::OutputContext& op_output,
               spark::rt::ExecutionContext&) override;

 private:
  spark::rt::Parameter<double> delay_ms_;  // lip-sync delay; >= the video processing latency
  std::unique_ptr<spark::st2110::AudioDelayLine> line_;  // sized from the first block's packet time
};

}  // namespace spark::ops
