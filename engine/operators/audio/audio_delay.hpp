// AudioDelayOp — holds audio by a fixed lip-sync delay so it stays aligned with the video path's
// added latency (resize + FRC). One AudioBlock in, zero-or-one (the delayed block) out.
#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/holoscan.hpp>

#include "audio_st2110.hpp"

namespace spark::ops {

class AudioDelayOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AudioDelayOp)
  AudioDelayOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override;

 private:
  holoscan::Parameter<double> delay_ms_;  // lip-sync delay; >= the video processing latency
  std::unique_ptr<spark::st2110::AudioDelayLine> line_;  // sized from the first block's packet time
};

}  // namespace spark::ops
