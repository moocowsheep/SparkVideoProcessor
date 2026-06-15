// FrcOp — motion-compensated frame interpolation (the headline FRC stage). Holds the previous frame;
// for each new frame it computes NVOF optical flow (prev->cur) and warps/blends an interpolated frame
// at phase `phase` in [0,1]. Pipeline role:  ... resize -> [frc] -> pack ...
//
// v1 emits one interpolated frame per input (1:1, motion-compensated retiming) — the core capability.
// A 2x rate-conversion mode (emit prev + mid) is a follow-on once TX pacing handles the rate change.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "../resize/gpu_frame.hpp"
#include "nvof_flow.hpp"

namespace spark::ops {

class FrcOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FrcOp)
  FrcOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;

 private:
  void ensure(uint32_t width, uint32_t height);
  void emit_live(bool force = false);  // periodic "spark_live frc_interpolated" line (1 Hz)

  holoscan::Parameter<double> phase_;        // interpolation t in [0,1] (0.5 = midpoint)
  holoscan::Parameter<uint32_t> grid_size_;  // NVOF output grid (1|2|4)

  spark::frc::NvofFlow flow_;
  bool inited_ = false;
  uint8_t* prevY8_ = nullptr;
  uint8_t* curY8_ = nullptr;
  spark::gpu::GpuFramePtr prev_;
  std::vector<spark::gpu::GpuFramePtr> pool_;
  size_t idx_ = 0;
  uint64_t frames_ = 0;
  double last_live_s_ = 0;
};

}  // namespace spark::ops
