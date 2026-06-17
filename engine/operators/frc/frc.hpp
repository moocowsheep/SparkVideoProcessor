// FrcOp — motion-compensated frame interpolation (the headline FRC stage). Holds the previous frame;
// for each new frame it computes NVOF optical flow (prev->cur) and warps/blends interpolated frames.
// Pipeline role:  ... resize -> [frc] -> pack ...
//
// Two modes (rate_mult):
//   1 (retime): emit one motion-compensated frame per input at phase `phase` (1:1). Synthesizes
//     every output frame — only useful as a retimer; in a 1:1 path it just adds OFA warp.
//   2 (up-convert): emit the REAL frame PLUS a motion-compensated midpoint per input -> 2x output
//     rate (e.g. 30->60). Real frames are preserved; only the new in-between frames are synthetic,
//     so interpolation does genuine work and artifacts land only on alternate (mid) frames. This
//     emits two messages per compute on "out", so the downstream pack input is sized to buffer them
//     and the TX runs at out_fps = 2x in_fps (see st2110_pipeline + the tx_pp shock-absorber ring).
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
  holoscan::Parameter<uint32_t> grid_size_;  // NVOF output grid (1|2|4; 1 = finest = best quality)
  holoscan::Parameter<uint32_t> rate_mult_;  // 1 = retime (1:1), 2 = up-convert (real + mid -> 2x)

  spark::frc::NvofFlow flow_;
  bool inited_ = false;
  cudaStream_t stream_ = nullptr;  // FRC's own CUDA stream (NVOF + interpolate); pipelined vs other ops
  uint8_t* prevY8_ = nullptr;
  uint8_t* curY8_ = nullptr;
  float* wmap_ = nullptr;  // per-pixel occlusion blend weight (luma res), shared luma->chroma per frame
  spark::gpu::GpuFramePtr prev_;
  std::vector<spark::gpu::GpuFramePtr> pool_;
  size_t idx_ = 0;
  uint64_t frames_ = 0;
  double last_live_s_ = 0;
};

}  // namespace spark::ops
