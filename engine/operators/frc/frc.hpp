// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

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
// Plus a timing variant (out_interval_ns > 0, SPARK_FRC=3): UNIFORM-GRID up-convert. Modes 1/2 are
//   source-locked — output capture_ts inherits every wobble of the source's frame spacing, so an
//   erratic source (e.g. a BMD with a bad reference) shows as motion wobble on air. Uniform mode
//   lays a rigid nominal output grid over the capture timeline and emits one frame per grid tick,
//   interpolated at the tick's TRUE phase between the actual capture times of the bracketing source
//   frames (real frames pass through when a tick lands on one). Output cadence is exactly nominal
//   regardless of source wobble; a source stall freezes output for the gap instead of synthesizing
//   fictitious motion. See frc_grid.hpp for the tick math.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "runtime/runtime.hpp"

#include "../resize/gpu_frame.hpp"
#include "frc_grid.hpp"
#include "frc_kernels.hpp"
#include "nvof_flow.hpp"

namespace spark::ops {

class FrcOp : public spark::rt::Operator {
 public:
  SPARK_OPERATOR_FORWARD_ARGS(FrcOp)
  FrcOp() = default;
  void setup(spark::rt::OperatorSpec& spec) override;
  void compute(spark::rt::InputContext& op_input, spark::rt::OutputContext& op_output,
               spark::rt::ExecutionContext& context) override;
  void stop() override;

 private:
  void ensure(uint32_t width, uint32_t height);
  void emit_live(bool force = false);  // periodic "spark_live frc_interpolated" line (1 Hz)
  void run_flow(const spark::gpu::GpuFramePtr& cur);  // NVOF prev_<->cur on stream_ (async)
  void compute_uniform(const spark::gpu::GpuFramePtr& cur, spark::rt::OutputContext& op_output);

  spark::rt::Parameter<double> phase_;        // interpolation t in [0,1] (0.5 = midpoint)
  spark::rt::Parameter<uint32_t> grid_size_;  // NVOF output grid (1|2|4; 1 = finest = best quality)
  spark::rt::Parameter<uint32_t> rate_mult_;  // 1 = retime (1:1), 2 = up-convert (real + mid -> 2x)
  spark::rt::Parameter<uint64_t> out_interval_ns_;  // >0 = uniform-grid mode at this output interval

  spark::frc::UniformGrid grid_;   // uniform-mode tick state (anchored on the first bracket)
  uint64_t src_prev_ts_ = 0;       // prev_'s ORIGINAL capture_ts (prev_ itself may be re-stamped)
  uint64_t grid_dropped_ = 0;      // ticks dropped (stall jumps / over-burst brackets)
  uint32_t burst_ = 1;             // max emits per compute (== pipeline_emit_burst(), cached)
  spark::frc::NvofFlow flow_;
  spark::frc::FlowView fview_{};  // what interpolate() consumes (raw or median-filtered flow)
  bool median_ = true;            // 3x3 flow median pre-filter (SPARK_FRC_MEDIAN=0 disables)
  bool inited_ = false;
  cudaStream_t stream_ = nullptr;  // FRC's own CUDA stream (NVOF + interpolate); pipelined vs other ops
  uint8_t* prevY8_ = nullptr;
  uint8_t* curY8_ = nullptr;
  short* flow_med_f_ = nullptr;  // median-filtered flow grids (packed SHORT2, grid_w x grid_h)
  short* flow_med_b_ = nullptr;
  spark::frc::WarpWorkspace ws_{};  // splat scratch (accum + phase-t grids), owned here
  float* wmap_ = nullptr;  // per-pixel occlusion blend weight (luma res), shared luma->chroma per frame
  spark::gpu::GpuFramePtr prev_;
  std::vector<spark::gpu::GpuFramePtr> pool_;
  size_t idx_ = 0;
  uint64_t frames_ = 0;
  double last_live_s_ = 0;
};

}  // namespace spark::ops
