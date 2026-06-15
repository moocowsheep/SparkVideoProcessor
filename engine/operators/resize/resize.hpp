// ResizeOp — scales a GpuFrame (planar YCbCr 4:2:2 10-bit) to a target resolution via NPP, plus a
// synthetic GPU source and sink for standalone benchmarking. Pipeline role:  ... unpack -> [resize]
// -> frc ... The M0 spike measured nppiResize 1080p->2160p 16u at ~0.1 ms/frame; this is that as a
// real operator (Y at full res, Cb/Cr at half width).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <npp.h>  // NppStreamContext (CUDA 13 NPP exposes only the _Ctx primitive variants)

#include "gpu_frame.hpp"

namespace spark::ops {

class ResizeOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ResizeOp)
  ResizeOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;

 private:
  holoscan::Parameter<uint32_t> out_width_;
  holoscan::Parameter<uint32_t> out_height_;
  holoscan::Parameter<std::string> interp_;  // linear | cubic | lanczos
  holoscan::Parameter<bool> measure_;  // per-frame cudaEvent timing (benchmark only; a sync/frame)

  int interp_code_ = 0;
  NppStreamContext npp_ctx_{};
  std::vector<spark::gpu::GpuFramePtr> pool_;  // output ring (target dims)
  size_t pool_idx_ = 0;
  void* ev_start_ = nullptr;  // cudaEvent_t (opaque here to keep CUDA out of the header)
  void* ev_stop_ = nullptr;

  uint64_t frames_ = 0;
  double ms_sum_ = 0.0, ms_min_ = 1e30, ms_max_ = 0.0;
};

// Synthetic GPU frame source: emits a gradient-filled GpuFrame at the input resolution each compute().
class TestGpuSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TestGpuSourceOp)
  TestGpuSourceOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<std::string> profile_;  // 1080p | 2160p (input size)
  spark::gpu::GpuFramePtr frame_;
  uint64_t n_ = 0;
};

// Trivial sink: synchronizes the device (so resize timing is real) and counts frames.
class GpuFrameSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GpuFrameSinkOp)
  GpuFrameSinkOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  uint64_t count_ = 0;
};

}  // namespace spark::ops
