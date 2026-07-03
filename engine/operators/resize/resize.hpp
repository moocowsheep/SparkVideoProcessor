// ResizeOp — scales a GpuFrame (planar YCbCr 4:2:2 10-bit) to a target resolution via NPP, plus a
// synthetic GPU source and sink for standalone benchmarking. Pipeline role:  ... unpack -> [resize]
// -> frc ... The M0 spike measured nppiResize 1080p->2160p 16u at ~0.1 ms/frame; this is that as a
// real operator (Y at full res, Cb/Cr at half width).
//
// interp also selects AI super-resolution ("fsrcnn" | "fsrcnn-s" | "espcn", alias "ai" = fsrcnn):
// an embedded pre-trained x2 CNN runs on the luma plane (sr_net.hpp) with cubic chroma. AI models
// are exact-2x only — any other geometry falls back to cubic (one-shot warn).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <npp.h>  // NppStreamContext (CUDA 13 NPP exposes only the _Ctx primitive variants)

#include "gpu_frame.hpp"
#include "sr_net.hpp"

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
  void ensure_sr(uint32_t in_width, uint32_t in_height);  // (re)build the SR engine for these dims

  holoscan::Parameter<uint32_t> out_width_;
  holoscan::Parameter<uint32_t> out_height_;
  holoscan::Parameter<std::string> interp_;  // auto | linear | cubic | lanczos | super | AI names
  holoscan::Parameter<bool> measure_;  // per-frame cudaEvent timing (benchmark only; a sync/frame)

  int interp_code_ = 0;
  std::string sr_model_;  // non-empty = AI SR luma path ("fsrcnn" | "fsrcnn-s" | "espcn")
  std::unique_ptr<spark::sr::Engine> sr_engine_;  // lazy: input dims known at first frame
  bool sr_fallback_logged_ = false;  // one-shot "not exact 2x -> cubic" warn
  cudaStream_t stream_ = nullptr;  // own CUDA stream (NPP resize); pipelined vs other ops
  NppStreamContext npp_ctx_{};
  std::vector<spark::gpu::GpuFramePtr> pool_;  // output ring (target dims)
  size_t pool_idx_ = 0;
  void* ev_start_ = nullptr;  // cudaEvent_t (opaque here to keep CUDA out of the header)
  void* ev_stop_ = nullptr;

  uint64_t frames_ = 0;
  double ms_sum_ = 0.0, ms_min_ = 1e30, ms_max_ = 0.0;
  bool identity_logged_ = false;  // one-shot log when the 1:1 passthrough engages
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
