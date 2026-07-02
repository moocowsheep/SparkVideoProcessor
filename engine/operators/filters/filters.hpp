// ProcAmpOp / SharpenOp — GpuFrame -> GpuFrame video filters for the modular processing chain.
// Pipeline role:  unpack -> [frc] -> [resize] -> [sharpen] -> [procamp] -> pack  (any subset active;
// st2110_pipeline composes the chain from config). Both follow the house operator contract: own CUDA
// stream, cudaStreamWaitEvent on the producer's ready event, out-of-place into a private pool (frames
// are shared downstream/held by FRC, so in-place mutation would corrupt other stages' references),
// record dst->ready, carry t_ingest_ns/capture_ts_ns through.
#pragma once

#include <cstdint>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "../resize/gpu_frame.hpp"

namespace spark::ops {

// Classic video proc amp on the native 10-bit YCbCr planes. Neutral = (0, 1, 1, 0); the pipeline
// only inserts the stage when a parameter is non-neutral (or SPARK_FILTERS lists it explicitly).
class ProcAmpOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ProcAmpOp)
  ProcAmpOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;

 private:
  void ensure(uint32_t width, uint32_t height);
  holoscan::Parameter<double> brightness_;  // black-level offset, ±1.0 = ±full Y swing; 0 = neutral
  holoscan::Parameter<double> contrast_;    // video gain about black (64); 1 = neutral
  holoscan::Parameter<double> saturation_;  // chroma gain about 512; 1 = neutral
  holoscan::Parameter<double> hue_deg_;     // chroma phase rotation, degrees; 0 = neutral
  cudaStream_t stream_ = nullptr;
  std::vector<spark::gpu::GpuFramePtr> pool_;
  size_t idx_ = 0;
  uint64_t frames_ = 0;
};

// Luma unsharp mask (broadcast detail-enhance). Chroma passes through by device copy.
class SharpenOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SharpenOp)
  SharpenOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;

 private:
  void ensure(uint32_t width, uint32_t height);
  holoscan::Parameter<double> amount_;  // unsharp gain; 0 = identity (stage normally omitted then)
  cudaStream_t stream_ = nullptr;
  std::vector<spark::gpu::GpuFramePtr> pool_;
  size_t idx_ = 0;
  uint64_t frames_ = 0;
};

}  // namespace spark::ops
