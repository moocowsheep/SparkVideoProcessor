// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// ProcAmpOp / SharpenOp — GpuFrame -> GpuFrame video filters for the modular processing chain.
// Pipeline role:  unpack -> [frc] -> [resize] -> [sharpen] -> [procamp] -> pack  (any subset active;
// st2110_pipeline composes the chain from config). Both follow the house operator contract: own CUDA
// stream, cudaStreamWaitEvent on the producer's ready event, out-of-place into a private pool (frames
// are shared downstream/held by FRC, so in-place mutation would corrupt other stages' references),
// record dst->ready, carry t_ingest_ns/capture_ts_ns through.
#pragma once

#include <cstdint>
#include <vector>

#include "runtime/runtime.hpp"

#include "../resize/gpu_frame.hpp"

namespace spark::ops {

// Classic video proc amp on the native 10-bit YCbCr planes. Neutral = (0, 1, 1, 0); the pipeline
// only inserts the stage when a parameter is non-neutral (or SPARK_FILTERS lists it explicitly).
class ProcAmpOp : public spark::rt::Operator {
 public:
  SPARK_OPERATOR_FORWARD_ARGS(ProcAmpOp)
  ProcAmpOp() = default;
  void setup(spark::rt::OperatorSpec& spec) override;
  void compute(spark::rt::InputContext& op_input, spark::rt::OutputContext& op_output,
               spark::rt::ExecutionContext& context) override;
  void stop() override;

 private:
  void ensure(uint32_t width, uint32_t height);
  spark::rt::Parameter<double> brightness_;  // black-level offset, ±1.0 = ±full Y swing; 0 = neutral
  spark::rt::Parameter<double> contrast_;    // video gain about black (64); 1 = neutral
  spark::rt::Parameter<double> saturation_;  // chroma gain about 512; 1 = neutral
  spark::rt::Parameter<double> hue_deg_;     // chroma phase rotation, degrees; 0 = neutral
  cudaStream_t stream_ = nullptr;
  std::vector<spark::gpu::GpuFramePtr> pool_;
  size_t idx_ = 0;
  uint64_t frames_ = 0;
};

// Luma unsharp mask (broadcast detail-enhance). Chroma passes through by device copy.
class SharpenOp : public spark::rt::Operator {
 public:
  SPARK_OPERATOR_FORWARD_ARGS(SharpenOp)
  SharpenOp() = default;
  void setup(spark::rt::OperatorSpec& spec) override;
  void compute(spark::rt::InputContext& op_input, spark::rt::OutputContext& op_output,
               spark::rt::ExecutionContext& context) override;
  void stop() override;

 private:
  void ensure(uint32_t width, uint32_t height);
  spark::rt::Parameter<double> amount_;  // unsharp gain; 0 = identity (stage normally omitted then)
  cudaStream_t stream_ = nullptr;
  std::vector<spark::gpu::GpuFramePtr> pool_;
  size_t idx_ = 0;
  uint64_t frames_ = 0;
};

// Spatial noise reduction: edge-preserving 5x5 bilateral, independent luma/chroma strengths
// (0 leaves that plane untouched — pass-through by device copy). Best placed BEFORE scale, at the
// native input resolution, where the noise actually lives.
class NrOp : public spark::rt::Operator {
 public:
  SPARK_OPERATOR_FORWARD_ARGS(NrOp)
  NrOp() = default;
  void setup(spark::rt::OperatorSpec& spec) override;
  void compute(spark::rt::InputContext& op_input, spark::rt::OutputContext& op_output,
               spark::rt::ExecutionContext& context) override;
  void stop() override;

 private:
  void ensure(uint32_t width, uint32_t height);
  spark::rt::Parameter<double> luma_;    // 0..1; 0 = Y untouched
  spark::rt::Parameter<double> chroma_;  // 0..1; 0 = Cb/Cr untouched
  cudaStream_t stream_ = nullptr;
  std::vector<spark::gpu::GpuFramePtr> pool_;
  size_t idx_ = 0;
  uint64_t frames_ = 0;
};

// Film grain (photochemical look; kernels ported from REDStreamer). The field re-draws every frame
// from an internal counter, so grain is alive on static frames too. mono = luma-only grain
// (silver-halide look; chroma passes through by device copy); color adds independent Cb/Cr fields
// (color-negative look).
class GrainOp : public spark::rt::Operator {
 public:
  SPARK_OPERATOR_FORWARD_ARGS(GrainOp)
  GrainOp() = default;
  void setup(spark::rt::OperatorSpec& spec) override;
  void compute(spark::rt::InputContext& op_input, spark::rt::OutputContext& op_output,
               spark::rt::ExecutionContext& context) override;
  void stop() override;

 private:
  void ensure(uint32_t width, uint32_t height);
  spark::rt::Parameter<double> amount_;     // 0..1; 0 = identity (stage normally omitted then)
  spark::rt::Parameter<double> size_;       // grain cell in pixels, 1..4
  spark::rt::Parameter<std::string> mode_;  // "mono" | "color"
  cudaStream_t stream_ = nullptr;
  std::vector<spark::gpu::GpuFramePtr> pool_;
  size_t idx_ = 0;
  uint64_t frames_ = 0;  // doubles as the per-frame grain seed
};

}  // namespace spark::ops
