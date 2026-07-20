// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

#include "filters.hpp"

#include <cuda_runtime.h>

#include "../pipeline_caps.hpp"
#include "filter_kernels.hpp"

namespace spark::ops {
namespace {
// Pool depth matches the other GpuFrame stages: > downstream queue capacity + frames in flight
// (see the aliasing note in resize.cpp) so a transiently-full queue can never lap a slot in use.
constexpr size_t kPool = 10;

// Shared setup shape for a GpuFrame->GpuFrame filter stage: input queue burst-deep only when this
// op directly receives FRC's multi-frame burst (SPARK_BURST_SINK), else 1-deep (1:1 hop; capacity
// == standing latency behind the paced TX). One frame per compute.
void filter_io(holoscan::OperatorSpec& spec, const std::string& op_name) {
  const auto cap = static_cast<uint64_t>(spark::pipeline_queue_cap(op_name));
  spec.input<spark::gpu::GpuFramePtr>("in")
      .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer, holoscan::Arg("capacity", cap),
                 holoscan::Arg("policy", static_cast<uint64_t>(2)))
      .condition(holoscan::ConditionType::kMessageAvailable,
                 holoscan::Arg("min_size", static_cast<uint64_t>(1)));
  spec.output<spark::gpu::GpuFramePtr>("out");
}
}  // namespace

// ---- ProcAmpOp ----
void ProcAmpOp::setup(holoscan::OperatorSpec& spec) {
  filter_io(spec, name());
  spec.param(brightness_, "brightness", "Brightness", "black-level offset (±1 = ±full swing)", 0.0);
  spec.param(contrast_, "contrast", "Contrast", "video gain about black; 1 = unity", 1.0);
  spec.param(saturation_, "saturation", "Saturation", "chroma gain; 1 = unity", 1.0);
  spec.param(hue_deg_, "hue_deg", "Hue", "chroma phase rotation (degrees)", 0.0);
}

void ProcAmpOp::ensure(uint32_t width, uint32_t height) {
  if (stream_ && !pool_.empty() && pool_[0]->width == width && pool_[0]->height == height) return;
  if (!stream_) cudaStreamCreate(&stream_);
  pool_.assign(kPool, nullptr);
  for (auto& f : pool_) f = std::make_shared<spark::gpu::GpuFrame>(width, height);
  HOLOSCAN_LOG_INFO("procamp: {}x{} bright={:.3f} contrast={:.3f} sat={:.3f} hue={:.1f}deg", width,
                    height, brightness_.get(), contrast_.get(), saturation_.get(), hue_deg_.get());
}

void ProcAmpOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                        holoscan::ExecutionContext&) {
  auto in = op_input.receive<spark::gpu::GpuFramePtr>("in");
  if (!in || !in.value()) return;
  const auto& src = *in.value();
  ensure(src.width, src.height);
  cudaStreamWaitEvent(stream_, src.ready, 0);  // order behind the producer's writes (cross-stream)
  auto dst = pool_[idx_];
  idx_ = (idx_ + 1) % pool_.size();
  spark::filters::procamp(src.y, src.cb, src.cr, dst->y, dst->cb, dst->cr, src.width, src.height,
                          static_cast<float>(brightness_.get()), static_cast<float>(contrast_.get()),
                          static_cast<float>(saturation_.get()), static_cast<float>(hue_deg_.get()),
                          stream_);
  cudaEventRecord(dst->ready, stream_);
  dst->t_ingest_ns = src.t_ingest_ns;
  dst->capture_ts_ns = src.capture_ts_ns;
  ++frames_;
  op_output.emit(dst, "out");
}

void ProcAmpOp::stop() {
  HOLOSCAN_LOG_INFO("procamp stopped: frames={}", frames_);
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

// ---- SharpenOp ----
void SharpenOp::setup(holoscan::OperatorSpec& spec) {
  filter_io(spec, name());
  spec.param(amount_, "amount", "Amount", "unsharp gain (0 = identity)", 1.0);
}

void SharpenOp::ensure(uint32_t width, uint32_t height) {
  if (stream_ && !pool_.empty() && pool_[0]->width == width && pool_[0]->height == height) return;
  if (!stream_) cudaStreamCreate(&stream_);
  pool_.assign(kPool, nullptr);
  for (auto& f : pool_) f = std::make_shared<spark::gpu::GpuFrame>(width, height);
  HOLOSCAN_LOG_INFO("sharpen: {}x{} amount={:.2f}", width, height, amount_.get());
}

void SharpenOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                        holoscan::ExecutionContext&) {
  auto in = op_input.receive<spark::gpu::GpuFramePtr>("in");
  if (!in || !in.value()) return;
  const auto& src = *in.value();
  ensure(src.width, src.height);
  cudaStreamWaitEvent(stream_, src.ready, 0);
  auto dst = pool_[idx_];
  idx_ = (idx_ + 1) % pool_.size();
  spark::filters::unsharp_y(src.y, dst->y, src.width, src.height,
                            static_cast<float>(amount_.get()), stream_);
  // Luma-only filter: chroma passes through untouched (device copy keeps the frame self-contained).
  const size_t cbytes = static_cast<size_t>(src.width / 2) * src.height * sizeof(uint16_t);
  cudaMemcpyAsync(dst->cb, src.cb, cbytes, cudaMemcpyDeviceToDevice, stream_);
  cudaMemcpyAsync(dst->cr, src.cr, cbytes, cudaMemcpyDeviceToDevice, stream_);
  cudaEventRecord(dst->ready, stream_);
  dst->t_ingest_ns = src.t_ingest_ns;
  dst->capture_ts_ns = src.capture_ts_ns;
  ++frames_;
  op_output.emit(dst, "out");
}

void SharpenOp::stop() {
  HOLOSCAN_LOG_INFO("sharpen stopped: frames={}", frames_);
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

// ---- NrOp ----
void NrOp::setup(holoscan::OperatorSpec& spec) {
  filter_io(spec, name());
  spec.param(luma_, "luma", "Luma NR", "bilateral strength on Y, 0..1 (0 = untouched)", 0.0);
  spec.param(chroma_, "chroma", "Chroma NR", "bilateral strength on Cb/Cr, 0..1 (0 = untouched)",
             0.0);
}

void NrOp::ensure(uint32_t width, uint32_t height) {
  if (stream_ && !pool_.empty() && pool_[0]->width == width && pool_[0]->height == height) return;
  if (!stream_) cudaStreamCreate(&stream_);
  pool_.assign(kPool, nullptr);
  for (auto& f : pool_) f = std::make_shared<spark::gpu::GpuFrame>(width, height);
  HOLOSCAN_LOG_INFO("nr: {}x{} luma={:.2f} chroma={:.2f}", width, height, luma_.get(),
                    chroma_.get());
}

void NrOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                   holoscan::ExecutionContext&) {
  auto in = op_input.receive<spark::gpu::GpuFramePtr>("in");
  if (!in || !in.value()) return;
  const auto& src = *in.value();
  ensure(src.width, src.height);
  cudaStreamWaitEvent(stream_, src.ready, 0);
  auto dst = pool_[idx_];
  idx_ = (idx_ + 1) % pool_.size();
  const auto luma = static_cast<float>(luma_.get());
  const auto chroma = static_cast<float>(chroma_.get());
  if (luma > 0.0f) {
    spark::filters::nr_y(src.y, dst->y, src.width, src.height, luma, stream_);
  } else {
    cudaMemcpyAsync(dst->y, src.y, (size_t)src.width * src.height * sizeof(uint16_t),
                    cudaMemcpyDeviceToDevice, stream_);
  }
  if (chroma > 0.0f) {
    spark::filters::nr_c(src.cb, src.cr, dst->cb, dst->cr, src.width, src.height, chroma, stream_);
  } else {
    const size_t cbytes = static_cast<size_t>(src.width / 2) * src.height * sizeof(uint16_t);
    cudaMemcpyAsync(dst->cb, src.cb, cbytes, cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(dst->cr, src.cr, cbytes, cudaMemcpyDeviceToDevice, stream_);
  }
  cudaEventRecord(dst->ready, stream_);
  dst->t_ingest_ns = src.t_ingest_ns;
  dst->capture_ts_ns = src.capture_ts_ns;
  ++frames_;
  op_output.emit(dst, "out");
}

void NrOp::stop() {
  HOLOSCAN_LOG_INFO("nr stopped: frames={}", frames_);
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

// ---- GrainOp ----
void GrainOp::setup(holoscan::OperatorSpec& spec) {
  filter_io(spec, name());
  spec.param(amount_, "amount", "Amount", "grain amount 0..1 (0 = identity)", 0.0);
  spec.param(size_, "size", "Size", "grain cell in pixels (1..4)", 1.5);
  spec.param(mode_, "mode", "Mode", "mono (luma only) | color (independent chroma fields)",
             std::string("mono"));
}

void GrainOp::ensure(uint32_t width, uint32_t height) {
  if (stream_ && !pool_.empty() && pool_[0]->width == width && pool_[0]->height == height) return;
  if (!stream_) cudaStreamCreate(&stream_);
  pool_.assign(kPool, nullptr);
  for (auto& f : pool_) f = std::make_shared<spark::gpu::GpuFrame>(width, height);
  HOLOSCAN_LOG_INFO("grain: {}x{} amount={:.2f} size={:.2f} mode={}", width, height, amount_.get(),
                    size_.get(), mode_.get());
}

void GrainOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                      holoscan::ExecutionContext&) {
  auto in = op_input.receive<spark::gpu::GpuFramePtr>("in");
  if (!in || !in.value()) return;
  const auto& src = *in.value();
  ensure(src.width, src.height);
  cudaStreamWaitEvent(stream_, src.ready, 0);
  auto dst = pool_[idx_];
  idx_ = (idx_ + 1) % pool_.size();
  const auto amount = static_cast<float>(amount_.get());
  const auto size = static_cast<float>(size_.get());
  const auto seed = static_cast<uint32_t>(frames_);  // new field every frame
  spark::filters::grain_y(src.y, dst->y, src.width, src.height, amount, size, seed, stream_);
  if (mode_.get() == "color") {
    spark::filters::grain_c(src.y, src.cb, src.cr, dst->cb, dst->cr, src.width, src.height, amount,
                            size, seed, stream_);
  } else {  // mono: chroma passes through untouched (device copy keeps the frame self-contained)
    const size_t cbytes = static_cast<size_t>(src.width / 2) * src.height * sizeof(uint16_t);
    cudaMemcpyAsync(dst->cb, src.cb, cbytes, cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(dst->cr, src.cr, cbytes, cudaMemcpyDeviceToDevice, stream_);
  }
  cudaEventRecord(dst->ready, stream_);
  dst->t_ingest_ns = src.t_ingest_ns;
  dst->capture_ts_ns = src.capture_ts_ns;
  ++frames_;
  op_output.emit(dst, "out");
}

void GrainOp::stop() {
  HOLOSCAN_LOG_INFO("grain stopped: frames={}", frames_);
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

}  // namespace spark::ops
