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

}  // namespace spark::ops
