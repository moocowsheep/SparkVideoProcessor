#include "codec_ops.hpp"

#include <stdexcept>

#include <cuda_runtime.h>

#include "pixel_codec.hpp"

namespace spark::ops {
namespace {
constexpr size_t kRing = 20;  // unpack/pack pools; > the 16-deep frc->resize / resize->pack queues + slack
void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("codec_ops: ") + what);
}
}  // namespace

// ---- UnpackOp: VideoFrame (host packed) -> GpuFrame (device planar) ----
void UnpackOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<spark::st2110::VideoFrame>("in");
  spec.output<spark::gpu::GpuFramePtr>("out");
}

void UnpackOp::ensure(uint32_t width, uint32_t height) {
  if (!stream_) cuda_check(cudaStreamCreate(&stream_), "unpack stream");
  const size_t octets = static_cast<size_t>(width / 2) * height * 5;
  if (dpacked_bytes_ == octets && !pool_.empty()) return;
  if (dpacked_) cudaFree(dpacked_);
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&dpacked_), octets), "cudaMalloc packed");
  dpacked_bytes_ = octets;
  pool_.assign(kRing, nullptr);
  for (auto& f : pool_) f = std::make_shared<spark::gpu::GpuFrame>(width, height);
  HOLOSCAN_LOG_INFO("unpack: {}x{} ({} octets)", width, height, octets);
}

void UnpackOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                       holoscan::ExecutionContext&) {
  auto in = op_input.receive<spark::st2110::VideoFrame>("in");
  if (!in || !in.value().data) return;
  const auto& vf = in.value();
  const auto& fmt = vf.format;
  ensure(fmt.width, fmt.height);

  // Host packed -> device, then unpack to the planar GpuFrame, all on our stream (async H2D is host-
  // synchronous for pageable memory but stays off the default stream, so it doesn't serialize the
  // other operators). The unpack kernel is stream-ordered after the upload; dpacked_ reuse across
  // frames is safe for the same reason.
  cuda_check(cudaMemcpyAsync(dpacked_, vf.data->data(), dpacked_bytes_, cudaMemcpyHostToDevice,
                             stream_),
             "H2D packed");
  auto dst = pool_[idx_];
  idx_ = (idx_ + 1) % pool_.size();
  spark::codec::unpack_422_10(dpacked_, dst->y, dst->cb, dst->cr, fmt.width, fmt.height, stream_);
  cuda_check(cudaEventRecord(dst->ready, stream_), "unpack record");  // consumers wait on this
  op_output.emit(dst, "out");
}

void UnpackOp::stop() {
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  if (dpacked_) {
    cudaFree(dpacked_);
    dpacked_ = nullptr;
  }
}

// ---- PackOp: GpuFrame (device planar) -> VideoFrame (host packed) ----
void PackOp::setup(holoscan::OperatorSpec& spec) {
  // Resize feeds one frame per compute here, but at up-convert/startup they can arrive in bursts ahead
  // of the paced TX. Buffer them (capacity 16) without batching (min_size 1) so PackOp runs once per
  // frame (one D2H + emit) and drains the burst across computes.
  spec.input<spark::gpu::GpuFramePtr>("in")
      .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,
                 holoscan::Arg("capacity", static_cast<uint64_t>(16)),
                 holoscan::Arg("policy", static_cast<uint64_t>(2)))
      .condition(holoscan::ConditionType::kMessageAvailable,
                 holoscan::Arg("min_size", static_cast<uint64_t>(1)));
  spec.output<spark::st2110::VideoFrame>("out");
  spec.param(out_fps_, "out_fps", "Output fps", "output RTP media rate (TX pacing)", 60000.0 / 1001.0);
}

void PackOp::ensure(uint32_t width, uint32_t height) {
  if (!stream_) cuda_check(cudaStreamCreate(&stream_), "pack stream");
  const size_t octets = static_cast<size_t>(width / 2) * height * 5;
  if (dpacked_bytes_ == octets && !host_pool_.empty()) return;
  if (dpacked_) cudaFree(dpacked_);
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&dpacked_), octets), "cudaMalloc packed");
  dpacked_bytes_ = octets;
  fmt_ = spark::st2110::VideoFormat{width, height, out_fps_.get()};
  host_pool_.assign(kRing, nullptr);
  for (auto& b : host_pool_) b = std::make_shared<std::vector<uint8_t>>(octets);
  HOLOSCAN_LOG_INFO("pack: {}x{} ({} octets)", width, height, octets);
}

void PackOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                     holoscan::ExecutionContext&) {
  auto in = op_input.receive<spark::gpu::GpuFramePtr>("in");
  if (!in || !in.value()) return;
  const auto& src = *in.value();
  ensure(src.width, src.height);

  // Order our stream behind whoever produced this frame (resize, on its own stream), then pack + D2H
  // on our stream. The blocking stream sync (not the default stream) waits only our work, so the host
  // buffer is valid before TX reads it without serializing the other operators.
  cuda_check(cudaStreamWaitEvent(stream_, src.ready, 0), "pack wait input");
  spark::codec::pack_422_10(dpacked_, src.y, src.cb, src.cr, src.width, src.height, stream_);
  auto host = host_pool_[idx_];
  idx_ = (idx_ + 1) % host_pool_.size();
  cuda_check(cudaMemcpyAsync(host->data(), dpacked_, dpacked_bytes_, cudaMemcpyDeviceToHost, stream_),
             "D2H packed");
  cuda_check(cudaStreamSynchronize(stream_), "pack D2H sync");

  spark::st2110::VideoFrame out;
  out.data = host;
  out.format = fmt_;
  op_output.emit(out, "out");
}

void PackOp::stop() {
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  if (dpacked_) {
    cudaFree(dpacked_);
    dpacked_ = nullptr;
  }
}

}  // namespace spark::ops
