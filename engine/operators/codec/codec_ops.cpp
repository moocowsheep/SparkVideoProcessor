#include "codec_ops.hpp"

#include <stdexcept>

#include <cuda_runtime.h>

#include "pixel_codec.hpp"

namespace spark::ops {
namespace {
constexpr size_t kRing = 4;
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

  // Host packed -> device, then unpack to the planar GpuFrame (default stream serializes with the
  // downstream GPU ops). The sync memcpy makes the upload visible before the kernel.
  cuda_check(cudaMemcpy(dpacked_, vf.data->data(), dpacked_bytes_, cudaMemcpyHostToDevice),
             "H2D packed");
  auto dst = pool_[idx_];
  idx_ = (idx_ + 1) % pool_.size();
  spark::codec::unpack_422_10(dpacked_, dst->y, dst->cb, dst->cr, fmt.width, fmt.height, 0);
  op_output.emit(dst, "out");
}

void UnpackOp::stop() {
  if (dpacked_) {
    cudaFree(dpacked_);
    dpacked_ = nullptr;
  }
}

// ---- PackOp: GpuFrame (device planar) -> VideoFrame (host packed) ----
void PackOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<spark::gpu::GpuFramePtr>("in");
  spec.output<spark::st2110::VideoFrame>("out");
  spec.param(out_fps_, "out_fps", "Output fps", "output RTP media rate (TX pacing)", 60000.0 / 1001.0);
}

void PackOp::ensure(uint32_t width, uint32_t height) {
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

  spark::codec::pack_422_10(dpacked_, src.y, src.cb, src.cr, src.width, src.height, 0);
  auto host = host_pool_[idx_];
  idx_ = (idx_ + 1) % host_pool_.size();
  // D2H sync waits for the pack kernel (same default stream) before the buffer is read on the host.
  cuda_check(cudaMemcpy(host->data(), dpacked_, dpacked_bytes_, cudaMemcpyDeviceToHost),
             "D2H packed");

  spark::st2110::VideoFrame out;
  out.data = host;
  out.format = fmt_;
  op_output.emit(out, "out");
}

void PackOp::stop() {
  if (dpacked_) {
    cudaFree(dpacked_);
    dpacked_ = nullptr;
  }
}

}  // namespace spark::ops
