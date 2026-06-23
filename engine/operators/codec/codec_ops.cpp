#include "codec_ops.hpp"

#include <chrono>
#include <stdexcept>

#include <cuda_runtime.h>

#include "../pipeline_caps.hpp"
#include "ip10_codec.hpp"
#include "pixel_codec.hpp"

namespace spark::ops {
namespace {
constexpr size_t kRing = 10;  // unpack/pack pools; > the 2-deep frc->resize / resize->pack queues +
                              // frames held in flight (frc keeps prev; TX holds host buffers while pacing)
void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("codec_ops: ") + what);
}
uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
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
  dst->t_ingest_ns = now_ns();  // frame enters the GPU graph here; pack reads this for the latency probe
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
  // Resize feeds one frame per compute and pack drains one. Size to the emit burst (2 under FRC
  // up-convert, where resize processes the pair back-to-back and two frames can land before the paced
  // TX drains one; else 1). min_size 1 runs pack once per frame (one D2H + emit). Capacity == standing
  // latency floor here, so passthrough/retime sit 1-deep.
  const auto cap = static_cast<uint64_t>(spark::pipeline_emit_burst());
  spec.input<spark::gpu::GpuFramePtr>("in")
      .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,
                 holoscan::Arg("capacity", cap),
                 holoscan::Arg("policy", static_cast<uint64_t>(2)))
      .condition(holoscan::ConditionType::kMessageAvailable,
                 holoscan::Arg("min_size", static_cast<uint64_t>(1)));
  spec.output<spark::st2110::VideoFrame>("out");
  spec.param(out_fps_, "out_fps", "Output fps", "output RTP media rate (TX pacing)", 60000.0 / 1001.0);
  spec.param(ip10_, "ip10", "IP10", "Blackmagic IP10 10:8 output (8-bit pgroups, ST 2110-22)", false);
}

void PackOp::ensure(uint32_t width, uint32_t height) {
  if (!stream_) cuda_check(cudaStreamCreate(&stream_), "pack stream");
  // IP10 emits the 10-bit samples as 8-bit codewords -> 4 octets/pgroup; raw stays at 5.
  const auto sampling =
      ip10_.get() ? spark::st2110::Sampling::YCbCr422_8 : spark::st2110::Sampling::YCbCr422_10;
  const size_t octets = static_cast<size_t>(width / 2) * height * (ip10_.get() ? 4 : 5);
  if (dpacked_bytes_ == octets && !host_pool_.empty()) return;
  if (dpacked_) cudaFree(dpacked_);
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&dpacked_), octets), "cudaMalloc packed");
  dpacked_bytes_ = octets;
  fmt_ = spark::st2110::VideoFormat{width, height, out_fps_.get(), sampling};
  host_pool_.assign(kRing, nullptr);
  for (auto& b : host_pool_) b = std::make_shared<std::vector<uint8_t>>(octets);
  HOLOSCAN_LOG_INFO("pack: {}x{} ({} octets, {})", width, height, octets, ip10_.get() ? "IP10 10:8" : "raw 10-bit");
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
  if (ip10_.get())
    spark::codec::ip10::ip10_pack_422(dpacked_, src.y, src.cb, src.cr, src.width, src.height, stream_);
  else
    spark::codec::pack_422_10(dpacked_, src.y, src.cb, src.cr, src.width, src.height, stream_);
  auto host = host_pool_[idx_];
  idx_ = (idx_ + 1) % host_pool_.size();
  cuda_check(cudaMemcpyAsync(host->data(), dpacked_, dpacked_bytes_, cudaMemcpyDeviceToHost, stream_),
             "D2H packed");
  cuda_check(cudaStreamSynchronize(stream_), "pack D2H sync");

  // Latency probe: unpack stamped t_ingest_ns when the frame entered the GPU graph; the host buffer is
  // ready for TX now, so (now - ingest) is the unpack->frc->resize->pack latency incl. the inter-op
  // queues — the trimmable part. (RX assembly upstream and TX pacing horizon downstream are extra.)
  if (src.t_ingest_ns) {
    const uint64_t lat = now_ns() - src.t_ingest_ns;
    lat_sum_ += lat;
    ++lat_n_;
    if (lat < lat_min_) lat_min_ = lat;
    if (lat > lat_max_) lat_max_ = lat;
    const double t = std::chrono::duration<double>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    if (t - last_live_s_ >= 1.0) {
      last_live_s_ = t;
      HOLOSCAN_LOG_INFO("spark_live pipe_latency_us cur={} min={} avg={} max={}", lat / 1000,
                        lat_min_ / 1000, (lat_sum_ / lat_n_) / 1000, lat_max_ / 1000);
    }
  }

  spark::st2110::VideoFrame out;
  out.data = host;
  out.format = fmt_;
  op_output.emit(out, "out");
}

void PackOp::stop() {
  if (lat_n_) {
    HOLOSCAN_LOG_INFO("pack stopped: pipe_latency_us (unpack->pack) min/avg/max = {}/{}/{}",
                      lat_min_ / 1000, (lat_sum_ / lat_n_) / 1000, lat_max_ / 1000);
  }
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
