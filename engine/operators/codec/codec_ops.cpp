#include "codec_ops.hpp"

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>

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

// Zero-copy host access: on GB10 (Grace Blackwell, cache-coherent unified LPDDR5x) the GPU can read/
// write ordinary malloc'd host memory directly (pageableMemoryAccess), so the packed-frame staging
// copies (H2D in unpack, D2H in pack) are pure overhead — the kernels touch each octet exactly once
// anyway. SPARK_ZEROCOPY=0 forces the copy path (A/B or fallback), =1 forces zero-copy (testing);
// unset auto-detects. Decided once, logged once.
bool host_zerocopy() {
  static const bool on = [] {
    bool v = false;
    const char* e = std::getenv("SPARK_ZEROCOPY");
    if (e && *e) {
      v = std::string(e) != "0";
    } else {
      cudaDeviceProp prop{};
      int dev = 0;
      if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        v = prop.pageableMemoryAccess != 0;
    }
    HOLOSCAN_LOG_INFO("codec_ops: host zero-copy {} ({})", v ? "ON" : "OFF",
                      e && *e ? "SPARK_ZEROCOPY" : "auto: pageableMemoryAccess");
    return v;
  }();
  return on;
}
}  // namespace

// ---- UnpackOp: VideoFrame (host packed) -> GpuFrame (device planar) ----
void UnpackOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<spark::st2110::VideoFrame>("in");
  spec.output<spark::gpu::GpuFramePtr>("out");
}

void UnpackOp::ensure(uint32_t width, uint32_t height, bool ip10) {
  if (!stream_) cuda_check(cudaStreamCreate(&stream_), "unpack stream");
  // IP10 sources arrive as 8-bit codeword pgroups (4 octets/pgroup); raw is 5. The planar GpuFrame
  // (10-bit Y/Cb/Cr) is identical either way — only the device staging buffer size differs.
  const size_t octets = static_cast<size_t>(width / 2) * height * (ip10 ? 4 : 5);
  if (dpacked_bytes_ == octets && !pool_.empty()) return;
  zerocopy_ = host_zerocopy();
  if (dpacked_) cudaFree(dpacked_);
  if (!zerocopy_)  // zero-copy reads the RX host buffer in place; no device staging needed
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&dpacked_), octets), "cudaMalloc packed");
  dpacked_bytes_ = octets;
  pool_.assign(kRing, nullptr);
  for (auto& f : pool_) f = std::make_shared<spark::gpu::GpuFrame>(width, height);
  HOLOSCAN_LOG_INFO("unpack: {}x{} ({} octets, {}{})", width, height, octets,
                    ip10 ? "IP10 10:8" : "raw 10-bit", zerocopy_ ? ", zero-copy" : "");
}

// Pop inflight entries whose GPU read finished, returning their RX buffers (and events) to the pools.
// wait=true (stop path) blocks on stragglers so no buffer outlives the op while a kernel reads it.
void UnpackOp::drain_inflight(bool wait) {
  while (!inflight_.empty()) {
    const cudaError_t st = wait ? cudaEventSynchronize(inflight_.front().first)
                                : cudaEventQuery(inflight_.front().first);
    if (st == cudaErrorNotReady) break;
    ev_pool_.push_back(inflight_.front().first);
    inflight_.pop_front();
  }
}

void UnpackOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                       holoscan::ExecutionContext&) {
  auto in = op_input.receive<spark::st2110::VideoFrame>("in");
  if (!in || !in.value().data) return;
  const auto& vf = in.value();
  const auto& fmt = vf.format;
  const bool ip10 = fmt.sampling == spark::st2110::Sampling::YCbCr422_8;
  ensure(fmt.width, fmt.height, ip10);
  drain_inflight(false);

  // Feed the unpack kernel. Zero-copy (GB10 coherent memory): the kernel reads the RX ring buffer in
  // place — no staging copy, but the buffer must stay referenced until the read completes (the RX
  // reuses any ring slot whose use_count drops to 1), so park the shared_ptr in inflight_ behind a
  // completion event. Copy path (non-coherent platforms): host packed -> device staging on our stream
  // (async H2D is host-synchronous for pageable memory, so the buffer is free on return), then unpack;
  // dpacked_ reuse across frames is safe because the copy and kernel are stream-ordered.
  const uint8_t* src = vf.data->data();
  if (!zerocopy_) {
    cuda_check(cudaMemcpyAsync(dpacked_, src, dpacked_bytes_, cudaMemcpyHostToDevice, stream_),
               "H2D packed");
    src = dpacked_;
  }
  auto dst = pool_[idx_];
  idx_ = (idx_ + 1) % pool_.size();
  if (ip10)
    spark::codec::ip10::ip10_unpack_422(src, dst->y, dst->cb, dst->cr, fmt.width, fmt.height, stream_);
  else
    spark::codec::unpack_422_10(src, dst->y, dst->cb, dst->cr, fmt.width, fmt.height, stream_);
  cuda_check(cudaEventRecord(dst->ready, stream_), "unpack record");  // consumers wait on this
  if (zerocopy_) {
    cudaEvent_t ev;
    if (!ev_pool_.empty()) {
      ev = ev_pool_.back();
      ev_pool_.pop_back();
    } else {
      cuda_check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "inflight event");
    }
    cuda_check(cudaEventRecord(ev, stream_), "inflight record");
    inflight_.emplace_back(ev, vf.data);
  }
  dst->t_ingest_ns = now_ns();  // frame enters the GPU graph here; pack reads this for the latency probe
  dst->capture_ts_ns = vf.capture_ts_ns;  // carry the source frame timing through for TX genlock
  op_output.emit(dst, "out");
}

void UnpackOp::stop() {
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  drain_inflight(true);
  for (cudaEvent_t ev : ev_pool_) cudaEventDestroy(ev);
  ev_pool_.clear();
  if (dpacked_) {
    cudaFree(dpacked_);
    dpacked_ = nullptr;
  }
}

// ---- PackOp: GpuFrame (device planar) -> VideoFrame (host packed) ----
void PackOp::setup(holoscan::OperatorSpec& spec) {
  // Burst-deep ONLY when pack directly receives FRC's multi-frame burst (frc last in the chain,
  // SPARK_BURST_SINK); otherwise the upstream feeds 1:1 and this sits 1-deep. min_size 1 runs pack
  // once per frame (one D2H + emit). Capacity == standing latency floor here.
  const auto cap = static_cast<uint64_t>(spark::pipeline_queue_cap(name()));
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
  zerocopy_ = host_zerocopy();
  if (dpacked_) cudaFree(dpacked_);
  if (!zerocopy_)  // zero-copy packs straight into the TX host buffer; no device staging needed
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&dpacked_), octets), "cudaMalloc packed");
  dpacked_bytes_ = octets;
  fmt_ = spark::st2110::VideoFormat{width, height, out_fps_.get(), sampling};
  host_pool_.assign(kRing, nullptr);
  for (auto& b : host_pool_) b = std::make_shared<std::vector<uint8_t>>(octets);
  HOLOSCAN_LOG_INFO("pack: {}x{} ({} octets, {}{})", width, height, octets,
                    ip10_.get() ? "IP10 10:8" : "raw 10-bit", zerocopy_ ? ", zero-copy" : "");
}

void PackOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                     holoscan::ExecutionContext&) {
  auto in = op_input.receive<spark::gpu::GpuFramePtr>("in");
  if (!in || !in.value()) return;
  const auto& src = *in.value();
  ensure(src.width, src.height);

  // Pick a host buffer the TX has finished transmitting. The TX holds the emitted shared_ptr for the
  // whole ~frame-long send, so use_count()==1 means only the pool still references it (TX released it).
  // A blind round-robin can lap the in-flight TX buffer under FRC up-convert + motion-driven GPU
  // latency spikes (pipeline runs many frames deep) and overwrite the frame mid-send — corrupting the
  // last-sent (bottom) lines (the bottom-tear, worse with motion). If every buffer is still in flight,
  // grow the pool rather than clobber one; it settles at the working depth and stops growing.
  // (Chosen before the kernel launch: the zero-copy path packs straight into it.)
  std::shared_ptr<std::vector<uint8_t>> host;
  for (size_t n = 0; n < host_pool_.size(); ++n) {
    auto& cand = host_pool_[idx_];
    idx_ = (idx_ + 1) % host_pool_.size();
    if (cand.use_count() == 1) { host = cand; break; }
  }
  if (!host) {
    host = std::make_shared<std::vector<uint8_t>>(dpacked_bytes_);
    host_pool_.push_back(host);
    HOLOSCAN_LOG_INFO("pack: grew host pool to {} buffers (TX holding the rest in flight)", host_pool_.size());
  }

  // Order our stream behind whoever produced this frame (resize, on its own stream), then pack on our
  // stream. Zero-copy: the kernel writes the TX host buffer directly over the coherent fabric (no D2H
  // staging); copy path: pack to device staging then D2H. Either way the blocking stream sync (not the
  // default stream) waits only our chain, so the host buffer is valid before TX reads it without
  // serializing the other operators.
  cuda_check(cudaStreamWaitEvent(stream_, src.ready, 0), "pack wait input");
  uint8_t* packed = zerocopy_ ? host->data() : dpacked_;
  if (ip10_.get())
    spark::codec::ip10::ip10_pack_422(packed, src.y, src.cb, src.cr, src.width, src.height, stream_);
  else
    spark::codec::pack_422_10(packed, src.y, src.cb, src.cr, src.width, src.height, stream_);
  if (!zerocopy_)
    cuda_check(cudaMemcpyAsync(host->data(), dpacked_, dpacked_bytes_, cudaMemcpyDeviceToHost, stream_),
               "D2H packed");
  const auto t_sync0 = std::chrono::steady_clock::now();
  cuda_check(cudaStreamSynchronize(stream_), "pack sync");
  const int64_t sync_us =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                            t_sync0)
          .count();

  // Latency probe: unpack stamped t_ingest_ns when the frame entered the GPU graph; the host buffer is
  // ready for TX now, so (now - ingest) is the unpack->frc->resize->pack latency incl. the inter-op
  // queues — the trimmable part. (RX assembly upstream and TX pacing horizon downstream are extra.)
  if (src.t_ingest_ns) {
    const uint64_t lat = now_ns() - src.t_ingest_ns;
    // Stall localization (the M8 determinism hunt): a long GPU fence right here = the CUDA chain
    // itself stalled (NVOF/warp/scale/pack kernels or GPU contention); a long chain latency with a
    // SHORT fence = the frame sat in inter-op queues / scheduler / an upstream CPU block. Warns are
    // rate-limited to 1/s; grep "pipe stall" and read the split.
    const double tw =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if ((sync_us > 25000 || lat > 250000000ULL) && tw - last_stall_s_ >= 1.0) {
      last_stall_s_ = tw;
      HOLOSCAN_LOG_WARN("pipe stall: chain latency {} ms, GPU fence {} ms -> {}", lat / 1000000,
                        sync_us / 1000,
                        sync_us > 25000 ? "GPU-side stall" : "queue/scheduler/CPU-side stall");
    }
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
  out.capture_ts_ns = src.capture_ts_ns;  // source frame timing -> TX genlock
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
