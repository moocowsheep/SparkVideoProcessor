#include "resize.hpp"

#include <stdexcept>

#include <cuda_runtime.h>
#include <npp.h>

#include "../pipeline_caps.hpp"

namespace spark::ops {
namespace {

int interp_code(const std::string& s) {
  if (s == "cubic") return NPPI_INTER_CUBIC;
  if (s == "lanczos") return NPPI_INTER_LANCZOS;
  return NPPI_INTER_LINEAR;
}

// This NPP build has no nppGetStreamContext helper — the app fills the context (per nppdefs.h).
void fill_npp_ctx(NppStreamContext& ctx) {
  ctx = NppStreamContext{};
  ctx.hStream = nullptr;  // default stream
  cudaGetDevice(&ctx.nCudaDeviceId);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, ctx.nCudaDeviceId);
  ctx.nMultiProcessorCount = prop.multiProcessorCount;
  ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
  cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor,
                         cudaDevAttrComputeCapabilityMajor, ctx.nCudaDeviceId);
  cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor,
                         cudaDevAttrComputeCapabilityMinor, ctx.nCudaDeviceId);
  ctx.nStreamFlags = 0;
}

// Resize one 16u single-channel plane src(sw x sh) -> dst(dw x dh).
void resize_plane(const uint16_t* src, uint32_t sw, uint32_t sh, uint16_t* dst, uint32_t dw,
                  uint32_t dh, int interp, const NppStreamContext& ctx) {
  const NppiSize src_size{static_cast<int>(sw), static_cast<int>(sh)};
  const NppiRect src_roi{0, 0, static_cast<int>(sw), static_cast<int>(sh)};
  const NppiSize dst_size{static_cast<int>(dw), static_cast<int>(dh)};
  const NppiRect dst_roi{0, 0, static_cast<int>(dw), static_cast<int>(dh)};
  const NppStatus st =
      nppiResize_16u_C1R_Ctx(src, static_cast<int>(sw) * 2, src_size, src_roi, dst,
                             static_cast<int>(dw) * 2, dst_size, dst_roi, interp, ctx);
  if (st != NPP_SUCCESS)
    throw std::runtime_error("nppiResize_16u_C1R_Ctx failed: " + std::to_string(st));
}

}  // namespace

// ---- ResizeOp ----
void ResizeOp::setup(holoscan::OperatorSpec& spec) {
  // Sized to the producer's per-compute emit burst: 2 under FRC up-convert (real + mid arrive together),
  // else 1 (retime / passthrough — unpack or retiming FRC feed one frame per tick). min_size 1 keeps
  // resize upscaling one frame per compute. This is the latency floor — capacity == standing latency
  // behind the paced TX, so passthrough/retime run 1-deep here instead of 2.
  const auto cap = static_cast<uint64_t>(spark::pipeline_emit_burst());
  spec.input<spark::gpu::GpuFramePtr>("in")
      .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,
                 holoscan::Arg("capacity", cap),
                 holoscan::Arg("policy", static_cast<uint64_t>(2)))
      .condition(holoscan::ConditionType::kMessageAvailable,
                 holoscan::Arg("min_size", static_cast<uint64_t>(1)));
  spec.output<spark::gpu::GpuFramePtr>("out");
  spec.param(out_width_, "out_width", "Out width", "target width", 3840u);
  spec.param(out_height_, "out_height", "Out height", "target height", 2160u);
  spec.param(interp_, "interp", "Interpolation", "linear | cubic | lanczos", std::string("cubic"));
  spec.param(measure_, "measure", "Measure", "per-frame GPU timing (benchmark only; adds a sync)",
             false);
}

void ResizeOp::start() {
  interp_code_ = interp_code(interp_.get());
  cudaStreamCreate(&stream_);  // resize runs here so it pipelines with FRC/pack on their own streams
  fill_npp_ctx(npp_ctx_);  // device context for the _Ctx primitives
  npp_ctx_.hStream = stream_;  // NPP enqueues on our stream, not the default
  // Resize outputs feed the pack input queue (capacity 2); size the pool above that + the 2160p frames
  // in flight (being packed / produced) so a transiently-full queue can never alias a slot still in use.
  pool_.resize(10);
  for (auto& f : pool_) f = std::make_shared<spark::gpu::GpuFrame>(out_width_.get(), out_height_.get());
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  ev_start_ = a;
  ev_stop_ = b;
  HOLOSCAN_LOG_INFO("resize started: -> {}x{} interp={}", out_width_.get(), out_height_.get(),
                    interp_.get());
}

void ResizeOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                       holoscan::ExecutionContext&) {
  auto in = op_input.receive<spark::gpu::GpuFramePtr>("in");
  if (!in || !in.value()) return;
  const auto& src = *in.value();
  cudaStreamWaitEvent(stream_, src.ready, 0);  // order behind the producer's writes (cross-stream)
  auto dst = pool_[pool_idx_];
  pool_idx_ = (pool_idx_ + 1) % pool_.size();

  const bool measure = measure_.get();
  auto a = static_cast<cudaEvent_t>(ev_start_);
  auto b = static_cast<cudaEvent_t>(ev_stop_);
  if (measure) cudaEventRecord(a);
  resize_plane(src.y, src.width, src.height, dst->y, dst->width, dst->height, interp_code_, npp_ctx_);
  resize_plane(src.cb, src.chroma_width(), src.height, dst->cb, dst->chroma_width(), dst->height,
               interp_code_, npp_ctx_);
  resize_plane(src.cr, src.chroma_width(), src.height, dst->cr, dst->chroma_width(), dst->height,
               interp_code_, npp_ctx_);
  if (measure) {  // benchmark path: a full GPU sync per frame. Off in the pipeline (stream-ordered).
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    ms_sum_ += ms;
    if (ms < ms_min_) ms_min_ = ms;
    if (ms > ms_max_) ms_max_ = ms;
  }
  cudaEventRecord(dst->ready, stream_);  // consumers (pack) wait on this before reading dst
  dst->t_ingest_ns = src.t_ingest_ns;    // carry the ingest time through resize for the latency probe
  dst->capture_ts_ns = src.capture_ts_ns;  // carry source frame timing through for TX genlock
  ++frames_;

  op_output.emit(dst, "out");
}

void ResizeOp::stop() {
  if (frames_ && ms_sum_ > 0.0) {
    HOLOSCAN_LOG_INFO(
        "resize stopped: frames={} | resize ms/frame min/avg/max = {:.3f}/{:.3f}/{:.3f}", frames_,
        ms_min_, ms_sum_ / frames_, ms_max_);
  } else {
    HOLOSCAN_LOG_INFO("resize stopped: frames={} (timing off)", frames_);
  }
  if (ev_start_) cudaEventDestroy(static_cast<cudaEvent_t>(ev_start_));
  if (ev_stop_) cudaEventDestroy(static_cast<cudaEvent_t>(ev_stop_));
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

// ---- TestGpuSourceOp ----
void TestGpuSourceOp::setup(holoscan::OperatorSpec& spec) {
  spec.output<spark::gpu::GpuFramePtr>("out");
  spec.param(profile_, "profile", "Profile", "1080p | 2160p (input size)", std::string("1080p"));
}

void TestGpuSourceOp::start() {
  const bool uhd = profile_.get() == "2160p";
  frame_ = std::make_shared<spark::gpu::GpuFrame>(uhd ? 3840u : 1920u, uhd ? 2160u : 1080u);
  spark::gpu::fill_gradient(*frame_, 1);
  HOLOSCAN_LOG_INFO("test_gpu_source: {}x{} (profile {})", frame_->width, frame_->height,
                    profile_.get());
}

void TestGpuSourceOp::compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
                              holoscan::ExecutionContext&) {
  ++n_;
  op_output.emit(frame_, "out");  // reused input frame (resize reads it read-only)
}

// ---- GpuFrameSinkOp ----
void GpuFrameSinkOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<spark::gpu::GpuFramePtr>("in");
}

void GpuFrameSinkOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext&,
                             holoscan::ExecutionContext&) {
  auto in = op_input.receive<spark::gpu::GpuFramePtr>("in");
  if (in && in.value()) {
    cudaDeviceSynchronize();  // ensure the resize actually completed
    ++count_;
  }
}

}  // namespace spark::ops
