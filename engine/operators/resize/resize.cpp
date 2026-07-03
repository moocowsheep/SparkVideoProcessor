#include "resize.hpp"

#include <cstdlib>
#include <stdexcept>

#include <cuda_runtime.h>
#include <npp.h>

#include "../pipeline_caps.hpp"
#include "sr_model.hpp"

namespace spark::ops {
namespace {

// "auto" resolves per-frame in compute(): supersampling for a downscale (cubic/lanczos on a
// downscale alias — they only interpolate, they don't prefilter), cubic for an upscale.
constexpr int kAutoInterp = -1;

int interp_code(const std::string& s) {
  if (s == "auto") return kAutoInterp;
  if (s == "cubic") return NPPI_INTER_CUBIC;
  if (s == "lanczos") return NPPI_INTER_LANCZOS;
  if (s == "super") return NPPI_INTER_SUPER;
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
  // Burst-deep ONLY when this op directly receives FRC's multi-frame burst (SPARK_BURST_SINK);
  // every other placement is 1:1 and sits 1-deep. min_size 1 keeps resize upscaling one frame per
  // compute. This is the latency floor — capacity == standing latency behind the paced TX.
  const auto cap = static_cast<uint64_t>(spark::pipeline_queue_cap(name()));
  spec.input<spark::gpu::GpuFramePtr>("in")
      .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,
                 holoscan::Arg("capacity", cap),
                 holoscan::Arg("policy", static_cast<uint64_t>(2)))
      .condition(holoscan::ConditionType::kMessageAvailable,
                 holoscan::Arg("min_size", static_cast<uint64_t>(1)));
  spec.output<spark::gpu::GpuFramePtr>("out");
  spec.param(out_width_, "out_width", "Out width", "target width", 3840u);
  spec.param(out_height_, "out_height", "Out height", "target height", 2160u);
  spec.param(interp_, "interp", "Interpolation",
             "auto | linear | cubic | lanczos | super | fsrcnn | fsrcnn-s | espcn (AI x2 SR)",
             std::string("cubic"));
  spec.param(measure_, "measure", "Measure", "per-frame GPU timing (benchmark only; adds a sync)",
             false);
}

void ResizeOp::start() {
  std::string interp = interp_.get();
  if (interp == "ai") interp = "fsrcnn";  // friendly alias for the default AI model
  if (spark::sr::is_sr_interp(interp)) {
    sr_model_ = interp;
    interp_code_ = NPPI_INTER_CUBIC;  // chroma planes + the non-2x geometry fallback
  } else {
    interp_code_ = interp_code(interp);
  }
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
  HOLOSCAN_LOG_INFO("resize started: -> {}x{} interp={}{}", out_width_.get(), out_height_.get(),
                    interp_.get(), sr_model_.empty() ? "" : " (AI x2 SR on luma, cubic chroma)");
}

// Build (or rebuild on an input-geometry change) the SR engine. A failure — bad SPARK_SR_WEIGHTS
// file, allocation — must not take the pipeline down: log once and drop to the cubic path.
void ResizeOp::ensure_sr(uint32_t in_width, uint32_t in_height) {
  if (sr_engine_ && sr_engine_->in_width() == in_width && sr_engine_->in_height() == in_height)
    return;
  try {
    const char* file = std::getenv("SPARK_SR_WEIGHTS");
    const auto net = (file && *file) ? spark::sr::load_file(file, sr_model_)
                                     : spark::sr::load_embedded(sr_model_);
    sr_engine_ = std::make_unique<spark::sr::Engine>(net, in_width, in_height);
    HOLOSCAN_LOG_INFO("resize: AI SR engaged: {}  ({}x{} -> {}x{}){}", sr_engine_->plan(),
                      in_width, in_height, 2 * in_width, 2 * in_height,
                      (file && *file) ? std::string(" weights=") + file : std::string());
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("resize: AI SR init failed ({}) — falling back to cubic", e.what());
    sr_engine_.reset();
    sr_model_.clear();
  }
}

void ResizeOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                       holoscan::ExecutionContext&) {
  auto in = op_input.receive<spark::gpu::GpuFramePtr>("in");
  if (!in || !in.value()) return;
  const auto& src = *in.value();
  // 1:1 passthrough: same geometry in and out (e.g. 2160p -> 2160p IP10 transcode) means the NPP
  // resample would only burn GPU time and a pool buffer to produce an identical image — forward the
  // input frame instead (its ready event and timing metadata travel with it). Skipped when measuring
  // (the benchmark wants the kernel).
  if (src.width == out_width_.get() && src.height == out_height_.get() && !measure_.get()) {
    if (!identity_logged_) {
      identity_logged_ = true;
      HOLOSCAN_LOG_INFO("resize: {}x{} == target — 1:1 passthrough (NPP skipped)", src.width, src.height);
    }
    ++frames_;
    op_output.emit(in.value(), "out");
    return;
  }
  cudaStreamWaitEvent(stream_, src.ready, 0);  // order behind the producer's writes (cross-stream)
  auto dst = pool_[pool_idx_];
  pool_idx_ = (pool_idx_ + 1) % pool_.size();

  // Resolve the kernel for this direction. NPP's SUPER (supersampling) is the proper anti-aliased
  // DOWNSCALE and is downscale-only — it errors when a dimension grows — so "auto" picks it for
  // shrinks and cubic otherwise, and an explicit "super" falls back to cubic on a non-shrink.
  const bool down = dst->width < src.width && dst->height < src.height;
  int interp = interp_code_;
  if (interp == kAutoInterp) interp = down ? NPPI_INTER_SUPER : NPPI_INTER_CUBIC;
  else if (interp == NPPI_INTER_SUPER && !down) interp = NPPI_INTER_CUBIC;

  // AI SR is exact-2x only (the embedded nets are x2); other geometries take the cubic fallback
  // resolved above. The 4:2:2 chroma planes are (w/2 x h), so a 2x frame upscale is 2x for them
  // too — but the nets are trained on luma statistics, so chroma goes through NPP cubic instead.
  const bool sr_active = !sr_model_.empty() && dst->width == src.width * 2 &&
                         dst->height == src.height * 2;
  if (!sr_model_.empty() && !sr_active && !sr_fallback_logged_) {
    sr_fallback_logged_ = true;
    HOLOSCAN_LOG_WARN("resize: interp={} supports exact 2x only — {}x{} -> {}x{} uses cubic",
                      sr_model_, src.width, src.height, dst->width, dst->height);
  }
  if (sr_active) ensure_sr(src.width, src.height);  // clears sr_model_ if init fails

  const bool measure = measure_.get();
  auto a = static_cast<cudaEvent_t>(ev_start_);
  auto b = static_cast<cudaEvent_t>(ev_stop_);
  if (measure) cudaEventRecord(a);
  if (sr_active && sr_engine_) {
    sr_engine_->run(src.y, dst->y, stream_);
  } else {
    resize_plane(src.y, src.width, src.height, dst->y, dst->width, dst->height, interp, npp_ctx_);
  }
  resize_plane(src.cb, src.chroma_width(), src.height, dst->cb, dst->chroma_width(), dst->height,
               interp, npp_ctx_);
  resize_plane(src.cr, src.chroma_width(), src.height, dst->cr, dst->chroma_width(), dst->height,
               interp, npp_ctx_);
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
  sr_engine_.reset();  // after the sync: no SR kernels in flight
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
