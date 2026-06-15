#include "frc.hpp"

#include <chrono>
#include <stdexcept>

#include <cuda_runtime.h>

#include "frc_kernels.hpp"

namespace spark::ops {

void FrcOp::emit_live(bool force) {
  const double t =
      std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if (!force && t - last_live_s_ < 1.0) return;
  last_live_s_ = t;
  HOLOSCAN_LOG_INFO("spark_live frc_interpolated={}", frames_);
}

void FrcOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<spark::gpu::GpuFramePtr>("in");
  spec.output<spark::gpu::GpuFramePtr>("out");
  spec.param(phase_, "phase", "Phase", "interpolation t in [0,1] (0.5 = midpoint)", 0.5);
  spec.param(grid_size_, "grid_size", "NVOF grid", "flow output grid 1|2|4", 4u);
}

void FrcOp::ensure(uint32_t width, uint32_t height) {
  if (inited_) return;
  flow_.init(width, height, grid_size_.get());
  cudaMalloc(reinterpret_cast<void**>(&prevY8_), static_cast<size_t>(width) * height);
  cudaMalloc(reinterpret_cast<void**>(&curY8_), static_cast<size_t>(width) * height);
  pool_.assign(4, nullptr);
  for (auto& f : pool_) f = std::make_shared<spark::gpu::GpuFrame>(width, height);
  inited_ = true;
  HOLOSCAN_LOG_INFO("frc: {}x{} grid={} phase={}", width, height, grid_size_.get(), phase_.get());
}

void FrcOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                    holoscan::ExecutionContext&) {
  auto in = op_input.receive<spark::gpu::GpuFramePtr>("in");
  if (!in || !in.value()) return;
  auto cur = in.value();
  ensure(cur->width, cur->height);
  emit_live();  // 1 Hz live stats (throttled)

  if (!prev_) {  // first frame: nothing to interpolate from -> pass it through
    prev_ = cur;
    op_output.emit(cur, "out");
    return;
  }

  // 10-bit luma -> 8-bit for the optical-flow engine. All on the default stream, which orders these
  // kernels before NVOF's cuMemcpy2D — no explicit sync needed (NVOF's own ctxSync gates the flow,
  // and downstream pack is default-stream-ordered after interpolate). Removing the per-frame
  // cudaDeviceSynchronize() cuts the pipeline's pacing jitter / latency.
  spark::frc::y10_to_y8(prev_->y, prevY8_, cur->width, cur->height, 0);
  spark::frc::y10_to_y8(cur->y, curY8_, cur->width, cur->height, 0);

  flow_.compute(prevY8_, curY8_);  // NVOF prev->cur (ctxSync inside; flow ready on return)

  auto out = pool_[idx_];
  idx_ = (idx_ + 1) % pool_.size();
  spark::frc::interpolate(*prev_, *cur, flow_.flow_dev(), flow_.flow_pitch_bytes(), flow_.grid_w(),
                          flow_.grid_h(), flow_.grid_size(), *out,
                          static_cast<float>(phase_.get()), 0);
  ++frames_;
  prev_ = cur;
  op_output.emit(out, "out");
}

void FrcOp::stop() {
  emit_live(true);  // final live snapshot for the daemon
  HOLOSCAN_LOG_INFO("frc stopped: interpolated {} frames", frames_);
  if (prevY8_) {
    cudaFree(prevY8_);
    prevY8_ = nullptr;
  }
  if (curY8_) {
    cudaFree(curY8_);
    curY8_ = nullptr;
  }
}

}  // namespace spark::ops
