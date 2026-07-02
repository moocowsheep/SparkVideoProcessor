#include "frc.hpp"

#include <chrono>
#include <stdexcept>

#include <cuda_runtime.h>

#include "../pipeline_caps.hpp"
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
  // Up-convert emits TWO frames per compute (real + mid) on "out"; retime emits one. Size the
  // transmitter to hold the burst until GXF delivers it downstream, and gate FRC's execution on the
  // downstream (resize) having room for the WHOLE burst. Without that gate the default room-for-1
  // condition lets FRC run with a single free slot, so a second emit overflows and GXF logs "Sync
  // failed" every frame; requiring room for `burst` makes FRC backpressure cleanly. burst==1 in retime
  // keeps the queue (and its latency) minimal.
  const auto burst = static_cast<uint64_t>(spark::pipeline_emit_burst());
  spec.output<spark::gpu::GpuFramePtr>("out")
      .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,
                 holoscan::Arg("capacity", burst),  // exactly the emit burst (floor)
                 holoscan::Arg("policy", static_cast<uint64_t>(2)))  // 2 = fault: warn, don't drop
      .condition(holoscan::ConditionType::kDownstreamMessageAffordable,
                 holoscan::Arg("min_size", burst));
  spec.param(phase_, "phase", "Phase", "interpolation t in [0,1] (0.5 = midpoint)", 0.5);
  spec.param(grid_size_, "grid_size", "NVOF grid", "flow output grid 1|2|4", 4u);
  spec.param(rate_mult_, "rate_mult", "Rate multiplier", "1 = retime (1:1), 2 = up-convert (2x)", 1u);
}

void FrcOp::ensure(uint32_t width, uint32_t height) {
  if (inited_) return;
  cudaStreamCreate(&stream_);  // default-flags: concurrent with other ops' streams, ordered vs stream 0
  flow_.init(width, height, grid_size_.get());
  cudaMalloc(reinterpret_cast<void**>(&prevY8_), static_cast<size_t>(width) * height);
  cudaMalloc(reinterpret_cast<void**>(&curY8_), static_cast<size_t>(width) * height);
  cudaMalloc(reinterpret_cast<void**>(&wmap_), static_cast<size_t>(width) * height * sizeof(float));
  // Up-convert mids feed the resize input queue (capacity 2); the pool must exceed that + the mids in
  // flight (being read by resize / built here) so a transiently-full queue can never alias a slot still
  // in use. Mids are at native input resolution now (FRC precedes resize), so this is cheap.
  pool_.assign(10, nullptr);
  for (auto& f : pool_) f = std::make_shared<spark::gpu::GpuFrame>(width, height);
  inited_ = true;
  HOLOSCAN_LOG_INFO("frc: {}x{} grid={} phase={} rate_mult={}", width, height, grid_size_.get(),
                    phase_.get(), rate_mult_.get());
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
    op_output.emit(cur, "out");  // carries the producer's (unpack) ready event downstream unchanged
    return;
  }

  // Order our stream behind the producers of both inputs (cross-stream), then run the whole chain on
  // stream_: 10-bit luma -> 8-bit, NVOF flow (bound to stream_, no ctxSync), and the warp/blend. All
  // same-stream so they order without explicit syncs; the shared prevY8_/curY8_/NVOF buffers reuse
  // safely because FrcOp::compute is serial.
  cudaStreamWaitEvent(stream_, prev_->ready, 0);
  cudaStreamWaitEvent(stream_, cur->ready, 0);
  spark::frc::y10_to_y8(prev_->y, prevY8_, cur->width, cur->height, stream_);
  spark::frc::y10_to_y8(cur->y, curY8_, cur->width, cur->height, stream_);

  flow_.compute(prevY8_, curY8_, stream_);  // NVOF fwd (prev->cur) + bwd (cur->prev), async on stream_

  auto mid = pool_[idx_];
  idx_ = (idx_ + 1) % pool_.size();
  spark::frc::interpolate(*prev_, *cur, flow_.flow_dev(), flow_.flow_dev_bwd(),
                          flow_.flow_pitch_bytes(), flow_.grid_w(), flow_.grid_h(),
                          flow_.grid_size(), *mid, wmap_, static_cast<float>(phase_.get()), stream_);
  cudaEventRecord(mid->ready, stream_);  // mid is ready once interpolate completes on stream_
  mid->t_ingest_ns = cur->t_ingest_ns;   // mid rides cur's ingest time for the latency probe
  // The mid is the temporal MIDPOINT of (prev, cur), so stamp it halfway between their capture times.
  // Stamping it cur's time (as before) made the up-convert pair (mid, cur) carry IDENTICAL capture_ts,
  // so the genlock TX scheduled both on the same base — the second frame of every pair past-stamped
  // into an unpaced burst. Midpoint stamping spaces the pair exactly one output interval apart; for
  // retime (mid only) it's a constant half-frame shift the genlock calibration absorbs.
  const uint64_t pc = prev_->capture_ts_ns, cc = cur->capture_ts_ns;
  mid->capture_ts_ns = (pc != 0 && cc > pc) ? cc - (cc - pc) / 2 : cc;
  ++frames_;
  prev_ = cur;

  // Emit. Up-convert (rate_mult>=2): the motion-comp MIDPOINT then the REAL current frame, so the
  // output stream is ... F0, M01, F1, M12, F2 ... at 2x the input rate (real frames preserved, only
  // the in-between frames synthetic). Retime (rate_mult==1): just the single motion-comp frame (1:1).
  op_output.emit(mid, "out");
  if (rate_mult_.get() >= 2) op_output.emit(cur, "out");
}

void FrcOp::stop() {
  emit_live(true);  // final live snapshot for the daemon
  HOLOSCAN_LOG_INFO("frc stopped: interpolated {} frames", frames_);
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  if (prevY8_) {
    cudaFree(prevY8_);
    prevY8_ = nullptr;
  }
  if (curY8_) {
    cudaFree(curY8_);
    curY8_ = nullptr;
  }
  if (wmap_) {
    cudaFree(wmap_);
    wmap_ = nullptr;
  }
}

}  // namespace spark::ops
