// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

#include "frc.hpp"

#include <chrono>
#include <cstdlib>
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
  SPARK_LOG_INFO("spark_live frc_interpolated={}", frames_);
}

void FrcOp::setup(spark::rt::OperatorSpec& spec) {
  spec.input<spark::gpu::GpuFramePtr>("in");
  // Up-convert emits TWO frames per compute (real + mid) on "out"; retime emits one. Size the
  // transmitter to hold the burst until GXF delivers it downstream, and gate FRC's execution on the
  // downstream (resize) having room for the WHOLE burst. Without that gate the default room-for-1
  // condition lets FRC run with a single free slot, so a second emit overflows and GXF logs "Sync
  // failed" every frame; requiring room for `burst` makes FRC backpressure cleanly. burst==1 in retime
  // keeps the queue (and its latency) minimal.
  const auto burst = static_cast<uint64_t>(spark::pipeline_emit_burst());
  spec.output<spark::gpu::GpuFramePtr>("out")
      .connector(spark::rt::IOSpec::ConnectorType::kDoubleBuffer,
                 spark::rt::Arg("capacity", burst),  // exactly the emit burst (floor)
                 spark::rt::Arg("policy", static_cast<uint64_t>(2)))  // 2 = fault: warn, don't drop
      .condition(spark::rt::ConditionType::kDownstreamMessageAffordable,
                 spark::rt::Arg("min_size", burst));
  spec.param(phase_, "phase", "Phase", "interpolation t in [0,1] (0.5 = midpoint)", 0.5);
  spec.param(grid_size_, "grid_size", "NVOF grid", "flow output grid 1|2|4", 4u);
  spec.param(rate_mult_, "rate_mult", "Rate multiplier", "1 = retime (1:1), 2 = up-convert (2x)", 1u);
  spec.param(out_interval_ns_, "out_interval_ns", "Uniform output interval",
             ">0 = uniform-grid mode: one frame per nominal grid tick, phase from true capture times",
             uint64_t(0));
}

void FrcOp::ensure(uint32_t width, uint32_t height) {
  if (inited_) return;
  cudaStreamCreate(&stream_);  // default-flags: concurrent with other ops' streams, ordered vs stream 0
  flow_.init(width, height, grid_size_.get());
  cudaMalloc(reinterpret_cast<void**>(&prevY8_), static_cast<size_t>(width) * height);
  cudaMalloc(reinterpret_cast<void**>(&curY8_), static_cast<size_t>(width) * height);
  cudaMalloc(reinterpret_cast<void**>(&wmap_), static_cast<size_t>(width) * height * sizeof(float));
  const char* mv = std::getenv("SPARK_FRC_MEDIAN");
  median_ = mv ? std::atoi(mv) != 0 : true;
  const size_t grid_bytes =
      static_cast<size_t>(flow_.grid_w()) * flow_.grid_h() * 2 * sizeof(short);
  cudaMalloc(reinterpret_cast<void**>(&flow_med_f_), grid_bytes);
  cudaMalloc(reinterpret_cast<void**>(&flow_med_b_), grid_bytes);
  cudaMalloc(reinterpret_cast<void**>(&ws_.accum),
             spark::frc::warp_workspace_floats(flow_.grid_w(), flow_.grid_h()) * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&ws_.fwd_t), grid_bytes);
  cudaMalloc(reinterpret_cast<void**>(&ws_.bwd_t), grid_bytes);
  // Synthetic frames feed the downstream input queue (capacity == emit burst); the pool must exceed
  // that + the frames in flight (being read downstream / built here) so a transiently-full queue can
  // never alias a slot still in use. Frames are at native input resolution (FRC precedes resize).
  burst_ = spark::pipeline_emit_burst();
  grid_.interval_ns = out_interval_ns_.get();
  pool_.assign(2 * burst_ + 6, nullptr);
  for (auto& f : pool_) f = std::make_shared<spark::gpu::GpuFrame>(width, height);
  inited_ = true;
  SPARK_LOG_INFO(
      "frc: {}x{} grid={} phase={} rate_mult={} uniform_interval={}ns burst={} median={} cost={} "
      "temporal_hints={}",
      width, height, grid_size_.get(), phase_.get(), rate_mult_.get(), out_interval_ns_.get(),
      burst_, median_, flow_.has_cost(), flow_.temporal_hints());
}

void FrcOp::run_flow(const spark::gpu::GpuFramePtr& cur) {
  // Order our stream behind the producers of both inputs (cross-stream), then run the whole chain on
  // stream_: 10-bit luma -> 8-bit, NVOF flow (bound to stream_, no ctxSync), and the warp/blend. All
  // same-stream so they order without explicit syncs; the shared prevY8_/curY8_/NVOF buffers reuse
  // safely because FrcOp::compute is serial.
  cudaStreamWaitEvent(stream_, prev_->ready, 0);
  cudaStreamWaitEvent(stream_, cur->ready, 0);
  spark::frc::y10_to_y8(prev_->y, prevY8_, cur->width, cur->height, stream_);
  spark::frc::y10_to_y8(cur->y, curY8_, cur->width, cur->height, stream_);
  flow_.compute(prevY8_, curY8_, stream_);  // NVOF fwd (prev->cur) + bwd (cur->prev), async on stream_
  fview_ = flow_.view();
  if (median_) {  // 3x3 vector median: kills single-cell outliers before they warp
    const uint32_t packed_pitch = flow_.grid_w() * 2 * sizeof(short);
    spark::frc::median3x3_flow(fview_, flow_med_f_, flow_med_b_, packed_pitch, stream_);
    fview_.fwd = flow_med_f_;
    fview_.bwd = flow_med_b_;
    fview_.pitch_bytes = packed_pitch;
  }
}

void FrcOp::compute_uniform(const spark::gpu::GpuFramePtr& cur, spark::rt::OutputContext& op_output) {
  const uint64_t pc = src_prev_ts_, cc = cur->capture_ts_ns;
  src_prev_ts_ = cc;
  if (pc == 0 || cc <= pc) {  // no usable bracket (missing/non-monotonic capture stamps): pass through
    prev_ = cur;
    op_output.emit(cur, "out");
    return;
  }
  // Collect this bracket's grid ticks. A stall (bracket > 8 output intervals) freezes rather than
  // synthesizes — the grid jumps a whole number of intervals so the wire cadence never shifts.
  spark::frc::UniformGrid::Tick ticks[16];
  const int max_out = static_cast<int>(burst_ < 16u ? burst_ : 16u);
  const uint64_t before = grid_dropped_;
  const int n = grid_.ticks(pc, cc, ticks, max_out, 8 * grid_.interval_ns, &grid_dropped_);
  if (grid_dropped_ != before && ((grid_dropped_ / 16) != (before / 16)))
    SPARK_LOG_WARN("frc: uniform grid dropped {} ticks total (source stall / over-burst bracket)",
                      grid_dropped_);

  // A tick landing on cur itself (phase ~1) passes the REAL frame through, re-stamped onto the grid
  // — on a clean source the grid stays phase-aligned with the real frames, so this preserves their
  // full sharpness and only the in-between ticks are synthetic (same artifact profile as mode 2).
  constexpr float kPassHi = 0.99f;
  bool need_flow = false;
  for (int i = 0; i < n; ++i) need_flow |= !(i == n - 1 && ticks[i].phase >= kPassHi);
  if (need_flow) run_flow(cur);

  for (int i = 0; i < n; ++i) {
    if (i == n - 1 && ticks[i].phase >= kPassHi) {
      cur->capture_ts_ns = ticks[i].ts;  // re-stamp onto the grid (source ts lives in src_prev_ts_)
      op_output.emit(cur, "out");
      continue;
    }
    auto out = pool_[idx_];
    idx_ = (idx_ + 1) % pool_.size();
    spark::frc::interpolate(*prev_, *cur, fview_, ws_, prevY8_, curY8_, *out, wmap_,
                            ticks[i].phase, stream_);
    cudaEventRecord(out->ready, stream_);
    out->t_ingest_ns = cur->t_ingest_ns;  // rides cur's ingest time for the latency probe
    out->capture_ts_ns = ticks[i].ts;
    ++frames_;
    op_output.emit(out, "out");
  }
  prev_ = cur;
}

void FrcOp::compute(spark::rt::InputContext& op_input, spark::rt::OutputContext& op_output,
                    spark::rt::ExecutionContext&) {
  auto in = op_input.receive<spark::gpu::GpuFramePtr>("in");
  if (!in || !in.value()) return;
  auto cur = in.value();
  ensure(cur->width, cur->height);
  emit_live();  // 1 Hz live stats (throttled)
  // Stall localization (M8 determinism hunt): this compute is enqueue-only (NVOF Execute + warp
  // launches, all async), so any long wall time here is a CPU-side block — an NVOF API internal
  // sync, allocator, or lock — as opposed to the GPU-side stalls pack's fence catches.
  struct WallWarn {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ~WallWarn() {
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
      if (ms > 20) SPARK_LOG_WARN("frc stall: compute blocked {} ms on the CPU side", ms);
    }
  } wall_warn;

  if (!prev_) {  // first frame: nothing to interpolate from -> pass it through
    prev_ = cur;
    src_prev_ts_ = cur->capture_ts_ns;
    op_output.emit(cur, "out");  // carries the producer's (unpack) ready event downstream unchanged
    return;
  }

  if (grid_.interval_ns > 0) {  // uniform-grid mode (SPARK_FRC=3)
    compute_uniform(cur, op_output);
    return;
  }

  run_flow(cur);

  auto mid = pool_[idx_];
  idx_ = (idx_ + 1) % pool_.size();
  spark::frc::interpolate(*prev_, *cur, fview_, ws_, prevY8_, curY8_, *mid, wmap_,
                          static_cast<float>(phase_.get()), stream_);
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
  src_prev_ts_ = cc;

  // Emit. Up-convert (rate_mult>=2): the motion-comp MIDPOINT then the REAL current frame, so the
  // output stream is ... F0, M01, F1, M12, F2 ... at 2x the input rate (real frames preserved, only
  // the in-between frames synthetic). Retime (rate_mult==1): just the single motion-comp frame (1:1).
  op_output.emit(mid, "out");
  if (rate_mult_.get() >= 2) op_output.emit(cur, "out");
}

void FrcOp::stop() {
  emit_live(true);  // final live snapshot for the daemon
  SPARK_LOG_INFO("frc stopped: interpolated {} frames, uniform-grid ticks dropped {}", frames_,
                    grid_dropped_);
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
  if (flow_med_f_) {
    cudaFree(flow_med_f_);
    flow_med_f_ = nullptr;
  }
  if (flow_med_b_) {
    cudaFree(flow_med_b_);
    flow_med_b_ = nullptr;
  }
  if (ws_.accum) {
    cudaFree(ws_.accum);
    cudaFree(ws_.fwd_t);
    cudaFree(ws_.bwd_t);
    ws_ = {};
  }
}

}  // namespace spark::ops
