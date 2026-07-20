// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// FRC CUDA kernels: 10-bit luma -> 8-bit (for NVOF input), a 3x3 median post-filter for the flow
// grids, forward-splatting of the flow to the output phase, and an occlusion-aware
// motion-compensated warp/blend that synthesizes an interpolated frame at phase t in [0,1] from
// prev/cur using BOTH the forward (prev->cur) and backward (cur->prev) flow fields (see FlowView).
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "../resize/gpu_frame.hpp"
#include "flow_view.hpp"

namespace spark::frc {

void y10_to_y8(const uint16_t* y10, uint8_t* y8, uint32_t w, uint32_t h, cudaStream_t s);

// 3x3 component-wise median of BOTH flow grids in `in` (borders clamped), written to
// fwd_out/bwd_out (SHORT2, out_pitch_bytes row pitch). Kills single-cell outlier vectors before
// they warp; run it and point the FlowView at the filtered copies.
void median3x3_flow(const FlowView& in, void* fwd_out, void* bwd_out, uint32_t out_pitch_bytes,
                    cudaStream_t s);

// Scratch device buffers interpolate() needs for the per-phase forward splat. accum holds
// warp_workspace_floats(grid_w, grid_h) floats; fwd_t/bwd_t are packed SHORT2 grids
// (grid_w x grid_h each). Contents are transient — valid only within one interpolate() call.
struct WarpWorkspace {
  float* accum = nullptr;
  short* fwd_t = nullptr;  // forward field re-indexed at phase t
  short* bwd_t = nullptr;  // backward field re-indexed at phase 1-t
};
size_t warp_workspace_floats(uint32_t grid_w, uint32_t grid_h);

// Occlusion-aware bidirectional interpolation at phase t. Both flow fields are first
// forward-splatted to the output phase (confidence-weighted, holes fall back to the raw field), so
// each output pixel reads the trajectory that actually crosses it; the splatted grids are sampled
// edge-aware (joint-bilateral against the prevY8/curY8 luma guides) so motion boundaries snap to
// image edges instead of the flow-grid cells. The luma kernel then blends a forward candidate
// (warp prev) and a backward candidate (warp cur) weighted by photometric end-to-end residual x
// fwd<->bwd vector agreement x OFA cost confidence (when flow.cost_* is set) x the temporal prior —
// killing the 50/50 ghost in revealed/occluded regions. The per-pixel luma blend weight is written
// to `wmap` (width x height floats) and reused for chroma so both planes share one decision.
// prevY8/curY8: the same tightly-packed 8-bit luma planes fed to NvofFlow::compute().
void interpolate(const spark::gpu::GpuFrame& prev, const spark::gpu::GpuFrame& cur,
                 const FlowView& flow, const WarpWorkspace& ws, const uint8_t* prevY8,
                 const uint8_t* curY8, spark::gpu::GpuFrame& out, float* wmap, float t,
                 cudaStream_t s);

}  // namespace spark::frc
