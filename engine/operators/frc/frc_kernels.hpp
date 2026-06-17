// FRC CUDA kernels: 10-bit luma -> 8-bit (for NVOF input), and occlusion-aware motion-compensated
// warp/blend that synthesizes an interpolated frame at phase t in [0,1] from prev/cur using BOTH the
// forward (prev->cur) and backward (cur->prev) flow fields.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "../resize/gpu_frame.hpp"

namespace spark::frc {

void y10_to_y8(const uint16_t* y10, uint8_t* y8, uint32_t w, uint32_t h, cudaStream_t s);

// Occlusion-aware bidirectional interpolation. flow_fwd/flow_bwd: NVOF SHORT2 grids (grid_w x grid_h,
// S10.5, luma coords) for prev->cur and cur->prev respectively (same pitch). For each output pixel the
// luma kernel forms a forward candidate (warp prev) and a backward candidate (warp cur), measures each
// trajectory's end-to-end photometric residual (occlusion test), and blends by reliability x temporal
// prior — killing the 50/50 ghost in revealed/occluded regions. The per-pixel luma blend weight is
// written to `wmap` (width x height floats) and reused for chroma so both planes share one decision.
void interpolate(const spark::gpu::GpuFrame& prev, const spark::gpu::GpuFrame& cur,
                 const void* flow_fwd, const void* flow_bwd, uint32_t flow_pitch_bytes,
                 uint32_t grid_w, uint32_t grid_h, uint32_t grid_size, spark::gpu::GpuFrame& out,
                 float* wmap, float t, cudaStream_t s);

}  // namespace spark::frc
