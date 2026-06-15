// FRC CUDA kernels: 10-bit luma -> 8-bit (for NVOF input), and motion-compensated warp/blend that
// synthesizes an interpolated frame at phase t in [0,1] from prev/cur using a forward flow field.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "../resize/gpu_frame.hpp"

namespace spark::frc {

void y10_to_y8(const uint16_t* y10, uint8_t* y8, uint32_t w, uint32_t h, cudaStream_t s);

// flow_dev: NVOF SHORT2 grid (grid_w x grid_h, S10.5, luma coords). Warps prev/cur into `out` (same
// dims as prev) at phase t: out = (1-t)*prev(p - t*f) + t*cur(p + (1-t)*f), per plane.
void interpolate(const spark::gpu::GpuFrame& prev, const spark::gpu::GpuFrame& cur,
                 const void* flow_dev, uint32_t flow_pitch_bytes, uint32_t grid_w, uint32_t grid_h,
                 uint32_t grid_size, spark::gpu::GpuFrame& out, float t, cudaStream_t s);

}  // namespace spark::frc
