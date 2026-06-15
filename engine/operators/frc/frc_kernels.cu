#include "frc_kernels.hpp"

namespace spark::frc {
namespace {

__global__ void y10_to_y8_k(const uint16_t* __restrict__ y10, uint8_t* __restrict__ y8,
                            unsigned long long n) {
  const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y8[i] = (uint8_t)(y10[i] >> 2);  // 10-bit -> 8-bit luma
}

__device__ inline float bilinear(const uint16_t* p, int w, int h, float x, float y) {
  x = fminf(fmaxf(x, 0.0f), w - 1.0f);
  y = fminf(fmaxf(y, 0.0f), h - 1.0f);
  const int x0 = (int)x, y0 = (int)y;
  const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
  const float fx = x - x0, fy = y - y0;
  const float a = p[(long)y0 * w + x0], b = p[(long)y0 * w + x1];
  const float c = p[(long)y1 * w + x0], d = p[(long)y1 * w + x1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

// One thread per output pixel. xsub = 1 for luma, 2 for 4:2:2 chroma (half width). The flow grid is
// in luma coordinates; chroma motion in x is flow_x / xsub.
__global__ void warp_k(const uint16_t* __restrict__ prev, const uint16_t* __restrict__ cur,
                       const short* __restrict__ flow, uint32_t flow_pitch_elems, uint32_t grid_w,
                       uint32_t grid_h, uint32_t grid_size, uint16_t* __restrict__ out, uint32_t pw,
                       uint32_t ph, uint32_t xsub, float t) {
  const uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= pw || py >= ph) return;

  const uint32_t lumax = px * xsub;
  const uint32_t gx = min(lumax / grid_size, grid_w - 1);
  const uint32_t gy = min(py / grid_size, grid_h - 1);
  const short* fv = flow + (unsigned long long)gy * flow_pitch_elems + (unsigned long long)gx * 2;
  const float fx = (fv[0] / 32.0f) / xsub;  // plane-x pixels
  const float fy = fv[1] / 32.0f;           // plane-y pixels (no vertical subsample in 4:2:2)

  const float pv = bilinear(prev, pw, ph, px - t * fx, py - t * fy);
  const float cv = bilinear(cur, pw, ph, px + (1.0f - t) * fx, py + (1.0f - t) * fy);
  out[(unsigned long long)py * pw + px] = (uint16_t)(pv * (1.0f - t) + cv * t + 0.5f);
}

}  // namespace

void y10_to_y8(const uint16_t* y10, uint8_t* y8, uint32_t w, uint32_t h, cudaStream_t s) {
  const unsigned long long n = (unsigned long long)w * h;
  const int block = 256;
  y10_to_y8_k<<<(unsigned int)((n + block - 1) / block), block, 0, s>>>(y10, y8, n);
}

void interpolate(const spark::gpu::GpuFrame& prev, const spark::gpu::GpuFrame& cur,
                 const void* flow_dev, uint32_t flow_pitch_bytes, uint32_t grid_w, uint32_t grid_h,
                 uint32_t grid_size, spark::gpu::GpuFrame& out, float t, cudaStream_t s) {
  const short* flow = reinterpret_cast<const short*>(flow_dev);
  const uint32_t fpe = flow_pitch_bytes / 2;  // int16 elements per flow row
  const dim3 blk(16, 16);
  auto launch = [&](const uint16_t* pp, const uint16_t* cc, uint16_t* oo, uint32_t pw, uint32_t ph,
                    uint32_t xsub) {
    const dim3 grd((pw + 15) / 16, (ph + 15) / 16);
    warp_k<<<grd, blk, 0, s>>>(pp, cc, flow, fpe, grid_w, grid_h, grid_size, oo, pw, ph, xsub, t);
  };
  launch(prev.y, cur.y, out.y, prev.width, prev.height, 1);
  launch(prev.cb, cur.cb, out.cb, prev.chroma_width(), prev.height, 2);
  launch(prev.cr, cur.cr, out.cr, prev.chroma_width(), prev.height, 2);
}

}  // namespace spark::frc
