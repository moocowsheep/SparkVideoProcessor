#include "filter_kernels.hpp"

#include <cmath>

namespace spark::filters {
namespace {

// 10-bit reserved code words are 0-3 and 1020-1023 (SMPTE); a proc amp may push past legal video
// levels (that's what it's for) but must never emit protected codes.
__device__ inline uint16_t clamp10(float v) {
  const int i = __float2int_rn(v);
  return static_cast<uint16_t>(min(max(i, 4), 1019));
}

__global__ void procamp_y_k(const uint16_t* __restrict__ in, uint16_t* __restrict__ out, size_t n,
                            float gain, float offset_cv) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = clamp10(64.0f + (static_cast<float>(in[i]) - 64.0f) * gain + offset_cv);
}

// Chroma: one thread per 4:2:2 sample pair — hue rotation mixes Cb and Cr, so both planes are read
// and written together.
__global__ void procamp_c_k(const uint16_t* __restrict__ cb_in, const uint16_t* __restrict__ cr_in,
                            uint16_t* __restrict__ cb_out, uint16_t* __restrict__ cr_out, size_t n,
                            float sat, float cos_h, float sin_h) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float b = static_cast<float>(cb_in[i]) - 512.0f;
  const float r = static_cast<float>(cr_in[i]) - 512.0f;
  cb_out[i] = clamp10(512.0f + sat * (b * cos_h - r * sin_h));
  cr_out[i] = clamp10(512.0f + sat * (b * sin_h + r * cos_h));
}

// Unsharp: y + amount * (y - blur), blur = 3x3 gaussian (1 2 1 / 2 4 2 / 1 2 1)/16, edge-clamped.
__global__ void unsharp_y_k(const uint16_t* __restrict__ in, uint16_t* __restrict__ out, uint32_t w,
                            uint32_t h, float amount) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const uint32_t xl = x > 0 ? x - 1 : 0, xr = x + 1 < w ? x + 1 : w - 1;
  const uint32_t yu = y > 0 ? y - 1 : 0, yd = y + 1 < h ? y + 1 : h - 1;
  const float c = in[(size_t)y * w + x];
  const float blur = (static_cast<float>(in[(size_t)yu * w + xl]) + 2.0f * in[(size_t)yu * w + x] +
                      in[(size_t)yu * w + xr] + 2.0f * in[(size_t)y * w + xl] + 4.0f * c +
                      2.0f * in[(size_t)y * w + xr] + in[(size_t)yd * w + xl] +
                      2.0f * in[(size_t)yd * w + x] + in[(size_t)yd * w + xr]) *
                     (1.0f / 16.0f);
  out[(size_t)y * w + x] = clamp10(c + amount * (c - blur));
}

constexpr int kBlock = 256;
inline unsigned int grid1d(size_t n) { return (unsigned int)((n + kBlock - 1) / kBlock); }

}  // namespace

void procamp(const uint16_t* y_in, const uint16_t* cb_in, const uint16_t* cr_in, uint16_t* y_out,
             uint16_t* cb_out, uint16_t* cr_out, uint32_t width, uint32_t height, float brightness,
             float contrast, float saturation, float hue_deg, cudaStream_t stream) {
  const size_t ny = (size_t)width * height;
  const size_t nc = (size_t)(width / 2) * height;
  const float offset_cv = brightness * 876.0f;  // ±1.0 = ±the full 10-bit Y swing (940-64)
  const float rad = hue_deg * 3.14159265358979f / 180.0f;
  procamp_y_k<<<grid1d(ny), kBlock, 0, stream>>>(y_in, y_out, ny, contrast, offset_cv);
  procamp_c_k<<<grid1d(nc), kBlock, 0, stream>>>(cb_in, cr_in, cb_out, cr_out, nc, saturation,
                                                 std::cos(rad), std::sin(rad));
}

void unsharp_y(const uint16_t* y_in, uint16_t* y_out, uint32_t width, uint32_t height, float amount,
               cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  unsharp_y_k<<<grid, block, 0, stream>>>(y_in, y_out, width, height, amount);
}

}  // namespace spark::filters
