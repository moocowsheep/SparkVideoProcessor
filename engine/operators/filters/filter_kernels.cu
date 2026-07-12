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

// --- film grain (REDStreamer grainKernel, adapted to planar 10-bit YCbCr) ---

// murmur3-finalizer mix over a lattice coordinate + per-frame/channel salt; deterministic,
// stateless, whole-image parallel.
__device__ __forceinline__ uint32_t grain_hash(uint32_t x, uint32_t y, uint32_t salt) {
  uint32_t h = x * 0x8da6b343u + y * 0xd8163841u + salt * 0xcb1ab31fu;
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  h *= 0xc2b2ae35u;
  h ^= h >> 16;
  return h;
}

// Zero-mean triangular noise in [-1,1] from one hash (sum of the two 16-bit halves): closer to
// Gaussian than raw uniform, one draw per tap.
__device__ __forceinline__ float grain_tri(uint32_t h) {
  return ((h & 0xffffu) + (h >> 16)) * (1.0f / 65535.0f) - 1.0f;
}

// One value-noise sample: smoothstep-bilerped lattice of triangular draws. size 1 degenerates to
// per-pixel noise (fx = fy = 0 exactly).
__device__ __forceinline__ float grain_noise(float gx, float gy, uint32_t salt) {
  const float fxi = floorf(gx), fyi = floorf(gy);
  const int ix = static_cast<int>(fxi), iy = static_cast<int>(fyi);
  float fx = gx - fxi, fy = gy - fyi;
  fx = fx * fx * (3.0f - 2.0f * fx);
  fy = fy * fy * (3.0f - 2.0f * fy);
  const float n00 = grain_tri(grain_hash(ix, iy, salt));
  const float n10 = grain_tri(grain_hash(ix + 1, iy, salt));
  const float n01 = grain_tri(grain_hash(ix, iy + 1, salt));
  const float n11 = grain_tri(grain_hash(ix + 1, iy + 1, salt));
  const float top = n00 + fx * (n10 - n00);
  const float bot = n01 + fx * (n11 - n01);
  return top + fy * (bot - top);
}

// Film density response from 10-bit luma: 0 at/below black (64) and at/above white (940), peaking
// at mid-gray — grain lives in the mid-tones and the extremes stay bit-exact.
__device__ __forceinline__ float grain_weight(float y_cv) {
  const float L = fminf(fmaxf((y_cv - 64.0f) * (1.0f / 876.0f), 0.0f), 1.0f);
  return 4.0f * L * (1.0f - L);
}

__global__ void grain_y_k(const uint16_t* __restrict__ in, uint16_t* __restrict__ out, uint32_t w,
                          uint32_t h, float strength_cv, float inv_size, uint32_t seed) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const size_t i = (size_t)y * w + x;
  const float c = in[i];
  const float wgt = grain_weight(c);
  if (wgt <= 0.0f) {  // pure black/white: bit-exact passthrough
    out[i] = in[i];
    return;
  }
  const float gx = (x + 0.5f) * inv_size - 0.5f;
  const float gy = (y + 0.5f) * inv_size - 0.5f;
  out[i] = clamp10(c + strength_cv * wgt * grain_noise(gx, gy, seed * 4u + 3u));
}

// Chroma grain: one thread per 4:2:2 sample pair, weight from the co-sited (left) luma sample.
// The lattice is walked in luma-pixel units (2x per chroma sample) so the grain cell size matches
// the Y field visually; salts 0/1 keep Cb/Cr fields independent of each other and of Y (salt 3).
__global__ void grain_c_k(const uint16_t* __restrict__ y_in, const uint16_t* __restrict__ cb_in,
                          const uint16_t* __restrict__ cr_in, uint16_t* __restrict__ cb_out,
                          uint16_t* __restrict__ cr_out, uint32_t w, uint32_t h, float strength_cv,
                          float inv_size, uint32_t seed) {
  const uint32_t cw = w / 2;
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cw || y >= h) return;
  const size_t i = (size_t)y * cw + x;
  const float wgt = grain_weight(y_in[(size_t)y * w + x * 2]);
  if (wgt <= 0.0f) {
    cb_out[i] = cb_in[i];
    cr_out[i] = cr_in[i];
    return;
  }
  const float gx = (x * 2 + 0.5f) * inv_size - 0.5f;
  const float gy = (y + 0.5f) * inv_size - 0.5f;
  const float s = strength_cv * wgt;
  cb_out[i] = clamp10(static_cast<float>(cb_in[i]) + s * grain_noise(gx, gy, seed * 4u));
  cr_out[i] = clamp10(static_cast<float>(cr_in[i]) + s * grain_noise(gx, gy, seed * 4u + 1u));
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
  // Positive hue = CLOCKWISE on the vectorscope (skin toward red/warm), matching the common analog
  // phase-control expectation — hence the negation into the CCW (Cb,Cr) rotation below.
  const float rad = -hue_deg * 3.14159265358979f / 180.0f;
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

// amount 1 -> peak amplitude 0.08 of the swing at mid-gray; triangular noise makes that ~3.3% RMS
// (0.08 / sqrt(6)) — same tuning as REDStreamer's grainRgb48.
void grain_y(const uint16_t* y_in, uint16_t* y_out, uint32_t width, uint32_t height, float amount,
             float size, uint32_t seed, cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  const float strength_cv = 0.08f * fminf(fmaxf(amount, 0.0f), 1.0f) * 876.0f;  // Y swing 64..940
  const float inv_size = 1.0f / fminf(fmaxf(size, 1.0f), 4.0f);
  grain_y_k<<<grid, block, 0, stream>>>(y_in, y_out, width, height, strength_cv, inv_size, seed);
}

void grain_c(const uint16_t* y_in, const uint16_t* cb_in, const uint16_t* cr_in, uint16_t* cb_out,
             uint16_t* cr_out, uint32_t width, uint32_t height, float amount, float size,
             uint32_t seed, cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid((width / 2 + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  const float strength_cv = 0.08f * fminf(fmaxf(amount, 0.0f), 1.0f) * 448.0f;  // chroma swing ±448
  const float inv_size = 1.0f / fminf(fmaxf(size, 1.0f), 4.0f);
  grain_c_k<<<grid, block, 0, stream>>>(y_in, cb_in, cr_in, cb_out, cr_out, width, height,
                                        strength_cv, inv_size, seed);
}

}  // namespace spark::filters
