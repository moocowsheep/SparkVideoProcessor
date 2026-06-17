#include "frc_kernels.hpp"

namespace spark::frc {
namespace {

// Photometric tolerance for the forward<->backward occlusion test, in 10-bit luma units. A matched
// (non-occluded) trajectory has a small end-to-end residual (sensor noise + warp); an occluded one
// jumps to the fore/background luma gap (often hundreds). exp(-residual/sigma) turns that into a soft
// per-side reliability, so the blend degrades gracefully rather than hard-switching at a threshold.
constexpr float kOcclSigma = 20.0f;

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

// Sample a SHORT2 flow grid at luma pixel (lumax, lumay); returns (dx, dy) in luma pixels.
__device__ inline float2 sample_flow(const short* flow, uint32_t pitch_elems, uint32_t grid_w,
                                     uint32_t grid_h, uint32_t grid_size, uint32_t lumax,
                                     uint32_t lumay) {
  const uint32_t gx = min(lumax / grid_size, grid_w - 1);
  const uint32_t gy = min(lumay / grid_size, grid_h - 1);
  const short* fv = flow + (unsigned long long)gy * pitch_elems + (unsigned long long)gx * 2;
  return make_float2(fv[0] / 32.0f, fv[1] / 32.0f);  // S10.5 fixed point
}

// Compute the occlusion-aware blend WEIGHT (weight of the cur/backward candidate, in [0,1]) for one
// luma output pixel, using both flows and the photometric end-to-end consistency of each trajectory.
__device__ inline float blend_weight(const uint16_t* __restrict__ prev,
                                      const uint16_t* __restrict__ cur, float2 Ff, float2 Fb,
                                      int W, int H, float px, float py, float t, float& cand_f_out,
                                      float& cand_b_out) {
  // Forward candidate: prev content that reaches output position px at time t (prev->cur motion Ff).
  const float cand_f = bilinear(prev, W, H, px - t * Ff.x, py - t * Ff.y);
  // Backward candidate: cur content that reaches px at time t (cur->prev motion Fb).
  const float cand_b = bilinear(cur, W, H, px - (1.0f - t) * Fb.x, py - (1.0f - t) * Fb.y);
  // End-to-end residuals: follow each trajectory to the *other* real frame. If the forward vector is
  // valid (visible in both), prev(source) ~= cur(source + Ff); a large gap means that path crosses an
  // occlusion and cand_f is unreliable. Symmetric for the backward path.
  const float endf = bilinear(cur, W, H, px + (1.0f - t) * Ff.x, py + (1.0f - t) * Ff.y);
  const float endb = bilinear(prev, W, H, px + t * Fb.x, py + t * Fb.y);
  const float relf = __expf(-fabsf(cand_f - endf) / kOcclSigma);  // forward-side reliability
  const float relb = __expf(-fabsf(cand_b - endb) / kOcclSigma);  // backward-side reliability
  // Temporal prior (cur weighs t, prev weighs 1-t) modulated by reliability. In consistent regions
  // relf==relb and this reduces to the plain temporal blend; in occlusions the visible side wins.
  const float nf = (1.0f - t) * relf;
  const float nb = t * relb;
  cand_f_out = cand_f;
  cand_b_out = cand_b;
  return nb / (nf + nb + 1e-6f);
}

// Luma: decide the occlusion-aware weight, write the interpolated luma AND publish the weight to wmap
// so the chroma planes reuse the exact same per-pixel decision (chroma is too noisy to re-derive it).
__global__ void warp_luma_k(const uint16_t* __restrict__ prev, const uint16_t* __restrict__ cur,
                            const short* __restrict__ ffwd, const short* __restrict__ fbwd,
                            uint32_t fpe, uint32_t grid_w, uint32_t grid_h, uint32_t grid_size,
                            uint16_t* __restrict__ out, float* __restrict__ wmap, uint32_t W,
                            uint32_t H, float t) {
  const uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= W || py >= H) return;
  const float2 Ff = sample_flow(ffwd, fpe, grid_w, grid_h, grid_size, px, py);
  const float2 Fb = sample_flow(fbwd, fpe, grid_w, grid_h, grid_size, px, py);
  float cand_f, cand_b;
  const float wb =
      blend_weight(prev, cur, Ff, Fb, W, H, px, py, t, cand_f, cand_b);
  wmap[(unsigned long long)py * W + px] = wb;
  out[(unsigned long long)py * W + px] = (uint16_t)((1.0f - wb) * cand_f + wb * cand_b + 0.5f);
}

// Chroma (4:2:2, half width): warp both candidates and blend with the luma-derived weight. The flow
// grid is in luma coords; chroma x-motion is flow_x / xsub (no vertical subsample in 4:2:2).
__global__ void warp_chroma_k(const uint16_t* __restrict__ prev, const uint16_t* __restrict__ cur,
                              const short* __restrict__ ffwd, const short* __restrict__ fbwd,
                              uint32_t fpe, uint32_t grid_w, uint32_t grid_h, uint32_t grid_size,
                              const float* __restrict__ wmap, uint32_t W, uint16_t* __restrict__ out,
                              uint32_t cw, uint32_t ch, uint32_t xsub, float t) {
  const uint32_t cx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t cy = blockIdx.y * blockDim.y + threadIdx.y;
  if (cx >= cw || cy >= ch) return;
  const uint32_t lumax = cx * xsub;
  float2 Ff = sample_flow(ffwd, fpe, grid_w, grid_h, grid_size, lumax, cy);
  float2 Fb = sample_flow(fbwd, fpe, grid_w, grid_h, grid_size, lumax, cy);
  Ff.x /= xsub;  // luma-px -> chroma-px in x
  Fb.x /= xsub;
  const float cand_f = bilinear(prev, cw, ch, cx - t * Ff.x, cy - t * Ff.y);
  const float cand_b = bilinear(cur, cw, ch, cx - (1.0f - t) * Fb.x, cy - (1.0f - t) * Fb.y);
  const float wb = wmap[(unsigned long long)cy * W + lumax];  // shared luma occlusion decision
  out[(unsigned long long)cy * cw + cx] = (uint16_t)((1.0f - wb) * cand_f + wb * cand_b + 0.5f);
}

}  // namespace

void y10_to_y8(const uint16_t* y10, uint8_t* y8, uint32_t w, uint32_t h, cudaStream_t s) {
  const unsigned long long n = (unsigned long long)w * h;
  const int block = 256;
  y10_to_y8_k<<<(unsigned int)((n + block - 1) / block), block, 0, s>>>(y10, y8, n);
}

void interpolate(const spark::gpu::GpuFrame& prev, const spark::gpu::GpuFrame& cur,
                 const void* flow_fwd, const void* flow_bwd, uint32_t flow_pitch_bytes,
                 uint32_t grid_w, uint32_t grid_h, uint32_t grid_size, spark::gpu::GpuFrame& out,
                 float* wmap, float t, cudaStream_t s) {
  const short* ff = reinterpret_cast<const short*>(flow_fwd);
  const short* fb = reinterpret_cast<const short*>(flow_bwd);
  const uint32_t fpe = flow_pitch_bytes / 2;  // int16 elements per flow row
  const uint32_t W = prev.width, H = prev.height;
  const dim3 blk(16, 16);
  // Luma first: it produces wmap, which the chroma kernels read. Same stream => ordered.
  const dim3 grd_y((W + 15) / 16, (H + 15) / 16);
  warp_luma_k<<<grd_y, blk, 0, s>>>(prev.y, cur.y, ff, fb, fpe, grid_w, grid_h, grid_size, out.y,
                                    wmap, W, H, t);
  const uint32_t cw = prev.chroma_width(), ch = prev.height;
  const dim3 grd_c((cw + 15) / 16, (ch + 15) / 16);
  warp_chroma_k<<<grd_c, blk, 0, s>>>(prev.cb, cur.cb, ff, fb, fpe, grid_w, grid_h, grid_size, wmap,
                                      W, out.cb, cw, ch, 2, t);
  warp_chroma_k<<<grd_c, blk, 0, s>>>(prev.cr, cur.cr, ff, fb, fpe, grid_w, grid_h, grid_size, wmap,
                                      W, out.cr, cw, ch, 2, t);
}

}  // namespace spark::frc
