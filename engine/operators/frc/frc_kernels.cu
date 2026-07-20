// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

#include "frc_kernels.hpp"

namespace spark::frc {
namespace {

// Photometric tolerance for the forward<->backward occlusion test, in 10-bit luma units. A matched
// (non-occluded) trajectory has a small end-to-end residual (sensor noise + warp); an occluded one
// jumps to the fore/background luma gap (often hundreds). exp(-residual/sigma) turns that into a soft
// per-side reliability, so the blend degrades gracefully rather than hard-switching at a threshold.
constexpr float kOcclSigma = 20.0f;
// Forward<->backward vector-agreement tolerance, in luma px. A valid forward vector cancels against
// the backward vector at its landing point (sum ~0); the sum grows fast across occlusions and on
// wrong-but-photometrically-plausible matches (repetitive texture) the residual test can't see.
constexpr float kVecSigma = 4.0f;
// Soft scale mapping the OFA per-vector matching cost to a confidence (k/(k+cost)). Only the
// RATIO of the two candidates' confidences survives the normalized blend, so the absolute scale is
// uncritical; in flat regions both costs rise together and the blend degrades to the temporal prior.
constexpr float kCostSoft = 32.0f;
// Luma-similarity range sigma for edge-aware flow sampling, in 8-bit units. Cells whose center luma
// matches the output pixel dominate the footprint, so flow edges snap to image edges instead of
// straddling grid_size-px cells.
constexpr float kEdgeSigma = 12.0f;
// Range-weight floor: keeps a residual bilinear term so the lookup never divides by ~0 when all
// four cells differ from the center (e.g. inside the ghost band of the averaged guide).
constexpr float kEdgeFloor = 0.05f;
// A splat cell whose accumulated weight is below this saw no trajectory land: fall back to the
// unsplatted field value there (rare; strong occlusions, where the consistency weights take over).
constexpr float kSplatEps = 1e-4f;
// Weight floor for splatted vectors: even a fwd/bwd-inconsistent vector should leave a trace where
// nothing more reliable lands, rather than punching a hole.
constexpr float kSplatFloor = 1e-3f;

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

// The 4-cell bilinear footprint of CONTINUOUS luma position (lx, ly) on a flow grid whose cell g
// covers luma [g*grid, (g+1)*grid) with center at (g+0.5)*grid - 0.5.
struct GridFoot {
  int x0, y0, x1, y1;
  float fx, fy;
};
__device__ inline GridFoot grid_foot(uint32_t grid_w, uint32_t grid_h, uint32_t grid_size, float lx,
                                     float ly) {
  float gx = (lx + 0.5f) / grid_size - 0.5f;
  float gy = (ly + 0.5f) / grid_size - 0.5f;
  gx = fminf(fmaxf(gx, 0.0f), grid_w - 1.0f);
  gy = fminf(fmaxf(gy, 0.0f), grid_h - 1.0f);
  GridFoot f;
  f.x0 = (int)gx;
  f.y0 = (int)gy;
  f.x1 = min(f.x0 + 1, (int)grid_w - 1);
  f.y1 = min(f.y0 + 1, (int)grid_h - 1);
  f.fx = gx - f.x0;
  f.fy = gy - f.y0;
  return f;
}

// Sample a SHORT2 flow grid at a continuous luma position: bilinear between the 4 nearest cells.
// (Nearest-cell lookup made the motion field piecewise-constant over grid_size-px blocks — every
// moving edge inherited the 4 px staircase.) Returns (dx, dy) in luma pixels.
__device__ inline float2 sample_flow(const short* __restrict__ flow, uint32_t pitch_elems,
                                     uint32_t grid_w, uint32_t grid_h, uint32_t grid_size, float lx,
                                     float ly) {
  const GridFoot f = grid_foot(grid_w, grid_h, grid_size, lx, ly);
  const short* r0 = flow + (unsigned long long)f.y0 * pitch_elems;
  const short* r1 = flow + (unsigned long long)f.y1 * pitch_elems;
  const float vx = ((float)r0[f.x0 * 2] * (1 - f.fx) + (float)r0[f.x1 * 2] * f.fx) * (1 - f.fy) +
                   ((float)r1[f.x0 * 2] * (1 - f.fx) + (float)r1[f.x1 * 2] * f.fx) * f.fy;
  const float vy =
      ((float)r0[f.x0 * 2 + 1] * (1 - f.fx) + (float)r0[f.x1 * 2 + 1] * f.fx) * (1 - f.fy) +
      ((float)r1[f.x0 * 2 + 1] * (1 - f.fx) + (float)r1[f.x1 * 2 + 1] * f.fx) * f.fy;
  return make_float2(vx / 32.0f, vy / 32.0f);  // S10.5 fixed point
}

// Edge-aware (joint-bilateral) flow sample: the 4-cell bilinear footprint re-weighted by luma
// similarity between the output pixel and each cell's center, read from the guide plane(s). guide1
// may be null (single-frame guide); with both set the guide is their average — exact wherever the
// scene is static (so static edges snap), and ~uniform inside ghosted moving bands (degrading to
// plain bilinear there, never worse).
__device__ inline float2 sample_flow_edge(const short* __restrict__ flow, uint32_t pitch_elems,
                                          uint32_t grid_w, uint32_t grid_h, uint32_t grid_size,
                                          const uint8_t* __restrict__ guide0,
                                          const uint8_t* __restrict__ guide1, uint32_t W,
                                          uint32_t H, float lx, float ly) {
  const GridFoot f = grid_foot(grid_w, grid_h, grid_size, lx, ly);
  auto guide_at = [&](int px, int py) {
    px = min(max(px, 0), (int)W - 1);
    py = min(max(py, 0), (int)H - 1);
    const unsigned long long i = (unsigned long long)py * W + px;
    return guide1 ? 0.5f * ((float)guide0[i] + (float)guide1[i]) : (float)guide0[i];
  };
  const float gc = guide_at((int)(lx + 0.5f), (int)(ly + 0.5f));
  const short* r0 = flow + (unsigned long long)f.y0 * pitch_elems;
  const short* r1 = flow + (unsigned long long)f.y1 * pitch_elems;
  float sx = 0, sy = 0, sw = 0;
  const int cx[4] = {f.x0, f.x1, f.x0, f.x1}, cy[4] = {f.y0, f.y0, f.y1, f.y1};
  const float bw[4] = {(1 - f.fx) * (1 - f.fy), f.fx * (1 - f.fy), (1 - f.fx) * f.fy,
                       f.fx * f.fy};
  for (int i = 0; i < 4; ++i) {
    const float gi = guide_at((int)((cx[i] + 0.5f) * grid_size - 0.5f + 0.5f),
                              (int)((cy[i] + 0.5f) * grid_size - 0.5f + 0.5f));
    const float w = bw[i] * (__expf(-fabsf(gi - gc) / kEdgeSigma) + kEdgeFloor);
    const short* v = (cy[i] == f.y0 ? r0 : r1) + cx[i] * 2;
    sx += w * v[0];
    sy += w * v[1];
    sw += w;
  }
  return make_float2(sx / sw / 32.0f, sy / sw / 32.0f);
}

// OFA matching cost at a luma position (nearest cell — the cost field is too coarse/quantized to
// warrant interpolation). Returns 0 (perfect confidence) when there is no cost buffer.
__device__ inline float sample_cost(const uint8_t* __restrict__ cost, uint32_t pitch_bytes,
                                    uint32_t elem_bytes, uint32_t grid_w, uint32_t grid_h,
                                    uint32_t grid_size, float lx, float ly) {
  if (!cost) return 0.0f;
  const int gx = min(max((int)(lx / grid_size), 0), (int)grid_w - 1);
  const int gy = min(max((int)(ly / grid_size), 0), (int)grid_h - 1);
  const uint8_t* row = cost + (unsigned long long)gy * pitch_bytes;
  return elem_bytes == 1 ? (float)row[gx] : (float)((const uint32_t*)row)[gx];
}

// ---- forward splatting: re-index both flow fields at output phase t ----
//
// The raw fields answer "where does the content AT THIS CELL go", indexed at the source frame; the
// warp needs "which trajectory CROSSES this output pixel at phase t". Splatting answers it exactly:
// each vector deposits itself (bilinearly, weighted by its fwd<->bwd agreement and cost confidence)
// onto the grid cell its trajectory crosses at t. Collisions resolve toward the higher-confidence
// vector (foreground wins over disoccluded background); holes fall back to the unsplatted field.

// One direction. field: the flow being splatted (pitch pe); other: the opposite direction's field
// used for the agreement weight (pitch ope). f = the fraction of `field`'s vector from its source
// frame to phase t (t for fwd, 1-t for bwd). Accumulates px-unit sums into sumx/sumy/sumw (packed
// grid_w x grid_h planes).
__global__ void splat_k(const short* __restrict__ field, uint32_t pe,
                        const short* __restrict__ other, uint32_t ope,
                        const uint8_t* __restrict__ cost, uint32_t cost_pitch, uint32_t cost_elem,
                        float f, int gw, int gh, uint32_t gs, float* __restrict__ sumx,
                        float* __restrict__ sumy, float* __restrict__ sumw) {
  const int gx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= gw || gy >= gh) return;
  const short* v = field + (unsigned long long)gy * pe + gx * 2;
  const float Fx = v[0] / 32.0f, Fy = v[1] / 32.0f;
  const float cxl = (gx + 0.5f) * gs - 0.5f, cyl = (gy + 0.5f) * gs - 0.5f;  // cell center, luma px
  // Reliability: does the other direction cancel this vector at its endpoint? Plus cost confidence.
  const float2 ov = sample_flow(other, ope, gw, gh, gs, cxl + Fx, cyl + Fy);
  float w = __expf(-hypotf(Fx + ov.x, Fy + ov.y) / kVecSigma);
  if (cost) {
    const uint8_t* row = cost + (unsigned long long)gy * cost_pitch;
    const float c = cost_elem == 1 ? (float)row[gx] : (float)((const uint32_t*)row)[gx];
    w *= kCostSoft / (kCostSoft + c);
  }
  w += kSplatFloor;
  // Landing cell (grid coords) of this trajectory at phase t; bilinear deposit into 4 neighbors.
  const float lgx = (cxl + f * Fx + 0.5f) / gs - 0.5f;
  const float lgy = (cyl + f * Fy + 0.5f) / gs - 0.5f;
  const int x0 = (int)floorf(lgx), y0 = (int)floorf(lgy);
  const float fx = lgx - x0, fy = lgy - y0;
  const float bw[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
  const int dx[4] = {0, 1, 0, 1}, dy[4] = {0, 0, 1, 1};
  for (int i = 0; i < 4; ++i) {
    const int xx = x0 + dx[i], yy = y0 + dy[i];
    if (xx < 0 || xx >= gw || yy < 0 || yy >= gh || bw[i] == 0.0f) continue;
    const unsigned long long o = (unsigned long long)yy * gw + xx;
    atomicAdd(&sumx[o], w * bw[i] * Fx);
    atomicAdd(&sumy[o], w * bw[i] * Fy);
    atomicAdd(&sumw[o], w * bw[i]);
  }
}

// Normalize the splat sums into a dense packed SHORT2 grid; holes take the unsplatted field's value
// at the same cell (the old sample-at-output approximation) so downstream sampling needs no mask.
__global__ void splat_resolve_k(const float* __restrict__ sumx, const float* __restrict__ sumy,
                                const float* __restrict__ sumw,
                                const short* __restrict__ fallback, uint32_t fpe,
                                short* __restrict__ out, int gw, int gh) {
  const int gx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= gw || gy >= gh) return;
  const unsigned long long i = (unsigned long long)gy * gw + gx;
  short ox, oy;
  const float w = sumw[i];
  if (w > kSplatEps) {
    ox = (short)fminf(fmaxf(rintf(sumx[i] / w * 32.0f), -32768.0f), 32767.0f);
    oy = (short)fminf(fmaxf(rintf(sumy[i] / w * 32.0f), -32768.0f), 32767.0f);
  } else {
    const short* v = fallback + (unsigned long long)gy * fpe + gx * 2;
    ox = v[0];
    oy = v[1];
  }
  out[i * 2] = ox;
  out[i * 2 + 1] = oy;
}

// Everything the warp kernels read besides the frames themselves: the raw (median-filtered) fields
// + cost, the phase-t splatted fields, and the 8-bit luma guides for edge-aware sampling.
struct WarpCtx {
  FlowView fv;         // raw fields (consistency lookups, cost)
  const short* fwd_t;  // fwd field splatted to phase t (packed, pitch = grid_w elems)
  const short* bwd_t;  // bwd field splatted to phase 1-t
  const uint8_t* prevY8;
  const uint8_t* curY8;
};

// Compute the occlusion-aware blend WEIGHT (weight of the cur/backward candidate, in [0,1]) for one
// luma output pixel. The splatted fields give the trajectories crossing (px,py) at t directly;
// reliability per side = photometric end-to-end consistency x forward<->backward vector agreement
// x OFA cost confidence, modulating the temporal prior.
__device__ inline float blend_weight(const uint16_t* __restrict__ prev,
                                     const uint16_t* __restrict__ cur, const WarpCtx c, int W,
                                     int H, float px, float py, float t, float& cand_f_out,
                                     float& cand_b_out) {
  const uint32_t tpe = c.fv.grid_w * 2;  // splatted grids are packed
  // Trajectory through (px,py) per direction, edge-aware sampled (guide = avg of both frames:
  // exact on static edges, neutral inside moving bands).
  const float2 Ff = sample_flow_edge(c.fwd_t, tpe, c.fv.grid_w, c.fv.grid_h, c.fv.grid_size,
                                     c.prevY8, c.curY8, W, H, px, py);
  const float2 Fb = sample_flow_edge(c.bwd_t, tpe, c.fv.grid_w, c.fv.grid_h, c.fv.grid_size,
                                     c.prevY8, c.curY8, W, H, px, py);
  // Forward: source in prev at px - t*Ff, endpoint in cur at source + Ff. Backward symmetric.
  const float sfx = px - t * Ff.x, sfy = py - t * Ff.y;
  const float sbx = px - (1.0f - t) * Fb.x, sby = py - (1.0f - t) * Fb.y;
  const float cand_f = bilinear(prev, W, H, sfx, sfy);
  const float cand_b = bilinear(cur, W, H, sbx, sby);
  // End-to-end photometric residuals along each trajectory. If the forward vector is valid
  // (visible in both frames), prev(sf) ~= cur(sf + Ff); a large gap means that path crosses an
  // occlusion and cand_f is unreliable. Symmetric for the backward path.
  const float endf = bilinear(cur, W, H, sfx + Ff.x, sfy + Ff.y);
  const float endb = bilinear(prev, W, H, sbx + Fb.x, sby + Fb.y);
  float relf = __expf(-fabsf(cand_f - endf) / kOcclSigma);
  float relb = __expf(-fabsf(cand_b - endb) / kOcclSigma);
  // Vector-space consistency against the RAW opposite field at each landing point.
  const short* ffwd = (const short*)c.fv.fwd;
  const short* fbwd = (const short*)c.fv.bwd;
  const uint32_t pe = c.fv.pitch_bytes / 2;
  const float2 fb_at_f =
      sample_flow(fbwd, pe, c.fv.grid_w, c.fv.grid_h, c.fv.grid_size, sfx + Ff.x, sfy + Ff.y);
  const float2 ff_at_b =
      sample_flow(ffwd, pe, c.fv.grid_w, c.fv.grid_h, c.fv.grid_size, sbx + Fb.x, sby + Fb.y);
  relf *= __expf(-hypotf(Ff.x + fb_at_f.x, Ff.y + fb_at_f.y) / kVecSigma);
  relb *= __expf(-hypotf(Fb.x + ff_at_b.x, Fb.y + ff_at_b.y) / kVecSigma);
  // OFA matching-cost confidence, sampled at each trajectory's source cell.
  if (c.fv.cost_elem_bytes) {
    const float cf =
        sample_cost((const uint8_t*)c.fv.cost_fwd, c.fv.cost_pitch_bytes, c.fv.cost_elem_bytes,
                    c.fv.grid_w, c.fv.grid_h, c.fv.grid_size, sfx, sfy);
    const float cb =
        sample_cost((const uint8_t*)c.fv.cost_bwd, c.fv.cost_pitch_bytes, c.fv.cost_elem_bytes,
                    c.fv.grid_w, c.fv.grid_h, c.fv.grid_size, sbx, sby);
    relf *= kCostSoft / (kCostSoft + cf);
    relb *= kCostSoft / (kCostSoft + cb);
  }
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
                            const WarpCtx c, uint16_t* __restrict__ out, float* __restrict__ wmap,
                            uint32_t W, uint32_t H, float t) {
  const uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= W || py >= H) return;
  float cand_f, cand_b;
  const float wb = blend_weight(prev, cur, c, W, H, px, py, t, cand_f, cand_b);
  wmap[(unsigned long long)py * W + px] = wb;
  out[(unsigned long long)py * W + px] = (uint16_t)((1.0f - wb) * cand_f + wb * cand_b + 0.5f);
}

// Chroma (4:2:2, half width): the same splatted trajectories sampled in luma coordinates, warped
// candidates blended with the luma-derived weight. The flow grid is in luma coords; chroma x-motion
// is flow_x / xsub (no vertical subsample in 4:2:2). W/H are the LUMA dims (guide + wmap reads).
__global__ void warp_chroma_k(const uint16_t* __restrict__ prev, const uint16_t* __restrict__ cur,
                              const WarpCtx c, const float* __restrict__ wmap, uint32_t W,
                              uint32_t H, uint16_t* __restrict__ out, uint32_t cw, uint32_t ch,
                              uint32_t xsub, float t) {
  const uint32_t cx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t cy = blockIdx.y * blockDim.y + threadIdx.y;
  if (cx >= cw || cy >= ch) return;
  const uint32_t lumax = cx * xsub;
  const float px = lumax, py = cy;
  const uint32_t tpe = c.fv.grid_w * 2;
  const float2 Ff = sample_flow_edge(c.fwd_t, tpe, c.fv.grid_w, c.fv.grid_h, c.fv.grid_size,
                                     c.prevY8, c.curY8, W, H, px, py);
  const float2 Fb = sample_flow_edge(c.bwd_t, tpe, c.fv.grid_w, c.fv.grid_h, c.fv.grid_size,
                                     c.prevY8, c.curY8, W, H, px, py);
  const float cand_f = bilinear(prev, cw, ch, (px - t * Ff.x) / xsub, py - t * Ff.y);
  const float cand_b =
      bilinear(cur, cw, ch, (px - (1.0f - t) * Fb.x) / xsub, py - (1.0f - t) * Fb.y);
  const float wb = wmap[(unsigned long long)cy * W + lumax];  // shared luma occlusion decision
  out[(unsigned long long)cy * cw + cx] = (uint16_t)((1.0f - wb) * cand_f + wb * cand_b + 0.5f);
}

__device__ inline short median9(short v[9]) {
  for (int i = 1; i < 9; ++i) {  // insertion sort; 9 elements, register-resident
    const short k = v[i];
    int j = i - 1;
    for (; j >= 0 && v[j] > k; --j) v[j + 1] = v[j];
    v[j + 1] = k;
  }
  return v[4];
}

// 3x3 component-wise median over the flow grid (clamped borders): kills single-cell outlier
// vectors — the classic FRUC vector post-filter — while leaving coherent motion untouched.
__global__ void median3x3_k(const short* __restrict__ in, uint32_t in_pe, short* __restrict__ out,
                            uint32_t out_pe, int gw, int gh) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= gw || y >= gh) return;
  short vx[9], vy[9];
  int n = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    const int yy = min(max(y + dy, 0), gh - 1);
    for (int dx = -1; dx <= 1; ++dx) {
      const int xx = min(max(x + dx, 0), gw - 1);
      const short* v = in + (unsigned long long)yy * in_pe + xx * 2;
      vx[n] = v[0];
      vy[n] = v[1];
      ++n;
    }
  }
  out[(unsigned long long)y * out_pe + x * 2] = median9(vx);
  out[(unsigned long long)y * out_pe + x * 2 + 1] = median9(vy);
}

}  // namespace

void y10_to_y8(const uint16_t* y10, uint8_t* y8, uint32_t w, uint32_t h, cudaStream_t s) {
  const unsigned long long n = (unsigned long long)w * h;
  const int block = 256;
  y10_to_y8_k<<<(unsigned int)((n + block - 1) / block), block, 0, s>>>(y10, y8, n);
}

void median3x3_flow(const FlowView& in, void* fwd_out, void* bwd_out, uint32_t out_pitch_bytes,
                    cudaStream_t s) {
  const dim3 blk(16, 16);
  const dim3 grd((in.grid_w + 15) / 16, (in.grid_h + 15) / 16);
  median3x3_k<<<grd, blk, 0, s>>>((const short*)in.fwd, in.pitch_bytes / 2, (short*)fwd_out,
                                  out_pitch_bytes / 2, in.grid_w, in.grid_h);
  median3x3_k<<<grd, blk, 0, s>>>((const short*)in.bwd, in.pitch_bytes / 2, (short*)bwd_out,
                                  out_pitch_bytes / 2, in.grid_w, in.grid_h);
}

size_t warp_workspace_floats(uint32_t grid_w, uint32_t grid_h) {
  return (size_t)6 * grid_w * grid_h;  // sumx/sumy/sumw for each direction
}

void interpolate(const spark::gpu::GpuFrame& prev, const spark::gpu::GpuFrame& cur,
                 const FlowView& flow, const WarpWorkspace& ws, const uint8_t* prevY8,
                 const uint8_t* curY8, spark::gpu::GpuFrame& out, float* wmap, float t,
                 cudaStream_t s) {
  const uint32_t W = prev.width, H = prev.height;
  const uint32_t gw = flow.grid_w, gh = flow.grid_h;
  const uint32_t pe = flow.pitch_bytes / 2;
  const size_t plane = (size_t)gw * gh;
  const dim3 blk(16, 16);
  const dim3 grd_g((gw + 15) / 16, (gh + 15) / 16);

  // Splat both fields to phase t (per-call: uniform-grid mode interpolates several phases from one
  // flow pair). Layout of ws.accum: [sumx_f, sumy_f, sumw_f, sumx_b, sumy_b, sumw_b] planes.
  cudaMemsetAsync(ws.accum, 0, 6 * plane * sizeof(float), s);
  float* ax_f = ws.accum;
  float* ay_f = ws.accum + plane;
  float* aw_f = ws.accum + 2 * plane;
  float* ax_b = ws.accum + 3 * plane;
  float* ay_b = ws.accum + 4 * plane;
  float* aw_b = ws.accum + 5 * plane;
  splat_k<<<grd_g, blk, 0, s>>>((const short*)flow.fwd, pe, (const short*)flow.bwd, pe,
                                (const uint8_t*)flow.cost_fwd, flow.cost_pitch_bytes,
                                flow.cost_elem_bytes, t, gw, gh, flow.grid_size, ax_f, ay_f, aw_f);
  splat_k<<<grd_g, blk, 0, s>>>((const short*)flow.bwd, pe, (const short*)flow.fwd, pe,
                                (const uint8_t*)flow.cost_bwd, flow.cost_pitch_bytes,
                                flow.cost_elem_bytes, 1.0f - t, gw, gh, flow.grid_size, ax_b, ay_b,
                                aw_b);
  splat_resolve_k<<<grd_g, blk, 0, s>>>(ax_f, ay_f, aw_f, (const short*)flow.fwd, pe, ws.fwd_t, gw,
                                        gh);
  splat_resolve_k<<<grd_g, blk, 0, s>>>(ax_b, ay_b, aw_b, (const short*)flow.bwd, pe, ws.bwd_t, gw,
                                        gh);

  WarpCtx ctx;
  ctx.fv = flow;
  ctx.fwd_t = ws.fwd_t;
  ctx.bwd_t = ws.bwd_t;
  ctx.prevY8 = prevY8;
  ctx.curY8 = curY8;

  // Luma first: it produces wmap, which the chroma kernels read. Same stream => ordered.
  const dim3 grd_y((W + 15) / 16, (H + 15) / 16);
  warp_luma_k<<<grd_y, blk, 0, s>>>(prev.y, cur.y, ctx, out.y, wmap, W, H, t);
  const uint32_t cw = prev.chroma_width(), ch = prev.height;
  const dim3 grd_c((cw + 15) / 16, (ch + 15) / 16);
  warp_chroma_k<<<grd_c, blk, 0, s>>>(prev.cb, cur.cb, ctx, wmap, W, H, out.cb, cw, ch, 2, t);
  warp_chroma_k<<<grd_c, blk, 0, s>>>(prev.cr, cur.cr, ctx, wmap, W, H, out.cr, cw, ch, 2, t);
}

}  // namespace spark::frc
