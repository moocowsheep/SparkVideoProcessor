// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Fused CUDA executor for SRW1 super-resolution nets (see sr_net.hpp for the contract).
//
// Layout/precision choices, sized for 1080p->2160p at 59.94 fps on the GB10:
//   - activations: planar [c][h][w] __half between layers, fp32 accumulation inside kernels.
//     fp16 storage halves DRAM traffic — the budget here is the shared LPDDR bus, not FLOPs.
//   - weights: fp32, pre-reordered on the host so the innermost loop reads consecutive floats
//     warp-uniformly ([tap][cout] / [cin][tap][cout]); every thread of a warp reads the same
//     address, which the read-only path broadcasts for free.
//   - each kernel computes ALL output channels of one pixel in registers (channel counts are
//     tiny — 4..64), so a layer is one pass over the input with no im2col/workspace.
//   - fusion: a 1x1 conv never touches DRAM — it runs in registers on its neighbor's output
//     (FSRCNN: conv5+conv1x1 head, conv1x1+conv1x1+shuffle tail). The pixel shuffle is always
//     fused into the final conv, which writes 2x2 uint16 HR pixels directly.
//   - SAME padding is zero-fill, matching the TF graphs the weights were trained in.
//
// The three embedded nets compile to entirely-fused plans:
//   fsrcnn   head5(1->56->12) + 4x conv3(12->12) + tail(1x1 12->56->4 + shuffle)   6 kernels
//   fsrcnn-s head5(1->32->5)  +    conv3(5->5)   + tail(1x1 5->32->4 + shuffle)    3 kernels
//   espcn    head5(1->64)     +    conv3(64->32) + tail(3x3 32->4 + shuffle,tanh)  3 kernels
// Other SRW1 blobs (SPARK_SR_WEIGHTS retrains with new shapes) fall back to generic per-layer
// kernels — correct, just slower.
#include "sr_net.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace spark::sr {
namespace {

constexpr int BW = 32, BH = 8;  // spatial tile per block (256 threads)

#define SR_CK(x)                                                                        \
  do {                                                                                  \
    const cudaError_t err_ = (x);                                                       \
    if (err_ != cudaSuccess)                                                            \
      throw std::runtime_error(std::string("sr_net CUDA error: ") +                     \
                               cudaGetErrorString(err_) + " @ " #x);                    \
  } while (0)

// ---- device helpers -------------------------------------------------------------------------

// Activation codes match sr_model.hpp Act. alpha may be null for none/relu/tanh.
__device__ __forceinline__ float act_apply(float v, int act, const float* alpha, int c) {
  switch (act) {
    case 1: return v >= 0.0f ? v : __ldg(&alpha[c]) * v;  // prelu
    case 2: return v >= 0.0f ? v : 0.0f;                  // relu
    case 3: return tanhf(v);                              // tanh
    default: return v;
  }
}

// 10-bit video range <-> the [0,1] luma domain the models were trained on (black 64, white 940).
// Input is NOT clamped (sub-black / super-white ride through the convs); output clamps to the
// non-reserved code range like every other stage in the chain.
__device__ __forceinline__ float norm10(uint16_t v) {
  return (static_cast<float>(v) - 64.0f) * (1.0f / 876.0f);
}
__device__ __forceinline__ uint16_t denorm10(float v) {
  const int i = __float2int_rn(64.0f + v * 876.0f);
  return static_cast<uint16_t>(min(max(i, 4), 1019));
}

// ---- specialized kernels --------------------------------------------------------------------
//
// Structure (all of them): weights staged into shared memory once per block (the naive
// per-FMA global __ldg made every kernel issue/latency-bound at a uniform ~0.18 T-MAC/s), and
// the middle/tail convs register-tile PX strided pixels per thread (pixel p sits at
// x + p*BW), so one weight load feeds PX FMAs and each thread carries COUT*PX independent
// accumulator chains for latency hiding.

// Head: KxK conv on normalized luma -> C1 channels (+act1), optionally fused with a following
// 1x1 conv -> C2 channels (+act2). C2 == 0 means "no second stage, write C1 planes".
// w1: [K*K][C1], w2: [C1][C2]. PX=1: C1 accumulators per pixel already cap the registers.
template <int K, int C1, int C2>
__global__ void __launch_bounds__(BW* BH) k_head(const uint16_t* __restrict__ src,
                                                 const float* __restrict__ w1,
                                                 const float* __restrict__ b1,
                                                 const float* __restrict__ a1, int act1,
                                                 const float* __restrict__ w2,
                                                 const float* __restrict__ b2,
                                                 const float* __restrict__ a2, int act2,
                                                 __half* __restrict__ dst, int w, int h) {
  constexpr int TW = BW + K - 1, TH = BH + K - 1;
  // Raw u16 tile; the zero-pad of TF SAME is normalized 0.0 == code 64, so pad with 64.
  __shared__ uint16_t tile[TH * TW];
  __shared__ float wsm1[K * K * C1];
  __shared__ float wsm2[C2 > 0 ? C1 * C2 + 1 : 1];
  const int tid = threadIdx.y * BW + threadIdx.x;
  for (int i = tid; i < K * K * C1; i += BW * BH) wsm1[i] = w1[i];
  if constexpr (C2 > 0)
    for (int i = tid; i < C1 * C2; i += BW * BH) wsm2[i] = w2[i];
  const int x0 = blockIdx.x * BW, y0 = blockIdx.y * BH;
  for (int i = tid; i < TW * TH; i += BW * BH) {
    const int ty = i / TW, tx = i - ty * TW;
    const int gx = x0 - K / 2 + tx, gy = y0 - K / 2 + ty;
    tile[i] = (gx >= 0 && gx < w && gy >= 0 && gy < h) ? src[(size_t)gy * w + gx] : uint16_t(64);
  }
  __syncthreads();
  const int x = x0 + threadIdx.x, y = y0 + threadIdx.y;
  if (x >= w || y >= h) return;

  float acc[C1];
#pragma unroll
  for (int c = 0; c < C1; ++c) acc[c] = __ldg(&b1[c]);
#pragma unroll
  for (int ky = 0; ky < K; ++ky)
#pragma unroll
    for (int kx = 0; kx < K; ++kx) {
      const float px = norm10(tile[(threadIdx.y + ky) * TW + threadIdx.x + kx]);
      const float* wrow = &wsm1[(ky * K + kx) * C1];
#pragma unroll
      for (int c = 0; c < C1; ++c) acc[c] = fmaf(px, wrow[c], acc[c]);
    }
#pragma unroll
  for (int c = 0; c < C1; ++c) acc[c] = act_apply(acc[c], act1, a1, c);

  const size_t plane = (size_t)w * h, at = (size_t)y * w + x;
  if constexpr (C2 == 0) {
#pragma unroll
    for (int c = 0; c < C1; ++c) dst[c * plane + at] = __float2half(acc[c]);
  } else {
    float acc2[C2];
#pragma unroll
    for (int j = 0; j < C2; ++j) acc2[j] = __ldg(&b2[j]);
#pragma unroll
    for (int c = 0; c < C1; ++c) {
      const float* wrow = &wsm2[c * C2];
#pragma unroll
      for (int j = 0; j < C2; ++j) acc2[j] = fmaf(acc[c], wrow[j], acc2[j]);
    }
#pragma unroll
    for (int j = 0; j < C2; ++j)
      dst[j * plane + at] = __float2half(act_apply(acc2[j], act2, a2, j));
  }
}

// Shared tile+weights staging for the PX-tiled conv kernels: CHUNK input planes (fp16, straight
// copies — the activations are already fp16-rounded) plus that chunk's weight slice.
template <int K, int COUT, int CHUNK, int TW, int TH>
__device__ void stage_chunk(const __half* __restrict__ src, size_t plane, int cin, int c0,
                            const float* __restrict__ w, __half (*tile)[TH * TW], float* wsm,
                            int x0, int y0, int wpx, int hpx) {
  const int tid = threadIdx.y * BW + threadIdx.x;
  const int nw = (c0 + CHUNK < cin ? CHUNK : cin - c0) * K * K * COUT;
  for (int i = tid; i < nw; i += BW * BH) wsm[i] = w[(size_t)c0 * K * K * COUT + i];
  for (int cc = 0; cc < CHUNK && c0 + cc < cin; ++cc) {
    const __half* pl = src + (size_t)(c0 + cc) * plane;
    for (int i = tid; i < TW * TH; i += BW * BH) {
      const int ty = i / TW, tx = i - ty * TW;
      const int gx = x0 - K / 2 + tx, gy = y0 - K / 2 + ty;
      tile[cc][i] =
          (gx >= 0 && gx < wpx && gy >= 0 && gy < hpx) ? pl[(size_t)gy * wpx + gx] : __half(0);
    }
  }
}

// Middle KxK conv, CIN -> COUT (+act), fp16 planes in and out. Input channels stream through
// shared memory CHUNK planes at a time; each thread computes PX strided pixels so a weight read
// feeds PX FMAs and COUT*PX accumulator chains hide latency. w: [CIN][K*K][COUT].
template <int K, int CIN, int COUT, int CHUNK, int PX>
__global__ void __launch_bounds__(BW* BH) k_conv(const __half* __restrict__ src,
                                                 const float* __restrict__ w,
                                                 const float* __restrict__ b,
                                                 const float* __restrict__ a, int act,
                                                 __half* __restrict__ dst, int wpx, int hpx) {
  constexpr int TW = BW * PX + K - 1, TH = BH + K - 1;
  __shared__ __half tile[CHUNK][TH * TW];
  __shared__ float wsm[CHUNK * K * K * COUT];
  const int x0 = blockIdx.x * (BW * PX), y0 = blockIdx.y * BH;
  const size_t plane = (size_t)wpx * hpx;

  float acc[COUT][PX];
#pragma unroll
  for (int j = 0; j < COUT; ++j) {
    const float bj = __ldg(&b[j]);
#pragma unroll
    for (int p = 0; p < PX; ++p) acc[j][p] = bj;
  }

  for (int c0 = 0; c0 < CIN; c0 += CHUNK) {
    __syncthreads();  // previous iteration's readers are done before the tile is overwritten
    stage_chunk<K, COUT, CHUNK, TW, TH>(src, plane, CIN, c0, w, tile, wsm, x0, y0, wpx, hpx);
    __syncthreads();
#pragma unroll
    for (int cc = 0; cc < CHUNK; ++cc) {
      if (c0 + cc >= CIN) break;
#pragma unroll
      for (int ky = 0; ky < K; ++ky) {
        float t[PX][K];
#pragma unroll
        for (int p = 0; p < PX; ++p)
#pragma unroll
          for (int kx = 0; kx < K; ++kx)
            t[p][kx] = __half2float(
                tile[cc][(threadIdx.y + ky) * TW + threadIdx.x + p * BW + kx]);
#pragma unroll
        for (int kx = 0; kx < K; ++kx) {
          const float* wrow = &wsm[((cc * K + ky) * K + kx) * COUT];
#pragma unroll
          for (int j = 0; j < COUT; ++j) {
            const float wv = wrow[j];
#pragma unroll
            for (int p = 0; p < PX; ++p) acc[j][p] = fmaf(t[p][kx], wv, acc[j][p]);
          }
        }
      }
    }
  }
  const int y = y0 + threadIdx.y;
  if (y >= hpx) return;
#pragma unroll
  for (int p = 0; p < PX; ++p) {
    const int x = x0 + threadIdx.x + p * BW;
    if (x >= wpx) continue;
    const size_t at = (size_t)y * wpx + x;
#pragma unroll
    for (int j = 0; j < COUT; ++j)
      dst[j * plane + at] = __float2half(act_apply(acc[j][p], act, a, j));
  }
}

// FSRCNN tail: 1x1 expand A->B (+actE), 1x1 B->4 (+bias), pixel shuffle, optional output act,
// denormalize -> 2x2 uint16 HR pixels per thread. Per-pixel register work; weights in smem.
// wE: [A][B], wO: [B][4]. Shuffle channel order: j = 2*dy + dx (TF depth_to_space, C_out=1).
template <int A, int B>
__global__ void __launch_bounds__(256) k_tail11(const __half* __restrict__ src,
                                                const float* __restrict__ wE,
                                                const float* __restrict__ bE,
                                                const float* __restrict__ aE, int actE,
                                                const float* __restrict__ wO,
                                                const float* __restrict__ bO, int act_out,
                                                uint16_t* __restrict__ dst, int w, int h) {
  __shared__ float wsmE[A * B], wsmO[B * 4];
  for (int i = threadIdx.x; i < A * B; i += blockDim.x) wsmE[i] = wE[i];
  for (int i = threadIdx.x; i < B * 4; i += blockDim.x) wsmO[i] = wO[i];
  __syncthreads();
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
  if (x >= w) return;
  const size_t plane = (size_t)w * h, at = (size_t)y * w + x;

  float xb[B];
#pragma unroll
  for (int j = 0; j < B; ++j) xb[j] = __ldg(&bE[j]);
#pragma unroll
  for (int c = 0; c < A; ++c) {
    const float xa = __half2float(src[c * plane + at]);
    const float* wrow = &wsmE[c * B];
#pragma unroll
    for (int j = 0; j < B; ++j) xb[j] = fmaf(xa, wrow[j], xb[j]);
  }
#pragma unroll
  for (int j = 0; j < B; ++j) xb[j] = act_apply(xb[j], actE, aE, j);

  float o[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) o[j] = bO ? __ldg(&bO[j]) : 0.0f;
#pragma unroll
  for (int c = 0; c < B; ++c) {
    const float* wrow = &wsmO[c * 4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = fmaf(xb[c], wrow[j], o[j]);
  }
  const int ow = 2 * w, ox = 2 * x, oy = 2 * y;
  dst[(size_t)oy * ow + ox] = denorm10(act_apply(o[0], act_out, nullptr, 0));
  dst[(size_t)oy * ow + ox + 1] = denorm10(act_apply(o[1], act_out, nullptr, 0));
  dst[(size_t)(oy + 1) * ow + ox] = denorm10(act_apply(o[2], act_out, nullptr, 0));
  dst[(size_t)(oy + 1) * ow + ox + 1] = denorm10(act_apply(o[3], act_out, nullptr, 0));
}

// ESPCN tail: KxK conv CIN->4 (+bias), pixel shuffle, output act, denormalize. Same PX-tiled
// structure as k_conv. w: [CIN][K*K][4].
template <int K, int CIN, int CHUNK, int PX>
__global__ void __launch_bounds__(BW* BH) k_tailk(const __half* __restrict__ src,
                                                  const float* __restrict__ w,
                                                  const float* __restrict__ b, int act_out,
                                                  uint16_t* __restrict__ dst, int wpx, int hpx) {
  constexpr int TW = BW * PX + K - 1, TH = BH + K - 1;
  __shared__ __half tile[CHUNK][TH * TW];
  __shared__ float wsm[CHUNK * K * K * 4];
  const int x0 = blockIdx.x * (BW * PX), y0 = blockIdx.y * BH;
  const size_t plane = (size_t)wpx * hpx;

  float acc[4][PX];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float bj = b ? __ldg(&b[j]) : 0.0f;
#pragma unroll
    for (int p = 0; p < PX; ++p) acc[j][p] = bj;
  }

  for (int c0 = 0; c0 < CIN; c0 += CHUNK) {
    __syncthreads();
    stage_chunk<K, 4, CHUNK, TW, TH>(src, plane, CIN, c0, w, tile, wsm, x0, y0, wpx, hpx);
    __syncthreads();
#pragma unroll
    for (int cc = 0; cc < CHUNK; ++cc) {
      if (c0 + cc >= CIN) break;
#pragma unroll
      for (int ky = 0; ky < K; ++ky) {
        float t[PX][K];
#pragma unroll
        for (int p = 0; p < PX; ++p)
#pragma unroll
          for (int kx = 0; kx < K; ++kx)
            t[p][kx] = __half2float(
                tile[cc][(threadIdx.y + ky) * TW + threadIdx.x + p * BW + kx]);
#pragma unroll
        for (int kx = 0; kx < K; ++kx) {
          const float* wrow = &wsm[((cc * K + ky) * K + kx) * 4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float wv = wrow[j];
#pragma unroll
            for (int p = 0; p < PX; ++p) acc[j][p] = fmaf(t[p][kx], wv, acc[j][p]);
          }
        }
      }
    }
  }
  const int y = y0 + threadIdx.y;
  if (y >= hpx) return;
  const int ow = 2 * wpx;
#pragma unroll
  for (int p = 0; p < PX; ++p) {
    const int x = x0 + threadIdx.x + p * BW;
    if (x >= wpx) continue;
    const int ox = 2 * x, oy = 2 * y;
    dst[(size_t)oy * ow + ox] = denorm10(act_apply(acc[0][p], act_out, nullptr, 0));
    dst[(size_t)oy * ow + ox + 1] = denorm10(act_apply(acc[1][p], act_out, nullptr, 0));
    dst[(size_t)(oy + 1) * ow + ox] = denorm10(act_apply(acc[2][p], act_out, nullptr, 0));
    dst[(size_t)(oy + 1) * ow + ox + 1] = denorm10(act_apply(acc[3][p], act_out, nullptr, 0));
  }
}

// ---- generic fallback kernels (arbitrary SRW1 shapes; correctness over speed) -----------------

__global__ void k_norm_in(const uint16_t* __restrict__ src, __half* __restrict__ dst, int w,
                          int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
  if (x < w) dst[(size_t)y * w + x] = __float2half(norm10(src[(size_t)y * w + x]));
}

// One thread per pixel, all output channels serially. w: raw OIHW.
__global__ void k_conv_gen(const __half* __restrict__ src, const float* __restrict__ w,
                           const float* __restrict__ b, const float* __restrict__ a, int act,
                           __half* __restrict__ dst, int wpx, int hpx, int k, int cin, int cout) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
  if (x >= wpx) return;
  const size_t plane = (size_t)wpx * hpx;
  for (int co = 0; co < cout; ++co) {
    float acc = b ? __ldg(&b[co]) : 0.0f;
    for (int ci = 0; ci < cin; ++ci)
      for (int ky = 0; ky < k; ++ky) {
        const int gy = y + ky - k / 2;
        if (gy < 0 || gy >= hpx) continue;
        for (int kx = 0; kx < k; ++kx) {
          const int gx = x + kx - k / 2;
          if (gx < 0 || gx >= wpx) continue;
          acc = fmaf(__half2float(src[ci * plane + (size_t)gy * wpx + gx]),
                     __ldg(&w[(((size_t)co * cin + ci) * k + ky) * k + kx]), acc);
        }
      }
    dst[co * plane + (size_t)y * wpx + x] = __float2half(act_apply(acc, act, a, co));
  }
}

__global__ void k_tail_gen(const __half* __restrict__ src, int act_out,
                           uint16_t* __restrict__ dst, int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
  if (x >= w) return;
  const size_t plane = (size_t)w * h, at = (size_t)y * w + x;
  const int ow = 2 * w, ox = 2 * x, oy = 2 * y;
  for (int j = 0; j < 4; ++j) {
    const float v = act_apply(__half2float(src[j * plane + at]), act_out, nullptr, 0);
    dst[(size_t)(oy + (j >> 1)) * ow + ox + (j & 1)] = denorm10(v);
  }
}

// ---- launch plan -----------------------------------------------------------------------------

// Which shapes have specialized instantiations (must mirror the dispatch if-chains below).
bool head_fused_ok(uint32_t k, uint32_t c1, uint32_t c2) {
  return k == 5 && ((c1 == 56 && c2 == 12) || (c1 == 32 && c2 == 5));
}
bool head_ok(uint32_t k, uint32_t c1) { return k == 5 && c1 == 64; }
bool conv_ok(uint32_t k, uint32_t cin, uint32_t cout) {
  return k == 3 && ((cin == 12 && cout == 12) || (cin == 5 && cout == 5) ||
                    (cin == 64 && cout == 32));
}
bool tail11_ok(uint32_t a, uint32_t b) { return (a == 12 && b == 56) || (a == 5 && b == 32); }
bool tailk_ok(uint32_t k, uint32_t cin) { return k == 3 && cin == 32; }

struct Step {
  enum Kind { kHead, kConv, kTail11, kTailK, kNormIn, kConvGen, kTailGen } kind;
  uint32_t k = 0, c1 = 0, c2 = 0;
  int act1 = 0, act2 = 0;
  // Weight arena offsets during plan build, patched to device pointers after upload.
  const float *w1 = nullptr, *b1 = nullptr, *a1 = nullptr;
  const float *w2 = nullptr, *b2 = nullptr, *a2 = nullptr;
  const __half* src = nullptr;  // fp16 activation input (null for the u16 head/norm-in)
  __half* dst = nullptr;        // fp16 activation output (null for the u16 tails)
};

}  // namespace

struct Engine::Impl {
  uint32_t w = 0, h = 0;
  std::string plan_str;
  float* arena = nullptr;  // all weights/biases/alphas, one device allocation
  __half* buf[2] = {nullptr, nullptr};
  std::vector<Step> steps;

  ~Impl() {
    cudaFree(arena);
    cudaFree(buf[0]);
    cudaFree(buf[1]);
  }
};

namespace {

// Weight staging: reorders push into one host vector; offsets become device pointers post-upload.
class Arena {
 public:
  size_t push(std::vector<float> v) {
    const size_t off = data_.size();
    data_.insert(data_.end(), v.begin(), v.end());
    return off;
  }
  size_t push_raw(const std::vector<float>& v) { return push(v); }
  const std::vector<float>& data() const { return data_; }

 private:
  std::vector<float> data_;
};

std::vector<float> reorder_tap_cout(const Layer& l) {  // OIHW -> [tap][cout]  (cin == 1 heads)
  const uint32_t kk = l.k * l.k;
  std::vector<float> r(kk * l.cout);
  for (uint32_t co = 0; co < l.cout; ++co)
    for (uint32_t t = 0; t < kk; ++t) r[t * l.cout + co] = l.w[co * kk + t];
  return r;
}

std::vector<float> reorder_cin_cout(const Layer& l) {  // 1x1 OIHW -> [cin][cout]
  std::vector<float> r(l.cin * l.cout);
  for (uint32_t co = 0; co < l.cout; ++co)
    for (uint32_t ci = 0; ci < l.cin; ++ci) r[ci * l.cout + co] = l.w[co * l.cin + ci];
  return r;
}

std::vector<float> reorder_cin_tap_cout(const Layer& l) {  // OIHW -> [cin][tap][cout]
  const uint32_t kk = l.k * l.k;
  std::vector<float> r((size_t)l.cin * kk * l.cout);
  for (uint32_t co = 0; co < l.cout; ++co)
    for (uint32_t ci = 0; ci < l.cin; ++ci)
      for (uint32_t t = 0; t < kk; ++t)
        r[((size_t)ci * kk + t) * l.cout + co] = l.w[((size_t)co * l.cin + ci) * kk + t];
  return r;
}

const float* off_ptr(size_t off) { return reinterpret_cast<const float*>(off + 1); }  // +1: keep 0 == null
size_t ptr_off(const float* p) { return reinterpret_cast<size_t>(p) - 1; }

void patch(const float*& p, const float* base) {
  if (p) p = base + ptr_off(p);
}

std::string act_name(Act a) {
  switch (a) {
    case Act::prelu: return "prelu";
    case Act::relu: return "relu";
    case Act::tanh: return "tanh";
    default: return "";
  }
}

}  // namespace

Engine::Engine(const NetDef& net, uint32_t in_w, uint32_t in_h) : impl_(new Impl) {
  if (net.scale != 2) throw std::runtime_error("sr_net: only scale-2 nets are supported");
  if (in_w < 32 || in_h < 16)
    throw std::runtime_error("sr_net: input too small");
  impl_->w = in_w;
  impl_->h = in_h;

  const auto& L = net.layers;
  const size_t n_convs = L.size() - 1;  // parse_blob guarantees shuffle2-terminated
  const Layer& shuf = L.back();

  Arena arena;
  auto conv_w1 = [&](Step& s, const Layer& l, std::vector<float> w) {
    s.w1 = off_ptr(arena.push(std::move(w)));
    if (!l.bias.empty()) s.b1 = off_ptr(arena.push_raw(l.bias));
    if (!l.alpha.empty()) s.a1 = off_ptr(arena.push_raw(l.alpha));
    s.act1 = static_cast<int>(l.act);
  };
  auto conv_w2 = [&](Step& s, const Layer& l, std::vector<float> w) {
    s.w2 = off_ptr(arena.push(std::move(w)));
    if (!l.bias.empty()) s.b2 = off_ptr(arena.push_raw(l.bias));
    if (!l.alpha.empty()) s.a2 = off_ptr(arena.push_raw(l.alpha));
    s.act2 = static_cast<int>(l.act);
  };

  // Kernels expect a bias array; synthesize zeros when a conv has none (only ever tiny).
  auto with_bias = [&](const Layer& l) {
    Layer c = l;
    if (c.bias.empty()) c.bias.assign(c.cout, 0.0f);
    return c;
  };

  std::ostringstream plan;
  size_t head_used = 0, tail_used = 0;

  // Head: consume 1 or 2 leading convs if a specialized head fits.
  {
    const Layer h0 = with_bias(L[0]);
    if (n_convs >= 3 && L[1].kind == Kind::conv && L[1].k == 1 &&
        head_fused_ok(h0.k, h0.cout, L[1].cout)) {
      const Layer h1 = with_bias(L[1]);
      Step s;
      s.kind = Step::kHead;
      s.k = h0.k;
      s.c1 = h0.cout;
      s.c2 = h1.cout;
      conv_w1(s, h0, reorder_tap_cout(h0));
      conv_w2(s, h1, reorder_cin_cout(h1));
      impl_->steps.push_back(s);
      head_used = 2;
      plan << "head" << h0.k << "(1->" << h0.cout << "->" << h1.cout << ")";
    } else if (n_convs >= 2 && head_ok(h0.k, h0.cout)) {
      Step s;
      s.kind = Step::kHead;
      s.k = h0.k;
      s.c1 = h0.cout;
      s.c2 = 0;
      conv_w1(s, h0, reorder_tap_cout(h0));
      impl_->steps.push_back(s);
      head_used = 1;
      plan << "head" << h0.k << "(1->" << h0.cout << ")";
    } else {
      Step s;
      s.kind = Step::kNormIn;
      impl_->steps.push_back(s);
      plan << "norm-in";
    }
  }

  // Tail: consume trailing convs if a specialized shuffle-fused tail fits.
  std::vector<Step> tail_steps;
  std::ostringstream tail_plan;
  {
    const Layer& last = L[n_convs - 1];  // conv producing the 4 shuffle channels
    if (n_convs >= head_used + 2 && last.k == 1 && L[n_convs - 2].k == 1 &&
        tail11_ok(L[n_convs - 2].cin, last.cin)) {
      const Layer e = with_bias(L[n_convs - 2]);
      Step s;
      s.kind = Step::kTail11;
      s.c1 = e.cin;
      s.c2 = e.cout;
      s.act2 = static_cast<int>(shuf.act);
      conv_w1(s, e, reorder_cin_cout(e));
      s.w2 = off_ptr(arena.push(reorder_cin_cout(last)));
      if (!last.bias.empty()) s.b2 = off_ptr(arena.push_raw(last.bias));
      tail_steps.push_back(s);
      tail_used = 2;
      tail_plan << " tail(1x1 " << e.cin << "->" << e.cout << "->4+shuffle"
           << (shuf.act != Act::none ? "," + act_name(shuf.act) : "") << ")";
    } else if (n_convs >= head_used + 1 && tailk_ok(last.k, last.cin)) {
      Step s;
      s.kind = Step::kTailK;
      s.k = last.k;
      s.c1 = last.cin;
      s.act2 = static_cast<int>(shuf.act);
      s.w1 = off_ptr(arena.push(reorder_cin_tap_cout(last)));
      if (!last.bias.empty()) s.b1 = off_ptr(arena.push_raw(last.bias));
      tail_steps.push_back(s);
      tail_used = 1;
      tail_plan << " tail(" << last.k << "x" << last.k << " " << last.cin << "->4+shuffle"
           << (shuf.act != Act::none ? "," + act_name(shuf.act) : "") << ")";
    } else {
      Step s;  // generic: last conv runs as a middle layer into 4 planes, then shuffle out
      s.kind = Step::kTailGen;
      s.act2 = static_cast<int>(shuf.act);
      tail_steps.push_back(s);
      tail_plan << " shuffle-out" << (shuf.act != Act::none ? "(" + act_name(shuf.act) + ")" : "");
    }
  }

  // Middle convs (everything the head/tail didn't absorb).
  std::vector<Step> mid_steps;
  uint32_t conv3_run = 0;
  for (size_t i = head_used; i < n_convs - tail_used; ++i) {
    const Layer m = with_bias(L[i]);
    Step s;
    s.k = m.k;
    s.c1 = m.cin;
    s.c2 = m.cout;
    if (conv_ok(m.k, m.cin, m.cout)) {
      s.kind = Step::kConv;
      conv_w1(s, m, reorder_cin_tap_cout(m));
      ++conv3_run;
    } else {
      s.kind = Step::kConvGen;
      conv_w1(s, m, m.w);  // raw OIHW
      plan << " conv" << m.k << "x" << m.k << "*(" << m.cin << "->" << m.cout << ")";
    }
    mid_steps.push_back(s);
  }
  if (conv3_run) plan << " " << conv3_run << "x conv3";
  plan << tail_plan.str();
  for (auto& s : mid_steps) impl_->steps.push_back(s);
  for (auto& s : tail_steps) impl_->steps.push_back(s);

  // Ping-pong activation buffers between steps; track the widest channel count per slot.
  uint32_t ch_max[2] = {0, 0};
  int cur = 0;
  for (auto& s : impl_->steps) {
    switch (s.kind) {
      case Step::kHead:
      case Step::kNormIn: {
        const uint32_t ch = s.kind == Step::kNormIn ? 1 : (s.c2 ? s.c2 : s.c1);
        s.dst = reinterpret_cast<__half*>(static_cast<uintptr_t>(cur + 1));
        ch_max[cur] = std::max(ch_max[cur], ch);
        break;
      }
      case Step::kConv:
      case Step::kConvGen:
        s.src = reinterpret_cast<__half*>(static_cast<uintptr_t>(cur + 1));
        s.dst = reinterpret_cast<__half*>(static_cast<uintptr_t>((cur ^ 1) + 1));
        cur ^= 1;
        ch_max[cur] = std::max(ch_max[cur], s.c2);
        break;
      case Step::kTail11:
      case Step::kTailK:
      case Step::kTailGen:
        s.src = reinterpret_cast<__half*>(static_cast<uintptr_t>(cur + 1));
        break;
    }
  }

  // Upload the arena and patch offsets into device pointers.
  SR_CK(cudaMalloc(&impl_->arena, arena.data().size() * sizeof(float)));
  SR_CK(cudaMemcpy(impl_->arena, arena.data().data(), arena.data().size() * sizeof(float),
                   cudaMemcpyHostToDevice));
  const size_t plane = (size_t)in_w * in_h;
  for (int i = 0; i < 2; ++i)
    if (ch_max[i])
      SR_CK(cudaMalloc(&impl_->buf[i], (size_t)ch_max[i] * plane * sizeof(__half)));
  for (auto& s : impl_->steps) {
    patch(s.w1, impl_->arena);
    patch(s.b1, impl_->arena);
    patch(s.a1, impl_->arena);
    patch(s.w2, impl_->arena);
    patch(s.b2, impl_->arena);
    patch(s.a2, impl_->arena);
    if (s.src) s.src = impl_->buf[reinterpret_cast<uintptr_t>(s.src) - 1];
    if (s.dst) s.dst = impl_->buf[reinterpret_cast<uintptr_t>(s.dst) - 1];
  }
  impl_->plan_str = net.name + " " + plan.str();
}

Engine::~Engine() = default;

uint32_t Engine::in_width() const { return impl_->w; }
uint32_t Engine::in_height() const { return impl_->h; }
const std::string& Engine::plan() const { return impl_->plan_str; }

void Engine::run(const uint16_t* y_in, uint16_t* y_out, cudaStream_t stream) {
  const int w = static_cast<int>(impl_->w), h = static_cast<int>(impl_->h);
  const dim3 blk(BW, BH);
  const dim3 grd((w + BW - 1) / BW, (h + BH - 1) / BH);
  const dim3 grd_px2((w + 2 * BW - 1) / (2 * BW), (h + BH - 1) / BH);  // PX=2 pixel-tiled convs
  const dim3 grd_px4((w + 4 * BW - 1) / (4 * BW), (h + BH - 1) / BH);  // PX=4
  const dim3 row_blk(256), row_grd((w + 255) / 256, h);

  for (const auto& s : impl_->steps) {
    switch (s.kind) {
      case Step::kHead:
        if (s.c1 == 56 && s.c2 == 12)
          k_head<5, 56, 12><<<grd, blk, 0, stream>>>(y_in, s.w1, s.b1, s.a1, s.act1, s.w2, s.b2,
                                                     s.a2, s.act2, s.dst, w, h);
        else if (s.c1 == 32 && s.c2 == 5)
          k_head<5, 32, 5><<<grd, blk, 0, stream>>>(y_in, s.w1, s.b1, s.a1, s.act1, s.w2, s.b2,
                                                    s.a2, s.act2, s.dst, w, h);
        else if (s.c1 == 64 && s.c2 == 0)
          k_head<5, 64, 0><<<grd, blk, 0, stream>>>(y_in, s.w1, s.b1, s.a1, s.act1, nullptr,
                                                    nullptr, nullptr, 0, s.dst, w, h);
        else
          throw std::logic_error("sr_net: head dispatch mismatch");
        break;
      case Step::kConv:
        if (s.c1 == 12 && s.c2 == 12)
          k_conv<3, 12, 12, 6, 4><<<grd_px4, blk, 0, stream>>>(s.src, s.w1, s.b1, s.a1, s.act1,
                                                               s.dst, w, h);
        else if (s.c1 == 5 && s.c2 == 5)
          k_conv<3, 5, 5, 5, 4><<<grd_px4, blk, 0, stream>>>(s.src, s.w1, s.b1, s.a1, s.act1,
                                                             s.dst, w, h);
        else if (s.c1 == 64 && s.c2 == 32)
          k_conv<3, 64, 32, 8, 2><<<grd_px2, blk, 0, stream>>>(s.src, s.w1, s.b1, s.a1, s.act1,
                                                               s.dst, w, h);
        else
          throw std::logic_error("sr_net: conv dispatch mismatch");
        break;
      case Step::kTail11:
        if (s.c1 == 12 && s.c2 == 56)
          k_tail11<12, 56><<<row_grd, row_blk, 0, stream>>>(s.src, s.w1, s.b1, s.a1, s.act1,
                                                            s.w2, s.b2, s.act2, y_out, w, h);
        else if (s.c1 == 5 && s.c2 == 32)
          k_tail11<5, 32><<<row_grd, row_blk, 0, stream>>>(s.src, s.w1, s.b1, s.a1, s.act1, s.w2,
                                                           s.b2, s.act2, y_out, w, h);
        else
          throw std::logic_error("sr_net: tail11 dispatch mismatch");
        break;
      case Step::kTailK:
        if (s.k == 3 && s.c1 == 32)
          k_tailk<3, 32, 8, 4><<<grd_px4, blk, 0, stream>>>(s.src, s.w1, s.b1, s.act2, y_out, w,
                                                            h);
        else
          throw std::logic_error("sr_net: tailk dispatch mismatch");
        break;
      case Step::kNormIn:
        k_norm_in<<<row_grd, row_blk, 0, stream>>>(y_in, s.dst, w, h);
        break;
      case Step::kConvGen:
        k_conv_gen<<<row_grd, row_blk, 0, stream>>>(s.src, s.w1, s.b1, s.a1, s.act1, s.dst, w, h,
                                                    static_cast<int>(s.k),
                                                    static_cast<int>(s.c1),
                                                    static_cast<int>(s.c2));
        break;
      case Step::kTailGen:
        k_tail_gen<<<row_grd, row_blk, 0, stream>>>(s.src, s.act2, y_out, w, h);
        break;
    }
  }
  SR_CK(cudaGetLastError());  // catches launch-config errors; kernels themselves stay async
}

}  // namespace spark::sr
