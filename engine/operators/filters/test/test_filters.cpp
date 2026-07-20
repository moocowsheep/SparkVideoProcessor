// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Filter kernel round-trip tests: GPU procamp / unsharp vs simple CPU references on a synthetic
// 10-bit pattern, plus neutral-parameter identity checks. Raw device buffers (no GpuFrame/Holoscan),
// runs on any CUDA box without NIC/root.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "../filter_kernels.hpp"

namespace {

constexpr uint32_t W = 256, H = 64;

uint16_t clamp10(float v) {
  const int i = static_cast<int>(std::lround(v));
  return static_cast<uint16_t>(i < 4 ? 4 : (i > 1019 ? 1019 : i));
}

void cpu_procamp(const std::vector<uint16_t>& y, const std::vector<uint16_t>& cb,
                 const std::vector<uint16_t>& cr, std::vector<uint16_t>& yo,
                 std::vector<uint16_t>& cbo, std::vector<uint16_t>& cro, float bright, float con,
                 float sat, float hue_deg) {
  // Positive hue = clockwise on the vectorscope (skin toward warm) — negation matches procamp().
  const float off = bright * 876.0f, rad = -hue_deg * 3.14159265358979f / 180.0f;
  const float ch = std::cos(rad), sh = std::sin(rad);
  for (size_t i = 0; i < y.size(); ++i) yo[i] = clamp10(64.0f + (y[i] - 64.0f) * con + off);
  for (size_t i = 0; i < cb.size(); ++i) {
    const float b = cb[i] - 512.0f, r = cr[i] - 512.0f;
    cbo[i] = clamp10(512.0f + sat * (b * ch - r * sh));
    cro[i] = clamp10(512.0f + sat * (b * sh + r * ch));
  }
}

void cpu_unsharp(const std::vector<uint16_t>& y, std::vector<uint16_t>& yo, float amount) {
  auto at = [&](int x, int yy) {
    x = x < 0 ? 0 : (x >= (int)W ? W - 1 : x);
    yy = yy < 0 ? 0 : (yy >= (int)H ? H - 1 : yy);
    return static_cast<float>(y[(size_t)yy * W + x]);
  };
  for (int r = 0; r < (int)H; ++r)
    for (int c = 0; c < (int)W; ++c) {
      const float blur = (at(c - 1, r - 1) + 2 * at(c, r - 1) + at(c + 1, r - 1) + 2 * at(c - 1, r) +
                          4 * at(c, r) + 2 * at(c + 1, r) + at(c - 1, r + 1) + 2 * at(c, r + 1) +
                          at(c + 1, r + 1)) /
                         16.0f;
      yo[(size_t)r * W + c] = clamp10(at(c, r) + amount * (at(c, r) - blur));
    }
}

// Bilateral 5x5 reference, matching nr_y/nr_c: spatial sigma 1.5 px, range sigma 4 + 60*strength.
void cpu_nr(const std::vector<uint16_t>& in, std::vector<uint16_t>& out, uint32_t w, uint32_t h,
            float strength) {
  const float sr = 4.0f + 60.0f * strength;
  const float inv2sr2 = 1.0f / (2.0f * sr * sr);
  for (int y = 0; y < (int)h; ++y)
    for (int x = 0; x < (int)w; ++x) {
      const float c = in[(size_t)y * w + x];
      float acc = 0, wsum = 0;
      for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
          const int xx = std::min(std::max(x + dx, 0), (int)w - 1);
          const int yy = std::min(std::max(y + dy, 0), (int)h - 1);
          const float v = in[(size_t)yy * w + xx];
          const float d = v - c;
          const float wt = std::exp(-(dx * dx + dy * dy) / 4.5f) * std::exp(-d * d * inv2sr2);
          acc += wt * v;
          wsum += wt;
        }
      out[(size_t)y * w + x] = clamp10(acc / wsum);
    }
}

// GPU float rounding may differ from the CPU by one code value right at the .5 boundary — allow ±1.
int diff_count(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, int tol) {
  int bad = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::abs((int)a[i] - (int)b[i]) > tol) ++bad;
  return bad;
}

#define CK(x)                                                       \
  do {                                                              \
    cudaError_t e = (x);                                            \
    if (e != cudaSuccess) {                                         \
      std::fprintf(stderr, "CUDA fail %s: %s\n", #x, cudaGetErrorString(e)); \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

}  // namespace

int main() {
  const size_t ny = (size_t)W * H, nc = (size_t)(W / 2) * H;
  std::vector<uint16_t> y(ny), cb(nc), cr(nc);
  for (size_t i = 0; i < ny; ++i) y[i] = static_cast<uint16_t>(64 + (i * 7) % 877);   // 64..940
  for (size_t i = 0; i < nc; ++i) cb[i] = static_cast<uint16_t>(64 + (i * 11) % 897); // 64..960
  for (size_t i = 0; i < nc; ++i) cr[i] = static_cast<uint16_t>(64 + (i * 13) % 897);

  uint16_t *dy, *dcb, *dcr, *dyo, *dcbo, *dcro;
  CK(cudaMalloc(&dy, ny * 2));  CK(cudaMalloc(&dyo, ny * 2));
  CK(cudaMalloc(&dcb, nc * 2)); CK(cudaMalloc(&dcbo, nc * 2));
  CK(cudaMalloc(&dcr, nc * 2)); CK(cudaMalloc(&dcro, nc * 2));
  CK(cudaMemcpy(dy, y.data(), ny * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcb, cb.data(), nc * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcr, cr.data(), nc * 2, cudaMemcpyHostToDevice));

  std::vector<uint16_t> gy(ny), gcb(nc), gcr(nc), ry(ny), rcb(nc), rcr(nc);
  int failures = 0;
  auto run_procamp = [&](float b, float c, float s, float h, const char* name, bool expect_id) {
    spark::filters::procamp(dy, dcb, dcr, dyo, dcbo, dcro, W, H, b, c, s, h, nullptr);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(gy.data(), dyo, ny * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gcb.data(), dcbo, nc * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gcr.data(), dcro, nc * 2, cudaMemcpyDeviceToHost));
    cpu_procamp(y, cb, cr, ry, rcb, rcr, b, c, s, h);
    const int bad = diff_count(gy, ry, 1) + diff_count(gcb, rcb, 1) + diff_count(gcr, rcr, 1);
    const int id = expect_id ? diff_count(gy, y, 0) + diff_count(gcb, cb, 0) + diff_count(gcr, cr, 0) : 0;
    std::printf("procamp %-24s vs-ref bad=%d%s\n", name, bad, expect_id ? (id ? " IDENTITY BROKEN" : " (identity ok)") : "");
    failures += bad + id;
  };
  run_procamp(0.f, 1.f, 1.f, 0.f, "neutral", true);
  run_procamp(0.1f, 1.2f, 0.8f, 30.f, "bright/cont/sat/hue", false);
  run_procamp(-0.5f, 2.0f, 1.8f, -90.f, "extremes (clamps)", false);

  auto run_unsharp = [&](float amt, const char* name, bool expect_id) {
    spark::filters::unsharp_y(dy, dyo, W, H, amt, nullptr);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(gy.data(), dyo, ny * 2, cudaMemcpyDeviceToHost));
    cpu_unsharp(y, ry, amt);
    const int bad = diff_count(gy, ry, 1);
    const int id = expect_id ? diff_count(gy, y, 0) : 0;
    std::printf("unsharp %-24s vs-ref bad=%d%s\n", name, bad, expect_id ? (id ? " IDENTITY BROKEN" : " (identity ok)") : "");
    failures += bad + id;
  };
  run_unsharp(0.f, "amount=0", true);
  run_unsharp(1.0f, "amount=1", false);
  run_unsharp(2.0f, "amount=2 (clamps)", false);

  // NR vs the CPU bilateral reference. __expf on the GPU is a fast approximation, so the tap
  // weights differ slightly from std::exp — allow ±2 code values instead of the usual ±1.
  {
    std::vector<uint16_t> ncb(nc), ncr(nc);
    spark::filters::nr_y(dy, dyo, W, H, 0.6f, nullptr);
    spark::filters::nr_c(dcb, dcr, dcbo, dcro, W, H, 0.8f, nullptr);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(gy.data(), dyo, ny * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gcb.data(), dcbo, nc * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gcr.data(), dcro, nc * 2, cudaMemcpyDeviceToHost));
    cpu_nr(y, ry, W, H, 0.6f);
    cpu_nr(cb, ncb, W / 2, H, 0.8f);
    cpu_nr(cr, ncr, W / 2, H, 0.8f);
    const int bad = diff_count(gy, ry, 2) + diff_count(gcb, ncb, 2) + diff_count(gcr, ncr, 2);
    std::printf("nr      %-24s vs-ref bad=%d\n", "y=0.6 c=0.8", bad);
    failures += bad;
  }

  // Grain is hash noise, so instead of a CPU reference: determinism (same seed -> bit-identical),
  // frame-to-frame variation (different seed -> different field), the film response (black/white
  // bit-exact, mid-tones perturbed within the amplitude bound), and mono leaving chroma to the
  // caller while color writes both planes.
  {
    std::vector<uint16_t> g2(ny);
    spark::filters::grain_y(dy, dyo, W, H, 1.0f, 1.5f, 7u, nullptr);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(gy.data(), dyo, ny * 2, cudaMemcpyDeviceToHost));
    spark::filters::grain_y(dy, dyo, W, H, 1.0f, 1.5f, 7u, nullptr);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(g2.data(), dyo, ny * 2, cudaMemcpyDeviceToHost));
    const int nondet = diff_count(gy, g2, 0);
    spark::filters::grain_y(dy, dyo, W, H, 1.0f, 1.5f, 8u, nullptr);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(g2.data(), dyo, ny * 2, cudaMemcpyDeviceToHost));
    int reseed_same = 0;  // fields from different seeds must actually differ
    for (size_t i = 0; i < ny; ++i) reseed_same += gy[i] == g2[i];
    int resp_bad = 0, moved = 0;
    const float bound = 0.08f * 876.0f + 1.0f;  // peak amplitude at amount 1 (+1 rounding)
    for (size_t i = 0; i < ny; ++i) {
      const float d = std::abs((int)gy[i] - (int)y[i]);
      if (y[i] <= 64 || y[i] >= 940) resp_bad += d != 0;  // black/white: bit-exact
      else resp_bad += d > bound;                          // mid-tones: bounded amplitude
      moved += d != 0;
    }
    // >90% of mid-tone pixels should carry grain at amount 1 (weight only nulls the extremes).
    const int bad = nondet + resp_bad + (reseed_same > (int)ny * 95 / 100 ? 1 : 0) +
                    (moved < (int)ny / 2 ? 1 : 0);
    std::printf("grain   %-24s det=%d resp_bad=%d moved=%d/%zu\n", "y (amount=1)", nondet,
                resp_bad, moved, ny);
    failures += bad;
  }
  {
    spark::filters::grain_c(dy, dcb, dcr, dcbo, dcro, W, H, 1.0f, 1.5f, 7u, nullptr);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(gcb.data(), dcbo, nc * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gcr.data(), dcro, nc * 2, cudaMemcpyDeviceToHost));
    int moved_cb = 0, moved_cr = 0, indep = 0, resp_bad = 0;
    const float bound = 0.08f * 448.0f + 1.0f;
    for (size_t i = 0; i < nc; ++i) {
      const int db = std::abs((int)gcb[i] - (int)cb[i]), dr = std::abs((int)gcr[i] - (int)cr[i]);
      const uint16_t yv = y[(i / (W / 2)) * W + (i % (W / 2)) * 2];  // co-sited luma
      if (yv <= 64 || yv >= 940) resp_bad += (db != 0) + (dr != 0);
      else resp_bad += (db > bound) + (dr > bound);
      moved_cb += db != 0;
      moved_cr += dr != 0;
      indep += ((int)gcb[i] - (int)cb[i]) != ((int)gcr[i] - (int)cr[i]);
    }
    // Cb/Cr fields must be independent (different salts), not one field applied twice.
    const int bad = resp_bad + (moved_cb < (int)nc / 2 ? 1 : 0) + (moved_cr < (int)nc / 2 ? 1 : 0) +
                    (indep < (int)nc / 4 ? 1 : 0);
    std::printf("grain   %-24s resp_bad=%d moved=%d/%d indep=%d\n", "chroma (color mode)", resp_bad,
                moved_cb, moved_cr, indep);
    failures += bad;
  }

  std::printf(failures ? "FAIL (%d)\n" : "PASS\n", failures);
  return failures ? 1 : 0;
}
