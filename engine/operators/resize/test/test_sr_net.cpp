// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// AI super-resolution tests, three layers of proof:
//   1. GPU == CPU: every embedded net (and a synthetic net exercising the generic-fallback
//      kernels) vs a CPU reference forward of the same NetDef, on a deterministic 10-bit pattern
//      that includes sub-black/super-white. The CPU side emulates the GPU's fp16 activation
//      storage (aarch64 _Float16) at the exact layer boundaries the fused plan rounds at, so the
//      comparison is tight (±2 codes).
//   2. The weights actually super-resolve: SR of a box-downsampled detail pattern must not lose
//      to bilinear upsampling (and fsrcnn must beat it), and the mean level must hold — a weight
//      conversion bug (transposed/reordered tensors) fails this hard even though it would still
//      pass test 1.
//   3. A generous perf ceiling at 1080p->2160p (real numbers come from resize_smoke).
// Raw device buffers, no NIC, no root, no runtime.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../sr_model.hpp"
#include "../sr_net.hpp"

namespace {

#define CK(x)                                                                     \
  do {                                                                            \
    cudaError_t e = (x);                                                          \
    if (e != cudaSuccess) {                                                       \
      std::fprintf(stderr, "CUDA fail %s: %s\n", #x, cudaGetErrorString(e));      \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

using spark::sr::Act;
using spark::sr::Kind;
using spark::sr::Layer;
using spark::sr::NetDef;

// ---- CPU reference forward -------------------------------------------------------------------

float apply_act(float v, Act act, const std::vector<float>& alpha, uint32_t c) {
  switch (act) {
    case Act::prelu: return v >= 0.0f ? v : alpha[c] * v;
    case Act::relu: return v >= 0.0f ? v : 0.0f;
    case Act::tanh: return std::tanh(v);
    default: return v;
  }
}

uint16_t denorm10(float v) {
  const int i = static_cast<int>(std::nearbyintf(64.0f + v * 876.0f));  // rn, like __float2int_rn
  return static_cast<uint16_t>(i < 4 ? 4 : (i > 1019 ? 1019 : i));
}

void round_fp16(std::vector<std::vector<float>>& planes) {
  for (auto& p : planes)
    for (auto& v : p) v = static_cast<float>(static_cast<_Float16>(v));
}

// round_after: conv indices whose OUTPUT passes through fp16 DRAM on the GPU (fused stages don't
// round internally). round_input: the generic path stores the normalized input as fp16 too.
std::vector<uint16_t> cpu_forward(const NetDef& net, const std::vector<uint16_t>& y, uint32_t w,
                                  uint32_t h, const std::set<size_t>& round_after,
                                  bool round_input) {
  std::vector<std::vector<float>> act(1, std::vector<float>(size_t(w) * h));
  for (size_t i = 0; i < act[0].size(); ++i) act[0][i] = (float(y[i]) - 64.0f) / 876.0f;
  if (round_input) round_fp16(act);

  for (size_t li = 0; li + 1 < net.layers.size(); ++li) {  // convs; last layer is shuffle2
    const Layer& l = net.layers[li];
    const int k = int(l.k), r = k / 2;
    std::vector<std::vector<float>> out(l.cout, std::vector<float>(size_t(w) * h));
    for (uint32_t co = 0; co < l.cout; ++co)
      for (int yy = 0; yy < int(h); ++yy)
        for (int xx = 0; xx < int(w); ++xx) {
          float acc = l.bias.empty() ? 0.0f : l.bias[co];
          for (uint32_t ci = 0; ci < l.cin; ++ci)
            for (int ky = 0; ky < k; ++ky) {
              const int gy = yy + ky - r;
              if (gy < 0 || gy >= int(h)) continue;  // zero pad (TF SAME)
              for (int kx = 0; kx < k; ++kx) {
                const int gx = xx + kx - r;
                if (gx < 0 || gx >= int(w)) continue;
                acc = std::fmaf(act[ci][size_t(gy) * w + gx],
                                l.w[((size_t(co) * l.cin + ci) * k + ky) * k + kx], acc);
              }
            }
          out[co][size_t(yy) * w + xx] = apply_act(acc, l.act, l.alpha, co);
        }
    act = std::move(out);
    if (round_after.count(li)) round_fp16(act);
  }

  const Act shuffle_act = net.layers.back().act;
  std::vector<uint16_t> hr(size_t(2 * w) * 2 * h);
  for (uint32_t yy = 0; yy < h; ++yy)
    for (uint32_t xx = 0; xx < w; ++xx)
      for (int j = 0; j < 4; ++j) {
        const float v = apply_act(act[j][size_t(yy) * w + xx], shuffle_act, {}, 0);
        hr[size_t(2 * yy + (j >> 1)) * 2 * w + 2 * xx + (j & 1)] = denorm10(v);
      }
  return hr;
}

// ---- GPU run ----------------------------------------------------------------------------------

std::vector<uint16_t> gpu_forward(const NetDef& net, const std::vector<uint16_t>& y, uint32_t w,
                                  uint32_t h, std::string* plan = nullptr) {
  uint16_t *din, *dout;
  CK(cudaMalloc(&din, y.size() * 2));
  CK(cudaMalloc(&dout, y.size() * 8));
  CK(cudaMemcpy(din, y.data(), y.size() * 2, cudaMemcpyHostToDevice));
  spark::sr::Engine eng(net, w, h);
  if (plan) *plan = eng.plan();
  eng.run(din, dout, nullptr);
  CK(cudaDeviceSynchronize());
  std::vector<uint16_t> out(y.size() * 4);
  CK(cudaMemcpy(out.data(), dout, out.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaFree(din));
  CK(cudaFree(dout));
  return out;
}

// ---- helpers ----------------------------------------------------------------------------------

// Deterministic 10-bit test luma: smooth ramps + detail + excursions below black / above white.
std::vector<uint16_t> make_input(uint32_t w, uint32_t h) {
  std::vector<uint16_t> y(size_t(w) * h);
  uint32_t lcg = 12345;
  for (uint32_t yy = 0; yy < h; ++yy)
    for (uint32_t xx = 0; xx < w; ++xx) {
      lcg = lcg * 1664525u + 1013904223u;
      float v = 502.0f + 300.0f * std::sin(xx * 0.21f) * std::cos(yy * 0.13f) +
                120.0f * ((xx / 7 + yy / 5) % 2 ? 1 : -1) + float(lcg >> 27);
      if (((xx * 31 + yy * 17) & 127) == 0) v = 20.0f;    // sub-black
      if (((xx * 13 + yy * 29) & 127) == 1) v = 1000.0f;  // super-white
      y[size_t(yy) * w + xx] = uint16_t(v < 4 ? 4 : (v > 1019 ? 1019 : v));
    }
  return y;
}

// Band-limited "natural-ish" HR detail pattern for the PSNR check (legal video range).
std::vector<uint16_t> make_detail(uint32_t w, uint32_t h) {
  std::vector<uint16_t> y(size_t(w) * h);
  for (uint32_t yy = 0; yy < h; ++yy)
    for (uint32_t xx = 0; xx < w; ++xx) {
      float v = 400.0f + 180.0f * std::sin(2 * M_PI * xx / 37.0f) * std::sin(2 * M_PI * yy / 29.0f)
                + 160.0f * std::sin(2 * M_PI * (xx + yy) / 53.0f);
      const float dx = float(xx) - w / 2.0f, dy = float(yy) - h / 2.0f;
      const float rad = std::sqrt(dx * dx + dy * dy);
      v += 200.0f / (1.0f + std::exp((rad - h / 3.0f) * 0.7f));  // soft-edged disc
      if ((xx / 24 + yy / 24) % 2) v += 90.0f;                   // gentle checker
      y[size_t(yy) * w + xx] = uint16_t(v < 64 ? 64 : (v > 940 ? 940 : v));
    }
  return y;
}

std::vector<uint16_t> box_down2(const std::vector<uint16_t>& hr, uint32_t hw, uint32_t hh) {
  std::vector<uint16_t> lr(size_t(hw / 2) * (hh / 2));
  for (uint32_t yy = 0; yy < hh / 2; ++yy)
    for (uint32_t xx = 0; xx < hw / 2; ++xx) {
      const uint32_t a = hr[size_t(2 * yy) * hw + 2 * xx], b = hr[size_t(2 * yy) * hw + 2 * xx + 1];
      const uint32_t c = hr[size_t(2 * yy + 1) * hw + 2 * xx],
                     d = hr[size_t(2 * yy + 1) * hw + 2 * xx + 1];
      lr[size_t(yy) * (hw / 2) + xx] = uint16_t((a + b + c + d + 2) / 4);
    }
  return lr;
}

std::vector<uint16_t> bilinear_up2(const std::vector<uint16_t>& lr, uint32_t w, uint32_t h) {
  std::vector<uint16_t> hr(size_t(2 * w) * 2 * h);
  for (uint32_t oy = 0; oy < 2 * h; ++oy)
    for (uint32_t ox = 0; ox < 2 * w; ++ox) {
      const float sx = (ox + 0.5f) / 2.0f - 0.5f, sy = (oy + 0.5f) / 2.0f - 0.5f;
      const int x0 = int(std::floor(sx)), y0 = int(std::floor(sy));
      const float fx = sx - x0, fy = sy - y0;
      auto at = [&](int xx, int yy) {
        xx = xx < 0 ? 0 : (xx >= int(w) ? w - 1 : xx);
        yy = yy < 0 ? 0 : (yy >= int(h) ? h - 1 : yy);
        return float(lr[size_t(yy) * w + xx]);
      };
      const float v = (1 - fx) * (1 - fy) * at(x0, y0) + fx * (1 - fy) * at(x0 + 1, y0) +
                      (1 - fx) * fy * at(x0, y0 + 1) + fx * fy * at(x0 + 1, y0 + 1);
      hr[size_t(oy) * 2 * w + ox] = denorm10((v - 64.0f) / 876.0f);
    }
  return hr;
}

double psnr(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, uint32_t w, uint32_t h,
            uint32_t skip) {
  double mse = 0;
  size_t n = 0;
  for (uint32_t yy = skip; yy < h - skip; ++yy)
    for (uint32_t xx = skip; xx < w - skip; ++xx) {
      const double d = double(a[size_t(yy) * w + xx]) - b[size_t(yy) * w + xx];
      mse += d * d;
      ++n;
    }
  mse /= double(n);
  return 10.0 * std::log10(876.0 * 876.0 / mse);
}

double mean_diff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  double s = 0;
  for (size_t i = 0; i < a.size(); ++i) s += double(a[i]) - b[i];
  return s / double(a.size());
}

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("  FAIL: %s\n", what);
    ++failures;
  }
}

// GPU-vs-CPU comparison for one net on the standard pattern.
void test_vs_cpu(const NetDef& net, const std::set<size_t>& round_after, bool round_input) {
  const uint32_t W = 96, H = 64;
  const auto y = make_input(W, H);
  std::string plan;
  const auto g = gpu_forward(net, y, W, H, &plan);
  const auto c = cpu_forward(net, y, W, H, round_after, round_input);
  int bad = 0, maxd = 0;
  for (size_t i = 0; i < g.size(); ++i) {
    const int d = std::abs(int(g[i]) - int(c[i]));
    if (d > maxd) maxd = d;
    if (d > 2) ++bad;
  }
  std::printf("  %-42s vs-cpu maxdiff=%d bad(>2)=%d\n", plan.c_str(), maxd, bad);
  check(bad == 0, "GPU output differs from CPU reference beyond +-2 codes");
}

}  // namespace

int main() {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::printf("device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

  struct ModelSpec {
    const char* name;
    std::set<size_t> round_after;  // conv outputs that hit fp16 DRAM in the fused GPU plan
  };
  // fsrcnn: head fuses conv0+conv1, tail fuses conv6+conv7+shuffle -> rounds after 1,2,3,4,5.
  // fsrcnn-s: head conv0+conv1, tail conv3+conv4+shuffle -> rounds after 1,2.
  // espcn: unfused head conv0, tail fuses conv2+shuffle -> rounds after 0,1.
  const ModelSpec models[] = {
      {"fsrcnn", {1, 2, 3, 4, 5}},
      {"fsrcnn-s", {1, 2}},
      {"espcn", {0, 1}},
  };

  // --- 1. GPU vs CPU reference (embedded nets, fused kernels) ---
  for (const auto& m : models) {
    std::printf("%s:\n", m.name);
    test_vs_cpu(spark::sr::load_embedded(m.name), m.round_after, false);
  }

  // --- 1b. generic-fallback kernels via a synthetic net no specialization matches ---
  {
    std::printf("generic fallback:\n");
    NetDef net;
    net.name = "synthetic";
    net.scale = 2;
    uint32_t lcg = 777;
    auto rnd = [&]() {
      lcg = lcg * 1664525u + 1013904223u;
      return (float(lcg >> 8) / float(1 << 24) - 0.5f) * 0.6f;
    };
    Layer l1;  // 3x3 1->6, prelu — no conv_ok/head_ok match
    l1.kind = Kind::conv;
    l1.k = 3;
    l1.cin = 1;
    l1.cout = 6;
    l1.act = Act::prelu;
    for (int i = 0; i < 6 * 1 * 9; ++i) l1.w.push_back(rnd());
    for (int i = 0; i < 6; ++i) l1.bias.push_back(rnd() * 0.1f);
    for (int i = 0; i < 6; ++i) l1.alpha.push_back(0.1f + 0.05f * i);
    Layer l2;  // 1x1 6->4, relu
    l2.kind = Kind::conv;
    l2.k = 1;
    l2.cin = 6;
    l2.cout = 4;
    l2.act = Act::relu;
    for (int i = 0; i < 4 * 6; ++i) l2.w.push_back(rnd());
    for (int i = 0; i < 4; ++i) l2.bias.push_back(0.4f + rnd() * 0.1f);
    Layer sh;
    sh.kind = Kind::shuffle2;
    sh.cin = 4;
    sh.cout = 1;
    sh.act = Act::tanh;
    net.layers = {l1, l2, sh};
    test_vs_cpu(net, {0, 1}, /*round_input=*/true);  // norm-in + every conv stores fp16
  }

  // --- 2. the weights super-resolve: SR(box-down(HR)) vs bilinear, plus level hold ---
  {
    const uint32_t HW = 192, HH = 128;
    const auto hr = make_detail(HW, HH);
    const auto lr = box_down2(hr, HW, HH);
    const auto bil = bilinear_up2(lr, HW / 2, HH / 2);
    const double p_bil = psnr(bil, hr, HW, HH, 8);
    std::printf("psnr vs original (bilinear %.2f dB):\n", p_bil);
    for (const auto& m : models) {
      const auto net = spark::sr::load_embedded(m.name);
      const auto sr = gpu_forward(net, lr, HW / 2, HH / 2);
      const double p_sr = psnr(sr, hr, HW, HH, 8);
      const double dmean = mean_diff(sr, hr);
      std::printf("  %-9s %.2f dB (%+.2f vs bilinear), mean shift %+.2f codes\n", m.name, p_sr,
                  p_sr - p_bil, dmean);
      check(p_sr > p_bil - 0.5, "SR clearly loses to bilinear — weights look broken");
      if (std::string(m.name) == "fsrcnn")
        check(p_sr > p_bil, "fsrcnn should beat bilinear on detail");
      check(std::abs(dmean) < 2.0, "mean level shift — normalization/bias bug");
    }
  }

  // --- 3. perf ceiling at 1080p -> 2160p (informational; hard numbers via resize_smoke) ---
  {
    const uint32_t W = 1920, H = 1080;
    const auto y = make_input(W, H);
    uint16_t *din, *dout;
    CK(cudaMalloc(&din, y.size() * 2));
    CK(cudaMalloc(&dout, y.size() * 8));
    CK(cudaMemcpy(din, y.data(), y.size() * 2, cudaMemcpyHostToDevice));
    std::printf("1080p->2160p ms/frame (50-run avg):\n");
    for (const auto& m : models) {
      const auto net = spark::sr::load_embedded(m.name);
      spark::sr::Engine eng(net, W, H);
      for (int i = 0; i < 5; ++i) eng.run(din, dout, nullptr);
      CK(cudaDeviceSynchronize());
      cudaEvent_t a, b;
      CK(cudaEventCreate(&a));
      CK(cudaEventCreate(&b));
      CK(cudaEventRecord(a));
      for (int i = 0; i < 50; ++i) eng.run(din, dout, nullptr);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, a, b));
      // The GB10 DVFS governor idles the SM clock at ~0.5 GHz of 3 GHz unless clocks are locked
      // (sudo nvidia-smi -lgc), so absolute times here swing ~6x with system state. Only trip on
      // a structural regression; the 59.94 fps gate lives in the resize_smoke bench at locked
      // clocks (see docs/M9-filters.md).
      const float per = ms / 50.0f;
      std::printf("  %-9s %.3f ms%s\n", m.name, per,
                  per > 16.6f ? "  (over one 59.94 frame period — low GPU clocks?)" : "");
      check(per < 300.0f, "structurally slow — kernel regression, not just clocks");
      CK(cudaEventDestroy(a));
      CK(cudaEventDestroy(b));
    }
    CK(cudaFree(din));
    CK(cudaFree(dout));
  }

  std::printf(failures ? "FAIL (%d)\n" : "PASS\n", failures);
  return failures ? 1 : 0;
}
