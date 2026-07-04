// FRC validation (pure GPU, no NIC/root), three synthetic cases through the production path
// (NVOF fwd+bwd + cost, 3x3 flow median, occlusion-aware interpolate at t=0.5):
//   A translation    — cur = prev shifted by DX; also reads back the center flow vector.
//   B rotation+zoom  — spatially-varying flow; exercises the bilinear flow upsample and the
//                      source-position refinement (a piecewise-constant/unrefined warp smears).
//   C occlusion      — bright patch moving over a static textured background; exercises the
//                      photometric + vector consistency weighting in the reveal/cover bands.
// Pass = motion-comp MAE clearly beats the naive blend in every case (flow actually used).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "operators/frc/frc_kernels.hpp"
#include "operators/frc/nvof_flow.hpp"
#include "operators/resize/gpu_frame.hpp"

using spark::gpu::GpuFrame;

namespace {

constexpr uint32_t W = 1920, H = 1080;

void draw(std::vector<uint16_t>& img, int shift) {
  std::fill(img.begin(), img.end(), static_cast<uint16_t>(256));  // gray (10-bit)
  uint32_t s = 12345;
  auto rnd = [&]() { s = s * 1103515245u + 12345u; return (s >> 16) & 0x7fff; };
  for (int i = 0; i < 80; ++i) {
    const int bw = 8 + rnd() % 40, bh = 8 + rnd() % 40;
    const int px = rnd() % (static_cast<int>(W) - 64), py = rnd() % (static_cast<int>(H) - 64);
    const uint16_t c = static_cast<uint16_t>(200 + rnd() % 800);
    for (int yy = 0; yy < bh; ++yy)
      for (int xx = 0; xx < bw; ++xx) {
        const int X = px + xx + shift, Y = py + yy;
        if (X >= 0 && X < static_cast<int>(W) && Y >= 0 && Y < static_cast<int>(H)) img[Y * W + X] = c;
      }
  }
}

// Smooth multi-frequency pattern (analytic, so any affine sample of it is exact ground truth).
float pat(float x, float y) {
  return 512.0f + 200.0f * std::sin(0.050f * x) * std::cos(0.043f * y) +
         150.0f * std::sin(0.021f * (x + 0.7f * y)) + 100.0f * std::cos(0.017f * (x - 0.5f * y));
}

// Content rotated by `ang` and zoomed by `zoom` about the image center.
void draw_affine(std::vector<uint16_t>& img, float ang, float zoom) {
  const float cx = W * 0.5f, cy = H * 0.5f;
  const float ca = std::cos(ang), sa = std::sin(ang);
  for (uint32_t y = 0; y < H; ++y)
    for (uint32_t x = 0; x < W; ++x) {
      const float dx = x - cx, dy = y - cy;
      const float sx = (ca * dx + sa * dy) / zoom + cx;  // inverse transform
      const float sy = (-sa * dx + ca * dy) / zoom + cy;
      img[y * W + x] = static_cast<uint16_t>(std::fmin(std::fmax(pat(sx, sy), 0.0f), 1023.0f));
    }
}

void stamp_square(std::vector<uint16_t>& img, int x0, int y0, int side, uint16_t c) {
  for (int yy = y0; yy < y0 + side; ++yy)
    for (int xx = x0; xx < x0 + side; ++xx) img[static_cast<size_t>(yy) * W + xx] = c;
}

struct CaseResult {
  double mae_mc = 0, mae_naive = 0;        // over [margin, dim-margin)
  double band_mc = 0, band_naive = 0;      // over `band` pixels (0/0 if no band given)
  float cfx = 0, cfy = 0;                  // center flow vector (post-median, fwd)
  bool has_cost = false, thints = false;   // what the NVOF session pair actually enabled
};

// Run one prev/cur pair through the production interpolation path at t=0.5 and score against gt.
CaseResult run_case(const std::vector<uint16_t>& prevH, const std::vector<uint16_t>& curH,
                    const std::vector<uint16_t>& gtH, int margin,
                    const std::vector<uint8_t>* band) {
  GpuFrame prev(W, H), cur(W, H), mid(W, H);
  std::vector<uint16_t> cflat((W / 2) * H, 512);
  cudaMemcpy(prev.y, prevH.data(), (size_t)W * H * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(cur.y, curH.data(), (size_t)W * H * 2, cudaMemcpyHostToDevice);
  for (GpuFrame* f : {&prev, &cur}) {
    cudaMemcpy(f->cb, cflat.data(), (size_t)(W / 2) * H * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(f->cr, cflat.data(), (size_t)(W / 2) * H * 2, cudaMemcpyHostToDevice);
  }

  uint8_t *p8 = nullptr, *c8 = nullptr;
  cudaMalloc(&p8, (size_t)W * H);
  cudaMalloc(&c8, (size_t)W * H);
  spark::frc::y10_to_y8(prev.y, p8, W, H, 0);
  spark::frc::y10_to_y8(cur.y, c8, W, H, 0);

  // Fresh session pair per case: cases are unrelated scenes, so no temporal-hint carry-over.
  spark::frc::NvofFlow flow;
  flow.init(W, H, 4);
  flow.compute(p8, c8, nullptr);  // fwd + bwd (+ cost) on the default stream

  // Mirror FrcOp: 3x3 median on both grids, then interpolate off the filtered view.
  const spark::frc::FlowView raw = flow.view();
  spark::frc::FlowView fv = raw;
  const uint32_t packed_pitch = flow.grid_w() * 2 * sizeof(short);
  short *mf = nullptr, *mb = nullptr;
  cudaMalloc(&mf, (size_t)flow.grid_w() * flow.grid_h() * 2 * sizeof(short));
  cudaMalloc(&mb, (size_t)flow.grid_w() * flow.grid_h() * 2 * sizeof(short));
  spark::frc::median3x3_flow(raw, mf, mb, packed_pitch, nullptr);
  fv.fwd = mf;
  fv.bwd = mb;
  fv.pitch_bytes = packed_pitch;

  spark::frc::WarpWorkspace ws;
  cudaMalloc(reinterpret_cast<void**>(&ws.accum),
             spark::frc::warp_workspace_floats(flow.grid_w(), flow.grid_h()) * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&ws.fwd_t),
             (size_t)flow.grid_w() * flow.grid_h() * 2 * sizeof(short));
  cudaMalloc(reinterpret_cast<void**>(&ws.bwd_t),
             (size_t)flow.grid_w() * flow.grid_h() * 2 * sizeof(short));

  float* wmap = nullptr;
  cudaMalloc(&wmap, (size_t)W * H * sizeof(float));
  spark::frc::interpolate(prev, cur, fv, ws, p8, c8, mid, wmap, 0.5f, nullptr);
  cudaDeviceSynchronize();

  CaseResult r;
  r.has_cost = flow.has_cost();
  r.thints = flow.temporal_hints();
  std::vector<int16_t> frow(flow.grid_w() * 2);
  cudaMemcpy(frow.data(),
             reinterpret_cast<const char*>(mf) + (size_t)(flow.grid_h() / 2) * packed_pitch,
             flow.grid_w() * 2 * sizeof(int16_t), cudaMemcpyDeviceToHost);
  r.cfx = frow[(flow.grid_w() / 2) * 2] / 32.0f;
  r.cfy = frow[(flow.grid_w() / 2) * 2 + 1] / 32.0f;

  std::vector<uint16_t> midH(W * H);
  cudaMemcpy(midH.data(), mid.y, (size_t)W * H * 2, cudaMemcpyDeviceToHost);

  size_t n = 0, nb = 0;
  for (uint32_t y = margin; y < H - margin; ++y)
    for (uint32_t x = margin; x < W - margin; ++x) {
      const size_t i = (size_t)y * W + x;
      const double naive = 0.5 * prevH[i] + 0.5 * curH[i];
      const double e_mc = std::fabs(static_cast<double>(midH[i]) - gtH[i]);
      const double e_nv = std::fabs(naive - gtH[i]);
      r.mae_mc += e_mc;
      r.mae_naive += e_nv;
      ++n;
      if (band && (*band)[i]) {
        r.band_mc += e_mc;
        r.band_naive += e_nv;
        ++nb;
      }
    }
  r.mae_mc /= n;
  r.mae_naive /= n;
  if (nb) {
    r.band_mc /= nb;
    r.band_naive /= nb;
  }

  // Steady-state per-frame GPU cost of the whole FRC chain (production runs 2160p ~= 4x this).
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  constexpr int kIters = 10;
  auto bench = [&](auto&& fn) {
    cudaEventRecord(e0);
    for (int i = 0; i < kIters; ++i) fn();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    return ms / kIters;
  };
  const float ms_flow = bench([&] { flow.compute(p8, c8, nullptr); });
  const float ms_med =
      bench([&] { spark::frc::median3x3_flow(raw, mf, mb, packed_pitch, nullptr); });
  const float ms_interp = bench(
      [&] { spark::frc::interpolate(prev, cur, fv, ws, p8, c8, mid, wmap, 0.5f, nullptr); });
  std::printf(
      "  gpu ms/frame at 1080p: flow=%.2f median=%.2f interpolate(splat+warp)=%.2f total=%.2f\n",
      ms_flow, ms_med, ms_interp, ms_flow + ms_med + ms_interp);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);

  cudaFree(wmap);
  cudaFree(ws.accum);
  cudaFree(ws.fwd_t);
  cudaFree(ws.bwd_t);
  cudaFree(mf);
  cudaFree(mb);
  cudaFree(p8);
  cudaFree(c8);
  return r;
}

}  // namespace

int main() {
  bool pass = true;
  auto gate = [&](const char* name, const CaseResult& r, double factor) {
    const bool ok = r.mae_mc < r.mae_naive * factor;
    std::printf("%s: MAE motion-comp=%.2f naive-blend=%.2f (10-bit units)%s -> %s\n", name, r.mae_mc,
                r.mae_naive, ok ? "" : "  [mc not clearly better]", ok ? "ok" : "FAIL");
    pass &= ok;
  };

  {  // A: pure translation
    const int DX = 16;
    std::vector<uint16_t> prevH(W * H), curH(W * H), gtH(W * H);
    draw(prevH, 0);
    draw(curH, DX);
    draw(gtH, DX / 2);
    const CaseResult r = run_case(prevH, curH, gtH, DX, nullptr);
    std::printf("nvof: cost=%d temporal_hints=%d\n", r.has_cost, r.thints);
    std::printf("A translation: center flow=(%.2f, %.2f) px  [expect ~(%d, 0)]\n", r.cfx, r.cfy, DX);
    gate("A translation", r, 0.5);
  }

  {  // B: rotation + zoom about the center (max displacement ~30 px at the corners)
    const float ANG = 1.5f * 3.14159265f / 180.0f, ZOOM = 1.02f;
    std::vector<uint16_t> prevH(W * H), curH(W * H), gtH(W * H);
    draw_affine(prevH, 0.0f, 1.0f);
    draw_affine(curH, ANG, ZOOM);
    // Midpoint of the affine trajectory (half angle / half zoom; chord-vs-arc error < 0.1 px here).
    draw_affine(gtH, ANG * 0.5f, 1.0f + (ZOOM - 1.0f) * 0.5f);
    gate("B rotation+zoom", run_case(prevH, curH, gtH, 48, nullptr), 0.5);
  }

  {  // C: bright square moving DX over a STATIC textured background -> reveal/cover bands
    const int DX = 24, SIDE = 200, X0 = 860, Y0 = 440;
    std::vector<uint16_t> prevH(W * H), curH(W * H), gtH(W * H);
    draw(prevH, 0);
    curH = prevH;
    gtH = prevH;
    stamp_square(prevH, X0, Y0, SIDE, 960);
    stamp_square(curH, X0 + DX, Y0, SIDE, 960);
    stamp_square(gtH, X0 + DX / 2, Y0, SIDE, 960);
    // Band = within 40 px of the gt square's border (the occlusion/ghosting battleground).
    std::vector<uint8_t> band(W * H, 0);
    const int B = 40, gx = X0 + DX / 2;
    for (int y = Y0 - B; y < Y0 + SIDE + B; ++y)
      for (int x = gx - B; x < gx + SIDE + B; ++x) {
        const bool inner = x >= gx + B && x < gx + SIDE - B && y >= Y0 + B && y < Y0 + SIDE - B;
        if (!inner) band[(size_t)y * W + x] = 1;
      }
    const CaseResult r = run_case(prevH, curH, gtH, 48, &band);
    std::printf("C occlusion: boundary-band MAE motion-comp=%.2f naive-blend=%.2f\n", r.band_mc,
                r.band_naive);
    const bool band_ok = r.band_mc < r.band_naive;
    if (!band_ok) std::printf("C occlusion: [FAIL] band mc not better than naive\n");
    pass &= band_ok;
    gate("C occlusion", r, 0.5);
  }

  std::printf("%s\n", pass ? "[PASS] NVOF flow + motion-compensated interpolation works"
                           : "[FAIL] see cases above");
  return pass ? 0 : 1;
}
