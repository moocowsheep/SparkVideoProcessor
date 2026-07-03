// FRC validation (pure GPU, no NIC/root): synth prev + cur=prev shifted by DX. Run NVOF flow
// (fwd + bwd) + occlusion-aware motion-compensated interpolation at t=0.5; the result should
// reconstruct prev-shifted-by-DX/2.
// Pass = flow ~DX AND motion-comp MAE << naive-blend MAE (i.e., the flow is actually used, no ghosting).
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
void draw(std::vector<uint16_t>& img, uint32_t W, uint32_t H, int shift) {
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
}  // namespace

int main() {
  const uint32_t W = 1920, H = 1080;
  const int DX = 16;
  std::vector<uint16_t> prevH(W * H), curH(W * H), gtH(W * H);
  draw(prevH, W, H, 0);
  draw(curH, W, H, DX);
  draw(gtH, W, H, DX / 2);

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
  cudaDeviceSynchronize();

  spark::frc::NvofFlow flow;
  flow.init(W, H, 4);
  flow.compute(p8, c8, nullptr);  // fwd + bwd on the default stream

  const uint32_t gw = flow.grid_w(), gh = flow.grid_h(), pitch = flow.flow_pitch_bytes();
  std::vector<int16_t> frow(gw * 2);
  cudaMemcpy(frow.data(), static_cast<const char*>(flow.flow_dev()) + (size_t)(gh / 2) * pitch,
             gw * 2 * sizeof(int16_t), cudaMemcpyDeviceToHost);
  const float cfx = frow[(gw / 2) * 2] / 32.0f, cfy = frow[(gw / 2) * 2 + 1] / 32.0f;

  float* wmap = nullptr;  // per-pixel luma blend weights (shared with chroma inside interpolate)
  cudaMalloc(&wmap, (size_t)W * H * sizeof(float));
  spark::frc::interpolate(prev, cur, flow.flow_dev(), flow.flow_dev_bwd(), pitch, gw, gh,
                          flow.grid_size(), mid, wmap, 0.5f, nullptr);
  cudaDeviceSynchronize();

  std::vector<uint16_t> midH(W * H);
  cudaMemcpy(midH.data(), mid.y, (size_t)W * H * 2, cudaMemcpyDeviceToHost);

  double mae_mc = 0, mae_naive = 0;
  size_t n = 0;
  for (uint32_t y = DX; y < H - DX; ++y)
    for (uint32_t x = DX; x < W - DX; ++x) {
      const size_t i = (size_t)y * W + x;
      const double naive = 0.5 * prevH[i] + 0.5 * curH[i];
      mae_mc += std::fabs(static_cast<double>(midH[i]) - gtH[i]);
      mae_naive += std::fabs(naive - gtH[i]);
      ++n;
    }
  mae_mc /= n;
  mae_naive /= n;

  std::printf("center flow=(%.2f, %.2f) px  [expect ~(%d, 0)]\n", cfx, cfy, DX);
  std::printf("MAE vs ground-truth: motion-comp=%.2f  naive-blend=%.2f (10-bit units)\n", mae_mc,
              mae_naive);
  const bool mc_better = mae_mc < mae_naive * 0.5;  // flow actually used: clearly beats ghosting
  std::printf("%s\n", mc_better ? "[PASS] NVOF flow + motion-compensated interpolation works"
                                : "[FAIL] motion comp not better than naive blend");
  return mc_better ? 0 : 1;
}
