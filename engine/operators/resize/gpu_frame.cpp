// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

#include "gpu_frame.hpp"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace spark::gpu {
namespace {
void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("gpu_frame: ") + what + ": " + cudaGetErrorString(e));
}
}  // namespace

void GpuFrame::alloc(uint32_t w, uint32_t h) {
  free();
  width = w;
  height = h;
  const size_t y_bytes = static_cast<size_t>(w) * h * 2;
  const size_t c_bytes = static_cast<size_t>(w / 2) * h * 2;
  check(cudaMalloc(reinterpret_cast<void**>(&y), y_bytes), "cudaMalloc Y");
  check(cudaMalloc(reinterpret_cast<void**>(&cb), c_bytes), "cudaMalloc Cb");
  check(cudaMalloc(reinterpret_cast<void**>(&cr), c_bytes), "cudaMalloc Cr");
  // Disable timing: this event is only ever used for cross-stream ordering, never elapsed-time queries.
  check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "cudaEventCreate ready");
}

void GpuFrame::free() {
  if (y) cudaFree(y);
  if (cb) cudaFree(cb);
  if (cr) cudaFree(cr);
  if (ready) cudaEventDestroy(ready);
  y = cb = cr = nullptr;
  ready = nullptr;
}

void fill_gradient(GpuFrame& f, uint32_t seed) {
  // 10-bit values (0..1023) in a horizontal+vertical gradient, varied by seed. Built on the host
  // then copied up — content correctness isn't the point, only that resize has real data to work on.
  const uint32_t w = f.width, h = f.height, cw = f.chroma_width();
  std::vector<uint16_t> yp(static_cast<size_t>(w) * h);
  std::vector<uint16_t> cp(static_cast<size_t>(cw) * h);
  for (uint32_t r = 0; r < h; ++r)
    for (uint32_t c = 0; c < w; ++c)
      yp[static_cast<size_t>(r) * w + c] = static_cast<uint16_t>(((c + r + seed) & 0x3ff));
  for (uint32_t r = 0; r < h; ++r)
    for (uint32_t c = 0; c < cw; ++c)
      cp[static_cast<size_t>(r) * cw + c] = static_cast<uint16_t>(((c * 2 + r + seed) & 0x3ff));
  check(cudaMemcpy(f.y, yp.data(), yp.size() * 2, cudaMemcpyHostToDevice), "memcpy Y");
  check(cudaMemcpy(f.cb, cp.data(), cp.size() * 2, cudaMemcpyHostToDevice), "memcpy Cb");
  check(cudaMemcpy(f.cr, cp.data(), cp.size() * 2, cudaMemcpyHostToDevice), "memcpy Cr");
}

bool copy_plane_to_host(const uint16_t* dev, uint32_t w, uint32_t h, std::vector<uint16_t>& out) {
  out.resize(static_cast<size_t>(w) * h);
  return cudaMemcpy(out.data(), dev, out.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace spark::gpu
