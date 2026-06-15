// GpuFrame — the engine's intermediate GPU image format between unpack and pack.
//
// Planar YCbCr 4:2:2, 10-bit samples stored in uint16 (the M0 resize spike used 16u; ST 2110-20's
// 10-bit 4:2:2 unpacks to this). Three device planes: Y (w x h), Cb and Cr (w/2 x h, horizontally
// subsampled). RAII over cudaMalloc/cudaFree. Passed between operators as a shared_ptr so device
// memory lifetime follows the message.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace spark::gpu {

struct GpuFrame {
  uint16_t* y = nullptr;   // w x h
  uint16_t* cb = nullptr;  // (w/2) x h
  uint16_t* cr = nullptr;  // (w/2) x h
  uint32_t width = 0;
  uint32_t height = 0;

  GpuFrame() = default;
  GpuFrame(uint32_t w, uint32_t h) { alloc(w, h); }
  ~GpuFrame() { free(); }
  GpuFrame(const GpuFrame&) = delete;
  GpuFrame& operator=(const GpuFrame&) = delete;

  void alloc(uint32_t w, uint32_t h);  // throws std::runtime_error on cudaMalloc failure
  void free();

  uint32_t chroma_width() const { return width / 2; }
  int y_step_bytes() const { return static_cast<int>(width) * 2; }
  int c_step_bytes() const { return static_cast<int>(width / 2) * 2; }
};

using GpuFramePtr = std::shared_ptr<GpuFrame>;

// Fill all three planes with a deterministic gradient (host buffer -> device). `seed` varies it per
// frame so a downstream check can tell frames apart. For the resize smoke source only.
void fill_gradient(GpuFrame& f, uint32_t seed);

// Copy a plane's worth of device data back to a host vector (for validation). Returns false on error.
bool copy_plane_to_host(const uint16_t* dev, uint32_t w, uint32_t h, std::vector<uint16_t>& out);

}  // namespace spark::gpu
