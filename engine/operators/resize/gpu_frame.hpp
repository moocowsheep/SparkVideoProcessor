// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

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

#include <cuda_runtime.h>

namespace spark::gpu {

struct GpuFrame {
  uint16_t* y = nullptr;   // w x h
  uint16_t* cb = nullptr;  // (w/2) x h
  uint16_t* cr = nullptr;  // (w/2) x h
  uint32_t width = 0;
  uint32_t height = 0;
  // Cross-operator GPU ordering: the producing operator records this on its stream after the last
  // write to the planes; consumers cudaStreamWaitEvent on it before reading. Lets operators run on
  // independent CUDA streams (pipelined) instead of serializing on the default stream.
  cudaEvent_t ready = nullptr;
  // Latency probe: steady_clock ns stamped when the frame enters the GPU graph (unpack), propagated
  // through frc/resize, read at pack egress. 0 = unset (host-only; device kernels never touch it).
  uint64_t t_ingest_ns = 0;
  // Source frame timing (the RX's capture_ts_ns, from the sender's RTP clock), propagated unpack->pack
  // so the TX can genlock its send schedule to the source instead of a free-running grid. 0 = unset.
  uint64_t capture_ts_ns = 0;

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
