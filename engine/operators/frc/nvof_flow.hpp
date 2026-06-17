// NvofFlow — thin wrapper over the raw NVIDIA Optical Flow CUDA API (libnvidia-opticalflow), the
// hardware OFA proven in M0 (spike/nvof_probe.cpp). Computes a dense-ish flow field between two 8-bit
// luma frames; the FRC operator warps along it to synthesize interpolated frames (NvOFFRUC isn't
// available on aarch64, so we hand-roll FRC on the raw flow).
//
// Uses the CUDA driver API and the process's existing (runtime) primary context, so its flow buffer
// interops with the runtime-allocated GpuFrame planes. PIMPL keeps the NVOF headers out of this header.
#pragma once

#include <cstdint>
#include <memory>

namespace spark::frc {

class NvofFlow {
 public:
  NvofFlow();
  ~NvofFlow();

  void init(uint32_t width, uint32_t height, uint32_t grid_size = 4);

  // prevY8/curY8: device 8-bit luma, width x height, tightly packed. Enqueues BOTH flows on `stream`
  // (a cudaStream_t): forward prev->cur into the fwd buffer and backward cur->prev into the bwd buffer
  // (two nvOFExecute calls, frames swapped, no re-upload). The pair lets the warp do occlusion-aware
  // interpolation (forward<->backward consistency). ASYNC: NVOF's I/O is bound to `stream` and there is
  // no context sync, so the caller orders the consuming kernel on the same stream (or waits the output
  // via an event). `stream`=nullptr uses the default stream.
  void compute(const uint8_t* prevY8_dev, const uint8_t* curY8_dev, void* stream);

  // Flow outputs (device): grid_w x grid_h vectors, each {int16 x, int16 y} in S10.5 (value/32 = px).
  // fwd = prev->cur (indexed in prev coords); bwd = cur->prev (indexed in cur coords). Same pitch/grid.
  const void* flow_dev() const;      // forward (prev->cur)
  const void* flow_dev_bwd() const;  // backward (cur->prev)
  uint32_t flow_pitch_bytes() const;
  uint32_t grid_w() const;
  uint32_t grid_h() const;
  uint32_t grid_size() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace spark::frc
