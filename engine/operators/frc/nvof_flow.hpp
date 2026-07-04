// NvofFlow — thin wrapper over the raw NVIDIA Optical Flow CUDA API (libnvidia-opticalflow), the
// hardware OFA proven in M0 (spike/nvof_probe.cpp). Computes a dense-ish flow field between two 8-bit
// luma frames; the FRC operator warps along it to synthesize interpolated frames (NvOFFRUC isn't
// available on aarch64, so we hand-roll FRC on the raw flow).
//
// Runs TWO NVOF sessions — one per direction — so the driver's temporal hints stay
// direction-consistent (a single session alternating fwd/bwd would hint each solve with the other
// direction; that forced hints off). Also emits the OFA per-vector matching cost for the warp's
// confidence weighting. Env knobs: SPARK_FRC_TEMPORAL_HINTS=0, SPARK_FRC_COST=0 disable each.
//
// Uses the CUDA driver API and the process's existing (runtime) primary context, so its flow buffer
// interops with the runtime-allocated GpuFrame planes. PIMPL keeps the NVOF headers out of this header.
#pragma once

#include <cstdint>
#include <memory>

#include "flow_view.hpp"

namespace spark::frc {

class NvofFlow {
 public:
  NvofFlow();
  ~NvofFlow();

  void init(uint32_t width, uint32_t height, uint32_t grid_size = 4);

  // prevY8/curY8: device 8-bit luma, width x height, tightly packed. Enqueues BOTH flows on `stream`
  // (a cudaStream_t): forward prev->cur and backward cur->prev, each on its own NVOF session, plus
  // their cost planes. ASYNC: NVOF's I/O is bound to `stream` and there is no context sync, so the
  // caller orders the consuming kernel on the same stream (or waits the output via an event).
  // `stream`=nullptr uses the default stream.
  void compute(const uint8_t* prevY8_dev, const uint8_t* curY8_dev, void* stream);

  // The outputs as one bundle (device pointers valid until destruction): both flow grids
  // (grid_w x grid_h SHORT2, S10.5, luma coords) and, when available, both cost planes.
  FlowView view() const;

  // Individual accessors (subset of view(), kept for probes/tests).
  const void* flow_dev() const;      // forward (prev->cur), indexed in prev coords
  const void* flow_dev_bwd() const;  // backward (cur->prev), indexed in cur coords
  uint32_t flow_pitch_bytes() const;
  uint32_t grid_w() const;
  uint32_t grid_h() const;
  uint32_t grid_size() const;
  bool has_cost() const;        // per-vector cost planes present in view()
  bool temporal_hints() const;  // temporal hinting active on both sessions

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace spark::frc
