// AI super-resolution engine: runs an SRW1 net (sr_model.hpp) on the luma plane of a 10-bit
// GpuFrame, in (w x h) uint16 -> out (2w x 2h) uint16, entirely on the GPU. Hand-rolled CUDA —
// this box (GB10 aarch64) has no TensorRT/cuDNN, and the nets are tiny stacks of small convs
// where fused direct kernels beat a framework anyway. fp16 activations, fp32 accumulation.
//
// The constructor compiles the net into a fused launch plan (known FSRCNN/ESPCN shapes hit
// specialized kernels; anything else runs on generic fallback kernels), uploads weights, and
// allocates activation workspaces for the given input geometry. run() only enqueues kernels on
// the caller's stream — no syncs, so it slots into the house operator contract (record the
// frame's ready event on the same stream afterwards).
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <cuda_runtime.h>

#include "sr_model.hpp"

namespace spark::sr {

class Engine {
 public:
  // Throws std::runtime_error on an unsupported net or CUDA allocation failure.
  Engine(const NetDef& net, uint32_t in_w, uint32_t in_h);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // y_in: in_w x in_h, y_out: 2*in_w x 2*in_h, both device pointers (10-bit values in uint16).
  void run(const uint16_t* y_in, uint16_t* y_out, cudaStream_t stream);

  uint32_t in_width() const;
  uint32_t in_height() const;
  // One-line launch-plan description, e.g. "head5(1->56->12) 4x conv3(12->12) tail(12->56->4+shuffle)".
  const std::string& plan() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace spark::sr
