// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Zero-copy host access decision, shared by every operator that bridges a host wire buffer and the
// GPU (unpack/pack, jxs decode/encode).
//
// On GB10 (Grace Blackwell, cache-coherent unified LPDDR5x) the GPU can read/write ordinary malloc'd
// host memory directly (pageableMemoryAccess), so the staging copies (H2D on ingress, D2H on egress)
// are pure overhead — the kernels touch each octet exactly once anyway. SPARK_ZEROCOPY=0 forces the
// copy path (A/B or fallback on a non-coherent host), =1 forces zero-copy (testing); unset
// auto-detects. Decided once per process and logged once, so every operator agrees.
#pragma once

#include <cstdlib>
#include <string>

#include <cuda_runtime.h>
#include <holoscan/holoscan.hpp>

namespace spark::codec {

inline bool host_zerocopy() {
  static const bool on = [] {
    bool v = false;
    const char* e = std::getenv("SPARK_ZEROCOPY");
    if (e && *e) {
      v = std::string(e) != "0";
    } else {
      cudaDeviceProp prop{};
      int dev = 0;
      if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        v = prop.pageableMemoryAccess != 0;
    }
    HOLOSCAN_LOG_INFO("codec: host zero-copy {} ({})", v ? "ON" : "OFF",
                      e && *e ? "SPARK_ZEROCOPY" : "auto: pageableMemoryAccess");
    return v;
  }();
  return on;
}

}  // namespace spark::codec
