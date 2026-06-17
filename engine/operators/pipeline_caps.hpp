// Shared pipeline sizing helper. The only operator that emits more than one frame per compute is FRC
// in up-convert mode (real + mid = 2). In retime (SPARK_FRC=1) and passthrough (SPARK_FRC=0) every
// stage emits exactly one frame per compute, so the inter-operator queues only need to be 1 deep —
// halving the standing latency behind the paced TX (queue depth x output frame period == latency).
//
// This value is both the inter-op queue capacity floor AND the FRC output gate's min_size, so a single
// source keeps them consistent. It's read from the env the app is already driven by because Holoscan
// Parameters aren't populated yet when setup() runs (where connector capacity must be declared).
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace spark {

inline uint32_t pipeline_emit_burst() {
  const char* m = std::getenv("SPARK_FRC");
  return (m && std::strcmp(m, "2") == 0) ? 2u : 1u;  // 2 only for up-convert; 1 for retime/passthrough
}

}  // namespace spark
