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
#include <string>

namespace spark {

inline uint32_t pipeline_emit_burst() {
  const char* m = std::getenv("SPARK_FRC");
  if (m && std::strcmp(m, "3") == 0) return 4u;  // uniform grid: an erratic-slow source bracket can
                                                 // hold 3+ output ticks (nominal is 2), so allow 4
  return (m && std::strcmp(m, "2") == 0) ? 2u : 1u;  // 2 only for up-convert; 1 for retime/passthrough
}

// Per-hop input queue capacity. Only the hop fed directly by FRC's multi-frame burst needs
// burst-deep capacity — the app announces that op's name via SPARK_BURST_SINK (setup() runs at
// make_operator time, before Parameters bind, so env is the only channel). Every other hop is
// strictly 1:1 and sits at depth 1: each level of queue capacity behind the paced TX is a whole
// output frame period of standing latency (16.7ms at 59.94 — a uniform burst=4 at every hop cost
// ~100ms of latency ceiling, measured 2026-07-02).
inline uint32_t pipeline_queue_cap(const std::string& op_name) {
  const char* sink = std::getenv("SPARK_BURST_SINK");
  if (sink && *sink && op_name == sink) return pipeline_emit_burst();
  return 1u;
}

}  // namespace spark
