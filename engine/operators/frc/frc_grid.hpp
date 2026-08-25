// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// UniformGrid — the tick/phase math for FRC uniform-grid up-convert (SPARK_FRC=3), kept free of
// CUDA or the runtime, so it unit-tests standalone (test/test_frc_grid.cpp).
//
// Classic up-convert (mode 2) emits [mid, cur] per input frame, source-LOCKED: the output timeline
// inherits every wobble of the source's frame spacing (an erratic source -> visible motion wobble,
// see BiDirect-1). Uniform mode instead lays a fixed nominal grid over the capture timeline and,
// for each source bracket (prev_ts, cur_ts], emits one output frame per grid tick inside it,
// interpolated at the tick's TRUE phase between the actual capture times. Output cadence is then
// exactly nominal regardless of source wobble; timing error moves out of the frame clock and into
// (much less visible) interpolation phase.
//
// The grid is anchored once (first tick = the first bracket's cur) and advances by exactly
// interval_ns forever after, so every emitted capture_ts lands on the same rigid grid — the TX
// genlock (base = capture_ts + offset) then produces a perfectly even wire cadence.
#pragma once

#include <cstdint>

namespace spark::frc {

struct UniformGrid {
  uint64_t interval_ns = 0;  // nominal output frame interval; 0 = disabled
  uint64_t next_ns = 0;      // next tick to emit; 0 = not yet anchored

  struct Tick {
    uint64_t ts;  // grid timestamp for the output frame (its capture_ts downstream)
    float phase;  // interpolation t in (0,1]: (ts - prev_ts) / (cur_ts - prev_ts); 1.0 == cur
  };

  // Collect the grid ticks inside the source bracket (prev_ts, cur_ts]. Writes up to max_out ticks
  // (ascending, phases monotonic); ticks beyond max_out and ticks jumped over a stall are counted
  // into *dropped and the grid advances past them (grid PHASE is preserved — the jump is a whole
  // number of intervals — so the wire cadence never shifts). A bracket wider than stall_ns is a
  // source stall: interpolating across it would synthesize long fictitious motion, so the grid
  // jumps to just below cur_ts and the output freezes for the gap instead (receiver repeats).
  int ticks(uint64_t prev_ts, uint64_t cur_ts, Tick* out, int max_out, uint64_t stall_ns,
            uint64_t* dropped) {
    if (interval_ns == 0 || cur_ts <= prev_ts || max_out <= 0) return 0;
    if (next_ns == 0) next_ns = cur_ts;  // anchor: first tick == first real frame (phase 1)
    if (cur_ts - prev_ts > stall_ns && next_ns < cur_ts) {
      const uint64_t behind = cur_ts - next_ns;
      const uint64_t skip = behind / interval_ns;  // whole intervals: lands within one tick of cur
      if (dropped) *dropped += skip;
      next_ns += skip * interval_ns;
    }
    const double span = static_cast<double>(cur_ts - prev_ts);
    int n = 0;
    while (next_ns <= cur_ts) {
      if (n < max_out) {
        out[n].ts = next_ns;
        out[n].phase = next_ns <= prev_ts
                           ? 0.0f
                           : static_cast<float>(static_cast<double>(next_ns - prev_ts) / span);
        ++n;
      } else if (dropped) {
        ++*dropped;  // source ran slower than the grid for longer than the burst allows
      }
      next_ns += interval_ns;
    }
    return n;
  }
};

}  // namespace spark::frc
