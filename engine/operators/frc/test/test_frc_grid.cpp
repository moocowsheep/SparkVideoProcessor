// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Unit tests for UniformGrid — the tick/phase math behind FRC uniform-grid mode (SPARK_FRC=3).
// Pure host code, no CUDA/Holoscan.
#include "../frc_grid.hpp"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

using spark::frc::UniformGrid;

namespace {

constexpr uint64_t kOut = 16683333;         // 59.94 output interval, ns
constexpr uint64_t kIn = 2 * kOut;          // clean 29.97 source interval
constexpr uint64_t kStall = 8 * kOut;       // the operator's stall threshold
constexpr uint64_t kT0 = 1000000000000ULL;  // arbitrary capture epoch

struct Sim {
  UniformGrid g;
  uint64_t dropped = 0;
  std::vector<UniformGrid::Tick> emitted;

  explicit Sim(uint64_t interval) { g.interval_ns = interval; }

  int bracket(uint64_t prev_ts, uint64_t cur_ts, int max_out = 4) {
    UniformGrid::Tick t[16];
    const int n = g.ticks(prev_ts, cur_ts, t, max_out, kStall, &dropped);
    for (int i = 0; i < n; ++i) emitted.push_back(t[i]);
    return n;
  }
};

void test_clean_source_2x() {
  // Clean 29.97 source: after anchoring, every bracket must yield exactly 2 ticks — a midpoint
  // (phase 0.5) and the real frame (phase 1.0) — i.e. the mode degenerates to classic up-convert.
  Sim s(kOut);
  uint64_t prev = kT0;
  int n = s.bracket(prev, prev + kIn);  // anchor bracket: first tick == cur, phase 1
  assert(n == 1);
  assert(s.emitted[0].ts == prev + kIn);
  assert(s.emitted[0].phase > 0.999f);
  prev += kIn;
  for (int b = 0; b < 100; ++b) {
    const size_t base = s.emitted.size();
    n = s.bracket(prev, prev + kIn);
    assert(n == 2);
    assert(std::fabs(s.emitted[base].phase - 0.5f) < 0.01f);
    assert(s.emitted[base + 1].phase > 0.999f);
    assert(s.emitted[base + 1].ts == prev + kIn);
    prev += kIn;
  }
  assert(s.dropped == 0);
  std::puts("[ok] clean 29.97 source -> 2 ticks/bracket (mid + real), no drops");
}

void test_grid_rigidity() {
  // The whole point: every emitted ts must lie on ONE rigid grid (anchor + k*interval), no matter
  // how erratic the source spacing is. Erratic bracket widths jitter around 34.4ms (29.09fps-ish).
  Sim s(kOut);
  uint64_t prev = kT0;
  const uint64_t widths[] = {kIn,          kIn + 900000, kIn - 700000, kIn + 2400000,
                             kIn - 300000, kIn + 500000, kIn - 1200000, kIn + 1800000};
  s.bracket(prev, prev + widths[0]);
  const uint64_t anchor = s.emitted[0].ts;
  prev += widths[0];
  for (int b = 1; b < 200; ++b) {
    const uint64_t w = widths[b % 8];
    s.bracket(prev, prev + w);
    prev += w;
  }
  assert(s.emitted.size() > 300);  // ~2 per bracket
  for (const auto& t : s.emitted) {
    assert(t.ts >= anchor);
    assert((t.ts - anchor) % kOut == 0);  // on-grid, always
    assert(t.phase > 0.0f && t.phase <= 1.0f);
  }
  // Ticks must be strictly increasing by exactly one interval (no gaps: no stall in this input).
  for (size_t i = 1; i < s.emitted.size(); ++i)
    assert(s.emitted[i].ts - s.emitted[i - 1].ts == kOut);
  assert(s.dropped == 0);
  std::puts("[ok] erratic source -> every tick on the rigid grid, consecutive, no drops");
}

void test_slow_source_three_ticks() {
  // A slow patch (brackets wider than 2 output intervals) must yield 3 ticks in some brackets so
  // the OUTPUT rate stays nominal while the source underruns.
  Sim s(kOut);
  uint64_t prev = kT0;
  s.bracket(prev, prev + kIn);
  prev += kIn;
  const uint64_t slow = kIn + kOut / 2;  // 41.7ms brackets (~24fps source)
  size_t before = s.emitted.size();
  uint64_t span = 0;
  for (int b = 0; b < 40; ++b) {
    s.bracket(prev, prev + slow);
    prev += slow;
    span += slow;
  }
  const size_t made = s.emitted.size() - before;
  // Output ticks must match elapsed capture time / interval (rate preserved), +-1 for edges.
  const size_t expect = span / kOut;
  assert(made >= expect - 1 && made <= expect + 1);
  assert(s.dropped == 0);
  std::puts("[ok] slow (~24fps) source -> output rate preserved (3-tick brackets appear)");
}

void test_stall_freezes_not_interpolates() {
  // A source stall (bracket > stall threshold) must jump the grid rather than emit a long run of
  // synthetic frames, and the jump must preserve grid phase.
  Sim s(kOut);
  uint64_t prev = kT0;
  s.bracket(prev, prev + kIn);
  const uint64_t anchor = s.emitted[0].ts;
  prev += kIn;
  s.bracket(prev, prev + kIn);
  prev += kIn;
  const uint64_t stall = 20 * kOut;  // 333ms hole
  const size_t before = s.emitted.size();
  const uint64_t dropped_before = s.dropped;
  const int n = s.bracket(prev, prev + stall);
  prev += stall;
  assert(n <= 1);                          // at most the tick right at/under cur — no synthetic run
  assert(s.dropped > dropped_before);      // the hole was accounted, not synthesized
  for (size_t i = before; i < s.emitted.size(); ++i)
    assert((s.emitted[i].ts - anchor) % kOut == 0);  // grid phase preserved across the jump
  // And the stream resumes normally after the stall.
  const size_t base = s.emitted.size();
  const int m = s.bracket(prev, prev + kIn);
  assert(m >= 1 && m <= 3);
  for (size_t i = base; i < s.emitted.size(); ++i)
    assert((s.emitted[i].ts - anchor) % kOut == 0);
  std::puts("[ok] 333ms stall -> grid jump (phase preserved), no synthetic run, clean resume");
}

void test_burst_cap() {
  // A bracket holding more ticks than max_out (deep slow patch below the stall threshold) must cap
  // at max_out and count the overflow as dropped — never emit more than downstream can queue.
  Sim s(kOut);
  uint64_t prev = kT0;
  s.bracket(prev, prev + kIn);
  prev += kIn;
  const uint64_t wide = 6 * kOut;  // 100ms bracket: 6 ticks inside, cap is 4
  const int n = s.bracket(prev, prev + wide, 4);
  assert(n == 4);
  assert(s.dropped == 2);
  std::puts("[ok] over-burst bracket -> capped at max_out, overflow counted as dropped");
}

void test_disabled_and_degenerate() {
  UniformGrid g;  // interval 0 = disabled
  UniformGrid::Tick t[4];
  assert(g.ticks(kT0, kT0 + kIn, t, 4, kStall, nullptr) == 0);
  g.interval_ns = kOut;
  assert(g.ticks(kT0, kT0, t, 4, kStall, nullptr) == 0);      // empty bracket
  assert(g.ticks(kT0 + kIn, kT0, t, 4, kStall, nullptr) == 0);  // non-monotonic
  std::puts("[ok] disabled / degenerate brackets -> 0 ticks");
}

}  // namespace

int main() {
  test_clean_source_2x();
  test_grid_rigidity();
  test_slow_source_three_ticks();
  test_stall_freezes_not_interpolates();
  test_burst_cap();
  test_disabled_and_degenerate();
  std::puts("test_frc_grid: ALL PASS");
  return 0;
}
