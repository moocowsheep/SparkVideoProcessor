// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Unit test: RTP timestamp <-> absolute time (rtp_time.hpp). Pure CPU, no NIC/root.
//
// The properties the fixed-latency schedule and the audio relay stand on:
//  1. round-trip: unwrap(ticks(T) mod 2^32, near T+delay) recovers T within 1 tick,
//     for delays up to seconds, at 90 kHz (video) and 48 kHz (audio);
//  2. wrap-safe: holds across the 32-bit wrap boundary (13.25 h @90k, 24.9 h @48k);
//  3. skew-tolerant: a reference slightly BEFORE the stamp (sender clock ahead) still recovers T;
//  4. cross-rate: video and audio stamps of the same instant unwrap onto one timeline;
//  5. monotonic: consecutive frame stamps unwrap strictly increasing across a wrap.
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "../rtp_time.hpp"
#include "../../st2110_tx/rtp_st2110.hpp"

using spark::st2110::rtp_ticks_abs;
using spark::st2110::rtp_unwrap_ns;

// assert() vanishes under Release/-DNDEBUG — use an explicit check like the other test binaries.
#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

namespace {

uint64_t abs_diff(uint64_t a, uint64_t b) { return a > b ? a - b : b - a; }

// One tick in ns, rounded up — the unwrap's truncation bound.
uint64_t tick_ns(uint32_t rate) { return 1000000000ULL / rate + 1; }

void check_roundtrip(uint64_t t_ns, uint32_t rate, int64_t ref_delay_ns, const char* what) {
  const uint32_t stamp = static_cast<uint32_t>(rtp_ticks_abs(t_ns, rate));
  const uint64_t near = static_cast<uint64_t>(static_cast<int64_t>(t_ns) + ref_delay_ns);
  const uint64_t got = rtp_unwrap_ns(stamp, rate, near);
  if (abs_diff(got, t_ns) > tick_ns(rate)) {
    std::fprintf(stderr, "FAIL %s: T=%llu rate=%u delay=%lld -> got %llu (err %llu ns)\n", what,
                 (unsigned long long)t_ns, rate, (long long)ref_delay_ns, (unsigned long long)got,
                 (unsigned long long)abs_diff(got, t_ns));
    std::exit(1);
  }
}

}  // namespace

int main() {
  // A 2026-scale TAI instant (ns since 1970): the regime the engine actually runs in.
  const uint64_t kT2026 = 1782000000ULL * 1000000000ULL;

  const uint32_t rates[] = {90000, 48000};
  for (uint32_t rate : rates) {
    // 1. round-trip across realistic reference delays (arrival is capture + net/pacing delay)
    for (int64_t d : {int64_t(0), int64_t(100000), int64_t(16700000), int64_t(33400000),
                      int64_t(150000000), int64_t(2000000000), int64_t(60000000000)})
      check_roundtrip(kT2026 + 12345, rate, d, "roundtrip");
    // 3. sender clock slightly ahead of the reference clock (arrival "before" capture)
    for (int64_t d : {int64_t(-100000), int64_t(-3000000), int64_t(-500000000)})
      check_roundtrip(kT2026 + 999, rate, d, "skew");
    // 2. wrap boundary: instants straddling an exact multiple of 2^32 ticks
    const uint64_t wrap_ns =
        static_cast<uint64_t>((static_cast<unsigned __int128>(4294967296ULL) * 1000000000ULL) / rate);
    const uint64_t near_wrap = (kT2026 / wrap_ns) * wrap_ns;  // a wrap edge in the 2026 regime
    for (int64_t off : {int64_t(-50000000), int64_t(-1000), int64_t(0), int64_t(1000),
                        int64_t(50000000)})
      check_roundtrip(static_cast<uint64_t>(static_cast<int64_t>(near_wrap) + off), rate, 20000000,
                      "wrap");
  }

  // 4. cross-rate: one instant stamped at 90 kHz (video) and 48 kHz (audio) unwraps onto the same
  // timeline — the lip-sync premise.
  {
    const uint64_t t = kT2026 + 777777777;
    const uint64_t v = rtp_unwrap_ns(static_cast<uint32_t>(rtp_ticks_abs(t, 90000)), 90000, t + 5000000);
    const uint64_t a = rtp_unwrap_ns(static_cast<uint32_t>(rtp_ticks_abs(t, 48000)), 48000, t + 5000000);
    CHECK(abs_diff(v, a) <= tick_ns(48000) + tick_ns(90000));
  }

  // 5. monotonic across a wrap: 29.97 fps frame stamps (3003 ticks apart @90k) unwrap strictly
  // increasing through the boundary, each against its own arrival.
  {
    const uint64_t wrap_ns =
        static_cast<uint64_t>((static_cast<unsigned __int128>(4294967296ULL) * 1000000000ULL) / 90000ULL);
    uint64_t t = (kT2026 / wrap_ns) * wrap_ns - 10 * 33366700ULL;
    uint64_t prev = 0;
    for (int i = 0; i < 20; ++i, t += 33366700ULL) {
      const uint64_t got =
          rtp_unwrap_ns(static_cast<uint32_t>(rtp_ticks_abs(t, 90000)), 90000, t + 16000000);
      CHECK(prev == 0 || got > prev);
      prev = got;
    }
  }

  // TX re-stamp consistency: rtp_timestamp_90k(unwrap(ts) + L) == ts + ticks(L) within 1 tick —
  // what a receiver aligning output RTP timestamps against PTP actually checks.
  {
    const uint64_t L = 150000000;  // 150 ms
    const uint64_t t = kT2026 + 31415926;
    const uint32_t in_ts = static_cast<uint32_t>(rtp_ticks_abs(t, 90000));
    const uint64_t cap = rtp_unwrap_ns(in_ts, 90000, t + 20000000);
    const uint32_t out_ts = spark::st2110::rtp_timestamp_90k(cap + L);
    const uint32_t expect = in_ts + static_cast<uint32_t>(rtp_ticks_abs(L, 90000));
    const int32_t err = static_cast<int32_t>(out_ts - expect);
    CHECK(err >= -1 && err <= 1);
  }

  // 6. broken-sender-epoch rescue (rtp_restamp_delta): a stamp seconds off its arrival — the
  // BMD-1 failure observed live (audio epoch −14.8s then −8.26s vs TAI, video correct) — latches a
  // delta that (a) puts the corrected stamp within 1 tick of arrival, (b) applied unchanged to
  // later packets preserves the sender's exact tick cadence, (c) is 0 for a sane sender.
  {
    using spark::st2110::rtp_restamp_delta;
    const uint32_t rate = 48000;
    for (int64_t epoch_err_ns : {int64_t(-14795000000), int64_t(-8259000000), int64_t(22205000000)}) {
      const uint64_t arrival = kT2026 + 123456789;
      const uint64_t claimed = static_cast<uint64_t>(static_cast<int64_t>(arrival) + epoch_err_ns);
      const uint32_t bad_ts = static_cast<uint32_t>(rtp_ticks_abs(claimed, rate));
      const uint32_t delta = rtp_restamp_delta(bad_ts, rate, arrival);
      // (a) corrected stamp unwraps onto the arrival instant
      CHECK(abs_diff(rtp_unwrap_ns(bad_ts + delta, rate, arrival), arrival) <= tick_ns(rate));
      // (b) 10 s later (packets every 48 ticks / 1 ms), the SAME delta still lands each packet on
      // its own arrival, and corrected stamps advance by exactly the sender's cadence
      uint32_t prev_ts = bad_ts + delta;
      for (int i = 1; i <= 10000; i += 999) {
        const uint64_t arr_i = arrival + uint64_t(i) * 1000000ULL;
        const uint32_t ts_i = bad_ts + uint32_t(i) * 48u;
        CHECK(abs_diff(rtp_unwrap_ns(ts_i + delta, rate, arr_i), arr_i) <= tick_ns(rate));
        CHECK(uint32_t((ts_i + delta) - prev_ts) % 48u == 0u);
        prev_ts = ts_i + delta;
      }
    }
    // (c) sane sender: stamp == arrival instant -> delta 0 (the verbatim/bit-transparent path)
    const uint64_t t = kT2026 + 5555;
    CHECK(rtp_restamp_delta(static_cast<uint32_t>(rtp_ticks_abs(t, rate)), rate, t) == 0);
  }

  std::printf("test_rtp_time: all checks passed\n");
  return 0;
}
