// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// RTP <-> absolute-time conversion (ST 2110-10: RTP timestamps count media-clock ticks since the
// PTP epoch, modulo 2^32). Pure header, no DPDK/Holoscan — unit-tested in test_rtp_time.cpp.
//
// Why this exists: a 32-bit RTP timestamp is ambiguous (it wraps every ~13.25 h at 90 kHz, ~24.9 h
// at 48 kHz). Given ANY reference instant on the same clock the sender stamps against — a NIC HW
// arrival timestamp on a PTP-disciplined PHC is ideal — the ambiguity resolves to a unique absolute
// capture time within a +-2^31-tick window (+-6.6 h / +-12.4 h). That absolute time is what the
// fixed-latency schedule needs: wire_time = capture + L holds across engine restarts and never
// inherits a per-run base, and video (90 kHz) and audio (48 kHz) land on ONE shared timeline, so
// lip-sync is exact by construction.
#pragma once

#include <cstdint>

namespace spark::st2110 {

// Absolute media-clock ticks since the epoch at time `ns` — NOT truncated to 32 bits.
// __int128 intermediate: ns (~1.8e18 for 2027 TAI) * 90000 overflows 64 bits.
inline uint64_t rtp_ticks_abs(uint64_t ns, uint32_t rate_hz) {
  return static_cast<uint64_t>((static_cast<unsigned __int128>(ns) * rate_hz) / 1000000000u);
}

// Reconstruct the absolute nanosecond instant a 32-bit RTP timestamp refers to, given `near_ns` on
// the SAME clock the sender stamps against (for a PTP-locked ST 2110 sender: any PTP-synced clock,
// e.g. the NIC PHC arrival time of the packet). Chooses the unique instant with ticks == rtp_ts
// (mod 2^32) nearest `near_ns`; the signed cast tolerates the sender being slightly AHEAD of the
// reference (clock skew) as naturally as behind. Truncation error < 1 tick (11.1 us at 90 kHz).
inline uint64_t rtp_unwrap_ns(uint32_t rtp_ts, uint32_t rate_hz, uint64_t near_ns) {
  const uint64_t near_ticks = rtp_ticks_abs(near_ns, rate_hz);
  // ticks elapsed from the stamp to the reference, in [-2^31, 2^31): wrap-safe by construction
  const int32_t back = static_cast<int32_t>(static_cast<uint32_t>(near_ticks) - rtp_ts);
  const uint64_t ticks = near_ticks - static_cast<int64_t>(back);
  return static_cast<uint64_t>((static_cast<unsigned __int128>(ticks) * 1000000000u) / rate_hz);
}

// Correction for a sender whose RTP timestamps are NOT on the PTP epoch (observed: a BMD 2110-30
// sender with its audio epoch frozen seconds off while its video stamps were correct). Returns the
// mod-2^32 tick delta that moves `rtp_ts` onto the reference instant: unwrap(rtp_ts + delta) lands
// within 1 tick of `ref_ns`. Adding the SAME latched delta to every subsequent stamp preserves the
// sender's tick cadence exactly (integer add, no rounding), so the corrected stream is smooth and
// only the latch instant steps.
inline uint32_t rtp_restamp_delta(uint32_t rtp_ts, uint32_t rate_hz, uint64_t ref_ns) {
  return static_cast<uint32_t>(rtp_ticks_abs(ref_ns, rate_hz)) - rtp_ts;
}

}  // namespace spark::st2110
