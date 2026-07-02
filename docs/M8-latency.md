# M8 — end-to-end latency: model, floor, and the knobs that get us there

**Status:** implemented + built 2026-07-01; unit tests green; zero-copy A/B measured on the GB10.
Hardware validation on the BMD rig pending (checklist at the bottom).

## Where the latency actually lives

With the genlock TX (`reanchor_lead_ns > 0` + a `capture_ts`), every frame is scheduled at
`capture_ts + genlock_offset_` — so **the end-to-end latency (source capture → first bit on our TX
wire) IS `genlock_offset_`**, full stop. Everything else (queue depths, GPU time, scheduler hops)
only matters through how it shapes that offset and the safety margin (`lead`) it must contain.

Budget for 2160p59.94 IP10→IP10 (the flagship path), per frame:

| stage | cost | nature |
|---|---|---|
| source paces frame onto wire, RX reassembles to marker | ~16.0 ms (T_active) | inherent to frame-granularity processing |
| RX queue → unpack → frc → resize → pack (GPU + hops) | ~3–15 ms | trimmable (items 2–5 below) |
| TX lead (schedule base over NIC clock) | 8 ms default | trimmable margin (`SPARK_TX_LEAD_NS`) |
| TX paces frame to receiver | ~16.0 ms (T_active) | inherent (receiver drains at raster rate) |

The **floor for frame-at-a-time processing** is therefore ≈ `T_active + GPU-chain + lead` to first
bit out (~20–24 ms ≈ 1.3 frames at 59.94, FRC off), plus the receiver's own buffer. Going below
that means slice/line-level pipelining (process and re-transmit lines as they arrive) — a real
rearchitecture (chunked RX→GPU handoff, sliced kernels, per-slice TX scheduling; FRC excluded, it
needs the future frame). Feasible on this hardware (the per-line tx_pp stamps already exist) and
would land at ~1–3 ms, but it is a milestone of its own, not a tuning pass.

## What was wrong (and is now fixed)

1. **The offset was calibrated once, on the coldest possible frame, and never improved.**
   `genlock_offset_` was set by the FIRST frame through a cold pipeline (CUDA pool allocs, NVOF
   init, RX queue backlog) and re-centered only if the lead left a `[tgt/4, 2 frames]` band. Result:
   steady-state latency = *whatever startup baked in*, up to 2 frames (~33 ms at 59.94, ~66 ms at
   29.97) above the achievable floor, forever. → **Latency trim servo** (`trim_ns` param,
   `SPARK_TX_TRIM_NS`, default 5000 ns/frame): whenever the lead sits >500 µs above target, the
   base slews down by ≤5 µs per frame (≈300 ppm at 59.94 — smooth, no steps) until lead == target.
   The 1 Hz `spark_live` line now reports `tx_lead_us` / `tx_e2e_us` / `tx_trim_ms` so you can watch
   it converge.

2. **The RTP media clock never followed the wire.** After any re-anchor (or now, trim), the RTP
   timestamps stayed permanently offset from the wire schedule — so a timestamp-driven receiver kept
   playing at the OLD latency and wire-side wins bought nothing. → The media clock now **slews
   toward the base at the same bounded rate** (never steps — receivers keep clock lock), so wire
   time and RTP ts converge together and both receiver models (arrival-driven and ts-driven) see
   the reduction.

3. **The scheduler added up to 5 ms of polling latency per operator hop.** Holoscan's
   `MultiThreadScheduler` sleeps `check_recession_period_ms` (default **5.0 ms**) whenever a polling
   pass finds nothing ready; this graph has 5 hops. → Default is now the **EventBasedScheduler**
   (operators dispatch the moment an emit readies them). `SPARK_SCHED=mts` restores the exact
   legacy scheduler if anything misbehaves.

4. **The flagship 2160p→2160p path ran a full NPP cubic resample into an identical geometry.**
   → **1:1 passthrough** in ResizeOp: same in/out dims forwards the input frame (metadata, ready
   event and all). Saves GPU time and a pool hop; benchmark mode (`measure`) still runs the kernel.

5. **Both codec stages staged frames through device copies on a machine with coherent memory.**
   Measured on this GB10 (16.6 MB frame, steady state): zero-copy kernel touching malloc'd host
   memory **0.44 ms** vs 0.41 ms device-only vs **2.28 ms** for the H2D+kernel+D2H copy path. →
   **Zero-copy unpack/pack** when `cudaDeviceProp.pageableMemoryAccess=1` (auto-detected;
   `SPARK_ZEROCOPY=0/1` forces): unpack reads the RX ring buffer in place (holding the shared_ptr
   until the kernel completes — the RX reuses slots on `use_count==1`), pack writes straight into
   the TX host buffer. Removes ~2 ms/frame of copies plus host-thread blocking.

6. **FRC up-convert scheduled both frames of each pair at the same instant.** The interpolated mid
   carried `cur`'s `capture_ts`, so under genlock the pair (mid, real) got the SAME send base and
   the second frame past-stamped into an unpaced burst. → Mid is now stamped at the temporal
   **midpoint of (prev, cur)**, spacing the pair exactly one output interval. (For retime it's a
   constant half-frame shift the calibration absorbs.)

## Knobs (all env; daemon-launched runs inherit them)

| env | default | meaning |
|---|---|---|
| `SPARK_TX_LEAD_NS` | 8000000 | genlock lead target = TX-side latency + jitter margin. Lower stepwise (6→4→3 ms) while `tx_past_err` stays 0 and no dips. Every ns here is a ns of latency. |
| `SPARK_TX_TRIM_NS` | 5000 | max ns/frame the base+media clock slew toward target. 0 = exact legacy timing (no trim, no slew). Raise to converge faster (more receiver rate offset while trimming). |
| `SPARK_SCHED` | event | `event` = EventBasedScheduler (low latency); `mts` = legacy MultiThreadScheduler. |
| `SPARK_ZEROCOPY` | auto | host zero-copy in unpack/pack; auto-detects `pageableMemoryAccess` (GB10: on). |
| `SPARK_FRC` | 1 | **set 0 for lowest latency**: skips NVOF entirely AND removes retime's half-frame content age. 2 (up-convert) inherently needs the next frame. |
| (automatic) | — | identity resize passthrough when in==out geometry. |

Lowest-latency recipe: `SPARK_FRC=0 SPARK_TX_LEAD_NS=4000000` (after validating past_err stays 0),
everything else default. Expected: `tx_e2e_us` settling around ~20–24 ms (first bit out).

## Reading the telemetry

- `tx_lead_us` — schedule base over the NIC clock at compute time. Persistently above target ⇒ the
  servo is (or should be) trimming; pinned at target ⇒ converged. Falling toward `tgt/4` ⇒ pipeline
  delay growing (watch for a re-anchor).
- `tx_e2e_us` — `genlock_offset_ mod RTP-wrap`: capture → first-bit-out. Exact when the source
  stamps RTP on the PTP epoch (the BMDs do). A constant ~37 s reading means a TAI/UTC epoch skew
  between source and our PHC — the absolute value is then off, but its *changes* are still real.
- `tx_trim_ms` — total latency reclaimed by the servo this run.
- `pipe_latency_us` (existing, from pack) — unpack→pack GPU-chain + hop latency; this is where the
  scheduler and zero-copy wins show up.
- `st2110_tx: genlock re-anchor #N` — now logged with the lead it saw; should be ~1 early in a run
  (startup) and none after.

## Hardware validation checklist (BMD rig)

1. Baseline the old behavior first: `SPARK_TX_TRIM_NS=0 SPARK_SCHED=mts SPARK_ZEROCOPY=0` — record
   `tx_lead_us`/`tx_e2e_us`/`pipe_latency_us` after 5 min.
2. Defaults run: watch `tx_trim_ms` grow and `tx_lead_us` settle at 8000; picture must stay clean
   through the whole trim (the slew is ≈300 ppm — if the BMD shows any instability, halve
   `SPARK_TX_TRIM_NS` and retest).
3. `tx_past_err` must stay 0 and `rx_lost` 0 throughout; re-anchors ≤1 (startup).
4. Then push `SPARK_TX_LEAD_NS` down 8→6→4 ms, 10 min each, same checks.
5. FRC=2 (30→60) run: the pair-burst fix means past_err should now be 0 where it previously burst
   every second frame.
6. Optional glass-to-glass: camera → BMD encode → Spark → BMD decode → monitor, phone slo-mo on a
   running timecode; compare against `tx_e2e_us` + receiver buffer.
