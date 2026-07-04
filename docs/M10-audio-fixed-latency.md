# M10 — Fixed end-to-end latency + ST 2110-30 audio

Status: implemented + unit-tested; **hardware validation pending** (checklist at the bottom).

## The model

One rule for both essences:

```
wire_time = absolute_capture_time + L        (L = SPARK_LATENCY_MS, a configured CONSTANT)
```

- **Video**: each frame's RTP timestamp is unwrapped to its absolute PTP-epoch capture instant at
  the RX (against the frame's first-packet NIC PHC arrival — `rtp_time.hpp`), carried through the
  GPU chain as before, and the TX schedules the frame at `capture + L`. No first-frame calibration,
  no trim servo, no re-anchor: the latency is a printable constant across restarts.
- **Audio**: the companion 2110-30 flow is received on the same RX port (second `rte_flow`, same
  queue, classified per packet), each packet's 48 kHz timestamp is unwrapped the same way, and a
  relay thread re-times it to `capture + L` on a dedicated TX queue with the same `tx_pp`
  send-on-timestamp pacing. The packet is otherwise **bit-transparent** (seq/SSRC/PCM verbatim;
  only the RTP timestamp is shifted by exactly L in ticks).

**Lip-sync is exact by construction**: both essences ride one timeline (the shared PHC/PTP clock)
with one constant. There is no A/V delay to calibrate and nothing that drifts. A receiver aligning
RTP-vs-PTP sees both essences at `capture + L`; output RTP timestamps are true PTP-epoch stamps
(the ST 2110-10 ideal — we previously stamped from an arbitrary anchored media clock).

A frame that misses its slot (lead < 2 ms) is **skipped unsent** — the receiver repeats a frame,
the schedule stands still. Latency never silently changes; chronic skipping is an operator signal
(raise L), reported via `tx_skipped` + a rate-limited WARN. This also absorbs host-load stalls up
to (L − pipeline floor) invisibly — margin the near-floor servo never had.

## Knobs

| Env | Default | Meaning |
|---|---|---|
| `SPARK_LATENCY_MS` | **150** | capture→wire latency for BOTH essences. `0` = legacy adaptive servo (M8 trim/re-anchor; audio disabled). |
| `SPARK_AV_OFFSET_MS` | 0 | extra audio delay (+) or advance (−) vs video — operator lip-sync trim. |
| `SPARK_RX_AUDIO_MCAST/SRC/PORT` | — | audio source (set by the daemon from NMOS IS-05, or by hand). Port 0 → 5004. |
| `SPARK_TX_AUDIO_MCAST/PORT` | — | audio egress (from NMOS audio sender activation, or by hand). |
| `SPARK_AUDIO_RATE` | 48000 | audio RTP clock (2110-30 is 48 kHz). |

Audio activates only when **both** RX and TX audio groups are set **and** `SPARK_LATENCY_MS > 0`.
The RX frame queue auto-sizes to hold the standing store: `max(8, L·fps_in + 4)` frames.

## Prerequisites (why the ERROR log exists)

`capture + L` is only meaningful if the source's RTP stamps and our NIC clocks share the PTP
timeline: **ptp4l must be running** (`sudo bash deploy/ptp.sh start`; both media ports share PHC0)
and **the source must be PTP-locked** (the BMDs are). The TX logs a one-time
`FIXED-LATENCY CLOCK MISMATCH` ERROR if the first frame's lead is impossible (≫L or negative
seconds) — the fix is starting PTP, not tuning L. `SPARK_LATENCY_MS=0` falls back to the servo,
which tolerates unsynced clocks (it calibrates the offset away).

Sizing L: true e2e floor ≈ RX assembly (~1 source frame) + GPU chain (`pipe_latency_us`, sized to
its MAX, not its average — spikes to ~136 ms observed on the 1080p→2160p59.94 IP10 FRC chain
against a ~75 ms average) + submit lead. Probe with **`tx_skipped`**: step L down until skips
appear, then back off. (`tx_margin_us` cannot size L at 59.94/60 line-burst — see limitation 6;
it gauges TX-arrival slack, ~12.5 ms steady when healthy.)

## Telemetry

- `spark_live tx_…` gains `tx_fixed_ms` (configured L; 0 = servo) and `tx_margin_us`.
  `tx_e2e_us` is now **exact** in fixed mode (== L).
- New `spark_live audio_rx_pkts/audio_rx_bad/audio_rx_drop` (RX side) and
  `audio_tx_pkts/audio_tx_late/audio_tx_drop` (relay side).
- Daemon `PipelineStats` additions: `tx_reanchors, tx_skipped, fixed_latency_ms, tx_margin_us,
  pipe_latency_us, audio_rx_packets, audio_tx_packets, audio_late`; dashboard shows
  Latency/Margin/Pipe latency/Audio rows (audio late/drop and zero-margin highlight red).

## Control plane

Already wired end to end: the NMOS node's `reconcile()` pushes `rxAudio*` (receiver activation)
and `txAudio*` (audio sender activation) into the daemon config; the daemon setenvs
`SPARK_RX_AUDIO_*` / `SPARK_TX_AUDIO_*`; the engine now consumes them. Note the engine only runs
when the **video** receiver is active — audio is a companion essence, not a standalone route.
`SPARK_LATENCY_MS` is daemon-launch-env inherited (like `SPARK_TX_TP`), not yet a config field.

## Known limitations / follow-ups

1. **Audio SDP mirroring**: our NMOS audio sender still advertises 48 kHz/L24/2ch/1 ms + pt 97,
   but the relay is bit-transparent — a source with a different channel count/ptime/pt will play
   wrong (or not at all) on receivers that trust our SDP. Mirror the source's `rtpmap/fmtp/ptime`
   into the sender SDP when the audio receiver is active. (BMD↔BMD defaults match today.)
2. Audio and video flows must differ in group or port (same-group+same-port would classify as video).
3. Non-PTP sources: fixed mode paces correctly (unwrap is arrival-anchored) but absolute latency
   equals L only up to the source's clock error, and drift eventually skips a frame (video) or
   drops/dups a packet (audio blip). Servo mode remains the fallback for free-running sources.
4. Startup: the first few frames skip while the cold pipeline exceeds L−floor (CUDA/NVOF warmup) —
   honest repeats instead of the servo's baked-in transient. Expected, brief.
5. `tx_margin_us` clamps at 0 (digits-only parser); negative margin shows as 0 + rising skips.
6. **`tx_margin_us` does not measure L-headroom at 59.94/60 line-burst.** compute() blocks until
   its frame is nearly submitted (pacing horizon), so the next frame's arrival lead is pinned at
   `interval − (T_active − horizon)` ≈ 13.4 ms regardless of L (the L-store stands in the upstream
   queues). Margin gauges TX-arrival slack — ~12.5 ms steady when healthy, dips = TX-side stalls.
   To size L, probe with `tx_skipped` (step L down until skips appear), not margin.
7. NIC stats must stay off the pacing thread: the mlx5 `rte_eth_xstats` sweep costs 10–20 ms of
   firmware round-trips. It originally ran inside emit_live() on the compute thread and skipped
   exactly the 1 Hz stats frame at ANY L (found on the rig 2026-07-03); a 1 Hz poller thread now
   caches it. Symptom signature if it regresses: ~1 skip/s whose WARN directly follows the
   `spark_live tx_…` line, with small late leads (−8..+1 ms).
8. **Broken sender audio epoch (auto-rescued)**: a sender whose audio RTP timestamps are not on
   the PTP epoch (observed live 2026-07-04: BMD-1 audio frozen −14.8 s, then −8.26 s vs TAI
   mid-run, while its video stamps stayed correct) would unwrap to a capture seconds in the past
   and the relay would drop 100% of audio. `audio_ingest` now sanity-checks each unwrapped capture
   against the packet's NIC arrival; beyond ±500 ms it latches a constant mod-2³² tick delta
   (`rtp_restamp_delta`) that re-anchors the stream to arrival — smooth (exact integer add,
   cadence preserved), one step at the latch instant, and it returns to the verbatim stamp when
   the sender recovers. Lip-sync while latched is arrival-anchored: exact to within network+ptime
   (~1–2 ms), not to the sender's (broken) claim. WARN on each latch (1/s rate-limited);
   `audio_rx_relatch` counts them — any nonzero value means the sender needs fixing.
   **The VIDEO stamps get the same guard** (added 2026-07-04 after BMD-1's video epoch stepped
   +2.0 s: frames scheduled 2 s early, `tx_lead_us`≈2.06 s, true wire latency ~2.1 s while
   `tx_e2e_us` echoed L — and with a future epoch nothing skips, so a fixed-mode floor probe
   silently measures nothing). `video_capture_ns()` applies the same latch per frame at a tighter
   **±150 ms** (healthy video stamps sit within ~20 ms of first-packet arrival; the audio 500 ms
   bound let sub-threshold epoch hops accumulate into a 65→565 ms latency sawtooth, observed live);
   since both the fixed schedule and the outgoing wire stamps derive from `capture_ts` (wire =
   capture + L), the one correction keeps latency real and the output 2110-10 stamps on the PTP
   epoch. `rx_relatch` on the `spark_live rx_…` line counts video latches. Note this also
   subsumes most `FIXED-LATENCY CLOCK MISMATCH` cases: capture is now always arrival-sane, so
   that error should no longer fire even when the sender epoch is broken.

## Rig validation checklist

1. **PTP first**: `sudo bash deploy/ptp.sh start`; confirm rms lock + both ports on PHC0.
2. **Video fixed-latency**: usual 1080p29.97→2160p59.94 IP10 FRC route with defaults
   (`SPARK_LATENCY_MS=150`). Confirm: `tx_fixed_ms=150`, `tx_e2e_us=150000`, re-anchors 0,
   `past_err` 0, skips 0 after warmup, `tx_margin_us` steady ≫0; BMD picture clean.
   Restart the engine → identical `tx_e2e_us` (the whole point).
3. **Latency floor probe**: step `SPARK_LATENCY_MS` 150→120→100→80 until skips appear; note the
   floor + margin at each step. Pick the shipping default from floor + jitter headroom.
4. **Load immunity**: 8–16 CPU burners; confirm skips stay ~0 at L=150 (the stall episodes that
   plagued the near-floor servo should vanish inside the margin).
5. **Audio path**: route BMD audio (IS-05 audio receiver + audio sender activation, or env by
   hand). Confirm `audio_rx_pkts`≈`audio_tx_pkts` climbing, late/drop 0, BMD audio output present.
6. **Lip-sync**: clap test / BMD test signal with embedded tone+flash at the monitor. Should be
   exact; trim `SPARK_AV_OFFSET_MS` only if the display chain itself skews.
7. **Erratic source** (BiDirect-1 until its genlock is fixed): expect FRC mode 3 to absorb cadence;
   pipe spikes > L−floor show as brief freezes (skips) — count them/10 min and decide L.
8. **Servo regression check**: `SPARK_LATENCY_MS=0` restores the exact pre-M10 behavior (M8 trim,
   re-anchor, audio off) — the validated fallback stays intact.
