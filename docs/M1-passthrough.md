# M1 — `st2110_rx → st2110_tx` pass-through (first deliverable)

Status: **✅ COMPLETE (2026-06-15). One-process rx→tx frame forwarding, shared EAL, clean at 1080p and
2160p/12G (zero loss, ~2 µs ingest latency, TX `future_err=0`).** Processing operators
(`unpack → resize → frc → pack`) graft between rx and tx next.

Builds on gate 4 (`docs/M1-gate4-pacing.md`): the `st2110_tx`/`st2110_rx` operators + the proven mlx5
`tx_pp` pacing.

## What it is
`engine/apps/st2110_passthrough.cpp`: `st2110_rx → st2110_tx` in **one process**, both CX-7 ports under
**one EAL** (the shared-`DpdkEal` refactor). RX reassembles each ST 2110-20 frame and TX re-transmits
it `tx_pp`-paced. A `MultiThreadScheduler` runs RX (assembling frame N+1) concurrently with TX (pacing
frame N).

## Validation (2-process, 4-port switched fabric)
A generator feeds the forwarder; the forwarder re-transmits to a third port:

```
generator (st2110_tx_smoke, port A 0000:01:00.0) ──▶ port B 0000:01:00.1
        st2110_passthrough:  rx port B ──▶ tx port C 0002:01:00.0 ──▶ port D 0002:01:00.1
```

300 frames each:

| profile | RX in (matched) | TX out | lost | bad | hw_missed | q_dropped | ingest latency avg | TX future_err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1080p | 1 113 300 | 1 113 300 | 0 | 0 | 0 | 0 | ~2.5 µs | 0 |
| 2160p (12G) | 4 448 100 | 4 448 100 | 0 | 0 | 0 | 0 | ~1.9 µs | 0 |

Every received frame was reassembled (`bad=0`) and re-transmitted (`future_err=0`) with zero loss at
line rate.

Reproduce:
```bash
# forwarder (rx port B -> tx port C -> dst port D)
sudo -n SPARK_PROFILE=2160p SPARK_RX_PCI=0000:01:00.1 SPARK_TX_PCI=0002:01:00.0 \
        SPARK_DST_MAC=30:c5:99:3e:9d:30 ./engine/build/st2110_passthrough &
# generator -> port B (start after the forwarder's "st2110_rx started" line)
sudo -n SPARK_PROFILE=2160p SPARK_WARMUP_MS=800 SPARK_TX_PCI=0000:01:00.0 \
        SPARK_DST_MAC=30:c5:99:3e:9d:2c ./engine/build/st2110_tx_smoke
```

## Key design — decouple NIC draining from the emit cadence
`st2110_rx` has two modes: a terminal **sink** (`emit_frames=false`, the gate-4 measuring path) and a
**source** (`emit_frames=true`, used here). The source mode's crux:

A naive "poll the NIC only inside `compute()`, one frame per call" version **blocked on emit while TX
paced the previous frame**, so the NIC RX ring overflowed during the gap → ~10% `hw_missed`, ~10 ms
latency. A deeper ring only traded loss for latency (it just buffers longer). The fix is a **dedicated
poll thread** that drains the NIC continuously into a bounded frame queue; `compute()` merely pops a
finished frame and emits it. NIC draining never stalls → `lost=0`, ~2 µs latency.

(`st2110_rx` always declares a "frame" output — it can't know its mode at `setup()` time — and Holoscan
only ticks an operator whose output is connected, so sink-mode apps wire `st2110_rx → frame_sink`.)

## Co-location prerequisites (from the loopback groundwork)
Two issues were solved first so two busy operators can share a process (see `st2110_loopback.cpp`):
- `DpdkEal` widens thread affinity after `rte_eal_init` (else Holoscan workers inherit the single
  main-lcore mask and serialize on one core → ms-scale preemption).
- one-time TX `warmup_ms` so a co-located/peer RX is polling before TX floods (a startup race).

## Next
1. **Processing operators** between rx and tx: `unpack → resize (NPP) → frc (OFA/FRUC) → pack` (gates
   3/3b already proven on GB10).
2. **PTP discipline** (`ptp4l`/`phc2sys`, grandmaster) for production timing realism.
3. **Per-packet TX↔RX wire-latency** correlation (match TX `send_ts` to RX `hw_ts` by sequence).
4. **`deploy/` provisioning** — `REAL_TIME_CLOCK_ENABLE=1` + hugepages (M1 open item #5).
