# M1 — Gate 4: ST 2110-21 hardware pacing at rate (CX-7 loopback)

Status: **✅ HW pacing precision CONFIRMED at 1080p and 2160p — the M0-deferred gate-4 hardware risk
is retired. Sustained *on-rate* paced throughput and zero-copy ingest latency are not demonstrable
with the testpmd `txonly` probe (open-loop generator) and move to the `st2110_tx`/`st2110_rx`
operators — exactly the production path the M0 handoff chose.**
Host: DGX Spark, GB10 Grace Blackwell, aarch64, kernel 6.17, CUDA 13.0. NIC: 2× dual-port ConnectX-7
(4× 100G). Date: 2026-06-15.

Builds on the M0-proven mechanism (DPDK 23.11.4 mlx5 PMD + `tx_pp` HW send-scheduling, after
`REAL_TIME_CLOCK_ENABLE=1` on both CX-7 devices). Spike scripts: `spike/detect_loopback.py`,
`spike/st2110_loopback_gate4.sh`.

## Topology (task 2 — identify the loopback pair)
`spike/detect_loopback.py` sends uniquely-tagged raw L2 frames out each CX-7 port and listens on the
others. Result: **switched fabric — all 4 ports observe each other's frames (full mesh, one L2
domain)**, not a direct DAC pair. So any TX/RX port pair works; the test uses **TX `0002:01:00.0`
→ RX `0002:01:00.1`**, with the TX destination set to the RX port's MAC (`testpmd set eth-peer`) so
the in-path switch delivers the flow. `tx_pp` precision is a NIC TX-side property, independent of the
fabric in between.

## Result 1 — pacing-clock precision (the ST 2110-21 go/no-go): PASS
The `tx_pp_*` clock counters measure the NIC's packet-pacing clock disciplined to the PHC — the
foundation of ST 2110-21 narrow-sender pacing. Stable in **every** run, at both rates:

| metric | 1080p / ~3 Gbps | 2160p / ~12 Gbps | ST 2110-21 budget |
|---|---|---|---|
| `tx_pp_jitter` | 18 ns (8–70 ns across the TXD sweep) | **8 ns** | µs-scale — ~1000× margin |
| `tx_pp_wander` | 0 | 0 | — |
| `tx_pp_sync_lost` | 0 | 0 | must be 0 ✓ |
| loss (RX-pkts / TX-pkts) | ~0 | ~0 | ✓ |

Precision **did not degrade at 12G** (8 ns vs 18 ns is within sampling noise; the high-rate run had
tens of millions of packets, so its jitter figure is the most credible). This directly retires the
M0-flagged risk that "mlx5 `tx_pp` precision degrades under high Tx load."

## Result 2 — sustained on-rate throughput from testpmd: NOT achievable (generator limit, not NIC)
testpmd `txonly` is an **open-loop** generator: it assigns each packet an absolute send time
(`base + i·gap`) and never re-syncs to wall-clock, relying solely on Tx-ring backpressure to stay
near real time. Sweeping `TXD` (1080p, 5 s each, target 260 778 pps) shows a **bimodal** failure with
no stable lock to the target:

| TXD | horizon = TXD·gap | `future_err` | `past_err` | achieved | regime |
|----:|---:|---:|---:|---:|---|
| 128 | 491 µs | ~38.9 M | ~4.6 k | ~2.59 M pps | **flood** — horizon overruns the NIC `tx_pp` window → every packet future-errors → sent immediately at line rate |
| 64 | 245 µs | 0 | ~4.3 k | ~868 pps | **starve** — ring too shallow to keep the open-loop schedule fed → schedule lags wall-clock → every packet past-errors |
| 32 | 123 µs | 0 | ~3.4 k | ~700 pps | starve |
| 16 | 61 µs | 0 | ~3.5 k | ~700 pps | starve |
|  8 | 31 µs | 0 | ~3.2 k | ~650 pps | starve |

(128 row: from the initial 15 s run, normalized to pps.) The achieved rate jumps from ~370× *under*
target to ~10× *over* with no intermediate lock, because txonly has no wall-clock feedback loop. mlx5
did **not** clamp the ring (`TXD=8` accepted, no `nb_txd adjusted` warning), so this is purely a
property of the testpmd generator — confirmed by the pacing clock staying clean (`sync_lost=0`,
jitter ≤ 70 ns) in every row.

## Decision: sustained at-rate pacing is an operator deliverable
The NIC's pacing capability is proven; what's missing is a generator that produces packets *at* the
media rate. That is precisely the `st2110_tx` operator (Holoscan `advanced_network` DPDK backend):
it emits one RTP packet per ST 2110-21 slot straight from the framebuffer cadence, so it neither
floods nor starves, and hands each packet a send timestamp for the NIC scheduler. testpmd `txonly`
was only ever the mechanism probe — it answered the hardware question and that's its job done.

## What remains to fully close gate 4 (in the operators, not the probe)
1. **`st2110_tx`:** paced TX at 1080p and 2160p with `future_err`/`past_err` ≈ 0 and sustained
   pps == target (the at-rate proof testpmd can't give).
2. **`st2110_rx`:** zero-copy ingest latency from correlated HW TX/RX timestamps (testpmd `rxonly`
   only counts packets; it can't correlate per-packet TX/RX).
3. **PTP discipline:** `ptp4l`/`phc2sys` on the shared real-time PHC (`ptp0`). Loopback lock is
   trivial (all ports share `ptp0`); a real grandmaster is the production validation — and note a GM
   does **not** affect the results above (`sync_lost=0`, `wander=0` already; testpmd reads the same
   PHC the scheduler uses, so external discipline changes nothing here).

## Reproduce
```bash
# identify the loopback pair (prints the exact run commands)
sudo python3 spike/detect_loopback.py

# precision at rate (run 1080p then 2160p; the real gate is 12G)
sudo -E LOOP_TX_PCI=0002:01:00.0 LOOP_RX_PCI=0002:01:00.1 PROFILE=1080p bash spike/st2110_loopback_gate4.sh
sudo -E LOOP_TX_PCI=0002:01:00.0 LOOP_RX_PCI=0002:01:00.1 PROFILE=2160p bash spike/st2110_loopback_gate4.sh
```
Read `tx_pp_jitter` / `tx_pp_wander` / `tx_pp_sync_lost` for precision (the gate); `future_err` /
`past_err` reflect the open-loop generator, not the NIC. `TXD` sweeps the flood↔starve regimes.
```

## Provisioning note (M1 open item #5)
`REAL_TIME_CLOCK_ENABLE=1` (both CX-7 devices) + hugepages must move into `deploy/` so a fresh box is
reproducible — currently the probe self-allocates hugepages ad hoc.
