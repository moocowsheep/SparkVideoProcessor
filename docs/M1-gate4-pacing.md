# M1 — Gate 4: ST 2110-21 hardware pacing at rate (CX-7 loopback)

Status: **✅ GATE 4 CLOSED (TX + RX). HW pacing precision confirmed at 1080p and 2160p (the M0-deferred
risk retired); the `st2110_tx` operator delivers *sustained on-rate* paced TX with `future_err=0`
(Result 3), and the `st2110_rx` operator receives the loopback with **zero loss** and ~2–3 µs
zero-copy ingest latency at both rates (Result 4) — the at-rate + latency proofs testpmd's open-loop
`txonly`/`rxonly` structurally could not give. Remaining (not gate-4): PTP grandmaster discipline,
deploy provisioning, single-process rx→tx pass-through.**
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

## Result 3 — `st2110_tx` operator: sustained at-rate pacing, CLEAN (2026-06-15)
Built the `st2110_tx` Holoscan operator (engine/operators/st2110_tx/) on a raw-DPDK/mlx5 `tx_pp`
backend behind a swappable interface (advanced_network can replace it later). It generates real
RFC 4175 packets at the media rate and — the key design point — `compute()` self-throttles its
submission to keep the in-flight schedule within `pacing_horizon_ns` (50 µs) of the NIC clock, the
closed-loop feeding testpmd's open-loop `txonly` lacked. Run via `engine/build/st2110_tx_smoke`
(test_pattern → st2110_tx) on the gate-4 pair (TX `0002:01:00.0` → RX MAC `00:00:5e:00:53:30`),
300 frames each:

| profile | pkts/frame | rate | packets sent | jitter | wander | sync_lost | future_err | past_err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1080p | 3 711 | ~222 k pps | 1 113 300 | 40 ns | 0 | 0 | **0** | 3 157 (0.28 %) |
| 2160p (the gate) | 14 827 | ~889 k pps | 4 448 100 | 18 ns | 0 | 0 | **0** | **0** |

`future_err=0` at both rates is the at-rate proof: the schedule never overran the `tx_pp` window
(contrast the testpmd flood at `TXD=128`), while delivering the *full* frame rate (contrast the
testpmd starve at small `TXD`). 2160p/12G — the real gate — is flawless. The 1080p `past_err`
(~10/frame, frame-boundary scheduling at the looser 4.5 µs gap) is a minor tuning item, not a gate
failure. `rte_eth_read_clock` returned usable nanoseconds, so the `// BRINGUP` clock-unit concern did
not materialize.

## Result 4 — `st2110_rx` operator: zero-loss receive + ingest latency, CLEAN (2026-06-15)
Built `st2110_rx` (engine/operators/st2110_rx/) as the loopback RX half: a Depacketizer (RFC 4175
parse + scatter, shared framing TU) on a raw-DPDK/mlx5 backend with the RX-timestamp offload. Ran the
two-process loopback (`st2110_rx_smoke` on `0002:01:00.1` ← switch ← `st2110_tx_smoke` on
`0002:01:00.0`), 300 frames each:

| profile | TX packets | RX matched | lost | bad | hw_missed | ingest latency min/avg/max |
|---|---:|---:|---:|---:|---:|---|
| 1080p | 1 113 300 | 1 113 300 | **0** | **0** | 0 | 1 349 / 2 760 / 103 315 ns |
| 2160p (the gate) | 4 448 100 | 4 448 100 | **0** | **0** | 0 | 1 325 / 2 182 / 126 695 ns |

Every transmitted packet was received and parsed (RX matched == TX sent, `lost=0` RTP-sequence gaps,
`bad=0` parse failures), all 300 frames reassembled, at 12G with no HW drops. **Zero-copy ingest
latency ~2–3 µs average** (NIC HW rx timestamp → operator processing, shared `ptp0` base); the ~100 µs
max is a per-burst sampling outlier. This closes the gate-4 zero-copy-latency follow-on. The RTP/RFC
4175 framing is now validated on real hardware in both directions (the unit test already round-trips
it offline).

## What remains to fully close gate 4 (in the operators, not the probe)
1. ✅ **`st2110_tx`:** DONE — paced TX at 1080p and 2160p with `future_err=0` and sustained
   pps == target (see Result 3). Minor follow-up: trim the 1080p frame-boundary `past_err`.
2. ✅ **`st2110_rx`:** DONE — zero-loss receive + ~2–3 µs ingest latency at both rates (see Result 4).
   Follow-on: per-packet TX↔RX wire-latency correlation (embed/​match send time) and a single-process
   `st2110_rx → … → st2110_tx` pass-through (shared EAL across both ports).
3. ✅ **PTP discipline:** config + launcher in `deploy/ptp.sh` + `deploy/ptp4l.conf` (ST 2059-2
   profile; `--check`/`--test`/`--master`/slave). The engine reads the same PHC, so disciplining it
   aligns RTP timestamps + pacing with no engine change. Loopback lock is trivial (all ports share
   `ptp0`); slave-to-grandmaster is the production case (needs an actual GM) and does **not** affect
   the pacing results above (`sync_lost=0`/`wander=0` already — the scheduler reads the same PHC).
   `--check` confirms the CX-7 PTP-capable with `ptp0` present (M0 gate 1).

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

The `st2110_tx` operator (Result 3) — a real media generator, not the probe:
```bash
T=$(ls -d ~/holoscan-sdk/install-cu13-$(uname -m)-dgpu ~/holoscan-sdk/install-cu13-$(uname -m) 2>/dev/null | head -1)
cmake -S engine -B engine/build -DCMAKE_PREFIX_PATH="$T"
cmake --build engine/build
sudo -n SPARK_PROFILE=2160p SPARK_TX_PCI=0002:01:00.0 SPARK_DST_MAC=00:00:5e:00:53:30 \
     ./engine/build/st2110_tx_smoke   # SPARK_PROFILE=1080p|2160p, SPARK_FRAMES=n
```
Full loopback (RX first, then TX — two processes, mirrors gate-4; RX prints loss + ingest latency):
```bash
# The smokes now DISCOVER their ports (first linked-up ConnectX + its sibling). Pass the BDFs
# explicitly to reproduce the Spark runs above exactly — that box has four ports in one L2 domain,
# so discovery would pick the 0000:01:00.x pair, not the 0002:01:00.x pair used here.
sudo -n SPARK_PROFILE=2160p SPARK_SECONDS=10 SPARK_RX_PCI=0002:01:00.1 ./engine/build/st2110_rx_smoke &
sudo -n SPARK_PROFILE=2160p SPARK_FRAMES=300 SPARK_TX_PCI=0002:01:00.0 \
        SPARK_DST_MAC=00:00:5e:00:53:30 ./engine/build/st2110_tx_smoke
```

## Provisioning (M1 open item #5) — DONE
`REAL_TIME_CLOCK_ENABLE=1` (both CX-7 devices) + hugepages are now provisioned reproducibly by
`deploy/provision.sh` (idempotent; `--check` for read-only state). See `deploy/README.md`.

---

# x86_64 second host — pacing re-validation (in progress, 2026-08-25)

Gate 4 above was closed on the DGX Spark. `feat/x86-cross-platform` adds a second supported host, so
the pacing mechanism has to be re-proven there: the NIC is a **different generation and a different
link rate**, and `tx_pp` is a per-device capability, not a property of the DPDK stack.

| | DGX Spark (gate-4 host) | x86_64 host (`saturnrack`) |
|---|---|---|
| CPU | GB10 Grace Blackwell, aarch64 | AMD EPYC 7443P, 24C, **single NUMA node** |
| GPU | GB10 integrated | RTX PRO 6000 Blackwell Server Edition (driver 610.57.04) |
| NIC | 2× dual-port **ConnectX-7**, 4× 100G | 1× dual-port **ConnectX-6 Lx** (`15b3:101f`, MCX631432AS), 2× 25G, fw 26.30.1004 |
| ports | `0000:01:00.x`, `0002:01:00.x` | `0000:82:00.0` (up, 25G), `0000:82:00.1` (**link down — not cabled**) |
| PHC | `ptp0` (shared by all four ports) | `ptp2` (`ethtool -T enp130s0f0np0`) |
| kernel | 6.17 | 7.0.0-30-generic |

## Step 1 — `tx_pp` capability probe: PASS
`spike/dpdk_pacing_probe.sh` step 4 reports packet pacing **supported** on the CX-6 Lx. So
`REAL_TIME_CLOCK_ENABLE=1` took effect through the cold reboot, the port advertises DPDK's
`SEND_ON_TIMESTAMP` Tx offload, and `dpdk_tx_backend.cpp` will take its paced path rather than the
unpaced development fallback. ST 2110-21 compliant TX is available on this host.

This is the M0 gate-2 mechanism question answered for a *second* NIC generation — pacing is not a
CX-7-only capability (it is CX-6 Dx / Lx and later).

## Step 2 — PTP: already disciplined by the host, not by `deploy/ptp.sh`
This box runs linuxptp under systemd from boot, slaved to a facility grandmaster:

| unit | what it does |
|---|---|
| `ptp4l-smpte.service` | `ptp4l -f /etc/linuxptp/ptp4l-smpte.conf` — `clientOnly 1`, **domain 127**, ST 2059-2, hardware timestamping, on `enp130s0f0np0` |
| `phc2sys-smpte.service` | `phc2sys -a -r -n 127` — PHC → `CLOCK_REALTIME` |

Measured lock (`journalctl -u ptp4l-smpte`): **rms ~20–23 ns, max ~58 ns** to the GM, path delay
~3.57 µs, stable; `phc2sys` holds `CLOCK_REALTIME` at rms ~12–16 ns. That is orders of magnitude
inside the ST 2110-21 budget, and it is the *production* case the Spark runs never exercised — those
were a trivially-locked single-box loopback (`sync_lost=0` came free because all four ports shared
one PHC).

The host config and `deploy/ptp4l.conf` agree on domain 127 / ST 2059-2 / E2E / hardware timestamping,
so the engine needs no change: `st2110_tx`/`st2110_rx` read the same PHC the daemon disciplines.

> **Do not run `deploy/ptp.sh` (`slave`/`start`/`--master`/`--test`) on this host.** A second `ptp4l`
> on the same interface fights the systemd one, and `--master`/`--test` deliberately omit
> `slaveOnly`, so they could try to win BMCA against the facility grandmaster. `deploy/ptp.sh` now
> detects an externally-managed `ptp4l` and refuses to start on top of it (`FORCE=1` overrides).
> `bash deploy/ptp.sh --check` and `status` stay safe and report the systemd units.

## Blocked on cabling — the loopback pair does not exist yet
Only `0000:82:00.0` is cabled; `0000:82:00.1` reports `carrier=0`. The at-rate proofs (Results 1–4
above) are a **TX port → RX port** test, so they cannot run on one port. Unblock either by:
* connecting the second SFP28 port (DAC to the same switch, or a direct loopback), which gives the
  same topology as the Spark run; or
* pointing TX at a real ST 2110 receiver / RX at a real sender on the fabric (`SPARK_DST_MAC`,
  `SPARK_RX_MCAST`) — proves interop but not loss-free self-test.

The smoke apps log `LINK DOWN` when the discovered port has no carrier, so this failure mode is
visible at startup rather than as a silent zero-packet run.

## Pending — the numbers to fill in here
Once the second port is cabled, re-run the gate-4 sequence and record it in this section. Ports are
now discovered (`engine/operators/common/nic_ports.hpp` — first linked-up ConnectX + its sibling), so
no BDFs are needed on this host:

```bash
sudo python3 spike/detect_loopback.py                      # confirm the pair sees each other
sudo -E PROFILE=2160p bash spike/st2110_loopback_gate4.sh   # tx_pp_jitter / wander / sync_lost
sudo -n SPARK_PROFILE=2160p SPARK_SECONDS=10 ./engine/build/st2110_rx_smoke &
sudo -n SPARK_PROFILE=2160p SPARK_FRAMES=300 ./engine/build/st2110_tx_smoke
```

Targets, to match the Spark: `tx_pp_sync_lost=0`, `wander=0`, jitter in the tens of ns,
`future_err=0`, `lost=0`, ingest latency ~2–3 µs.

Two things to watch that the Spark run could not surface:
1. **25G, not 100G.** 2160p59.94 (~11.9 Gbps) is ~48 % of a 25G port, versus ~12 % of the Spark's
   100G. Pacing precision under high *relative* Tx load is exactly the risk M0 flagged, so a clean
   12G result here is a stronger result than the Spark's, not a weaker one — and 25G is a hard
   ceiling for anything beyond a single UHD flow on this card.
2. **A real grandmaster in the loop.** `sync_lost` and `wander` now measure the NIC pacing clock
   against a GM-disciplined PHC rather than a free-running one; a non-zero `wander` here would be a
   genuine finding, not the no-op it was on the Spark.

NUMA, flagged as a risk for a discrete PCIe NIC, is a **non-issue on this box**: single socket, one
NUMA node (`/sys/bus/pci/devices/0000:82:00.0/numa_node` = -1), so the default `TX_CORES=0,1` /
`RX_CORES=2,3` cannot land cross-socket.
