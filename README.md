# Spark Video Processor

A modular, low-latency live video processing platform for **NVIDIA Blackwell** GPUs — developed on
the **DGX Spark** (GB10 Grace Blackwell, aarch64) and also running on x86_64 hosts with a discrete
Blackwell card (validated on an RTX PRO 6000 Blackwell). The first module ingests uncompressed
**SMPTE ST 2110-20** video, changes resolution and/or frame rate (motion-compensated) on the GPU,
and emits uncompressed **SMPTE ST 2110-20** — all controllable from a web interface.

> **Status:** Milestone 0 (feasibility spike) **complete** (2026-06-15) — all hardware gates
> validated, Rivermax-free (DPDK mlx5 `tx_pp` pacing on the CX-7). See
> [`docs/M0-feasibility-findings.md`](docs/M0-feasibility-findings.md). Next: **M1**, the
> `st2110_rx → … → st2110_tx` pass-through, which also measures paced-loopback precision/latency (gate 4).

## Architecture

Two strictly separated planes (the real-time video path is never stalled by control/UI traffic):

- **Engine** (`engine/`) — C++/CUDA pipeline on **NVIDIA Holoscan SDK** (operator graph), with the
  Holoscan **`advanced_network` DPDK backend** (mlx5 `tx_pp` HW pacing; DOCA GPUNetIO in reserve) for
  ST 2110-20 IO, **NPP** for scaling, and **Optical Flow/FRUC** for frame-rate conversion. Zero-copy
  GPU tensors, PTP-locked. *(No Rivermax — see `docs/M0-feasibility-findings.md` "Networking IO".)*
- **Control daemon** (`control/`) — native C++ gRPC service owning pipeline lifecycle + telemetry,
  with a REST/WebSocket gateway.
- **Web UI** (`web/`) — vanilla-JS dashboard (no build step): NMOS routing, format config, a
  filter-chain panel rendered from the daemon's `/api/filters` descriptors (enable/reorder/tune
  each GPU stage), start/stop, and a live stats bar.

```
engine/    C++/CUDA Holoscan engine (operators = modules)
control/   C++ control daemon (gRPC + REST/WS)
web/       vanilla-JS dashboard (served by the daemon)
proto/     shared gRPC/protobuf contracts
deploy/    systemd units, container, NIC/PTP/core tuning
spike/     M0 validation scripts
docs/      design + findings
```

## Supported hardware

The engine targets **NVIDIA Blackwell + a ConnectX NIC** and builds from one tree on either
architecture. Nothing in the code is arch-specific: memory coherence, NPP stream context, SM clock
limits and TX pacing capability are all probed at runtime.

| | DGX Spark (primary) | x86_64 workstation |
|---|---|---|
| CPU / arch | GB10 Grace Blackwell, aarch64 | any x86_64, amd64 |
| GPU | GB10, sm_121 | RTX PRO 6000 Blackwell / GB20x, **sm_120** |
| Host↔GPU memory | 128 GB LPDDR5x **unified** (coherent) | discrete, over PCIe |
| ST 2110 + PTP NIC | integrated **ConnectX-7**, 2× QSFP hot-plug | any ConnectX with a PHC |

`CMAKE_CUDA_ARCHITECTURES` defaults to `120;121` — one fat binary runs on both. Narrow it with
`-DSPARK_CUDA_ARCHS=120` (or `=native`) for a faster build.

**Unified vs discrete memory** is the one real behavioural difference, and it is handled at
runtime: the unpack/pack operators probe `pageableMemoryAccess` and use zero-copy host access on
GB10, falling back to explicit H2D/D2H staging on a discrete GPU (`SPARK_ZEROCOPY` forces either
path — see `docs/M8-latency.md`). Correctness is identical; the discrete path adds one PCIe round
trip per frame to the latency budget.

**TX pacing** needs a NIC that advertises DPDK's `SEND_ON_TIMESTAMP` offload — i.e. a ConnectX with
`REAL_TIME_CLOCK_ENABLE=1` in firmware (`deploy/provision.sh` sets this; it needs a cold reboot).
Without it the pipeline still runs, unpaced, which is fine for development but is not ST 2110-21
compliant. Note the DGX Spark's onboard Realtek 10GbE has **no hardware PTP** and cannot be used for
2110 timing.

See the full plan and milestones in `docs/`.

## Quick diagnostics
```bash
bash spike/diagnose.sh        # read-only host / GPU / NIC / PTP inventory
bash spike/build_probes.sh    # build + run the OFA optical-flow and NPP resize gates (no sudo)
```
