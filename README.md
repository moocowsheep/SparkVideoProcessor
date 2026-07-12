# Spark Video Processor

A modular, low-latency live video processing platform for the **NVIDIA DGX Spark** (GB10 Grace
Blackwell). The first module ingests uncompressed **SMPTE ST 2110-20** video, changes resolution
and/or frame rate (motion-compensated) on the GPU, and emits uncompressed **SMPTE ST 2110-20** — all
controllable from a web interface.

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

## Hardware reality (DGX Spark)
- GB10 Grace Blackwell, aarch64, 128 GB LPDDR5x **unified** memory, CUDA 13, NPP + OFA present.
- Integrated **ConnectX-7** (2× QSFP, hot-plug, treat as 2× 100G) is the SMPTE 2110 + PTP NIC.
- Onboard Realtek 10GbE has **no hardware PTP** — not usable for 2110 timing.

See the full plan and milestones in `docs/`.

## Quick diagnostics
```bash
bash spike/diagnose.sh
```
