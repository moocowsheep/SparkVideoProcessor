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

## Wire codecs
The input and output legs each pick their own wire format, independently — so the engine can ingest
one codec and emit another, or transcode both ways:

| | uncompressed | Blackmagic IP10 (10:8) | JPEG XS |
|---|---|---|---|
| transport | ST 2110-20 (RFC 4175) | ST 2110-22 | ST 2110-22 (RFC 9134) |
| GUI | default | `IP10 output` / `source is IP10` | `JPEG XS output` / `source is JPEG XS` |
| env | — | `SPARK_IP10` / `SPARK_IN_IP10` | `SPARK_JXS` / `SPARK_IN_JXS` |

JPEG XS is GPU-encoded/decoded by **MooCUDAJXS** (a separate repository) and is **optional**: point
CMake at a checkout with `-DSPARK_JXS_DIR=/path/to/MooCUDAJXS` (a sibling checkout beside this repo
is found automatically). Without it the engine builds and runs unchanged, and only a run that
actually asks for JPEG XS fails — with a message saying how to enable it. See
[`docs/M11-jpegxs.md`](docs/M11-jpegxs.md).

> **JPEG XS licensing.** JPEG XS (ISO/IEC 21122) is a standardized codec that may be covered by
> patents. **You are solely responsible for determining if your use of jpeg-xs requires any
> additional licenses. The developer of this project is not responsible for obtaining any such
> licenses, nor liable for any licensing fees due, in connection with your use of jpeg-xs.**
> See [`NOTICE`](NOTICE).

See the full plan and milestones in `docs/`.

## Quick diagnostics
```bash
bash spike/diagnose.sh
```
