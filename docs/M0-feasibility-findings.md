# M0 — Feasibility Spike Findings

Status: **IN PROGRESS — GPU gates + Holoscan C++ build/engine skeleton CONFIRMED; networking gates blocked on ConnectX-7 cabling (user action).**
Host: DGX Spark `gx10-9d2a`, GB10 Grace Blackwell, aarch64, kernel 6.17, driver 580.159.03, CUDA 13.0.
Date started: 2026-06-15.

The spike validates four hardware unknowns before any module is built. Two are answered favorably
from read-only diagnostics; the two networking gates are **blocked** because the ConnectX-7 is not
currently on the bus.

## Gate results so far

| # | Gate | Status | Evidence |
|---|------|--------|----------|
| 1 | PTP hardware timestamping (ST 2110 timing) | ❌ **BLOCKED / not available on any active NIC** | CX-7 not enumerated; Realtek 10GbE has no HW PTP |
| 2 | ST 2110-20 RX+TX binds CX-7 (Rivermax-free: DPDK mlx5 + `tx_pp` HW pacing) | ⏸ **BLOCKED** — CX-7 not enumerated; spike staged (`spike/dpdk_pacing_probe.sh`) | See "Networking IO" decision below |
| 3 | OFA / FRUC for motion-comp FRC on GB10 | ✅ **CONFIRMED** | `spike/nvof_probe` ran HW optical flow on GB10; accurate flow `(8.00, 0.00)px` for an 8px shift; driver **OF API 5.0** (full FRUC); dims up to 8192×8192 |
| 3b | NPP resize path (M2) | ✅ **CONFIRMED** | `spike/npp_resize_test` 1080p→2160p 16u: LINEAR 0.09 / CUBIC 0.11 / LANCZOS 0.23 ms/frame |
| 4 | Zero-copy ingest latency | ⏸ Pending CX-7 up + DPDK/GPUNetIO path | — |

### Validated probe results (2026-06-15)
```
[nvof] CUDA device: NVIDIA GB10
[nvof] driver max OF API version: 5.0 (header is 2.0)
[nvof] caps: width 32..8192, height 32..8192
[nvof] EXECUTE ok. center flow (grid 32,32) = (8.00, 0.00) px  [expected ~|x|=8]
[nvof] PASS — NVOFA hardware optical flow works on this GB10.
[npp] resize 1920x1080 -> 3840x2160, 16u C1, 200 iters
[npp]   LINEAR 0.092  CUBIC 0.109  LANCZOS 0.230  ms/frame
```
Reproduce: `bash spike/build_probes.sh` (no sudo; uses on-system `libnvidia-opticalflow.so` + NPP + public NVOF headers in `spike/third_party/`). At 60 fps the per-frame budget is 16.7 ms, so resize + optical flow leave large headroom.

### Holoscan: runtime CONFIRMED, but conda-forge C++ build path is blocked on CUDA 13
- **Runtime works on GB10 ✅** — installed Miniforge + a CUDA-13 conda env (`moo`) with `libholoscan-dev`
  4.3.0 (cuda-version 13.3). The shipped `examples/ping_simple` ran the GXF executor and printed Rx
  1–10 (exit 0) once `librmm.so` was on the path. Holoscan headers report 4.3.0.
- **C++ *build* against conda-forge is blocked ❌.** `holoscan-config.cmake` calls
  `find_dependency()` on `nvtx3`, `rmm`, `rapids_logger`, `MATX` (lines 164–171). These are **not
  packaged on conda-forge** (no `rapids_logger`/`matx`/`nvtx3-c`; only ancient `rmm 22.08`). Installing
  conda-forge `librmm` also force-downgrades the whole env to CUDA 12.9 (librmm is CUDA-≤12 capped) and
  breaks the rmm/CCCL headers. So `find_package(holoscan)` cannot be satisfied from conda-forge here.
- pip `holoscan` = **`holoscan-cu12` 4.3.0** (CUDA-12, aarch64). Self-contained wheel (likely bundles
  the 3rdparty CMake configs), runs on the 580 driver via forward-compat, but is CUDA-12.
- Engine skeleton (`engine/`, mirrors the canonical Holoscan app) — **BUILDS + RUNS natively ✅**
  (2026-06-15). The source-built SDK (`install-cu13-aarch64-dgpu/`) is consumed from the host via
  `find_package(holoscan)`; the placeholder ping graph ran the GXF executor and printed Rx 1–10
  (exit 0). Host gcc 13.3 matches the Ubuntu-24.04 build container, and the install tree **bundles
  every dep that broke conda-forge** (rmm 26.02, nvtx3, rapids_logger, MATX, Eigen3 3.4, CCCL 3.2) —
  so the "not actively tested" native-consumption path just works here.
- **Net:** the Holoscan **C++ build environment** needs a path other than conda-forge. Decision below.

## Networking IO: Rivermax → open DPDK path (decision 2026-06-15)
**Decision:** Drop the Rivermax dependency. Do ST 2110-20 RX/TX over the Holoscan
**`advanced_network` operator's DPDK backend**, which drives the CX-7 directly via the **mlx5 PMD**
and uses the NIC's hardware accurate-send-scheduling (`tx_pp`/`tx_skew`) for ST 2110-21 pacing.
**DOCA GPUNetIO** is held in reserve as the GPUDirect-to-GPU option. No license required; the IO
plane stays open-source and aligned with the already-chosen Holoscan framework (whose `advanced_network`
DPDK backend is the only one NVIDIA documents as "actively tested").

- **Why not Rivermax:** turnkey and the ST 2110-21 compliance reference, but it needs a license
  (free 3-month eval, then a cost/bundling question) and ties the real-time IO plane to a proprietary
  SDK. The hard part it sells — narrow-sender TX pacing — is a *hardware* feature of the CX-7
  reachable without it.
- **Why not Intel MTL** (evaluated and rejected): complete BSD ST 2110 stack, but its hardware
  **narrow** pacing is **E810-only**; on a Mellanox NIC it falls back to **TSC software wide pacing**
  and does *not* use the CX-7's hardware scheduler. Mellanox + aarch64 are explicitly "not
  guaranteed," and the Spark has no PCIe slot to add an E810 — wrong NIC for that library.
- **What we now own:** the ST 2110-20 RTP framing + ST 2110-21 pacing implementation/validation that
  Rivermax would have provided. The risk to retire in the spike: mlx5 `tx_pp` precision is documented
  to **degrade under high Tx load** — must be verified at 2160p/12G rates, not just 1080p/3G.
- **Bonus:** the mlx5 PMD is **bifurcated** (runs alongside the kernel driver, no `vfio-pci` unbind),
  so the CX-7 stays usable by the host during probing.
- **PTP is unaffected** — gate 1 is `linuxptp` + the CX-7 PHC either way; dropping Rivermax changes
  nothing there.

### Gate 2 spike (Rivermax-free), staged so each step runs as the NIC comes up
1. **DPDK + mlx5 present** — `spike/dpdk_pacing_probe.sh` checks for `dpdk-testpmd`/`libdpdk` and the
   mlx5 PMD. *(Runs now; reports what's missing.)*
2. **CX-7 binds under DPDK** — once cabled, testpmd enumerates the `15b3` port via mlx5 (no unbind).
3. **HW send-scheduling capability** — bring the port up with the `tx_pp=<ns>` devarg; mlx5 fails
   probe cleanly if the NIC can't pace → this is the ST 2110-21 go/no-go.
4. **Paced loopback at rate** — generate ST 2110-20-rate UDP/RTP out one QSFP port, receive on the
   other (loopback topology); confirm pacing holds at 1080p (~3 Gbps) then 2160p (~12 Gbps). This is
   the real gate-2/4 validation, and the M1 `st2110_rx`/`st2110_tx` operators grow out of it.

## Detailed findings

### Networking — the blocker
- **ConnectX-7 is hot-plug and currently NOT on the PCI bus.**
  - `lspci -nnD | grep 15b3` → **no Mellanox device**.
  - `mlx5_core`, `mlx5_ib`, `ib_core`, `rdma_ucm` kernel modules **are loaded** (waiting for hotplug).
  - `/sys/class/infiniband` is **empty**; `rdma link show` returns nothing.
  - Hotplug handler: `/opt/nvidia/dgx-spark-mlnx-hotplug/mtk-hotplug-handler.sh`.
  - Interpretation: the CX-7 only enumerates when a **QSFP cable is connected** to a rear QSFP port.
    Nothing is currently plugged in.
- **Only active wired NIC: Realtek 10GbE** `enP7s7` (driver `r8127`), IP 192.0.2.224/24.
  - `ethtool -T enP7s7` → `PTP Hardware Clock: none`, `Hardware Transmit/Receive Timestamp Modes: none`.
  - **No `/dev/ptp*` devices on the system at all.**
- Other interfaces: `wlP9s9` (Wi-Fi 7, MediaTek MT7925, down), `tailscale0`, `docker0`.

**Consequence:** SMPTE ST 2110 requires PTP-disciplined timing (ST 2059 / IEEE 1588) and a NIC with
hardware timestamping + accurate send-scheduling. The 10GbE Realtek port has **neither**. The
CX-7 — which has both (HW PTP + mlx5 `tx_pp` pacing) — must be **cabled** (to a 2110 switch/source,
or a loopback to a second QSFP port / second device) before gates 1, 2, and 4 can be validated.

### GPU / compute — ready
- **OFA (Optical Flow Accelerator) present:** `nvidia-smi -q` lists an `OFA` utilization engine;
  `/lib/aarch64-linux-gnu/libnvidia-opticalflow.so(.1)` present. → motion-compensated FRC (v1 choice)
  is viable. Full confirmation = initialize the Optical Flow SDK / FRUC (`NvOFFRUC`) once installed.
- **NPP fully installed:** `libnpp-13-0` + `libnpp-dev-13-0` (13.0.1.2); all `nppi*` libs under
  `/usr/local/cuda/targets/sbsa-linux/lib`. → resize path primitive ready.
- **Media engines:** GB10 exposes NVENC, NVDEC, JPEG, OFA per `nvidia-smi`.
- **Toolchain:** CUDA 13.0 (`nvcc` V13.0.88), gcc/g++ 13.3, cmake 3.28, make, git present.
  Missing (install later): `ninja`, `node/npm/pnpm` (web UI), `protoc` (gRPC).

### Not yet installed (deferred until CX-7 is up)
- ~~NVIDIA **Rivermax** SDK~~ — **dropped** (see "Networking IO" decision). Replaced by DPDK via the
  Holoscan `advanced_network` operator; no license needed.
- NVIDIA **Holoscan** SDK (incl. the `advanced_network` DPDK backend / holohub).
- DOCA (for the GPUNetIO reserve path) / MLNX firmware tools (`mst`, `mlxconfig`) — `/dev/mst` absent.

## Plan decisions (locked)
- Test topology: **QSFP loopback self-test** (cable CX-7 port↔port; software PTP + a DPDK ST 2110-20
  generator on-box). Note: two ports on one CX-7 typically share a PHC, so loopback PTP "lock" is
  trivially perfect — it validates the *mechanism* (HW timestamping + mlx5 RX/TX with `tx_pp` pacing),
  not grandmaster discipline.
- Networking IO: **Rivermax dropped in favour of the Holoscan `advanced_network` DPDK backend**
  (DOCA GPUNetIO in reserve). See the "Networking IO" decision above.

## Still blocking gates 1, 2, 4 (needs user action)
1. **Cable the ConnectX-7** QSFP port↔port (QSFP56 DAC or loopback module) so it enumerates.
2. **Build the Holoscan `advanced_network` operator (DPDK backend)** from holohub — no license needed
   (replaces the former Rivermax license step). DPDK + mlx5 PMD come with it.
3. **One `sudo apt` install** for C++/web build tooling (no passwordless sudo here) — see below.
4. **Docker group access** — `./run build` (the source-build path, below) uses Docker; user is not in
   the `docker` group, so it currently needs sudo.

## Holoscan: chosen path = build from source (native, CUDA 13)
- Decision: **build Holoscan SDK v4.3.0 from source** (rapids-cmake/CPM auto-fetches the nvtx3 / rmm /
  rapids_logger / MATX that conda-forge lacked). The official build runs inside a **Docker
  build-container** (`./run build`) and emits a self-contained `install-cu13-aarch64-dgpu/` tree we
  then consume **natively** from the host (point `engine/` `find_package(holoscan)` at it). Best of
  both: reproducible build, native deployment, full CUDA-13.
- **GPU stack RESOLVED: `dgpu` + `cuda 13`** (not `igpu`, despite unified memory — the M0 guess was
  wrong). In Holoscan's taxonomy `igpu` means the **L4T/Tegra** stack, which v4.3.0 only ships for
  **CUDA 12** (`arm64-igpu_cu12_base`, from `l4t-cuda:12.6.11`); there is **no `igpu`+`cu13` base
  image**, so `./run build --gpu igpu --cuda 13` fails instantly at `docker build` (tries to pull a
  nonexistent `arm64-igpu_cu13_base`). The DGX Spark is **sbsa/server-arm, not Tegra** — evidence:
  `nvidia-smi` has no `nvgpu` marker, CUDA libs live under `targets/sbsa-linux/`, and there is no
  `/etc/nv_tegra_release`. Correct invocation: **`./run build --gpu dgpu --cuda 13`** (native arm64;
  base `nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04`). The `--arch` flag takes `arm64` (not `aarch64`)
  but defaults to the host arch, so it can be omitted. (The fully-local non-Docker CMake build is
  documented as "not actively tested".)
- Prereqs present: Docker 29.2.1, buildx 0.31.1, NVIDIA Container Toolkit 1.19.1. Blocker: **docker
  group access** (above). See `docs/SETUP.md` for exact steps.
- The CUDA-13 conda env `moo` (Holoscan **runtime** + Python) is kept as-is — useful for quick checks;
  not used for the C++ build.

## Reproduce
See `spike/diagnose.sh` (read-only diagnostics used to produce this report).
