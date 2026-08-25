# Setup / Unblock Checklist

Applies to both supported hosts — **DGX Spark** (GB10, aarch64) and an **x86_64** box with a
discrete Blackwell GPU (validated: RTX PRO 6000 Blackwell). The steps are identical on both.

Status after M0 spike: GPU capabilities **confirmed** on both (OFA optical flow, NPP resize).
Remaining setup needs host actions (sudo / cabling). Do these in order; steps 1–2 are independent
of the NIC and can run now.

> **No SDK to install.** The engine used to build against a source-built NVIDIA Holoscan SDK; it now
> runs on the in-tree `spark::rt` runtime (`engine/runtime/`, see `docs/M11-runtime.md`), so the
> dependencies are CUDA/NPP, DPDK and the NVOF SDK headers — all covered below. There is no
> containerized SDK build, no CMake >= 3.30.4 floor, and no install tree to point `CMAKE_PREFIX_PATH` at.

Run privileged commands from the Claude session with the `!` prefix, e.g. `! sudo apt ...`.

> **IO-plane runtime is automated:** `deploy/provision.sh` (idempotent; `--check` for read-only)
> installs `mft`/`dpdk`/`linuxptp`, sets `REAL_TIME_CLOCK_ENABLE=1` on every ConnectX it finds by PCI
> vendor 15b3 (the tx_pp prerequisite — applies to the Spark's CX-7 and to any ConnectX in an x86 box),
> and configures hugepages. This checklist covers the *dev/build* setup (protobuf, web, NVOF headers)
> that `provision.sh` does not. See `deploy/README.md`.

---

## 1. Build tooling (one sudo apt)
```bash
sudo apt update
sudo apt install -y linuxptp ninja-build protobuf-compiler protobuf-compiler-grpc \
                    libgrpc++-dev libprotobuf-dev nodejs npm
```
- `linuxptp` → `ptp4l`/`phc2sys` (PTP, gate 1 once the NIC is up)
- protobuf/gRPC → C++ control daemon (M4); nodejs/npm → web UI (M5)

Also vendor the NVIDIA Optical Flow SDK headers (header-only; the implementation ships in the
driver's `libnvidia-opticalflow.so`). `third_party/` is gitignored, so this is once per checkout:
```bash
bash spike/fetch_nvof_sdk.sh
```

## 2. Build the engine
Nothing to point at — CUDA, DPDK and the NVOF headers from step 1 are the whole dependency set:
```bash
cd ~/SparkVideoProcessor
cmake -G Ninja -S engine -B engine/build
cmake --build engine/build
ctest --test-dir engine/build    # 10 unit tests; no NIC or root needed
./engine/build/spark_engine      # placeholder ping graph (spark::rt self-test)
```
CMake defaults `CMAKE_CUDA_ARCHITECTURES` to `120;121`, covering GB20x (sm_120) and GB10 (sm_121)
in one binary. For a faster build on a known box: `-DSPARK_CUDA_ARCHS=native`.

## 3. ST 2110 IO stack — DPDK / mlx5 (SMPTE 2110 — gate 2)
Rivermax was **dropped** (no license, stays open-source — see `docs/M0-feasibility-findings.md`
"Networking IO" decision). The ST 2110-20 RX/TX runs over DPDK's **mlx5 PMD**, which drives the
ConnectX directly and uses its hardware accurate-send-scheduling (`tx_pp`) for ST 2110-21 pacing.
The mlx5 PMD is bifurcated, so this does **not** unbind the NIC from the kernel.
The engine talks to DPDK directly (`engine/operators/st2110_{tx,rx}/dpdk_*_backend.cpp`) — holohub's
`advanced_network` operator was evaluated and never adopted, and with the Holoscan SDK gone it is no
longer an option in reserve.
```bash
sudo apt install -y dpdk dpdk-dev          # distro DPDK 23.11 is what the engine builds against
bash spike/dpdk_pacing_probe.sh            # checks DPDK+mlx5, ConnectX bind, and tx_pp HW pacing
```
- No NVIDIA account / license needed. DOCA GPUNetIO is the reserve GPUDirect path for later.
- `tx_pp` is **not** a CX-7-only feature: it is confirmed on the Spark's ConnectX-7 (100G) and on the
  x86_64 host's dual-port **ConnectX-6 Lx** (25G, `15b3:101f`) — see `docs/M1-gate4-pacing.md`.
  ConnectX-6 Dx / Lx and later carry it; older cards do not.

`tx_pp` requires the NIC to advertise DPDK's `SEND_ON_TIMESTAMP` Tx offload, which in turn requires
`REAL_TIME_CLOCK_ENABLE=1` in firmware **plus a cold reboot** (a warm `reboot` does not apply it).
Confirm the knob exists on your card before counting on paced TX:
```bash
sudo mst start
sudo mlxconfig -d <pci-bdf> query | grep -i REAL_TIME_CLOCK_ENABLE
```
If the parameter is absent the card cannot pace, and the TX backend falls back to unpaced sends
(`dpdk_tx_backend.cpp` logs this) — usable for development, not ST 2110-21 compliant.

## 4. Cable + bring up the ConnectX (gates 1, 2, 4)
On the DGX Spark the CX-7 is hot-plug and invisible until something is connected; a PCIe card in an
x86 box enumerates regardless, but still needs a cable for link-up and the loopback self-test.
```bash
# Physically: connect QSFP port <-> QSFP port (QSFP56 DAC or loopback) for the self-test topology.
# Then verify it enumerated:
lspci -nnD | grep -i 15b3            # expect a Mellanox/ConnectX device
rdma link show                        # expect an mlx5 device
ls /sys/class/net                     # expect new high-speed interface(s)
ethtool -T <new_iface>                # expect a PTP Hardware Clock + HW timestamp modes
```
Then PTP. **First check whether the host already disciplines the PHC itself** — a box with its own
`ptp4l` systemd unit (the validated x86_64 host runs `ptp4l-smpte.service` + `phc2sys-smpte.service`,
ST 2059-2 domain 127, slaved to a facility grandmaster) needs nothing further: the engine reads the
same PHC, so it inherits that discipline.
```bash
bash deploy/ptp.sh --check          # read-only; reports a host-managed ptp4l + its lock offsets
```
If `--check` reports a host-managed `ptp4l`, do **not** start another one — `deploy/ptp.sh`'s running
modes refuse to (`FORCE=1` overrides), because a second `ptp4l` on the same interface fights the first
and `--master`/`--test` omit `slaveOnly`, so they could win BMCA against the real grandmaster.

Otherwise bring PTP up from here (loopback: software master on one port, slave on the other; both
ports usually share one PHC so lock is trivially perfect — validates the mechanism):
```bash
sudo bash deploy/ptp.sh start       # ptp4l slave + phc2sys; 'status' / 'stop' to manage
sudo ptp4l -i <iface> -m            # or drive ptp4l directly
```
Note the PHC index is **not** always `ptp0` — read it off the port (`ethtool -T <iface>` reports e.g.
`PTP Hardware Clock: 2` on the x86_64 host, vs `ptp0` on the Spark).

---

## Quick re-check anytime
```bash
bash spike/diagnose.sh        # read-only host/NIC/PTP/GPU inventory
bash spike/build_probes.sh    # rebuild+run OFA + NPP probes (no sudo; auto-fetches the NVOF SDK)
```

When steps 1–4 are done, M1 (the `st2110_rx → st2110_tx` pass-through) can begin.
