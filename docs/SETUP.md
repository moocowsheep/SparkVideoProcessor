# Setup / Unblock Checklist

Applies to both supported hosts — **DGX Spark** (GB10, aarch64) and an **x86_64** box with a
discrete Blackwell GPU (validated: RTX PRO 6000 Blackwell). The steps are the same; only the
Holoscan install-tree name differs by arch, and it is globbed rather than hardcoded throughout.

Status after M0 spike: GPU capabilities **confirmed** on both (OFA optical flow, NPP resize,
Holoscan runtime). Remaining setup needs host actions (sudo / cabling / NVIDIA account). Do these
in order; steps 1–3 are independent of the NIC and can run now.

Run privileged commands from the Claude session with the `!` prefix, e.g. `! sudo apt ...`.

> **IO-plane runtime is automated:** `deploy/provision.sh` (idempotent; `--check` for read-only)
> installs `mft`/`dpdk`/`linuxptp`, sets `REAL_TIME_CLOCK_ENABLE=1` on every ConnectX it finds by PCI
> vendor 15b3 (the tx_pp prerequisite — applies to the Spark's CX-7 and to any ConnectX in an x86 box),
> and configures hugepages. This checklist covers the *dev/build* setup (Holoscan, protobuf, web) that
> `provision.sh` does not. See `deploy/README.md`.

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

## 2. Docker access (for the Holoscan source build)
```bash
sudo usermod -aG docker "$USER"
# then log out/in (or: newgrp docker) so the group takes effect
docker ps        # should work without sudo afterwards
```

## 3. Build Holoscan SDK v4.3.0 from source (native install tree)
Produces a self-contained `install-cu13-<arch>-dgpu/` we consume natively from `engine/`. The
invocation is identical on both hosts — `--gpu dgpu` is correct for each, and `./run` picks the
arch up from the machine it runs on:
```bash
git clone --depth 1 --branch v4.3.0 \
  https://github.com/nvidia-holoscan/holoscan-sdk.git ~/holoscan-sdk
cd ~/holoscan-sdk
./run build --gpu dgpu --cuda 13    # DGX Spark is sbsa/dgpu, NOT igpu (igpu has no CUDA-13 base)
# result: ~/holoscan-sdk/install-cu13-aarch64-dgpu   (DGX Spark)
#      or ~/holoscan-sdk/install-cu13-x86_64-dgpu    (x86_64 host)
```
x86_64 dGPU is Holoscan's *primary* target, so this is the well-trodden path there — the aarch64
`dgpu`-not-`igpu` subtlety recorded in `docs/M0-feasibility-findings.md` is Spark-specific.

Then build the engine against it. Glob the install tree so the same command works on either arch:
```bash
cd ~/SparkVideoProcessor
T=$(echo ~/holoscan-sdk/install-cu13-*-dgpu)
cmake -G Ninja -S engine -B engine/build -DCMAKE_PREFIX_PATH="$T"
cmake --build engine/build
ctest --test-dir engine/build    # 8 unit tests; no NIC or root needed
./engine/build/spark_engine      # placeholder ping graph until M1 wires real operators
```
CMake defaults `CMAKE_CUDA_ARCHITECTURES` to `120;121`, covering GB20x (sm_120) and GB10 (sm_121)
in one binary. For a faster build on a known box: `-DSPARK_CUDA_ARCHS=native`.

## 4. ST 2110 IO stack — DPDK via Holoscan `advanced_network` (for SMPTE 2110 — gate 2)
Rivermax was **dropped** (no license, stays open-source — see `docs/M0-feasibility-findings.md`
"Networking IO" decision). The ST 2110-20 RX/TX runs over DPDK's **mlx5 PMD**, which drives the
ConnectX directly and uses its hardware accurate-send-scheduling (`tx_pp`) for ST 2110-21 pacing.
The mlx5 PMD is bifurcated, so this does **not** unbind the NIC from the kernel.
```bash
# Build the advanced_network operator (DPDK backend) from holohub — bundles a compatible DPDK.
git clone --depth 1 https://github.com/nvidia-holoscan/holohub.git ~/holohub
# then follow holohub/operators/advanced_network (DPDK manager) build for your arch / CUDA 13

# Standalone capability check (needs DPDK + mlx5 PMD; no NIC unbind):
sudo apt install -y dpdk dpdk-dev          # only if NOT using holohub's bundled DPDK
bash spike/dpdk_pacing_probe.sh            # checks DPDK+mlx5, ConnectX bind, and tx_pp HW pacing
```
- No NVIDIA account / license needed. DOCA GPUNetIO is the reserve GPUDirect path for later.

`tx_pp` requires the NIC to advertise DPDK's `SEND_ON_TIMESTAMP` Tx offload, which in turn requires
`REAL_TIME_CLOCK_ENABLE=1` in firmware **plus a cold reboot** (a warm `reboot` does not apply it).
Confirm the knob exists on your card before counting on paced TX:
```bash
sudo mst start
sudo mlxconfig -d <pci-bdf> query | grep -i REAL_TIME_CLOCK_ENABLE
```
If the parameter is absent the card cannot pace, and the TX backend falls back to unpaced sends
(`dpdk_tx_backend.cpp` logs this) — usable for development, not ST 2110-21 compliant.

## 5. Cable + bring up the ConnectX (gates 1, 2, 4)
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
Then PTP (loopback: run a software master on one port, slave on the other; note both ports usually
share one PHC so lock is trivially perfect — validates the mechanism):
```bash
sudo ptp4l -i <iface> -m            # add -2 (L2) or default UDP per your config
```

---

## Quick re-check anytime
```bash
bash spike/diagnose.sh        # read-only host/NIC/PTP/GPU inventory
bash spike/build_probes.sh    # rebuild+run OFA + NPP probes (no sudo; auto-fetches the NVOF SDK)
```

When steps 1–5 are done, M1 (the `st2110_rx → st2110_tx` pass-through) can begin.
