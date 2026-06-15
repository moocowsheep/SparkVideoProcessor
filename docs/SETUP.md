# Setup / Unblock Checklist — DGX Spark

Status after M0 spike: GPU capabilities **confirmed** (OFA optical flow, NPP resize, Holoscan
runtime). Remaining setup needs host actions (sudo / cabling / NVIDIA account). Do these in order;
steps 1–3 are independent of the NIC and can run now.

Run privileged commands from the Claude session with the `!` prefix, e.g. `! sudo apt ...`.

---

## 1. Build tooling (one sudo apt)
```bash
sudo apt update
sudo apt install -y linuxptp ninja-build protobuf-compiler protobuf-compiler-grpc \
                    libgrpc++-dev libprotobuf-dev nodejs npm
```
- `linuxptp` → `ptp4l`/`phc2sys` (PTP, gate 1 once the CX-7 is up)
- protobuf/gRPC → C++ control daemon (M4); nodejs/npm → web UI (M5)

## 2. Docker access (for the Holoscan source build)
```bash
sudo usermod -aG docker "$USER"
# then log out/in (or: newgrp docker) so the group takes effect
docker ps        # should work without sudo afterwards
```

## 3. Build Holoscan SDK v4.3.0 from source (native install tree)
Produces a self-contained `install-cu13-aarch64*/` we consume natively from `engine/`.
```bash
git clone --depth 1 --branch v4.3.0 \
  https://github.com/nvidia-holoscan/holoscan-sdk.git ~/holoscan-sdk
cd ~/holoscan-sdk
export CUDA_MAJOR=13 HOLOSCAN_BUILD_ARCH=aarch64
./run build --gpu igpu          # GB10 is unified-memory; if it fails, retry: --gpu dgpu
# result: ~/holoscan-sdk/install-cu13-aarch64-igpu  (point engine CMake here)
```
Then build the engine against it:
```bash
cd /home/saturn/claude/MooVideoProcessor
cmake -G Ninja -S engine -B engine/build \
  -DCMAKE_PREFIX_PATH="$HOME/holoscan-sdk/install-cu13-aarch64-igpu"
cmake --build engine/build
./engine/build/moo_engine        # placeholder ping graph until M1 wires real operators
```

## 4. ST 2110 IO stack — DPDK via Holoscan `advanced_network` (for SMPTE 2110 — gate 2)
Rivermax was **dropped** (no license, stays open-source — see `docs/M0-feasibility-findings.md`
"Networking IO" decision). The ST 2110-20 RX/TX runs over DPDK's **mlx5 PMD**, which drives the
CX-7 directly and uses its hardware accurate-send-scheduling (`tx_pp`) for ST 2110-21 pacing. The
mlx5 PMD is bifurcated, so this does **not** unbind the NIC from the kernel.
```bash
# Build the advanced_network operator (DPDK backend) from holohub — bundles a compatible DPDK.
git clone --depth 1 https://github.com/nvidia-holoscan/holohub.git ~/holohub
# then follow holohub/operators/advanced_network (DPDK manager) build for aarch64 / CUDA 13

# Standalone capability check (needs DPDK + mlx5 PMD; no NIC unbind):
sudo apt install -y dpdk dpdk-dev          # only if NOT using holohub's bundled DPDK
bash spike/dpdk_pacing_probe.sh            # checks DPDK+mlx5, CX-7 bind, and tx_pp HW pacing
```
- No NVIDIA account / license needed. DOCA GPUNetIO is the reserve GPUDirect path for later.

## 5. Cable + bring up the ConnectX-7 (gates 1, 2, 4)
The CX-7 is hot-plug and invisible until something is connected.
```bash
# Physically: connect QSFP port <-> QSFP port (QSFP56 DAC or loopback) for the self-test topology.
# Then verify it enumerated:
lspci -nnD | grep -i 15b3            # expect a Mellanox/ConnectX-7 device
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
bash spike/build_probes.sh    # rebuild+run OFA + NPP probes (no sudo)
```

When steps 1–5 are done, M1 (the `st2110_rx → st2110_tx` pass-through) can begin.
