#!/usr/bin/env bash
# M0 feasibility diagnostics — read-only. Safe to run anytime.
# Produces the evidence behind docs/M0-feasibility-findings.md.
set -u

hr() { printf '\n===== %s =====\n' "$1"; }

hr "Host"
uname -a
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null

hr "PCI: Mellanox/ConnectX (vendor 15b3)"
lspci -nnD 2>/dev/null | grep -iE '15b3|mellanox|connectx' || echo "NONE — ConnectX-7 not enumerated (hot-plug; connect a QSFP cable)"

hr "mlx5 / RDMA kernel modules"
lsmod | grep -iE 'mlx5|ib_core|rdma' || echo "(none loaded)"
ls /sys/class/infiniband 2>/dev/null || echo "(no /sys/class/infiniband)"
rdma link show 2>/dev/null || true

hr "Network interfaces"
ip -br addr

hr "PTP hardware timestamping per interface"
for i in $(ls /sys/class/net | grep -vE '^(lo|docker|veth|br-|tailscale)'); do
  echo "--- $i ---"; ethtool -T "$i" 2>/dev/null
done
echo "PTP clock devices:"; ls -l /dev/ptp* 2>/dev/null || echo "  (none)"

hr "GPU media engines"
nvidia-smi -q 2>/dev/null | grep -iE 'encoder|decoder|jpeg|ofa' | head

hr "Optical Flow / NPP / DPDK / DOCA / Holoscan libs"
ldconfig -p | grep -iE 'opticalflow|nvof|nppi|dpdk|rte_|doca|holoscan' || echo "(none)"
command -v dpdk-testpmd >/dev/null && echo "dpdk-testpmd: present" || echo "dpdk-testpmd: MISSING"

hr "Toolchain"
for t in nvcc gcc g++ cmake ninja make git node npm protoc; do
  printf '%-7s ' "$t"; (command -v "$t" >/dev/null && "$t" --version 2>/dev/null | head -1) || echo MISSING
done
