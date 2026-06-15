#!/usr/bin/env bash
# M0 gate 2 (Rivermax-free) — DPDK / mlx5 ST 2110-21 pacing capability probe.
#
# Validates the open networking path chosen in docs/M0-feasibility-findings.md ("Networking IO"):
#   ST 2110-20 RX/TX over DPDK's mlx5 PMD, using the CX-7's hardware accurate-send-scheduling
#   (tx_pp) for ST 2110-21 pacing — no Rivermax, no license.
#
# Staged so each step reports independently as the NIC comes up:
#   1. DPDK toolchain + mlx5 PMD present        (runs now, no NIC needed)
#   2. ConnectX-7 enumerated on the PCI bus      (needs the QSFP cable connected)
#   3. PHC present for RTP/PTP timestamps        (needs CX-7 up)
#   4. mlx5 hardware send-scheduling (tx_pp)     (gated: RUN_TESTPMD=1; needs hugepages + sudo)
#
# mlx5 is a *bifurcated* PMD: probing does NOT unbind the NIC from the kernel. Step 4 will briefly
# take over the port's DPDK queues, so only enable it on the loopback self-test ports, not a live feed.
#
# Usage:
#   bash spike/dpdk_pacing_probe.sh              # steps 1-3 (safe, read-only)
#   RUN_TESTPMD=1 sudo -E bash spike/dpdk_pacing_probe.sh   # also step 4 (tx_pp probe)
set -u

hr()   { printf '\n===== %s =====\n' "$1"; }
pass() { printf '  [PASS] %s\n' "$1"; }
skip() { printf '  [SKIP] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; }

TX_PP_NS="${TX_PP_NS:-500}"   # send-scheduling granularity to request, in ns

# --- 1. DPDK toolchain + mlx5 PMD -------------------------------------------------------------
hr "1. DPDK toolchain + mlx5 PMD"
HAVE_TESTPMD=0
if command -v dpdk-testpmd >/dev/null 2>&1; then
  pass "dpdk-testpmd present: $(command -v dpdk-testpmd)"
  HAVE_TESTPMD=1
else
  fail "dpdk-testpmd MISSING — 'sudo apt install dpdk dpdk-dev', or use holohub's bundled DPDK"
fi
if pkg-config --exists libdpdk 2>/dev/null; then
  pass "libdpdk (pkg-config): $(pkg-config --modversion libdpdk)"
else
  skip "libdpdk pkg-config not found (fine if only using the holohub-bundled DPDK)"
fi
if ldconfig -p 2>/dev/null | grep -qiE 'rte_net_mlx5|rte_common_mlx5'; then
  pass "mlx5 PMD libs present: $(ldconfig -p | grep -ioE 'librte_(net|common)_mlx5[^ ]*' | sort -u | tr '\n' ' ')"
else
  fail "mlx5 PMD libs (librte_net_mlx5) NOT found — DPDK must be built with mlx5 (needs rdma-core)"
fi

# --- 2. ConnectX-7 on the bus -----------------------------------------------------------------
hr "2. ConnectX-7 enumeration"
CX7_PCI="$(lspci -Dn 2>/dev/null | awk '/15b3:/{print $1; exit}')"
if [ -n "${CX7_PCI:-}" ]; then
  pass "ConnectX (15b3) at PCI ${CX7_PCI}"
  lspci -nnDs "${CX7_PCI}" 2>/dev/null | sed 's/^/        /'
else
  fail "no Mellanox/ConnectX (15b3) on the bus — connect a QSFP cable so the hot-plug CX-7 enumerates"
fi

# --- 3. PHC for RTP/PTP timestamps ------------------------------------------------------------
hr "3. PTP hardware clock (for RTP timestamps)"
if ls /dev/ptp* >/dev/null 2>&1; then
  pass "PHC device(s): $(ls /dev/ptp* | tr '\n' ' ')"
  for i in $(ls /sys/class/net 2>/dev/null | grep -vE '^(lo|docker|veth|br-|tailscale|wl)'); do
    phc="$(ethtool -T "$i" 2>/dev/null | awk -F': ' '/PTP Hardware Clock/{print $2}')"
    [ -n "${phc:-}" ] && [ "${phc}" != "none" ] && echo "        $i -> PHC index ${phc}"
  done
else
  fail "no /dev/ptp* — needs the CX-7 up (the Realtek 10GbE has no HW PTP)"
fi

# --- 4. mlx5 hardware send-scheduling (tx_pp) — the ST 2110-21 go/no-go ------------------------
hr "4. mlx5 tx_pp send-scheduling capability"
if [ "${RUN_TESTPMD:-0}" != "1" ]; then
  skip "set RUN_TESTPMD=1 (as root) to probe; needs the CX-7 up — see step 2"
elif [ "${HAVE_TESTPMD}" != "1" ] || [ -z "${CX7_PCI:-}" ]; then
  skip "prerequisites unmet (need dpdk-testpmd AND an enumerated CX-7)"
elif [ "$(id -u)" != "0" ]; then
  skip "must be root for testpmd — re-run: RUN_TESTPMD=1 sudo -E bash $0"
else
  # We are root: ensure hugepages exist + hugetlbfs mounted (harmless if already set).
  if [ "$(awk '/HugePages_Total/{print $2}' /proc/meminfo)" -eq 0 ]; then
    echo "  allocating 1024x2MB hugepages (none configured)"
    echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 2>/dev/null || true
  fi
  mkdir -p /dev/hugepages 2>/dev/null
  mountpoint -q /dev/hugepages || mount -t hugetlbfs none /dev/hugepages 2>/dev/null || true
  echo "  probing: dpdk-testpmd -a ${CX7_PCI},tx_pp=${TX_PP_NS} -- -i  (init port, then quit)"
  # EAL args BEFORE '--'; testpmd app args AFTER '--'. '-i' makes it read 'quit' from stdin.
  log="$(printf 'quit\n' | timeout 40 dpdk-testpmd \
          --file-prefix spark_txpp -a "${CX7_PCI},tx_pp=${TX_PP_NS}" \
          -- -i --total-num-mbufs=4096 2>&1)" || true
  echo "${log}" | sed 's/^/        /'
  # FAIL first: a port that fails to probe still drops into the testpmd> prompt with no devices,
  # so 'Done'/'testpmd>' alone is NOT success — require an actually-configured Port 0.
  if echo "${log}" | grep -qiE 'packet pacing is not supported|probe of PCI device .* aborted|No probed ethernet devices|Bus \(pci\) probe failed|cannot be used|Failed to process device configure'; then
    fail "mlx5 rejected tx_pp ('Packet pacing is not supported') — enable it in NIC firmware:"
    echo "         mlxconfig -d ${CX7_PCI} set REAL_TIME_CLOCK_ENABLE=1   (needs MFT/MST), then cold-reboot and re-probe."
  elif echo "${log}" | grep -qiE 'Configuring Port 0|Port 0: |Link up'; then
    pass "mlx5 accepted tx_pp + port initialized — HW send-scheduling (ST 2110-21 pacing) AVAILABLE"
  else
    skip "inconclusive — inspect the testpmd log above"
  fi
fi

# --- Summary ----------------------------------------------------------------------------------
hr "Next"
cat <<'EOF'
  When steps 1-4 PASS, the Rivermax-free pacing mechanism is proven. Remaining gate-2/4 work
  (becomes the M1 st2110_rx/st2110_tx operators):
    * generate ST 2110-20-rate UDP/RTP out one QSFP port, receive on the other (loopback);
    * confirm tx_pp pacing holds at 1080p (~3 Gbps) THEN 2160p (~12 Gbps) -- precision is known
      to degrade under high Tx load, so 12G is the real test, not 3G.
EOF
