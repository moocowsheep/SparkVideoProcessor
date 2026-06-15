#!/usr/bin/env bash
# M1 gate 4 — paced ST 2110-20-rate loopback over the CX-7 (Rivermax-free, raw DPDK).
#
# Builds directly on the M0-proven mechanism (dpdk-testpmd + mlx5 tx_pp HW send-scheduling).
# Drives the TX port at an ST 2110-20 *line rate* with every packet HW-scheduled at a fixed
# inter-packet interval, receives on the cabled partner port, and reports:
#   * pacing precision  — mlx5 'tx_pp_*' xstats (jitter / wander / timing errors, in ns), the
#                         direct hardware measure of how tightly sends matched the schedule;
#   * loss / throughput — RX packet count vs TX, at rate (a port that can't pace drops packets).
#
# Gate-4 verdict per the M0 handoff: pacing precision must hold at 1080p (~3 Gbps) AND, the real
# test, 2160p (~12 Gbps) — mlx5 tx_pp precision is documented to degrade under high Tx load, so
# 12G is the gate, not 3G. (Zero-copy ingest latency is the follow-on once the RX operator exists;
# testpmd rxonly only counts, it doesn't correlate per-packet TX/RX HW timestamps.)
#
# Two testpmd instances share the host: disjoint core lists + distinct --file-prefix, each binds
# only its own port via -a. mlx5 is bifurcated, so neither unbinds the NIC from the kernel.
#
# Usage (root, needs hugepages — allocated by the M0 probe / deploy provisioning):
#   sudo -E LOOP_TX_PCI=0000:01:00.0 LOOP_RX_PCI=0000:01:00.1 PROFILE=1080p \
#        bash spike/st2110_loopback_gate4.sh
#   # PROFILE = 1080p | 2160p   (run 1080p first to sanity-check, then 2160p for the real gate)
#   # If you ran spike/detect_loopback.py, source its LOOP_TX_PCI/LOOP_RX_PCI lines instead.
set -u

# --- ST 2110-20 rate profiles ----------------------------------------------------------------
# Representative 10-bit 4:2:2 line rates incl. ST 2110-21 gapped pacing overhead. PKT = on-wire
# L2 bytes (testpmd 'set txpkts'); GBPS = target line rate; pps + gap_ns are derived below.
PROFILE="${PROFILE:-1080p}"
PKT="${PKT:-1438}"                 # ST 2110-20 ~1420B RTP media payload + RTP/UDP/IP/Eth headers
case "$PROFILE" in
  1080p) GBPS="${GBPS:-3.0}"  ;;   # 1080p59.94 4:2:2 10-bit  ~2.97 Gbps
  2160p) GBPS="${GBPS:-12.0}" ;;   # 2160p59.94 4:2:2 10-bit  ~11.9 Gbps  (the real gate)
  *) echo "unknown PROFILE='$PROFILE' (use 1080p|2160p)"; exit 2 ;;
esac
DURATION="${DURATION:-15}"         # seconds at rate
TX_PP_NS="${TX_PP_NS:-500}"        # tx_pp clock granularity (M0 used 500)
# Tx ring depth bounds the scheduling horizon: in-flight packets are held for their send time,
# so horizon ~= TXD * gap must stay under the NIC's finite tx_pp window or every packet trips
# tx_pp_timestamp_future_errors and pacing collapses to line-rate. Small TXD => HW backpressures
# testpmd down to the scheduled rate. 128 * gap = ~490us (1080p) / ~123us (2160p).
TXD="${TXD:-128}"
TX_CORES="${TX_CORES:-0,1}"
RX_CORES="${RX_CORES:-2,3}"

LOOP_TX_PCI="${LOOP_TX_PCI:-}"
LOOP_RX_PCI="${LOOP_RX_PCI:-}"

hr()   { printf '\n===== %s =====\n' "$1"; }
pass() { printf '  [PASS] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; }
info() { printf '  [info] %s\n' "$1"; }

# Accept either a PCI BDF (e.g. 0002:01:00.0) or an interface name (e.g. enP2p1s0f0np0).
# DPDK's -a wants the PCI address; resolve a netdev name to its BDF via sysfs so both work.
resolve_pci() {
  local v="$1"
  if [ -e "/sys/class/net/$v/device" ]; then
    basename "$(readlink -f "/sys/class/net/$v/device")"
  else
    echo "$v"
  fi
}

# Resolve a PCI BDF to its netdev MAC. There's a switch in-path (all 4 ports share one L2
# domain), so the txonly destination MAC must be the RX port's MAC for the switch to deliver
# TX packets there (flooded as unknown-unicast, since rxonly never sources its own MAC).
pci_to_mac() {
  local pci="$1" d
  for d in /sys/class/net/*; do
    if [ "$(basename "$(readlink -f "$d/device" 2>/dev/null)")" = "$pci" ]; then
      cat "$d/address"; return 0
    fi
  done
}

# --- preflight -------------------------------------------------------------------------------
[ "$(id -u)" = "0" ] || { fail "must run as root (testpmd + hugepages)"; exit 1; }
command -v dpdk-testpmd >/dev/null || { fail "dpdk-testpmd missing"; exit 1; }
[ -n "$LOOP_TX_PCI" ] && [ -n "$LOOP_RX_PCI" ] || {
  fail "set LOOP_TX_PCI and LOOP_RX_PCI (run spike/detect_loopback.py to find the cabled pair)"; exit 1; }
# Tolerate an interface name in either var (common slip) and normalize to the PCI BDF DPDK needs.
LOOP_TX_PCI="$(resolve_pci "$LOOP_TX_PCI")"
LOOP_RX_PCI="$(resolve_pci "$LOOP_RX_PCI")"
RX_MAC="$(pci_to_mac "$LOOP_RX_PCI")"
[ -n "$RX_MAC" ] || { fail "could not resolve a netdev MAC for RX $LOOP_RX_PCI"; exit 1; }
if [ "$(awk '/HugePages_Total/{print $2}' /proc/meminfo)" -eq 0 ]; then
  info "no hugepages configured; allocating 2048x2MB"
  echo 2048 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 2>/dev/null || true
fi
mkdir -p /dev/hugepages; mountpoint -q /dev/hugepages || mount -t hugetlbfs none /dev/hugepages 2>/dev/null || true

# --- derive pacing from the profile ----------------------------------------------------------
# pps = bitrate / (bytes_on_wire * 8); per-packet gap = 1e9 / pps  (ns). testpmd 'set txtimes'
# takes <inter_burst_gap>,<intra_burst_gap>; with --burst=1 the intra-burst gap IS the per-packet
# interval, giving uniform ST 2110-21-style pacing (one packet per scheduled slot).
read -r PPS GAP_NS <<EOF
$(awk -v g="$GBPS" -v p="$PKT" 'BEGIN{pps=g*1e9/(p*8); printf "%d %d", pps, 1e9/pps}')
EOF
hr "gate-4 loopback: $PROFILE @ ${GBPS} Gbps"
info "TX ${LOOP_TX_PCI} (cores ${TX_CORES}) -> RX ${LOOP_RX_PCI} [${RX_MAC}] (cores ${RX_CORES})"
info "pkt=${PKT}B  pps=${PPS}  per-packet gap=${GAP_NS}ns  tx_pp=${TX_PP_NS}ns  txd=${TXD}  duration=${DURATION}s"

RXLOG="$(mktemp)"; TXLOG="$(mktemp)"

# --- RX instance (rxonly, counts packets for the loss check) ---------------------------------
# Plain bind: we only need RX-packets vs TX-packets here. Per-packet zero-copy latency (which
# needs correlated TX/RX HW timestamps) is the follow-on once the real RX operator exists.
hr "starting RX (rxonly)"
{
  echo "port stop all"; echo "set fwd rxonly"; echo "port start all"; echo "start"
  sleep "$((DURATION + 4))"
  echo "show port stats all"; echo "stop"; echo "quit"
} | timeout "$((DURATION + 20))" dpdk-testpmd \
      -l "$RX_CORES" --file-prefix spark_rx -a "${LOOP_RX_PCI}" \
      -- -i --total-num-mbufs=8192 --rxq=2 >"$RXLOG" 2>&1 &
RXPID=$!
sleep 3   # let RX bind + start before TX floods

# --- TX instance (txonly, tx_pp paced) -------------------------------------------------------
hr "starting TX (txonly, tx_pp paced)"
{
  echo "port stop all"
  # mlx5 only HW-paces if the Tx queue carries the send_on_timestamp offload; set txtimes alone
  # leaves the NIC sending at max rate (the first-run symptom). Must be set while the port is stopped.
  echo "port config 0 tx_offload send_on_timestamp on"
  echo "set fwd txonly"
  echo "set eth-peer 0 ${RX_MAC}"      # dst MAC = RX port so the in-path switch delivers there
  echo "set txpkts ${PKT}"
  echo "set burst 1"                   # 1 packet per scheduled slot => uniform per-packet pacing
  echo "set txtimes ${GAP_NS},${GAP_NS}"
  echo "port start all"
  echo "start"
  sleep "$DURATION"
  echo "stop"
  echo "show port xstats 0"
  echo "show port stats 0"
  echo "quit"
} | timeout "$((DURATION + 25))" dpdk-testpmd \
      -l "$TX_CORES" --file-prefix spark_tx -a "${LOOP_TX_PCI},tx_pp=${TX_PP_NS}" \
      -- -i --total-num-mbufs=8192 --txq=1 --txd="${TXD}" >"$TXLOG" 2>&1
wait "$RXPID" 2>/dev/null

# --- report ----------------------------------------------------------------------------------
hr "TX testpmd log (tail)"; tail -n 40 "$TXLOG" | sed 's/^/    /'
hr "tx_pp pacing xstats (the precision metric)"
grep -iE 'tx_pp_|clock' "$TXLOG" | sed 's/^/    /' || info "no tx_pp_* xstats found — see TX log above"
hr "RX testpmd log (tail)"; tail -n 25 "$RXLOG" | sed 's/^/    /'

TXPKTS="$(grep -iE 'TX-packets' "$TXLOG" | tail -1 | grep -oE '[0-9]+' | head -1)"
RXPKTS="$(grep -iE 'RX-packets' "$RXLOG" | tail -1 | grep -oE '[0-9]+' | head -1)"
EXPECT=$((PPS * DURATION))
hr "summary"
info "target pps=${PPS}  expected~${EXPECT} pkts over ${DURATION}s"
info "TX-packets=${TXPKTS:-?}  (achieved ~$(( ${TXPKTS:-0} / DURATION )) pps)   RX-packets=${RXPKTS:-?}"

# Pull the individual tx_pp counters for an actionable verdict.
ppval() { grep -E "^\s*$1:" "$TXLOG" | tail -1 | grep -oE '[0-9]+' | head -1; }
FUT="$(ppval tx_pp_timestamp_future_errors)"; PAST="$(ppval tx_pp_timestamp_past_errors)"
JIT="$(ppval tx_pp_jitter)"; WAN="$(ppval tx_pp_wander)"; SYNC="$(ppval tx_pp_sync_lost)"

# 1) Did the schedule actually throttle TX to the target rate (vs flooding at line rate)?
if [ -n "${TXPKTS:-}" ] && [ "$TXPKTS" -gt $((EXPECT * 3 / 2)) ]; then
  fail "rate NOT paced — TX ${TXPKTS} >> target ${EXPECT} (schedule horizon overran the tx_pp window)"
elif [ -n "${TXPKTS:-}" ] && [ "$TXPKTS" -lt $((EXPECT / 2)) ]; then
  fail "underrun — TX ${TXPKTS} << target ${EXPECT} (TX core couldn't feed the schedule)"
elif [ -n "${TXPKTS:-}" ]; then
  pass "rate paced — achieved ~$((TXPKTS / DURATION)) pps tracks the ${PROFILE} target ${PPS}"
fi
# 2) HW pacing precision — the ST 2110-21 go/no-go.
info "tx_pp: jitter=${JIT:-?}ns wander=${WAN:-?}ns sync_lost=${SYNC:-?} | future_err=${FUT:-?} past_err=${PAST:-?}"
if [ "${SYNC:-0}" -gt 0 ]; then
  fail "tx_pp_sync_lost>0 — the PP clock lost lock to the PHC (pacing unusable)"
elif [ "${FUT:-0}" -gt $((EXPECT / 100 + 1000)) ]; then
  fail "future_errors dominate — horizon > tx_pp window. Lower TXD (try TXD=$((TXD/2))) and re-run."
elif [ "${PAST:-0}" -gt $((EXPECT / 100 + 1000)) ]; then
  fail "past_errors high — TX core can't keep the schedule. Try larger TXD, more TX cores, or coarser TX_PP_NS."
else
  pass "clean pacing — jitter ${JIT:-?}ns / wander ${WAN:-?}ns, negligible schedule errors at $PROFILE"
fi
# 3) Loss through the fabric.
if [ -n "${RXPKTS:-}" ] && [ -n "${TXPKTS:-}" ] && [ "$TXPKTS" -gt 0 ]; then
  [ "$RXPKTS" -ge $((TXPKTS * 9 / 10)) ] && pass "no significant loss (RX >= 90% of TX)" \
    || fail "loss: RX-packets=${RXPKTS} << TX-packets=${TXPKTS} (check eth-peer / switch flooding)"
fi
cat <<EOF

  Gate 4 PASS = rate paced + sync_lost 0 + negligible future/past errors; jitter/wander are the
  residual HW precision (ns). tx_pp_jitter has already shown 4ns@3G / 32ns@12G — far inside the
  ST 2110-21 budget — so the remaining task is just to hold the schedule (TXD tuning above).
  (Logs: $TXLOG, $RXLOG)
EOF
