#!/usr/bin/env bash
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

# Spark Video Processor — host provisioning for the ST 2110 IO plane (M1 open item #5).
#
# Makes a fresh DGX Spark reproduce the runtime the engine needs, idempotently:
#   1. packages   — mft (mlxconfig), DPDK runtime+dev (mlx5 PMD), linuxptp
#   2. NIC firmware — REAL_TIME_CLOCK_ENABLE=1 on every ConnectX-7 (the tx_pp prerequisite; without
#                     it mlx5 reports "Packet pacing is not supported"). Needs a COLD REBOOT to apply.
#   3. hugepages  — runtime alloc + persistent (sysctl.d) + /dev/hugepages mount, for DPDK/EAL.
#
# Usage:
#   sudo bash deploy/provision.sh            # apply (idempotent; safe to re-run)
#   bash deploy/provision.sh --check         # read-only: report current state, change nothing
#   HUGEPAGES=4096 sudo bash deploy/provision.sh   # override hugepage count (default 2048 x 2MB)
#
# Dev/build setup (Holoscan SDK, protobuf, web) is separate — see docs/SETUP.md.
set -u

CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1
HUGEPAGES="${HUGEPAGES:-2048}"          # number of 2MB pages (2048 = 4 GiB)
HP_DIR=/sys/kernel/mm/hugepages/hugepages-2048kB
SYSCTL_FILE=/etc/sysctl.d/80-spark-hugepages.conf

reboot_needed=0
hr()   { printf '\n===== %s =====\n' "$1"; }
ok()   { printf '  [ok]   %s\n' "$1"; }
info() { printf '  [info] %s\n' "$1"; }
warn() { printf '  [warn] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; }

need_root() {
  if [ "$CHECK" = 0 ] && [ "$(id -u)" != 0 ]; then
    fail "must run as root to apply (or use --check). Try: sudo bash deploy/provision.sh"
    exit 1
  fi
}

# ConnectX-7 management BDFs = function-0 of each 15b3 card (mlxconfig is per-card).
detect_cx7() { lspci -D -d 15b3: 2>/dev/null | awk '{print $1}' | grep '\.0$'; }

# --- 1. packages ----------------------------------------------------------------------------------
provision_packages() {
  hr "packages (mft, dpdk, linuxptp)"
  local pkgs=(mft dpdk dpdk-dev linuxptp)
  if [ "$CHECK" = 1 ]; then
    for p in "${pkgs[@]}"; do
      dpkg -s "$p" >/dev/null 2>&1 && ok "$p installed" || warn "$p MISSING"
    done
    command -v mlxconfig >/dev/null && ok "mlxconfig present" || warn "mlxconfig missing (install mft)"
    command -v dpdk-testpmd >/dev/null && ok "dpdk-testpmd present" || warn "dpdk-testpmd missing"
    return
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq || warn "apt-get update failed (continuing)"
  apt-get install -y "${pkgs[@]}" && ok "packages installed" || fail "apt-get install failed"
}

# --- 2. NIC firmware: REAL_TIME_CLOCK_ENABLE=1 ----------------------------------------------------
rtc_value() {  # current REAL_TIME_CLOCK_ENABLE for a device, or "?" — prints 0/1/?
  mlxconfig -d "$1" query REAL_TIME_CLOCK_ENABLE 2>/dev/null \
    | awk '/REAL_TIME_CLOCK_ENABLE/{print ($NF ~ /1|True|ENABLE/) ? 1 : 0; found=1} END{if(!found)print "?"}'
}

provision_firmware() {
  hr "NIC firmware: REAL_TIME_CLOCK_ENABLE=1 (tx_pp prerequisite)"
  if ! command -v mlxconfig >/dev/null; then
    warn "mlxconfig not present — install mft (step 1) first"
    return
  fi
  [ "$CHECK" = 0 ] && { mst start >/dev/null 2>&1 || true; }
  local devs
  devs="$(detect_cx7)"
  if [ -z "$devs" ]; then
    warn "no ConnectX-7 (15b3) devices on the bus — cable the CX-7, then re-run"
    return
  fi
  for d in $devs; do
    local v
    v="$(rtc_value "$d")"
    if [ "$v" = "1" ]; then
      ok "$d REAL_TIME_CLOCK_ENABLE already 1"
    elif [ "$v" = "?" ]; then
      warn "$d REAL_TIME_CLOCK_ENABLE unknown — query needs root (re-run with sudo)"
    elif [ "$CHECK" = 1 ]; then
      warn "$d REAL_TIME_CLOCK_ENABLE = ${v} (needs 1 — run apply)"
    else
      if mlxconfig -y -d "$d" set REAL_TIME_CLOCK_ENABLE=1 >/dev/null 2>&1; then
        ok "$d set REAL_TIME_CLOCK_ENABLE=1"
        reboot_needed=1
      else
        fail "$d mlxconfig set failed"
      fi
    fi
  done
}

# --- 3. hugepages ---------------------------------------------------------------------------------
provision_hugepages() {
  hr "hugepages (${HUGEPAGES} x 2MB)"
  if [ ! -d "$HP_DIR" ]; then
    warn "no 2MB hugepage support at $HP_DIR"
    return
  fi
  local cur free
  cur="$(cat "$HP_DIR/nr_hugepages" 2>/dev/null || echo 0)"
  free="$(cat "$HP_DIR/free_hugepages" 2>/dev/null || echo 0)"
  if [ "$CHECK" = 1 ]; then
    [ "$cur" -ge "$HUGEPAGES" ] && ok "nr_hugepages=$cur (free=$free)" \
                                || warn "nr_hugepages=$cur < $HUGEPAGES"
    [ -f "$SYSCTL_FILE" ] && ok "persistent: $SYSCTL_FILE present" \
                          || warn "persistent config $SYSCTL_FILE missing"
    mountpoint -q /dev/hugepages && ok "/dev/hugepages mounted" || warn "/dev/hugepages not mounted"
    return
  fi
  # runtime
  echo "$HUGEPAGES" > "$HP_DIR/nr_hugepages" 2>/dev/null \
    && ok "allocated $(cat "$HP_DIR/nr_hugepages") pages now" || warn "runtime alloc failed"
  # persistent across reboot (vm.nr_hugepages sets the default 2MB pool)
  echo "# Spark Video Processor — DPDK/EAL hugepages (deploy/provision.sh)
vm.nr_hugepages = $HUGEPAGES" > "$SYSCTL_FILE" && ok "wrote $SYSCTL_FILE"
  sysctl -p "$SYSCTL_FILE" >/dev/null 2>&1 || true
  # mount
  mkdir -p /dev/hugepages
  mountpoint -q /dev/hugepages || mount -t hugetlbfs none /dev/hugepages 2>/dev/null || true
  mountpoint -q /dev/hugepages && ok "/dev/hugepages mounted" || warn "/dev/hugepages mount failed"
}

# --- 4. PTP host clock: NTP clients must not fight phc2sys ----------------------------------------
provision_ptp() {
  hr "PTP: NTP clients off (they fight phc2sys for CLOCK_REALTIME in slave mode)"
  if ! command -v systemctl >/dev/null; then
    warn "systemctl not present — skip NTP-client check"
    return
  fi
  local hit=0 s
  for s in systemd-timesyncd chronyd ntpd; do
    systemctl is-active --quiet "$s" 2>/dev/null || continue
    hit=1
    if [ "$CHECK" = 1 ]; then
      warn "$s active — will be disabled on apply (conflicts with phc2sys)"
    elif systemctl disable --now "$s" >/dev/null 2>&1; then
      ok "$s disabled"
    else
      warn "$s active but 'systemctl disable --now $s' failed"
    fi
  done
  [ "$hit" = 0 ] && ok "no NTP client active (phc2sys owns CLOCK_REALTIME)"
}

# --- run ------------------------------------------------------------------------------------------
need_root
printf 'Spark provisioning — mode: %s\n' "$([ "$CHECK" = 1 ] && echo CHECK || echo APPLY)"
provision_packages
provision_firmware
provision_hugepages
provision_ptp

hr "summary"
if [ "$CHECK" = 1 ]; then
  info "read-only check complete. Run 'sudo bash deploy/provision.sh' to apply any [warn] items."
elif [ "$reboot_needed" = 1 ]; then
  warn "REAL_TIME_CLOCK_ENABLE was changed — a COLD REBOOT is required for tx_pp to work."
  info "after reboot: bash deploy/provision.sh --check   should show all [ok]."
else
  ok "host provisioned (no reboot needed)."
fi
