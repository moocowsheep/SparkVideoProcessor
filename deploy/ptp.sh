#!/usr/bin/env bash
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

# Spark Video Processor — PTP discipline for the ConnectX (CX-7 on the Spark, CX-6 Lx on x86_64; M1 open item #3).
#
# Disciplines the shared real-time PHC (ptp0) with ptp4l + phc2sys so the engine's RTP timestamps and
# tx_pp pacing track a reference clock (ST 2110-10 / ST 2059). The engine reads the same PHC, so this
# needs no engine change. Loopback on one box doesn't need PTP (all ports share ptp0 — lock is
# trivial); this is for production interop with a facility grandmaster.
#
# Modes:
#   --check          read-only: HW-timestamp caps, /dev/ptp*, tools, config (run anytime, no root)
#   (default/slave)  ptp4l slaves the PHC to a network grandmaster + phc2sys syncs the system clock
#   start            slave mode in the background (setsid), log to /tmp/spark_ptp.log
#   stop             stop a backgrounded start (ptp4l + phc2sys)
#   status           is it running? show the latest lock/offset lines
#   --master         run this box AS the time source (no external GM) + phc2sys pushes system->PHC
#   --test           --master for ~15s, confirm ptp4l + HW timestamping work, then exit
# Interface: arg $2 or $IFACE; otherwise the first up ConnectX-7 port is auto-detected.
#
# If the host already runs ptp4l itself (e.g. a systemd ptp4l-smpte.service slaved to a facility
# grandmaster — the x86_64 host does), that daemon owns the PHC and this script must stay out of its
# way: a second ptp4l on the same interface fights it, and --master/--test omit slaveOnly, so they
# could win BMCA against the real GM. The running modes therefore refuse to start on top of an
# externally-managed ptp4l; FORCE=1 overrides. --check and status are always safe.
set -u

MODE="${1:-slave}"
CONF="$(dirname "$0")/ptp4l.conf"
LOG=/tmp/spark_ptp.log
PIDFILE=/tmp/spark_ptp.pid

ok()   { printf '  [ok]   %s\n' "$1"; }
info() { printf '  [info] %s\n' "$1"; }
warn() { printf '  [warn] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; }

detect_iface() {
  [ -n "${IFACE:-}" ] && { echo "$IFACE"; return; }
  [ -n "${2:-}" ] && { echo "$2"; return; }
  for d in /sys/class/net/*; do
    local n; n="$(basename "$d")"
    [ "$(cat "$d/device/vendor" 2>/dev/null)" = "0x15b3" ] && [ "$(cat "$d/carrier" 2>/dev/null)" = "1" ] \
      && { echo "$n"; return; }
  done
}

IFACE_RESOLVED="$(detect_iface "$@")"

# PIDs of any ptp4l we did NOT start via this script's `start` (i.e. systemd- or hand-managed).
external_ptp4l_pids() {
  local ours=""
  [ -f "$PIDFILE" ] && ours="$(cat "$PIDFILE" 2>/dev/null)"
  for pid in $(pgrep -x ptp4l 2>/dev/null); do
    [ -n "$ours" ] && [ "$pid" = "$ours" ] && continue
    # a `start` execs ptp4l in the pidfile's process group; skip anything in that group
    [ -n "$ours" ] && [ "$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$ours" ] && continue
    echo "$pid"
  done
}

# Systemd unit owning a pid, if any (empty when not systemd-managed).
unit_of_pid() {
  command -v systemctl >/dev/null || return 0
  systemctl status "$1" 2>/dev/null | awk 'NR==1 && $2 ~ /\.service$/ {print $2; exit}'
}

report_external_ptp4l() {
  local pids; pids="$(external_ptp4l_pids)"
  [ -z "$pids" ] && return 1
  for pid in $pids; do
    local unit; unit="$(unit_of_pid "$pid")"
    info "host-managed ptp4l already running (pid $pid${unit:+, $unit}): $(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null)"
    if [ -n "$unit" ]; then
      journalctl -u "$unit" -n 3 --no-pager 2>/dev/null | sed 's/^/         /'
    fi
  done
  return 0
}

# Guard the modes that would put a second ptp4l on the wire.
refuse_if_external() {
  external_ptp4l_pids | grep -q . || return 0
  report_external_ptp4l
  if [ "${FORCE:-0}" = "1" ]; then
    warn "FORCE=1 — starting a SECOND ptp4l anyway; expect the two to fight over the PHC"
    return 0
  fi
  fail "refusing '$MODE': this host already disciplines the PHC itself (see above)."
  info "the engine reads that same PHC, so nothing more is needed — verify with: bash $0 --check"
  info "to override anyway: FORCE=1 sudo bash $0 $MODE"
  exit 1
}

need_root() {
  [ "$(id -u)" = 0 ] || { fail "must run as root for $MODE (use --check for read-only)"; exit 1; }
}

case "$MODE" in
  --check)
    printf 'PTP check\n'
    command -v ptp4l   >/dev/null && ok "ptp4l present"   || warn "ptp4l missing (deploy/provision.sh installs linuxptp)"
    command -v phc2sys >/dev/null && ok "phc2sys present" || warn "phc2sys missing"
    [ -f "$CONF" ] && ok "config $CONF" || warn "config $CONF missing"
    ls /dev/ptp* >/dev/null 2>&1 && ok "PHC device(s): $(ls /dev/ptp* | tr '\n' ' ')" \
                                 || warn "no /dev/ptp* (REAL_TIME_CLOCK_ENABLE set? see deploy/provision.sh)"
    if [ -n "$IFACE_RESOLVED" ]; then
      ok "ConnectX iface: $IFACE_RESOLVED"
      if command -v ethtool >/dev/null; then
        if ethtool -T "$IFACE_RESOLVED" 2>/dev/null | grep -q 'hardware-transmit'; then
          ok "$IFACE_RESOLVED: HW TX/RX timestamping (PTP-capable)"
        else
          warn "$IFACE_RESOLVED: no HW timestamping reported"
        fi
        ethtool -T "$IFACE_RESOLVED" 2>/dev/null | grep -iE 'PTP Hardware Clock' | sed 's/^/  [info] /'
      fi
    else
      warn "no up ConnectX port found (cable it; on the DGX Spark the card is hot-plug)"
    fi
    if report_external_ptp4l; then
      ok "PHC is disciplined by the host — do NOT run '$0 slave|start|--master' on top of it"
    else
      info "no host-managed ptp4l — use '$0 start' to discipline the PHC from here"
    fi
    # NTP clients fight phc2sys for CLOCK_REALTIME in slave mode (PHC/ptp4l unaffected, but the OS
    # wall-clock sawtooths). deploy/provision.sh disables them; warn here if any are active.
    if command -v systemctl >/dev/null; then
      ntp_active=""
      for s in systemd-timesyncd chronyd ntpd; do
        systemctl is-active --quiet "$s" 2>/dev/null && ntp_active="$ntp_active $s"
      done
      [ -n "$ntp_active" ] \
        && warn "NTP client active:$ntp_active — fights phc2sys (disable: systemctl disable --now systemd-timesyncd)" \
        || ok "no NTP client contending for CLOCK_REALTIME"
    fi
    info "to run: sudo bash deploy/ptp.sh [--master|--test|slave|start|stop|status] [iface]"
    ;;

  --master)
    need_root
    refuse_if_external
    [ -n "$IFACE_RESOLVED" ] || { fail "no ConnectX iface"; exit 1; }
    info "ptp4l MASTER on $IFACE_RESOLVED (this box is the time source); Ctrl-C to stop"
    phc2sys -s CLOCK_REALTIME -c "$IFACE_RESOLVED" -O 0 -w -m >/tmp/spark_phc2sys.log 2>&1 &
    exec ptp4l -f "$CONF" -i "$IFACE_RESOLVED" -m
    ;;

  --test)
    need_root
    refuse_if_external
    [ -n "$IFACE_RESOLVED" ] || { fail "no ConnectX iface"; exit 1; }
    info "ptp4l master-mode self-test on $IFACE_RESOLVED for 15s ..."
    LOG="$(mktemp)"
    timeout 15 ptp4l -f "$CONF" -i "$IFACE_RESOLVED" -m >"$LOG" 2>&1
    tail -n 15 "$LOG" | sed 's/^/    /'
    if grep -qiE 'assuming the grand master|to MASTER|selected local clock|MASTER' "$LOG"; then
      ok "ptp4l initialized + assumed MASTER on $IFACE_RESOLVED — PTP stack works (HW timestamping)"
    else
      fail "ptp4l did not reach MASTER — see log above ($LOG)"
    fi
    ;;

  slave|"")
    need_root
    refuse_if_external
    [ -n "$IFACE_RESOLVED" ] || { fail "no ConnectX iface"; exit 1; }
    info "ptp4l SLAVE (slaveOnly) on $IFACE_RESOLVED (disciplines PHC to network grandmaster); Ctrl-C to stop"
    # phc2sys -a follows ptp4l over its UDS management socket, which is domain-scoped — it MUST use the
    # same domainNumber as ptp4l or it hangs forever at "Waiting for ptp4l..." and never syncs the
    # system clock. Pull the domain from CONF so the two can't drift apart.
    DOMAIN="$(awk '/^[[:space:]]*domainNumber/{print $2; exit}' "$CONF")"
    phc2sys -a -r -n "${DOMAIN:-0}" -m >/tmp/spark_phc2sys.log 2>&1 &   # auto-follow ptp4l; also sync CLOCK_REALTIME
    # -s = slaveOnly: a media slave node must NEVER win BMCA and become the facility grandmaster if the
    # real GM drops out. (--master / --test deliberately omit this.)
    exec ptp4l -f "$CONF" -i "$IFACE_RESOLVED" -s -m
    ;;

  start)
    need_root
    refuse_if_external
    [ -n "$IFACE_RESOLVED" ] || { fail "no ConnectX iface"; exit 1; }
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      info "already running (pid $(cat "$PIDFILE"))"; exit 0
    fi
    # setsid puts bash (-> exec ptp4l, same pid) and its backgrounded phc2sys in a fresh process
    # group whose pgid == that pid, so stop can take both down with one kill -- -PGID.
    setsid bash "$0" slave "$IFACE_RESOLVED" </dev/null >>"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    ok "started PTP slave on $IFACE_RESOLVED (pid $(cat "$PIDFILE"), log: $LOG, phc2sys: /tmp/spark_phc2sys.log)"
    info "watch lock: bash deploy/ptp.sh status"
    ;;

  stop)
    need_root
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      kill -- -"$(cat "$PIDFILE")" 2>/dev/null || kill "$(cat "$PIDFILE")" 2>/dev/null
      rm -f "$PIDFILE"
      ok "stopped ptp4l + phc2sys"
    else
      rm -f "$PIDFILE"
      pkill -x ptp4l 2>/dev/null && info "no pidfile; killed stray ptp4l" || info "not running"
      pkill -x phc2sys 2>/dev/null || true
    fi
    ;;

  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      ok "ptp4l running (pid $(cat "$PIDFILE"))"
    elif report_external_ptp4l; then
      ok "PHC disciplined by a host-managed ptp4l (not by '$0 start') — that is the expected setup
         on a box with its own PTP service"
    else
      fail "ptp4l not running"
    fi
    pgrep -x phc2sys >/dev/null && ok "phc2sys running" || warn "phc2sys not running (system clock not synced)"
    if [ -f "$LOG" ]; then
      echo "  -- last ptp4l lock/offset lines ($LOG):"
      grep -E 'master offset|port 1|selected|FAULT' "$LOG" | tail -n 6 | sed 's/^/    /'
    fi
    ;;

  *)
    fail "unknown mode '$MODE' (use --check | --master | --test | slave | start | stop | status)"
    exit 2
    ;;
esac
