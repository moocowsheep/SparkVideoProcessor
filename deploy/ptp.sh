#!/usr/bin/env bash
# Spark Video Processor — PTP discipline for the ConnectX-7 (M1 open item #3).
#
# Disciplines the shared real-time PHC (ptp0) with ptp4l + phc2sys so the engine's RTP timestamps and
# tx_pp pacing track a reference clock (ST 2110-10 / ST 2059). The engine reads the same PHC, so this
# needs no engine change. Loopback on one box doesn't need PTP (all ports share ptp0 — lock is
# trivial); this is for production interop with a facility grandmaster.
#
# Modes:
#   --check          read-only: HW-timestamp caps, /dev/ptp*, tools, config (run anytime, no root)
#   (default/slave)  ptp4l slaves the PHC to a network grandmaster + phc2sys syncs the system clock
#   --master         run this box AS the time source (no external GM) + phc2sys pushes system->PHC
#   --test           --master for ~15s, confirm ptp4l + HW timestamping work, then exit
# Interface: arg $2 or $IFACE; otherwise the first up ConnectX-7 port is auto-detected.
set -u

MODE="${1:-slave}"
CONF="$(dirname "$0")/ptp4l.conf"

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
      ok "CX-7 iface: $IFACE_RESOLVED"
      if command -v ethtool >/dev/null; then
        if ethtool -T "$IFACE_RESOLVED" 2>/dev/null | grep -q 'hardware-transmit'; then
          ok "$IFACE_RESOLVED: HW TX/RX timestamping (PTP-capable)"
        else
          warn "$IFACE_RESOLVED: no HW timestamping reported"
        fi
        ethtool -T "$IFACE_RESOLVED" 2>/dev/null | grep -iE 'PTP Hardware Clock' | sed 's/^/  [info] /'
      fi
    else
      warn "no up ConnectX-7 port found (cable it; the card is hot-plug)"
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
    info "to run: sudo bash deploy/ptp.sh [--master|--test|slave] [iface]"
    ;;

  --master)
    need_root
    [ -n "$IFACE_RESOLVED" ] || { fail "no CX-7 iface"; exit 1; }
    info "ptp4l MASTER on $IFACE_RESOLVED (this box is the time source); Ctrl-C to stop"
    phc2sys -s CLOCK_REALTIME -c "$IFACE_RESOLVED" -O 0 -w -m >/tmp/spark_phc2sys.log 2>&1 &
    exec ptp4l -f "$CONF" -i "$IFACE_RESOLVED" -m
    ;;

  --test)
    need_root
    [ -n "$IFACE_RESOLVED" ] || { fail "no CX-7 iface"; exit 1; }
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
    [ -n "$IFACE_RESOLVED" ] || { fail "no CX-7 iface"; exit 1; }
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

  *)
    fail "unknown mode '$MODE' (use --check | --master | --test | slave)"
    exit 2
    ;;
esac
