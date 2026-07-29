#!/usr/bin/env bash
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

# Install the source-tree systemd units:
#   spark-ptp.service   ptp4l slave + phc2sys (deploy/ptp.sh slave)
#   spark.service       control daemon + engine + NMOS node (deploy/spark_run.sh)
#
# The units run straight out of THIS checkout — @SPARK_ROOT@ is substituted with its path — so a
# rebuild takes effect on the next restart. (The .deb ships its own units under deploy/debian/.)
#
#   sudo bash deploy/systemd/install.sh              # install + enable at boot (does not start)
#   sudo bash deploy/systemd/install.sh --start      # ... and start both now
#   sudo bash deploy/systemd/install.sh --uninstall  # stop, disable, remove
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/deploy/systemd"
DEST=/etc/systemd/system
UNITS=(spark-ptp.service spark.service)

[ "$(id -u)" = 0 ] || { echo "must run as root: sudo bash deploy/systemd/install.sh $*" >&2; exit 1; }

if [ "${1:-}" = "--uninstall" ]; then
  systemctl disable --now "${UNITS[@]}" 2>/dev/null || true
  rm -f "${UNITS[@]/#/$DEST/}"
  systemctl daemon-reload
  echo "removed ${UNITS[*]}"
  exit 0
fi

for u in "${UNITS[@]}"; do
  sed "s|@SPARK_ROOT@|$ROOT|g" "$SRC/$u" > "$DEST/$u"
  echo "installed $DEST/$u (SPARK_ROOT=$ROOT)"
done
systemctl daemon-reload
systemctl enable "${UNITS[@]}"

# phc2sys owns CLOCK_REALTIME while spark-ptp runs; an NTP client fighting it sawtooths the wall
# clock. Conflicts= in the unit stops them at start, but they'd come back at boot if left enabled.
for s in systemd-timesyncd chrony chronyd ntp ntpsec; do
  if systemctl is-enabled --quiet "$s" 2>/dev/null; then
    echo "note: $s is enabled and contends with phc2sys — disable it: systemctl disable --now $s"
  fi
done

if [ "${1:-}" = "--start" ]; then
  systemctl start spark-ptp.service
  systemctl start spark.service
  systemctl --no-pager --lines=5 status "${UNITS[@]}" || true
else
  echo
  echo "enabled at boot. Start now with:"
  echo "    systemctl start spark-ptp spark"
  echo "Logs: journalctl -fu spark-ptp   /   journalctl -fu spark"
fi
