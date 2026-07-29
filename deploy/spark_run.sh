#!/usr/bin/env bash
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

# spark_run.sh — run the media plane: the control daemon (which owns/spawns the ST 2110 engine)
# and the NMOS node, as one foreground process for systemd (deploy/systemd/spark.service).
#
# The two are one failure domain: the node's IS-05 endpoints route the engine through the daemon's
# control API, so a node registered against a dead daemon is worse than no node at all. This script
# therefore starts the daemon, waits for its HTTP API to answer, starts the node, and exits as soon
# as EITHER exits — letting systemd's Restart= bring the pair back up together.
#
#   bash deploy/spark_run.sh          # foreground, both processes
#
# Env (all optional; deploy/lab.env supplies the site addressing for the node):
#   CONTROLD, PIPELINE, WEB   paths into the build tree (defaults below)
#   HTTP_PORT_CTRL            daemon HTTP/dashboard port (default 8080)
#   SPARK_NO_NMOS=1           run the daemon only (skip the node)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROLD="${CONTROLD:-$ROOT/control/build/spark_controld}"
PIPELINE="${PIPELINE:-$ROOT/engine/build/st2110_pipeline}"
WEB="${WEB:-$ROOT/web}"
HTTP_PORT_CTRL="${HTTP_PORT_CTRL:-8080}"

[ -x "$CONTROLD" ] || { echo "spark_controld not built at $CONTROLD" >&2; exit 1; }
[ -x "$PIPELINE" ] || { echo "st2110_pipeline not built at $PIPELINE" >&2; exit 1; }

CTRL_PID=""; NODE_PID=""
shutdown() {
  trap - TERM INT
  [ -n "$NODE_PID" ] && kill "$NODE_PID" 2>/dev/null
  [ -n "$CTRL_PID" ] && kill "$CTRL_PID" 2>/dev/null
  wait 2>/dev/null
  exit 0
}
trap shutdown TERM INT

echo "starting spark_controld (dashboard http://0.0.0.0:$HTTP_PORT_CTRL/)"
"$CONTROLD" --pipeline "$PIPELINE" --web "$WEB" --http "$HTTP_PORT_CTRL" &
CTRL_PID=$!

if [ "${SPARK_NO_NMOS:-0}" != 1 ]; then
  # The node's IS-05 handlers call the daemon on CONTROL_URL from the moment it registers, so don't
  # launch it until that API answers. 30s covers the daemon's DPDK/EAL + GPU init on a cold start.
  for _ in $(seq 1 60); do
    kill -0 "$CTRL_PID" 2>/dev/null || { echo "spark_controld exited during startup" >&2; wait "$CTRL_PID"; exit 1; }
    curl -sf --max-time 1 "http://127.0.0.1:$HTTP_PORT_CTRL/api/status" >/dev/null && break
    sleep 0.5
  done
  echo "starting spark_nmos_node"
  CONTROL_URL="http://127.0.0.1:$HTTP_PORT_CTRL" bash "$ROOT/deploy/nmos_node.sh" run &
  NODE_PID=$!
fi

# Exit on the first child to die; systemd restarts the pair.
wait -n
RC=$?
echo "a spark process exited (rc=$RC) — stopping the other" >&2
shutdown_rc=$RC
[ -n "$NODE_PID" ] && kill "$NODE_PID" 2>/dev/null
[ -n "$CTRL_PID" ] && kill "$CTRL_PID" 2>/dev/null
wait 2>/dev/null
exit "$shutdown_rc"
