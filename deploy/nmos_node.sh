#!/usr/bin/env bash
# nmos_node.sh — launch the Spark NMOS Node (control/build/spark_nmos_node) pinned to the
# facility registry. The node registers its IS-04 Senders/Receivers there and exposes IS-05
# connection management so a controller (or our dashboard) can route to/from us.
#
# Registry selection in nmos-cpp is DNS-SD-first: the node browses _nmos-register._tcp over
# mDNS and uses whatever it finds. We ALSO pin the registry statically (registry_address +
# registration_port + registry_version) so registration still works if mDNS is unavailable —
# both point at the same external registry, so the target is deterministic either way.
#
#   bash deploy/nmos_node.sh start    # launch in background (setsid), log to /tmp/spark_nmos_node.log
#   bash deploy/nmos_node.sh run      # launch in foreground
#   bash deploy/nmos_node.sh stop     # stop it
#   bash deploy/nmos_node.sh status   # is it registered in the registry? is its API up?
#
# Override anything via env, e.g.:
#   REGISTRY_HOST=192.168.18.41 REGISTRY_PORT=8010 HOST_ADDR=192.168.18.101 bash deploy/nmos_node.sh start
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE="$ROOT/control/build/spark_nmos_node"

# ---- config (env-overridable) ----
REGISTRY_HOST="${REGISTRY_HOST:-192.168.18.41}"   # facility registry (cc-pi400)
REGISTRY_PORT="${REGISTRY_PORT:-8010}"            # its registration API port
REGISTRY_VER="${REGISTRY_VER:-v1.3}"
HOST_ADDR="${HOST_ADDR:-192.168.18.101}"          # this box's media/mgmt IP the node advertises
HTTP_PORT="${HTTP_PORT:-3242}"                     # multiplexes all of the node's APIs onto one port
GMID="${GMID:-34-84-e4-ff-fe-aa-d9-88}"           # PTP grandmaster the node advertises on clk0
CONTROL_URL="${CONTROL_URL:-http://127.0.0.1:8080}" # the engine control daemon (IS-05 -> engine)
LOG_LEVEL="${LOG_LEVEL:-0}"
LOG="/tmp/spark_nmos_node.log"

settings_json() {
  cat <<JSON
{
  "logging_level": $LOG_LEVEL,
  "http_port": $HTTP_PORT,
  "host_address": "$HOST_ADDR",
  "registry_address": "$REGISTRY_HOST",
  "registration_port": $REGISTRY_PORT,
  "registry_version": "$REGISTRY_VER",
  "spark_ptp_gmid": "$GMID",
  "spark_ptp_traceable": true,
  "spark_ptp_locked": true,
  "spark_control_url": "$CONTROL_URL"
}
JSON
}

need_node() {
  [ -x "$NODE" ] || { echo "spark_nmos_node not built at $NODE (cmake -DSPARK_WITH_NMOS=ON)" >&2; exit 1; }
}

case "${1:-}" in
  run)
    need_node
    echo "registry: http://$REGISTRY_HOST:$REGISTRY_PORT/x-nmos/registration/$REGISTRY_VER  node API: http://$HOST_ADDR:$HTTP_PORT"
    exec "$NODE" "$(settings_json)"
    ;;
  start)
    need_node
    if pgrep -x spark_nmos_node >/dev/null; then echo "already running (pid $(pgrep -x spark_nmos_node | tr '\n' ' '))"; exit 0; fi
    setsid "$NODE" "$(settings_json)" >"$LOG" 2>&1 < /dev/null & disown
    echo "started spark_nmos_node -> registry http://$REGISTRY_HOST:$REGISTRY_PORT (log: $LOG)"
    ;;
  stop)
    pkill -x spark_nmos_node && echo "stopped" || echo "not running"
    ;;
  status)
    echo "== node self API (http://$HOST_ADDR:$HTTP_PORT) =="
    curl -s --max-time 4 "http://$HOST_ADDR:$HTTP_PORT/x-nmos/node/v1.3/self" \
      | grep -o '"label":"[^"]*"' | head -1 || echo "node API unreachable"
    echo "== are we in the registry (http://$REGISTRY_HOST:$REGISTRY_PORT)? =="
    curl -s --max-time 4 "http://$REGISTRY_HOST:$REGISTRY_PORT/x-nmos/query/$REGISTRY_VER/nodes" \
      | grep -o '"href":"[^"]*"' || echo "registry unreachable"
    ;;
  *)
    echo "usage: $0 {run|start|stop|status}"; exit 2;;
esac
