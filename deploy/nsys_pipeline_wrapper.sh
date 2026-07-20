#!/usr/bin/env bash
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

# nsys_pipeline_wrapper.sh — stand-in for engine/build/st2110_pipeline that captures a CPU-side
# trace alongside the engine, for chasing the residual ~1/10min pipeline stall (M8 determinism).
#
# Point the daemon at it:
#   sudo ./control/build/spark_controld --pipeline deploy/nsys_pipeline_wrapper.sh --web web
#
# nsys CANNOT be used here: its injection breaks DPDK EAL hugepage init ("Couldn't get fd on
# hugepage file" -> rte_service_init failure; verified 2026-07-02 — the same wrapper with nsys
# bypassed starts clean). Instead this runs a root SYSTEM-WIDE `perf record` (kernel events, no
# process injection) for 600s beside the engine: CPU samples + context-switch events, monotonic
# clock so the engine's "pipe stall ... (mono T)" warn lines index directly into the trace.
#
# Analyze around a stall at mono time T (thread names start with 'st2110' / worker pools):
#   perf script -i /tmp/spark_stall_perf.data --time <T-0.4>,<T+0.1> | less
set -u
REAL="$(dirname "$0")/../engine/build/st2110_pipeline"
[ -x "$REAL.real" ] && REAL="$REAL.real"   # support the moved-binary install mode
ulimit -s 32768 || true

# One capture at a time: engine restarts re-run this wrapper; don't stack perf sessions.
# (ps-state check, not pgrep: a finished capture leaves a zombie perf under the exec'd engine.)
if ! ps -C perf -o stat= 2>/dev/null | grep -qv '^Z'; then
  setsid perf record -a -g -F 99 --switch-events -k monotonic \
    -o /tmp/spark_stall_perf.data -- sleep 600 >/tmp/spark_perf.log 2>&1 &
  echo "[wrapper] perf capture started (600s, -> /tmp/spark_stall_perf.data)" >&2
else
  echo "[wrapper] perf already capturing — not starting another" >&2
fi

exec "$REAL"
