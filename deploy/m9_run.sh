#!/bin/bash
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

# M9 picture-check runner: (re)launch the pipeline with one filter variant (root; run via sudo).
# Kills any running st2110_pipeline, locks GPU clocks, launches the variant, logs to
# /tmp/spark_m9_<variant>.log. BD2 routing is separate (bd2_route_sr.sh — 2160p variants use the
# default 2160p29.97 SDP, frc60 uses tx_1080p5994_raw.sdp).
#
# Usage: sudo bash m9_run.sh cubic|sr|sr-s|espcn|sharpen|procamp|frc60
set -e
V="${1:?variant: cubic|sr|sr-s|espcn|sharpen|procamp|frc60}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
# Site-local addressing (gitignored). Everything below defaults to the IETF documentation
# ranges, so a fresh checkout is inert until this supplies real values. See lab.env.example.
# shellcheck disable=SC1091
[ -f "$DIR/deploy/lab.env" ] && . "$DIR/deploy/lab.env"

# BMD-1 1080p29.97 in -> BD2 monitor group out; L=105 (validated), run until killed/timeout.
# RX_SRC = BMD-1 (DHCP moved it .196 -> .104 on 2026-07-06). TX_SRC must match the NMOS node's
# sender manifest, which now advertises .110: the node gained a separate sender leg bound to the TX
# port (spark_tx_address), where it used to bind both legs to host_address .103. A mismatch here is
# invisible from this end — SSM receivers like the BiDirect drop every packet while the engine
# reports a clean paced stream.
COMMON="SPARK_LATENCY_MS=105 SPARK_RX_PCI=0000:01:00.0 SPARK_RX_IFACE=192.0.2.103
SPARK_RX_MCAST=239.255.194.137 SPARK_RX_PORT=16388 SPARK_RX_SRC=${BMD1_IP:-192.0.2.104}
SPARK_IN_W=1920 SPARK_IN_H=1080 SPARK_IN_FPS=30000/1001 SPARK_IP10=0
SPARK_TX_PCI=0002:01:00.0 SPARK_TX_SRC=${SPARK_TX_SRC:-192.0.2.110} SPARK_TX_MCAST=239.100.0.10
SPARK_TX_PORT=20000 SPARK_FRAMES=1000000"
UP="SPARK_FRC=0 SPARK_OUT_W=3840 SPARK_OUT_H=2160"  # 1080 -> 2160p29.97 upscale variants

case "$V" in
  cubic)   EXTRA="$UP SPARK_INTERP=cubic" ;;                              # SR A/B baseline
  sr)      EXTRA="$UP SPARK_INTERP=fsrcnn" ;;                             # AI x2 SR (best)
  sr-s)    EXTRA="$UP SPARK_INTERP=fsrcnn-s" ;;                           # AI x2 SR (fast)
  espcn)   EXTRA="$UP SPARK_INTERP=espcn" ;;                              # AI x2 SR (alt look)
  sharpen) EXTRA="$UP SPARK_INTERP=cubic SPARK_SHARPEN=${2:-0.8}" ;;      # unsharp on cubic; amount = arg 2
  procamp) EXTRA="$UP SPARK_INTERP=cubic SPARK_PA_SAT=1.6 SPARK_PA_HUE=30 SPARK_PA_BRIGHT=0.05" ;;
  frc60)   EXTRA="SPARK_FRC=3 SPARK_OUT_W=1920 SPARK_OUT_H=1080 SPARK_INTERP=auto" ;;  # 59.94 up-convert (uniform grid)
  frc60-flow) EXTRA="SPARK_FRC=2 SPARK_OUT_W=1920 SPARK_OUT_H=1080 SPARK_INTERP=auto" ;;  # 59.94 optical-flow up-convert;
             # knob A/Bs ride the inherited env: sudo SPARK_FRC_MEDIAN=0 bash m9_run.sh frc60-flow (also _COST, _TEMPORAL_HINTS, SPARK_FRC_GRID=4|2|1)
  *) echo "unknown variant: $V" >&2; exit 1 ;;
esac

nvidia-smi -lgc 3003 >/dev/null 2>&1 || true
P=$(pidof st2110_pipeline || true)
if [ -n "$P" ]; then kill -INT $P; sleep 3; fi
LOG="/tmp/spark_m9_${V}.log"
# shellcheck disable=SC2086
nohup timeout -s INT 3600 env $COMMON $EXTRA "$DIR/engine/build/st2110_pipeline" > "$LOG" 2>&1 &
echo "launched variant=$V log=$LOG"
