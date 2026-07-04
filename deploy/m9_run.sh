#!/bin/bash
# M9 picture-check runner: (re)launch the pipeline with one filter variant (root; run via sudo).
# Kills any running st2110_pipeline, locks GPU clocks, launches the variant, logs to
# /tmp/spark_m9_<variant>.log. BD2 routing is separate (bd2_route_sr.sh — 2160p variants use the
# default 2160p29.97 SDP, frc60 uses tx_1080p5994_raw.sdp).
#
# Usage: sudo bash m9_run.sh cubic|sr|sr-s|espcn|sharpen|procamp|frc60
set -e
V="${1:?variant: cubic|sr|sr-s|espcn|sharpen|procamp|frc60}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"

# BMD-1 1080p29.97 in -> BD2 monitor group out; L=105 (validated), run until killed/timeout.
COMMON="SPARK_LATENCY_MS=105 SPARK_RX_PCI=0000:01:00.0 SPARK_RX_IFACE=192.168.18.103
SPARK_RX_MCAST=239.255.194.137 SPARK_RX_PORT=16388 SPARK_RX_SRC=192.168.18.196
SPARK_IN_W=1920 SPARK_IN_H=1080 SPARK_IN_FPS=30000/1001 SPARK_IP10=0
SPARK_TX_PCI=0002:01:00.0 SPARK_TX_SRC=192.168.18.104 SPARK_TX_MCAST=239.100.0.10
SPARK_TX_PORT=20000 SPARK_FRAMES=1000000"
UP="SPARK_FRC=0 SPARK_OUT_W=3840 SPARK_OUT_H=2160"  # 1080 -> 2160p29.97 upscale variants

case "$V" in
  cubic)   EXTRA="$UP SPARK_INTERP=cubic" ;;                              # SR A/B baseline
  sr)      EXTRA="$UP SPARK_INTERP=fsrcnn" ;;                             # AI x2 SR (best)
  sr-s)    EXTRA="$UP SPARK_INTERP=fsrcnn-s" ;;                           # AI x2 SR (fast)
  espcn)   EXTRA="$UP SPARK_INTERP=espcn" ;;                              # AI x2 SR (alt look)
  sharpen) EXTRA="$UP SPARK_INTERP=cubic SPARK_SHARPEN=${2:-0.8}" ;;      # unsharp on cubic; amount = arg 2
  procamp) EXTRA="$UP SPARK_INTERP=cubic SPARK_PA_SAT=1.6 SPARK_PA_HUE=30 SPARK_PA_BRIGHT=0.05" ;;
  frc60)   EXTRA="SPARK_FRC=3 SPARK_OUT_W=1920 SPARK_OUT_H=1080 SPARK_INTERP=auto" ;;  # 59.94 up-convert
  *) echo "unknown variant: $V" >&2; exit 1 ;;
esac

nvidia-smi -lgc 3003 >/dev/null 2>&1 || true
P=$(pidof st2110_pipeline || true)
if [ -n "$P" ]; then kill -INT $P; sleep 3; fi
LOG="/tmp/spark_m9_${V}.log"
# shellcheck disable=SC2086
nohup timeout -s INT 3600 env $COMMON $EXTRA "$DIR/engine/build/st2110_pipeline" > "$LOG" 2>&1 &
echo "launched variant=$V log=$LOG"
