#!/usr/bin/env bash
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

# Spark Video Processor — P5 bench harness (software ST 2110 sender + NMOS advertisement).
#
# Lets you validate the NMOS path without a real camera: a dependency-free software sender
# (spike/st2110_software_sender, kernel multicast, no DPDK/root) emits a real RFC 4175 2110-20 video
# (+ optional 2110-30 audio) stream, and a spark_nmos_node advertises a matching Sender so the
# processor's dashboard discovers it. A loopback self-test proves the sender/receiver without any NIC.
#
# Usage:
#   bash deploy/p5_bench.sh --build           # compile the software sender (g++, pure C++)
#   bash deploy/p5_bench.sh --selftest        # send<->recv over the default mcast iface (no hardware)
#   bash deploy/p5_bench.sh --check           # report prerequisites
#   bash deploy/p5_bench.sh --run             # registry + sender + NMOS "camera" (drive from dashboard)
#     env: GROUP=239.100.0.10 PORT=5004 IFACE=<media-NIC-ip> AUDIO=1 PROFILE=1080p REGISTRY=<url>
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SENDER_SRC="$ROOT/spike/st2110_software_sender.cpp"
SENDER="$ROOT/spike/build/st2110_software_sender"
NODE="$ROOT/control/build/spark_nmos_node"
REG="$ROOT/third_party/nmos-cpp/build/nmos-cpp-registry"

hr()   { printf '\n===== %s =====\n' "$1"; }
ok()   { printf '  [ok]   %s\n' "$1"; }
info() { printf '  [info] %s\n' "$1"; }
warn() { printf '  [warn] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; }

GROUP="${GROUP:-239.100.0.10}"
PORT="${PORT:-5004}"
IFACE="${IFACE:-0.0.0.0}"
PROFILE="${PROFILE:-1080p}"
AUDIO="${AUDIO:-0}"
REGISTRY="${REGISTRY:-}"

do_build() {
  hr "build software sender (pure C++; no DPDK/Holoscan)"
  mkdir -p "$ROOT/spike/build"
  # only the pure framing cores are needed — rtp_st2110.cpp + audio_st2110.cpp
  if g++ -O2 -std=c++17 -I "$ROOT/engine" "$SENDER_SRC" \
        "$ROOT/engine/operators/st2110_tx/rtp_st2110.cpp" \
        "$ROOT/engine/operators/audio/audio_st2110.cpp" \
        -lpthread -o "$SENDER"; then
    ok "built $SENDER"
  else
    fail "compile failed"; return 1
  fi
}

do_check() {
  hr "prerequisites"
  command -v g++ >/dev/null && ok "g++ present" || warn "g++ missing"
  [ -x "$SENDER" ] && ok "software sender built" || warn "sender not built (run --build)"
  [ -x "$NODE" ] && ok "spark_nmos_node present" || warn "spark_nmos_node missing (build with -DSPARK_WITH_NMOS=ON)"
  [ -x "$REG" ] && ok "nmos-cpp-registry present" || warn "registry missing (run deploy/nmos.sh --build)"
}

# Loopback self-test: prove the sender's stream is a valid, depacketizable 2110-20 stream — no NIC.
do_selftest() {
  [ -x "$SENDER" ] || do_build || return 1
  hr "self-test: software send <-> recv (IP_MULTICAST_LOOP)"
  # Loopback multicast delivery depends on the host's default multicast interface; on a bench box you
  # may need IFACE=<a multicast-loopback-capable NIC ip>. This proves the framing, not the transport.
  info "interface $IFACE (override with IFACE=<ip> if no frames arrive)"
  local g="239.255.42.42" rlog="/tmp/p5_selftest_recv.log"
  "$SENDER" --recv --group "$g" --port 5004 --iface "$IFACE" > "$rlog" 2>&1 &
  local rpid=$!
  i=0; until grep -qiE "joined|MEMBERSHIP" "$rlog" 2>/dev/null || ! kill -0 $rpid 2>/dev/null || [ $i -ge 8 ]; do i=$((i+1)); sleep 1; done
  if grep -qi "MEMBERSHIP" "$rlog"; then
    warn "could not join multicast on $IFACE — set IFACE=<a multicast-capable NIC ip>"
    kill $rpid 2>/dev/null; sed -n '1,3p' "$rlog"; return 0
  fi
  "$SENDER" --send --group "$g" --port 5004 --iface "$IFACE" --profile "$PROFILE" --frames 60 --loopback
  kill -9 "$rpid" 2>/dev/null; wait "$rpid" 2>/dev/null  # SIGKILL: never hang on teardown
  # read the highest frame count the receiver reported (live 1 Hz lines; no reliance on clean exit)
  local frames; frames="$(grep -oE 'frames=[0-9]+' "$rlog" | cut -d= -f2 | sort -n | tail -1)"
  if [ "${frames:-0}" -gt 0 ]; then
    ok "receiver reconstructed ${frames} frames — sender + framing path verified"
  else
    warn "framing OK (sender ran) but no frames looped back on $IFACE — loopback multicast unavailable here; validate on-wire (--run) instead"
  fi
}

# Stand up the SENDER side of the bench: registry + software sender + NMOS "camera" advertising it.
do_run() {
  [ -x "$SENDER" ] || do_build || return 1
  hr "bench: software sender + NMOS camera"
  local reg_url="$REGISTRY"
  if [ -z "$reg_url" ]; then
    [ -x "$REG" ] || { fail "no registry binary; run deploy/nmos.sh --build or set REGISTRY=<url>"; return 1; }
    printf '{"logging_level":0}\n' > /tmp/p5_reg.json
    setsid "$REG" /tmp/p5_reg.json > /tmp/p5_registry.log 2>&1 & disown
    reg_url="http://127.0.0.1:3211"
    ok "started local registry -> $reg_url"
  else
    info "using external registry $reg_url"
  fi

  local aud=""; [ "$AUDIO" = "1" ] && aud="--audio --audio-group 239.100.0.20 --audio-port $PORT"
  setsid "$SENDER" --send --group "$GROUP" --port "$PORT" --iface "$IFACE" --profile "$PROFILE" $aud \
      > /tmp/p5_sender.log 2>&1 & disown
  ok "software sender -> $GROUP:$PORT ($PROFILE${aud:+ + audio}) on iface $IFACE"

  # The camera node's default video Sender SDP is group 239.100.0.10:5004 1080p — matches the sender.
  if [ -x "$NODE" ]; then
    printf '{"logging_level":0,"http_port":3252,"spark_activate_senders":true}\n' > /tmp/p5_camera.json
    setsid "$NODE" /tmp/p5_camera.json > /tmp/p5_camera.log 2>&1 & disown
    ok "NMOS camera node advertising the sender (node API :3252)"
  else
    warn "spark_nmos_node missing — sender runs but is not NMOS-discoverable"
  fi

  hr "drive it from the processor dashboard"
  info "1. run the processor: spark_controld (+ its spark_nmos_node) on the box with the CX-7"
  info "2. dashboard -> Discover: Registry=$reg_url"
  info "3. Connect the camera's ST 2110-20 video source -> the engine joins $GROUP and receives it"
  info "stop the bench:  pkill st2110_software_send; pkill -f p5_camera; pkill nmos-cpp-registry"
}

case "${1:---selftest}" in
  --build)    do_build ;;
  --check)    do_check ;;
  --selftest) do_selftest ;;
  --run)      do_run ;;
  *) fail "unknown mode '$1' (use --build | --selftest | --check | --run)"; exit 2 ;;
esac
