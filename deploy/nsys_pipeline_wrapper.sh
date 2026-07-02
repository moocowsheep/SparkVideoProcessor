#!/usr/bin/env bash
# nsys_pipeline_wrapper.sh — stand-in for engine/build/st2110_pipeline that launches the real
# binary under Nsight Systems, for chasing the episodic 100-250ms pipeline stalls (M8 determinism).
#
# Usage: point the daemon at this script instead of the binary (or temporarily move the binary to
# st2110_pipeline.real and copy this script into its place), then start the engine as usual:
#   sudo ./control/build/spark_controld --pipeline deploy/nsys_pipeline_wrapper.sh --web web
#
# Collects 240s of CUDA + NVTX + OS-runtime traces, writes /tmp/spark_stall_<pid>.nsys-rep, and
# leaves the engine RUNNING (--kill none). Analyze with:
#   nsys stats --report cuda_gpu_kern_sum,osrt_sum /tmp/spark_stall_*.nsys-rep
# and look for the stall window in the timeline GUI (gaps in kernel launches vs long osrt waits).
set -u
REAL="$(dirname "$0")/../engine/build/st2110_pipeline"
[ -x "$REAL.real" ] && REAL="$REAL.real"   # support the moved-binary install mode
exec /usr/local/cuda/bin/nsys profile \
  --trace=cuda,nvtx,osrt \
  --duration=240 --kill=none \
  --output="/tmp/spark_stall_$$" --force-overwrite=true \
  "$REAL"
