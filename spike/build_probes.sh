#!/usr/bin/env bash
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

# Build + run the M0 gate probes (OFA optical flow, NPP resize). No sudo required.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
NVOF_INC="third_party/NVIDIAOpticalFlowSDK"
if [ ! -f "$NVOF_INC/nvOpticalFlowCuda.h" ]; then
  echo "== fetching NVIDIA Optical Flow SDK headers =="
  bash fetch_nvof_sdk.sh
fi

# The driver libs live under the multiarch triplet dir, which differs by platform
# (aarch64-linux-gnu on the DGX Spark, x86_64-linux-gnu on an x86 host). Ask gcc for the local
# triplet instead of hardcoding one, and fall back to uname if that ever comes up empty.
TRIPLET="$(gcc -dumpmachine 2>/dev/null || true)"
[ -n "$TRIPLET" ] || TRIPLET="$(uname -m)-linux-gnu"
NVOF_LIBDIR=""
for d in "/lib/$TRIPLET" "/usr/lib/$TRIPLET"; do
  if ls "$d"/libnvidia-opticalflow.so* >/dev/null 2>&1; then NVOF_LIBDIR="$d"; break; fi
done
if [ -z "$NVOF_LIBDIR" ]; then
  echo "ERROR: libnvidia-opticalflow.so not found under /lib/$TRIPLET or /usr/lib/$TRIPLET" >&2
  echo "       (it ships with the NVIDIA driver — is the driver installed?)" >&2
  exit 1
fi
echo "== NVOF driver lib: $NVOF_LIBDIR (triplet $TRIPLET) =="

# Some driver installs ship only libnvidia-opticalflow.so.1 with no unversioned .so symlink, so
# `-lnvidia-opticalflow` cannot resolve. Link the versioned file by path when that is the case.
NVOF_LINK="-lnvidia-opticalflow"
[ -e "$NVOF_LIBDIR/libnvidia-opticalflow.so" ] || NVOF_LINK="$NVOF_LIBDIR/libnvidia-opticalflow.so.1"

echo "== building nvof_probe =="
g++ -O2 -std=c++17 -I "$NVOF_INC" -I/usr/local/cuda/include \
    nvof_probe.cpp -o build/nvof_probe \
    -L"$NVOF_LIBDIR" -lcuda $NVOF_LINK

echo "== building npp_resize_test =="
nvcc -O2 -std=c++17 npp_resize_test.cu -o build/npp_resize_test \
    -lnppig -lnppisu -lnppc -lcudart

echo "== running nvof_probe =="
./build/nvof_probe || echo "nvof_probe exit=$?"
echo
echo "== running npp_resize_test =="
./build/npp_resize_test || echo "npp_resize_test exit=$?"
