#!/usr/bin/env bash
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

# Build + run the M0 gate probes (OFA optical flow, NPP resize). No sudo required.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
NVOF_INC="third_party/NVIDIAOpticalFlowSDK"

echo "== building nvof_probe =="
g++ -O2 -std=c++17 -I "$NVOF_INC" -I/usr/local/cuda/include \
    nvof_probe.cpp -o build/nvof_probe \
    -L/lib/aarch64-linux-gnu -lcuda -lnvidia-opticalflow

echo "== building npp_resize_test =="
nvcc -O2 -std=c++17 npp_resize_test.cu -o build/npp_resize_test \
    -lnppig -lnppisu -lnppc -lcudart

echo "== running nvof_probe =="
./build/nvof_probe || echo "nvof_probe exit=$?"
echo
echo "== running npp_resize_test =="
./build/npp_resize_test || echo "npp_resize_test exit=$?"
