#!/usr/bin/env bash
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

# Vendor the NVIDIA Optical Flow SDK headers into spike/third_party/.
#
# `spark_frc` (engine/operators/frc) drives the raw NVOF CUDA API directly — NvOFFRUC has no
# aarch64 build, so FRC is hand-rolled on the flow field and we keep that same path on x86 so both
# targets run identical code. That needs the SDK's public headers (nvOpticalFlowCuda.h /
# nvOpticalFlowCommon.h). They are header-only interface declarations; the implementation is in the
# driver's libnvidia-opticalflow.so, already installed with the NVIDIA driver on both platforms.
#
# third_party/ is gitignored (vendored SDKs are not committed), so this runs once per checkout.
# Idempotent — re-running is a no-op if the headers are already present.
#
#   bash spike/fetch_nvof_sdk.sh
set -euo pipefail
cd "$(dirname "$0")"

DEST="third_party/NVIDIAOpticalFlowSDK"
REPO="${NVOF_SDK_REPO:-https://github.com/NVIDIA/NVIDIAOpticalFlowSDK.git}"
REF="${NVOF_SDK_REF:-master}"

if [ -f "$DEST/nvOpticalFlowCuda.h" ]; then
  echo "[nvof-sdk] already vendored at spike/$DEST — nothing to do"
  exit 0
fi

echo "[nvof-sdk] cloning $REPO ($REF) -> spike/$DEST"
mkdir -p third_party
rm -rf "$DEST"
git clone --depth 1 --branch "$REF" "$REPO" "$DEST"

if [ ! -f "$DEST/nvOpticalFlowCuda.h" ]; then
  echo "[nvof-sdk] ERROR: clone succeeded but nvOpticalFlowCuda.h is missing." >&2
  echo "           Check the SDK layout, or point NVOF_SDK_REPO at a local copy." >&2
  exit 1
fi
echo "[nvof-sdk] OK — headers vendored:"
ls "$DEST"/nvOpticalFlow*.h | sed 's/^/    /'
