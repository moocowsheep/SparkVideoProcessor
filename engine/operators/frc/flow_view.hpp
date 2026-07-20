// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// FlowView — device pointers + geometry for one bidirectional NVOF flow field (and its optional
// per-vector matching cost), decoupled from who produced it: NvofFlow::view() fills one straight
// from the NVOF buffers; FrcOp swaps in median-filtered copies before handing it to interpolate().
#pragma once

#include <cstdint>

namespace spark::frc {

struct FlowView {
  const void* fwd = nullptr;  // SHORT2 grid (S10.5 px), prev->cur, indexed in prev coords
  const void* bwd = nullptr;  // SHORT2 grid (S10.5 px), cur->prev, indexed in cur coords
  uint32_t pitch_bytes = 0;   // row pitch of fwd/bwd
  uint32_t grid_w = 0;
  uint32_t grid_h = 0;
  uint32_t grid_size = 4;  // luma px per flow cell (1|2|4)
  // OFA per-vector matching cost, same grid as the flow; higher = worse match. Optional.
  const void* cost_fwd = nullptr;
  const void* cost_bwd = nullptr;
  uint32_t cost_pitch_bytes = 0;
  uint32_t cost_elem_bytes = 0;  // 1 (UINT8) or 4 (legacy UINT); 0 = no cost available
};

}  // namespace spark::frc
