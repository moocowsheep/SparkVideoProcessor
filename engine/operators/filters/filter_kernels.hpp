// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Video filter CUDA kernels — proc amp (levels/color) and unsharp sharpening on the planar 10-bit
// GpuFrame planes (uint16 storage, 10-bit video range: Y 64..940, chroma centered on 512). Plane
// pointers + stream only, no GpuFrame/Holoscan deps, so the unit test drives them standalone.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace spark::filters {

// Proc amp, classic video semantics, neutral = (0, 1, 1, 0):
//   brightness: black-level offset, normalized (±1.0 = ±full Y swing of 876 code values)
//   contrast:   video gain about black (64), 1.0 = unity
//   saturation: chroma gain about 512, 1.0 = unity
//   hue_deg:    chroma phase rotation in degrees
// Y and chroma are processed out-of-place; outputs clamp to the 10-bit non-reserved range [4, 1019].
// width/height describe the Y plane; chroma planes are (width/2) x height (4:2:2).
void procamp(const uint16_t* y_in, const uint16_t* cb_in, const uint16_t* cr_in, uint16_t* y_out,
             uint16_t* cb_out, uint16_t* cr_out, uint32_t width, uint32_t height, float brightness,
             float contrast, float saturation, float hue_deg, cudaStream_t stream);

// Unsharp mask on the LUMA plane only (broadcast detail-enhance style; sharpening chroma just rings):
//   y_out = y + amount * (y - gauss3x3(y)),  edges clamped, output clamped to [4, 1019].
// amount 0 = identity. Chroma is untouched — the caller forwards/copies those planes.
void unsharp_y(const uint16_t* y_in, uint16_t* y_out, uint32_t width, uint32_t height, float amount,
               cudaStream_t stream);

// Film grain on the LUMA plane (ported from REDStreamer's grainKernel): value noise — a smoothstep-
// bilerped lattice of triangular hash draws — with a photochemical density response 4L(1-L), so
// grain peaks in the mid-tones and vanishes at black/white (those stay bit-exact). `size` is the
// grain cell in pixels (1..4; 1 = per-pixel), `seed` re-draws the field (pass a frame counter),
// amount 1 = 0.08 of the Y swing peak amplitude at mid-gray. Output clamps to [4, 1019].
void grain_y(const uint16_t* y_in, uint16_t* y_out, uint32_t width, uint32_t height, float amount,
             float size, uint32_t seed, cudaStream_t stream);

// Chroma leg of "color" grain: independent noise fields on Cb and Cr (color-negative look; mono
// grain leaves chroma untouched — the caller copies those planes). The film-response weight is
// taken from the co-sited luma sample, amplitude scales the ±448 chroma swing. 4:2:2 planes,
// (width/2) x height; salts are derived from `seed` so the fields decorrelate from the Y grain.
void grain_c(const uint16_t* y_in, const uint16_t* cb_in, const uint16_t* cr_in, uint16_t* cb_out,
             uint16_t* cr_out, uint32_t width, uint32_t height, float amount, float size,
             uint32_t seed, cudaStream_t stream);

// Spatial noise reduction: edge-preserving 5x5 bilateral. Range sigma maps from strength
// (0..1 -> 4..64 code values), spatial sigma fixed at 1.5 px; flat regions average toward clean,
// edges bigger than ~sigma_r survive. strength <= 0 = identity (the caller skips/copies instead).
void nr_y(const uint16_t* y_in, uint16_t* y_out, uint32_t width, uint32_t height, float strength,
          cudaStream_t stream);

// Chroma pair: same bilateral independently on the Cb and Cr planes ((width/2) x height, 4:2:2),
// one thread per sample pair. Chroma noise usually tolerates a stronger setting than luma.
void nr_c(const uint16_t* cb_in, const uint16_t* cr_in, uint16_t* cb_out, uint16_t* cr_out,
          uint32_t width, uint32_t height, float strength, cudaStream_t stream);

}  // namespace spark::filters
