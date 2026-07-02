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

}  // namespace spark::filters
