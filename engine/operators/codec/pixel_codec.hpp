// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// ST 2110-20 / RFC 4175 pixel codec — CUDA bridge between packed octets and the planar GpuFrame.
//
// Format: YCbCr 4:2:2 10-bit. A pgroup = 2 pixels = 5 octets packing {Cb0,Y0,Cr0,Y1} (each 10-bit,
// big-endian bit order per RFC 4175). unpack splits that into planar Y (w x h) + Cb/Cr (w/2 x h)
// uint16; pack is the exact inverse (a bijection — round-trips byte-identically). Device pointers;
// the kernels run on `stream`.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace spark::codec {

// packed_dev: device buffer of (w/2)*h*5 octets. y/cb/cr: device planes (uint16).
void unpack_422_10(const uint8_t* packed_dev, uint16_t* y, uint16_t* cb, uint16_t* cr, uint32_t width,
                   uint32_t height, cudaStream_t stream);

void pack_422_10(uint8_t* packed_dev, const uint16_t* y, const uint16_t* cb, const uint16_t* cr,
                 uint32_t width, uint32_t height, cudaStream_t stream);

}  // namespace spark::codec
