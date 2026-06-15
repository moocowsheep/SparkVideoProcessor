#include "pixel_codec.hpp"

namespace spark::codec {
namespace {

// One thread per pgroup (2 luma pixels). packed pgroup = 5 octets:
//   b0=Cb[9:2] b1=Cb[1:0]|Y0[9:4] b2=Y0[3:0]|Cr[9:6] b3=Cr[5:0]|Y1[9:8] b4=Y1[7:0]
__global__ void unpack422_10(const uint8_t* __restrict__ packed, uint16_t* __restrict__ y,
                             uint16_t* __restrict__ cb, uint16_t* __restrict__ cr, uint32_t width,
                             uint32_t height) {
  const uint32_t ppl = width / 2;  // pgroups per line
  const unsigned long long total = (unsigned long long)ppl * height;
  const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const uint32_t r = (uint32_t)(idx / ppl);
  const uint32_t pg = (uint32_t)(idx % ppl);

  const uint8_t* p = packed + idx * 5ULL;
  const uint32_t b0 = p[0], b1 = p[1], b2 = p[2], b3 = p[3], b4 = p[4];
  const uint16_t cb0 = (uint16_t)((b0 << 2) | (b1 >> 6));
  const uint16_t y0 = (uint16_t)(((b1 & 0x3f) << 4) | (b2 >> 4));
  const uint16_t cr0 = (uint16_t)(((b2 & 0x0f) << 6) | (b3 >> 2));
  const uint16_t y1 = (uint16_t)(((b3 & 0x03) << 8) | b4);

  const unsigned long long yrow = (unsigned long long)r * width + (unsigned long long)pg * 2;
  y[yrow] = y0;
  y[yrow + 1] = y1;
  const unsigned long long crow = (unsigned long long)r * ppl + pg;
  cb[crow] = cb0;
  cr[crow] = cr0;
}

__global__ void pack422_10(uint8_t* __restrict__ packed, const uint16_t* __restrict__ y,
                           const uint16_t* __restrict__ cb, const uint16_t* __restrict__ cr,
                           uint32_t width, uint32_t height) {
  const uint32_t ppl = width / 2;
  const unsigned long long total = (unsigned long long)ppl * height;
  const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const uint32_t r = (uint32_t)(idx / ppl);
  const uint32_t pg = (uint32_t)(idx % ppl);

  const unsigned long long yrow = (unsigned long long)r * width + (unsigned long long)pg * 2;
  const unsigned long long crow = (unsigned long long)r * ppl + pg;
  const uint32_t cb0 = cb[crow] & 0x3ff, cr0 = cr[crow] & 0x3ff;
  const uint32_t y0 = y[yrow] & 0x3ff, y1 = y[yrow + 1] & 0x3ff;

  uint8_t* p = packed + idx * 5ULL;
  p[0] = (uint8_t)(cb0 >> 2);
  p[1] = (uint8_t)(((cb0 & 0x3) << 6) | (y0 >> 4));
  p[2] = (uint8_t)(((y0 & 0xf) << 4) | (cr0 >> 6));
  p[3] = (uint8_t)(((cr0 & 0x3f) << 2) | (y1 >> 8));
  p[4] = (uint8_t)(y1 & 0xff);
}

}  // namespace

void unpack_422_10(const uint8_t* packed_dev, uint16_t* y, uint16_t* cb, uint16_t* cr,
                   uint32_t width, uint32_t height, cudaStream_t stream) {
  const unsigned long long total = (unsigned long long)(width / 2) * height;
  const int block = 256;
  const unsigned long long grid = (total + block - 1) / block;
  unpack422_10<<<(unsigned int)grid, block, 0, stream>>>(packed_dev, y, cb, cr, width, height);
}

void pack_422_10(uint8_t* packed_dev, const uint16_t* y, const uint16_t* cb, const uint16_t* cr,
                 uint32_t width, uint32_t height, cudaStream_t stream) {
  const unsigned long long total = (unsigned long long)(width / 2) * height;
  const int block = 256;
  const unsigned long long grid = (total + block - 1) / block;
  pack422_10<<<(unsigned int)grid, block, 0, stream>>>(packed_dev, y, cb, cr, width, height);
}

}  // namespace spark::codec
