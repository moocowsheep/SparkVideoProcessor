// CUDA kernels for the Blackmagic IP10 (10:8) codec — the GPU counterpart of the CPU reference
// (ip10_codec.cpp). The per-sample arithmetic (make_cfg / encode_sample / decode_sample / next_table)
// is shared verbatim from ip10_codec.hpp as __host__ __device__ inlines, so the kernels can't drift
// from the unit-tested reference.
//
// Parallelism: IP10 is sequential WITHIN a row (each sample's table comes from the previous decoded
// sample of its lane) but rows are independent (context resets per row), so we map one CUDA thread to
// one image row and walk it left-to-right. A 4:2:2 row interleaves four lanes — Y0/Y1 (the two luma of
// each pgroup, hence luma's "2-back") and Cb/Cr — each carried as its own running table. Output is laid
// straight into 8-bit RFC 4175 4:2:2 pgroups {Cb,Y0,Cr,Y1} (4 octets/pgroup) so the existing packetizer
// transmits it unchanged. Row-only parallelism (height threads) is ample for real-time 2160p60.
#include <cuda_runtime.h>

#include "ip10_codec.hpp"

namespace spark::codec::ip10 {
namespace {

__global__ void ip10_encode422(const uint16_t* __restrict__ y, const uint16_t* __restrict__ cb,
                               const uint16_t* __restrict__ cr, uint8_t* __restrict__ packed,
                               uint32_t width, uint32_t height) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= height) return;
  const Cfg c = make_cfg();
  const uint32_t ppl = width / 2;  // pgroups (and Cb/Cr samples) per line
  const uint16_t* yr = y + static_cast<size_t>(row) * width;
  const uint16_t* cbr = cb + static_cast<size_t>(row) * ppl;
  const uint16_t* crr = cr + static_cast<size_t>(row) * ppl;
  uint8_t* out = packed + static_cast<size_t>(row) * ppl * 4;  // 4 octets/pgroup

  int t_y0 = c.default_table, t_y1 = c.default_table, t_cb = c.default_table, t_cr = c.default_table;
  for (uint32_t pg = 0; pg < ppl; ++pg) {
    const int cw_cb = encode_sample(c, t_cb, cbr[pg] & 0x3ff);
    const int cw_y0 = encode_sample(c, t_y0, yr[2 * pg] & 0x3ff);
    const int cw_cr = encode_sample(c, t_cr, crr[pg] & 0x3ff);
    const int cw_y1 = encode_sample(c, t_y1, yr[2 * pg + 1] & 0x3ff);
    out[pg * 4 + 0] = static_cast<uint8_t>(cw_cb);  // RFC 4175 4:2:2 pgroup order: Cb Y0 Cr Y1
    out[pg * 4 + 1] = static_cast<uint8_t>(cw_y0);
    out[pg * 4 + 2] = static_cast<uint8_t>(cw_cr);
    out[pg * 4 + 3] = static_cast<uint8_t>(cw_y1);
    // Advance each lane's table from what the decoder will reconstruct (encoder/decoder lockstep).
    t_cb = next_table(c, decode_sample(c, t_cb, cw_cb));
    t_y0 = next_table(c, decode_sample(c, t_y0, cw_y0));
    t_cr = next_table(c, decode_sample(c, t_cr, cw_cr));
    t_y1 = next_table(c, decode_sample(c, t_y1, cw_y1));
  }
}

__global__ void ip10_decode422(const uint8_t* __restrict__ packed, uint16_t* __restrict__ y,
                               uint16_t* __restrict__ cb, uint16_t* __restrict__ cr, uint32_t width,
                               uint32_t height) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= height) return;
  const Cfg c = make_cfg();
  const uint32_t ppl = width / 2;
  const uint8_t* in = packed + static_cast<size_t>(row) * ppl * 4;
  uint16_t* yr = y + static_cast<size_t>(row) * width;
  uint16_t* cbr = cb + static_cast<size_t>(row) * ppl;
  uint16_t* crr = cr + static_cast<size_t>(row) * ppl;

  int t_y0 = c.default_table, t_y1 = c.default_table, t_cb = c.default_table, t_cr = c.default_table;
  for (uint32_t pg = 0; pg < ppl; ++pg) {
    const int r_cb = decode_sample(c, t_cb, in[pg * 4 + 0]);
    const int r_y0 = decode_sample(c, t_y0, in[pg * 4 + 1]);
    const int r_cr = decode_sample(c, t_cr, in[pg * 4 + 2]);
    const int r_y1 = decode_sample(c, t_y1, in[pg * 4 + 3]);
    cbr[pg] = static_cast<uint16_t>(r_cb);
    yr[2 * pg] = static_cast<uint16_t>(r_y0);
    crr[pg] = static_cast<uint16_t>(r_cr);
    yr[2 * pg + 1] = static_cast<uint16_t>(r_y1);
    t_cb = next_table(c, r_cb);
    t_y0 = next_table(c, r_y0);
    t_cr = next_table(c, r_cr);
    t_y1 = next_table(c, r_y1);
  }
}

}  // namespace

void ip10_pack_422(uint8_t* packed, const uint16_t* y, const uint16_t* cb, const uint16_t* cr,
                   uint32_t width, uint32_t height, cudaStream_t stream) {
  const int block = 128;
  const uint32_t grid = (height + block - 1) / block;
  ip10_encode422<<<grid, block, 0, stream>>>(y, cb, cr, packed, width, height);
}

void ip10_unpack_422(const uint8_t* packed, uint16_t* y, uint16_t* cb, uint16_t* cr, uint32_t width,
                     uint32_t height, cudaStream_t stream) {
  const int block = 128;
  const uint32_t grid = (height + block - 1) / block;
  ip10_decode422<<<grid, block, 0, stream>>>(packed, y, cb, cr, width, height);
}

}  // namespace spark::codec::ip10
