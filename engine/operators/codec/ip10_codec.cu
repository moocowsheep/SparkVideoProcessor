// CUDA kernels for the Blackmagic IP10 (10:8) codec — the GPU counterpart of the CPU reference
// (ip10_codec.cpp). The per-sample arithmetic (make_cfg / encode_sample / decode_sample / next_table)
// is shared verbatim from ip10_codec.hpp as __host__ __device__ inlines, so the kernels can't drift
// from the unit-tested reference.
//
// Parallelism: IP10 is sequential WITHIN a lane along a row (each sample's table comes from the previous
// decoded sample of its lane) but lanes and rows are independent. A 4:2:2 row has four independent lanes
// — Cb, Y0, Cr, Y1 (Y's two-per-pgroup is why luma is "2-back") — so we map ONE THREAD PER (row, lane):
// height*4 threads. Each thread serial-scans its single lane down the row, reading its plane with the
// right stride and writing its column of the 8-bit RFC 4175 4:2:2 pgroup {Cb,Y0,Cr,Y1} (4 octets/pgroup,
// stride 4). That's 4x the threads of a one-thread-per-row design — enough occupancy to keep 2160p60
// real-time and cut pipeline latency (the prior 1-thread/row layout left the GPU badly under-occupied).
#include <cuda_runtime.h>

#include "ip10_codec.hpp"

namespace spark::codec::ip10 {
namespace {

// Resolve a thread's lane to its plane pointer + element stride. lane: 0=Cb,1=Y0,2=Cr,3=Y1.
// Cb/Cr are the half-width chroma planes (stride 1); Y0/Y1 are the even/odd luma of each pgroup
// (stride 2, offset 0/1) within the full-width Y plane.
template <typename T>
__device__ inline T* lane_ptr(T* y, T* cb, T* cr, uint32_t row, uint32_t width, uint32_t ppl,
                              uint32_t lane, uint32_t& stride) {
  switch (lane) {
    case 0: stride = 1; return cb + static_cast<size_t>(row) * ppl;            // Cb
    case 2: stride = 1; return cr + static_cast<size_t>(row) * ppl;            // Cr
    case 1: stride = 2; return y + static_cast<size_t>(row) * width;           // Y0
    default: stride = 2; return y + static_cast<size_t>(row) * width + 1;      // Y1
  }
}

__global__ void ip10_encode422(const uint16_t* __restrict__ y, const uint16_t* __restrict__ cb,
                               const uint16_t* __restrict__ cr, uint8_t* __restrict__ packed,
                               uint32_t width, uint32_t height) {
  const unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t row = (uint32_t)(tid >> 2), lane = (uint32_t)(tid & 3);  // 4 lanes per row
  if (row >= height) return;
  const Cfg c = make_cfg();
  const uint32_t ppl = width / 2;
  uint32_t istride;
  const uint16_t* in = lane_ptr<const uint16_t>(y, cb, cr, row, width, ppl, lane, istride);
  uint8_t* out = packed + static_cast<size_t>(row) * ppl * 4 + lane;  // this lane's column, stride 4

  int table = c.default_table;
  for (uint32_t pg = 0; pg < ppl; ++pg) {
    const int code_word = encode_sample(c, table, in[(size_t)pg * istride] & 0x3ff);
    out[(size_t)pg * 4] = static_cast<uint8_t>(code_word);
    table = next_table(c, decode_sample(c, table, code_word));  // encoder/decoder lockstep
  }
}

__global__ void ip10_decode422(const uint8_t* __restrict__ packed, uint16_t* __restrict__ y,
                               uint16_t* __restrict__ cb, uint16_t* __restrict__ cr, uint32_t width,
                               uint32_t height) {
  const unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t row = (uint32_t)(tid >> 2), lane = (uint32_t)(tid & 3);
  if (row >= height) return;
  const Cfg c = make_cfg();
  const uint32_t ppl = width / 2;
  uint32_t ostride;
  uint16_t* out = lane_ptr<uint16_t>(y, cb, cr, row, width, ppl, lane, ostride);
  const uint8_t* in = packed + static_cast<size_t>(row) * ppl * 4 + lane;

  int table = c.default_table;
  for (uint32_t pg = 0; pg < ppl; ++pg) {
    const int recon = decode_sample(c, table, in[(size_t)pg * 4]);
    out[(size_t)pg * ostride] = static_cast<uint16_t>(recon);
    table = next_table(c, recon);
  }
}

constexpr int kBlock = 128;
inline unsigned int grid_for(uint32_t height) {
  const unsigned long long threads = (unsigned long long)height * 4;  // 4 lanes per row
  return (unsigned int)((threads + kBlock - 1) / kBlock);
}

}  // namespace

void ip10_pack_422(uint8_t* packed, const uint16_t* y, const uint16_t* cb, const uint16_t* cr,
                   uint32_t width, uint32_t height, cudaStream_t stream) {
  ip10_encode422<<<grid_for(height), kBlock, 0, stream>>>(y, cb, cr, packed, width, height);
}

void ip10_unpack_422(const uint8_t* packed, uint16_t* y, uint16_t* cb, uint16_t* cr, uint32_t width,
                     uint32_t height, cudaStream_t stream) {
  ip10_decode422<<<grid_for(height), kBlock, 0, stream>>>(packed, y, cb, cr, width, height);
}

}  // namespace spark::codec::ip10
