// Blackmagic IP10 (10:8) codec — clean-room implementation of the published, license-free codec.
//
// IP10 is a Context-Adaptive Mixed Lossy/Lossless, Absolutely-Constant-Bit-Rate (ACBR) codec: every
// 10-bit source sample is mapped to a fixed-size 8-bit symbol (no VLC, no buffers), so 12G-SDI-rate
// 2160p59.94/60 fits within a 10G link. Each symbol's top bit flags lossless('0')/lossy('1'); the code
// table (which positions the 128-wide lossless window) is chosen *only* from the previously DECODED
// sample, so encoder and decoder stay in lockstep with no side channel. Lossless on smooth gradients,
// strictly bounded ±3 error on edges (which perceptual masking hides).
//
// This header carries the per-sample primitives as inline functions so the CPU reference (ip10_codec.cpp,
// unit-tested) and the CUDA kernels (ip10_codec.cu) share ONE definition of the arithmetic and can't
// drift. Constants are derived exactly as Blackmagic's reference SetupBMIP10(sample_bits, coded_bits);
// the only flavour we transmit is 10:8 ("professional video"). Validated byte-exact against Blackmagic's
// conformance vector (decode(encode(DucksTakeOff)) == their reference reconstruction, all 4.1M samples).
//
// Transport (ST 2110-22): the 8-bit codewords are packetized exactly like an 8-bit YCbCr-4:2:2 RFC 4175
// raw stream — each codeword sits in the 8-bit component position of the pgroup ({Cb,Y0,Cr,Y1} = 4
// octets / 2 px). See st2110_format.hpp (pgroup geometry) and spark_node.cpp (the vnd.blackmagicdesign
// .ip10 SDP). The receiver reverses it: depacketize 8-bit pgroups -> ip10 decode -> 10-bit samples.
#pragma once

#include <cstdint>

// Usable from plain C++ (CPU reference / tests) and from NVCC (.cu device code).
#if defined(__CUDACC__)
#define SPARK_IP10_HD __host__ __device__
#else
#define SPARK_IP10_HD
#endif

namespace spark::codec::ip10 {

// Codec constants for a given flavour. For 10:8: lossy_code_width=7, 128 lossless + 128 lossy codes,
// 129 tables, default/middle table 64, lossless-window selection thresholds 64..960, lossy flag bit 0x80.
struct Cfg {
  int lossy_code_width;
  int lossy_rounding;
  int lossless_codes;
  int lossy_codes;
  int code_tables;
  int default_table;
  int lowest_table_thresh;
  int highest_table_thresh;
  int lossy_flag;
};

// Derive the constants for `sample_bits`->`coded_bits` (we only ever call make_cfg(10, 8)).
SPARK_IP10_HD inline Cfg make_cfg(int sample_bits = 10, int coded_bits = 8) {
  const int sample_states = 1 << sample_bits;
  const int coded_states = 1 << coded_bits;
  Cfg c{};
  c.lossy_code_width = (1 << (sample_bits - coded_bits + 1)) - 1;
  c.lossy_rounding = c.lossy_code_width / 2;
  c.lossless_codes = coded_states / 2;
  c.lossy_codes = coded_states / 2;
  c.code_tables = 1 + coded_states / 2;
  c.default_table = c.code_tables / 2;
  c.lowest_table_thresh = c.lossless_codes / 2;
  c.highest_table_thresh = sample_states - (c.lossless_codes / 2);
  c.lossy_flag = coded_states / 2;
  return c;
}

// Pick the table for the NEXT sample from the current sample's decoded value (the shared "context").
SPARK_IP10_HD inline int next_table(const Cfg& c, int sample) {
  if (sample < c.lowest_table_thresh) return 0;
  if (sample >= c.highest_table_thresh) return c.lossy_codes;
  return (sample - c.lowest_table_thresh + c.lossy_rounding) / c.lossy_code_width;
}

// Map a 10-bit sample to its 8-bit codeword under `table`.
SPARK_IP10_HD inline int encode_sample(const Cfg& c, int table, int sample) {
  const int lossless_low = table * c.lossy_code_width;
  const int lossless_high = lossless_low + c.lossless_codes;
  if (sample >= lossless_low && sample < lossless_high) return sample - lossless_low;  // lossless
  if (sample < lossless_low) return c.lossy_flag | (sample / c.lossy_code_width);
  return c.lossy_flag | ((sample - lossless_high) / c.lossy_code_width + table);
}

// Map an 8-bit codeword back to a 10-bit sample under `table` (the inverse of encode_sample).
SPARK_IP10_HD inline int decode_sample(const Cfg& c, int table, int code_word) {
  const int index = code_word & ~c.lossy_flag;
  if (code_word & c.lossy_flag) {
    if (index < table) return index * c.lossy_code_width + c.lossy_rounding;
    return index * c.lossy_code_width + c.lossy_rounding + c.lossless_codes;
  }
  return table * c.lossy_code_width + index;
}

// --- whole-plane helpers (CPU reference; the CUDA kernels parallelize the same logic across rows) ---
//
// Context resets at the start of every row to the middle table (= the "previous sample 0x200" convention),
// so rows are independent (-> embarrassingly parallel on the GPU). `back` selects the per-component
// reference distance within the interleaved sample order: 2 for luma (the two Y of a 4:2:2 pgroup form
// two independent sub-lanes), 1 for chroma. So call with back=2 on the Y plane and back=1 on Cb/Cr.

// in: width*height 10-bit samples (0..1023). out: width*height 8-bit codewords.
void encode_plane(const uint16_t* in, uint8_t* out, int width, int height, int back);

// in: width*height 8-bit codewords. out: width*height reconstructed 10-bit samples.
void decode_plane(const uint8_t* in, uint16_t* out, int width, int height, int back);

// --- CUDA host-callable entry points (ip10_codec.cu); one thread per image row -------------------
// pack:   planar 10-bit Y(w×h)/Cb/Cr(w/2×h) -> device buffer `packed` of (w/2)*h*4 octets, IP10-coded
//         and laid out as 8-bit RFC 4175 4:2:2 pgroups {Cb,Y0,Cr,Y1} ready for the existing packetizer.
// unpack: the inverse (received 8-bit IP10 pgroups -> reconstructed planar 10-bit), for an RX path.
// cudaStream_t is forward-declared so this header stays usable from plain C++ (CPU reference + tests)
// without pulling in <cuda_runtime.h>; the typedef is identical to CUDA's, so .cu units stay consistent.
typedef struct CUstream_st* cudaStream_t;
void ip10_pack_422(uint8_t* packed, const uint16_t* y, const uint16_t* cb, const uint16_t* cr,
                   uint32_t width, uint32_t height, cudaStream_t stream);
void ip10_unpack_422(const uint8_t* packed, uint16_t* y, uint16_t* cb, uint16_t* cr, uint32_t width,
                     uint32_t height, cudaStream_t stream);

}  // namespace spark::codec::ip10
