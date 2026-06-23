// CPU reference for the Blackmagic IP10 (10:8) codec — see ip10_codec.hpp for the algorithm.
//
// This is the authoritative, unit-tested definition of the row walk; the CUDA kernels (ip10_codec.cu)
// reproduce it one-thread-per-row. Per row, each of `back` interleaved sub-lanes is an independent
// 1-back sequence whose code table is derived from that lane's previous reconstructed sample, and every
// lane starts from the middle table (the per-row context reset).
#include "ip10_codec.hpp"

#include <cstddef>

namespace spark::codec::ip10 {

void encode_plane(const uint16_t* in, uint8_t* out, int width, int height, int back) {
  const Cfg c = make_cfg();
  for (int row = 0; row < height; ++row) {
    const uint16_t* sin = in + static_cast<size_t>(row) * width;
    uint8_t* sout = out + static_cast<size_t>(row) * width;
    int last_recon[2] = {0, 0};  // per-lane previous reconstruction (back is 1 or 2)
    for (int i = 0; i < width; ++i) {
      const int lane = i % back;
      const int table = (i < back) ? c.default_table : next_table(c, last_recon[lane]);
      const int code_word = encode_sample(c, table, sin[i] & 0x3ff);
      sout[i] = static_cast<uint8_t>(code_word);
      last_recon[lane] = decode_sample(c, table, code_word);  // encoder must track what the decoder sees
    }
  }
}

void decode_plane(const uint8_t* in, uint16_t* out, int width, int height, int back) {
  const Cfg c = make_cfg();
  for (int row = 0; row < height; ++row) {
    const uint8_t* cin = in + static_cast<size_t>(row) * width;
    uint16_t* sout = out + static_cast<size_t>(row) * width;
    int last_recon[2] = {0, 0};
    for (int i = 0; i < width; ++i) {
      const int lane = i % back;
      const int table = (i < back) ? c.default_table : next_table(c, last_recon[lane]);
      const int recon = decode_sample(c, table, cin[i]);
      sout[i] = static_cast<uint16_t>(recon);
      last_recon[lane] = recon;
    }
  }
}

}  // namespace spark::codec::ip10
