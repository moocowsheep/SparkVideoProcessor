// Unit test for the Blackmagic IP10 (10:8) codec reference (operators/codec/ip10_codec.{hpp,cpp}).
// Pure CPU — no GPU/NIC/root. Verifies: (1) hand-computed golden codewords from the published spec's
// worked example, (2) the strictly bounded ±3 reconstruction error, (3) lossless on smooth content,
// (4) idempotence (re-encoding a reconstruction is lossless), (5) decode∘encode determinism per row.
//
// (Byte-exact conformance against Blackmagic's DucksTakeOff vector is validated out-of-tree — that
// asset is proprietary and not committed; see docs and the scratchpad validator.)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "operators/codec/ip10_codec.hpp"

using namespace spark::codec::ip10;

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++failures;
}

int main() {
  const Cfg c = make_cfg(10, 8);

  // (0) Setup constants match the spec's 10:8 flavour.
  check(c.lossy_code_width == 7 && c.lossless_codes == 128 && c.lossy_codes == 128 &&
            c.code_tables == 129 && c.default_table == 64 && c.lowest_table_thresh == 64 &&
            c.highest_table_thresh == 960 && c.lossy_flag == 128,
        "10:8 constants (width 7, 128/128 codes, 129 tables, default 64, flag 0x80)");

  // (1) Golden codewords: at a row start every lane uses the middle table 64 (lossless window [448,576)).
  //     Spec worked example (DucksTakeOff first samples): 137 -> lossy 147 -> decodes 136; 140 -> 148 -> 143.
  check(encode_sample(c, 64, 137) == 147 && decode_sample(c, 64, 147) == 136, "golden 137 -> cw147 -> 136");
  check(encode_sample(c, 64, 140) == 148 && decode_sample(c, 64, 148) == 143, "golden 140 -> cw148 -> 143");
  //     A value inside the lossless window encodes losslessly (codeword has the flag bit clear).
  check(encode_sample(c, 64, 500) == (500 - 64 * 7) && (encode_sample(c, 64, 500) & c.lossy_flag) == 0 &&
            decode_sample(c, 64, encode_sample(c, 64, 500)) == 500,
        "in-window value 500 is lossless under table 64");

  // (2) Every codeword fits in 8 bits and reconstruction error never exceeds ±3, across all tables/samples.
  int worst = 0;
  bool byte_ok = true;
  for (int table = 0; table < c.code_tables; ++table)
    for (int s = 0; s < 1024; ++s) {
      const int cw = encode_sample(c, table, s);
      if (cw < 0 || cw > 255) byte_ok = false;
      const int r = decode_sample(c, table, cw);
      worst = std::max(worst, std::abs(r - s));
    }
  check(byte_ok, "all codewords are 8-bit (0..255)");
  check(worst <= 3, "reconstruction error bounded by ±3 over every table × sample");

  // (3) Lossless on a smooth gradient (luma plane, 2-back): once the context settles, error is zero.
  const int W = 256;
  std::vector<uint16_t> ramp(W), rrec(W);
  std::vector<uint8_t> rcw(W);
  for (int i = 0; i < W; ++i) ramp[i] = static_cast<uint16_t>(400 + i);  // gentle slope within one window
  encode_plane(ramp.data(), rcw.data(), W, 1, 2);
  decode_plane(rcw.data(), rrec.data(), W, 1, 2);
  int ramp_tail_err = 0;
  for (int i = 4; i < W; ++i) ramp_tail_err = std::max(ramp_tail_err, std::abs((int)rrec[i] - (int)ramp[i]));
  check(ramp_tail_err == 0, "smooth gradient is mathematically lossless once context settles");

  // (4) Bounded error + idempotence on a real-ish 2D plane (random edges). A second encode of the
  //     reconstruction must be lossless (the spec's idempotence guarantee).
  const int PW = 320, PH = 200;
  std::vector<uint16_t> src(PW * PH), rec(PW * PH), rec2(PW * PH);
  std::vector<uint8_t> cw1(PW * PH), cw2(PW * PH);
  std::srand(12345);
  for (auto& s : src) s = static_cast<uint16_t>(std::rand() & 0x3ff);
  encode_plane(src.data(), cw1.data(), PW, PH, 2);
  decode_plane(cw1.data(), rec.data(), PW, PH, 2);
  encode_plane(rec.data(), cw2.data(), PW, PH, 2);
  decode_plane(cw2.data(), rec2.data(), PW, PH, 2);
  int plane_worst = 0;
  bool idempotent = true;
  for (int i = 0; i < PW * PH; ++i) {
    plane_worst = std::max(plane_worst, std::abs((int)rec[i] - (int)src[i]));
    if (rec2[i] != rec[i]) idempotent = false;
  }
  check(plane_worst <= 3, "2D plane reconstruction error bounded by ±3");
  check(idempotent, "idempotence: re-encoding a reconstruction is lossless");

  // (5) Chroma uses 1-back; sanity that the plane helpers run and round-trip within bound there too.
  std::vector<uint16_t> csrc(PW * PH), crec(PW * PH);
  std::vector<uint8_t> ccw(PW * PH);
  for (int i = 0; i < PW * PH; ++i) csrc[i] = src[i];
  encode_plane(csrc.data(), ccw.data(), PW, PH, 1);
  decode_plane(ccw.data(), crec.data(), PW, PH, 1);
  int chroma_worst = 0;
  for (int i = 0; i < PW * PH; ++i) chroma_worst = std::max(chroma_worst, std::abs((int)crec[i] - (int)csrc[i]));
  check(chroma_worst <= 3, "chroma plane (1-back) reconstruction bounded by ±3");

  std::printf("%s\n", failures == 0 ? "[PASS] ip10 codec" : "[FAIL] ip10 codec");
  return failures == 0 ? 0 : 1;
}
