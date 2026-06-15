// Unit test for the ST 2110-20 / RFC 4175 packetizer. Pure CPU, no NIC/root — runs in any build.
// Strategy: packetize a synthetic frame, then parse each packet back the way a receiver would
// (read SRDs until the continuation bit clears, copy pixel runs to their line/offset) and assert the
// reconstruction is byte-identical to the original. That exercises the planner AND the byte layout.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../rtp_st2110.hpp"

using namespace spark::st2110;

namespace {
int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("  [FAIL] %s\n", what);
    ++g_failures;
  }
}

uint16_t be16(const uint8_t* p) { return static_cast<uint16_t>((p[0] << 8) | p[1]); }

// Round-trip one profile through the packetizer and a receiver-style parser.
void test_profile(const char* name, VideoFormat fmt, uint32_t max_payload) {
  std::printf("----- %s (%ux%u, payload<=%u) -----\n", name, fmt.width, fmt.height, max_payload);

  // Deterministic, position-varying source frame (catches line/offset copy bugs).
  std::vector<uint8_t> src(fmt.octets_per_frame());
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>((i * 131u + 7u) & 0xff);

  std::vector<uint8_t> recon(src.size(), 0);
  Packetizer pkt(fmt, max_payload, /*pt=*/96, /*ssrc=*/0xCAFEBABE);

  const uint32_t expect_pkts = pkt.packets_per_frame();
  pkt.start_frame(rtp_timestamp_90k(1234567890123ULL));

  std::vector<uint8_t> buf(max_payload + 64);
  uint64_t total_data = 0;
  uint32_t count = 0, markers = 0;
  uint32_t prev_seq = 0;
  bool seq_ok = true, len_ok = true, align_ok = true, marker_last_ok = true;

  for (PacketPlan p; pkt.next(p);) {
    if (count == 0) prev_seq = p.sequence;
    else if (p.sequence != prev_seq + 1) seq_ok = false, prev_seq = p.sequence;
    else prev_seq = p.sequence;

    if (p.payload_len > max_payload) len_ok = false;
    pkt.write_payload(p, src.data(), buf.data());

    // --- parse like a receiver ---
    const uint8_t* r = buf.data();
    const bool marker = (r[1] & 0x80) != 0;
    if (marker) ++markers;
    const uint32_t seq = (static_cast<uint32_t>(be16(r + 12)) << 16) | be16(r + 2);  // ESN|low16
    if (seq != p.sequence) seq_ok = false;
    r += 14;  // RTP(12) + ESN(2)

    struct PSrd { uint16_t len, line, off; };
    std::vector<PSrd> srds;
    for (;;) {
      const uint16_t len = be16(r);
      const uint16_t line = be16(r + 2) & 0x7fff;
      const uint16_t f3 = be16(r + 4);
      const bool cont = (f3 & 0x8000) != 0;
      const uint16_t off = f3 & 0x7fff;
      srds.push_back({len, line, off});
      r += 6;
      if (!cont) break;
    }
    for (const auto& s : srds) {
      if (s.len % VideoFormat::kOctetsPerPgroup != 0) align_ok = false;
      std::memcpy(recon.data() + fmt.byte_offset(s.line, s.off), r, s.len);
      r += s.len;
      total_data += s.len;
    }
    // marker must coincide with the very last packet
    const bool is_last = (count + 1 == expect_pkts);
    if (marker != is_last) marker_last_ok = false;
    ++count;
  }

  check(count == expect_pkts, "packets_per_frame() matches actual packet count");
  check(total_data == fmt.octets_per_frame(), "all frame octets carried exactly once");
  check(recon == src, "reconstructed frame is byte-identical (round trip)");
  check(markers == 1, "exactly one RTP marker bit set");
  check(seq_ok, "sequence numbers contiguous and consistent across RTP/ESN");
  check(len_ok, "no packet exceeds the payload budget");
  check(align_ok, "every SRD length is pgroup-aligned");
  check(marker_last_ok, "marker bit is on the final packet only");
  std::printf("  packets=%u  data=%llu octets\n", count,
              static_cast<unsigned long long>(total_data));
}
}  // namespace

int main() {
  std::printf("===== ST 2110-20 / RFC 4175 packetizer test =====\n");
  test_profile("1080p", profile_1080p(), 1420);
  test_profile("2160p", profile_2160p(), 1420);
  test_profile("1080p tiny-payload", profile_1080p(), 200);  // forces many small packets / 2-SRD edges
  if (g_failures == 0) {
    std::printf("\n[PASS] all packetizer checks passed.\n");
    return 0;
  }
  std::printf("\n[FAIL] %d check(s) failed.\n", g_failures);
  return 1;
}
