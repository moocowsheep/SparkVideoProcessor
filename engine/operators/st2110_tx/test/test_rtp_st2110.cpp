// Unit test for the ST 2110-20 / RFC 4175 packetizer + depacketizer. Pure CPU, no NIC/root.
// Strategy: packetize a synthetic frame, then run each packet through the *real* Depacketizer (the
// same parse/scatter the RX operator uses) and assert the reconstruction is byte-identical. That
// exercises both directions and the on-wire byte layout in one round trip.
#include <cstdint>
#include <cstdio>
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

void test_profile(const char* name, VideoFormat fmt, uint32_t max_payload) {
  std::printf("----- %s (%ux%u, payload<=%u) -----\n", name, fmt.width, fmt.height, max_payload);

  // Deterministic, position-varying source frame (catches line/offset copy bugs).
  std::vector<uint8_t> src(fmt.octets_per_frame());
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>((i * 131u + 7u) & 0xff);

  std::vector<uint8_t> recon(src.size(), 0);
  Packetizer pkt(fmt, max_payload, /*pt=*/96, /*ssrc=*/0xCAFEBABE);
  Depacketizer depkt(fmt);

  const uint32_t expect_pkts = pkt.packets_per_frame();
  pkt.start_frame(rtp_timestamp_90k(1234567890123ULL));

  std::vector<uint8_t> buf(max_payload + 64);
  uint64_t total_data = 0;
  uint32_t count = 0, markers = 0, prev_seq = 0;
  bool seq_ok = true, len_ok = true, align_ok = true, marker_last_ok = true, parse_ok = true;

  for (PacketPlan p; pkt.next(p);) {
    if (count == 0) prev_seq = p.sequence;
    else if (p.sequence != prev_seq + 1) seq_ok = false;
    prev_seq = p.sequence;
    if (p.payload_len > max_payload) len_ok = false;

    pkt.write_payload(p, src.data(), buf.data());

    RxPacketInfo info;
    if (!depkt.parse(buf.data(), p.payload_len, info)) {
      parse_ok = false;
      ++count;
      continue;
    }
    if (info.sequence != p.sequence) seq_ok = false;
    if (info.marker) ++markers;
    depkt.scatter(info, buf.data(), recon.data());
    for (int k = 0; k < info.nsrd; ++k) {
      if (info.srd[k].length % fmt.octets_per_pgroup() != 0) align_ok = false;
      total_data += info.srd[k].length;
    }
    if (info.marker != (count + 1 == expect_pkts)) marker_last_ok = false;
    ++count;
  }

  check(parse_ok, "every packet parses via Depacketizer");
  check(count == expect_pkts, "packets_per_frame() matches actual packet count");
  check(total_data == fmt.octets_per_frame(), "all frame octets carried exactly once");
  check(recon == src, "reconstructed frame is byte-identical (packetize -> depacketize round trip)");
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
  std::printf("===== ST 2110-20 / RFC 4175 packetizer + depacketizer test =====\n");
  test_profile("1080p", profile_1080p(), 1420);
  test_profile("2160p", profile_2160p(), 1420);
  test_profile("1080p tiny-payload", profile_1080p(), 200);  // forces many small / 2-SRD packets
  // IP10 carries an 8-bit 4:2:2 codeword stream (4 octets/pgroup) — the generalized geometry must
  // packetize/depacketize it just as faithfully as the 10-bit raw path.
  test_profile("2160p60 IP10 (8-bit)", VideoFormat{3840, 2160, 60000.0 / 1001.0, Sampling::YCbCr422_8}, 1420);
  if (g_failures == 0) {
    std::printf("\n[PASS] all packetizer/depacketizer checks passed.\n");
    return 0;
  }
  std::printf("\n[FAIL] %d check(s) failed.\n", g_failures);
  return 1;
}
