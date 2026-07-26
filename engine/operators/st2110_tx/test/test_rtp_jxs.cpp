// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Unit test for the ST 2110-22 / RFC 9134 JPEG XS packetizer + depacketizer. Pure CPU, no NIC/root.
// Same strategy as test_rtp_st2110: fragment a synthetic codestream, run every packet through the
// *real* JxsDepacketizer the RX operator uses, and assert byte-identical reassembly. Because JPEG XS
// packets are position-implicit (concatenation, not addressed writes), the test also pins the header
// fields that carry that position — the {SEP,P} packet counter, the frame counter, and L == M.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../rtp_jxs.hpp"

using namespace spark::st2110;

namespace {
int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("  [FAIL] %s\n", what);
    ++g_failures;
  }
}

// Fragment `size` bytes, reassemble through the depacketizer, and verify the round trip.
void test_codestream(const char* name, uint32_t size, uint32_t max_payload) {
  std::printf("----- %s (%u byte codestream, payload<=%u) -----\n", name, size, max_payload);

  std::vector<uint8_t> src(size);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>((i * 197u + 23u) & 0xff);

  JxsPacketizer pkt(max_payload, /*pt=*/96, /*ssrc=*/0xCAFEBABE);
  JxsDepacketizer depkt;

  const uint32_t expect_pkts = pkt.packets_for(size);
  pkt.start_frame(/*rtp_timestamp=*/0x12345678u, size);

  std::vector<uint8_t> buf(max_payload + 64);
  std::vector<uint8_t> recon;
  recon.reserve(size);
  uint32_t count = 0, markers = 0;
  uint16_t prev_seq = 0;
  bool seq_ok = true, len_ok = true, parse_ok = true, ctr_ok = true, lm_ok = true, mode_ok = true;
  uint8_t frame_no = 0xff;

  for (JxsPacketPlan p; pkt.next(p);) {
    if (count == 0) prev_seq = p.sequence;
    else if (p.sequence != static_cast<uint16_t>(prev_seq + 1)) seq_ok = false;
    prev_seq = p.sequence;
    if (p.payload_len > max_payload) len_ok = false;

    pkt.write_payload(p, src.data(), buf.data());

    JxsRxPacketInfo info;
    if (!depkt.parse(buf.data(), p.payload_len, info)) {
      parse_ok = false;
      break;
    }
    // Position is implicit: the receiver must see a dense 0,1,2,... {SEP,P} counter, because that
    // is the only thing that tells it a fragment is missing before it concatenates.
    if (info.packet_counter != count) ctr_ok = false;
    if (info.last != info.marker || info.marker != p.marker) lm_ok = false;
    if (info.slice_mode || !info.progressive) mode_ok = false;
    if (count == 0) frame_no = info.frame_counter;
    else if (info.frame_counter != frame_no) ctr_ok = false;

    recon.insert(recon.end(), buf.data() + info.data_offset,
                 buf.data() + info.data_offset + info.data_len);
    if (info.marker) ++markers;
    ++count;
  }

  check(parse_ok, "every packet parses");
  check(count == expect_pkts, "packet count matches packets_for()");
  check(seq_ok, "RTP sequence increments by 1");
  check(len_ok, "payload stays within the budget");
  check(ctr_ok, "{SEP,P} counter is dense and F is constant across the frame");
  check(lm_ok, "L bit mirrors the RTP marker (codestream mode)");
  check(mode_ok, "K=0 (codestream) and I=00 (progressive)");
  check(markers == 1, "exactly one marker, on the last packet");
  check(recon.size() == src.size(), "reassembled size matches");
  check(recon == src, "reassembled codestream is byte-identical");
  std::printf("  %u packets, %zu bytes reassembled\n", count, recon.size());
}

// The 11-bit P counter wraps every 2048 packets; SEP must take over as the high half so the
// receiver's dense-counter check keeps working on a large codestream.
void test_counter_overrun() {
  std::printf("----- packet counter overrun (P wraps into SEP) -----\n");
  const uint32_t frag = 100;
  const uint32_t max_payload = kJxsHeaderOctets + frag;
  const uint32_t size = frag * 5000;  // > 2048 packets, so P overruns twice
  std::vector<uint8_t> src(size, 0xa5);

  JxsPacketizer pkt(max_payload);
  JxsDepacketizer depkt;
  pkt.start_frame(1u, size);

  std::vector<uint8_t> buf(max_payload + 64);
  uint32_t count = 0;
  bool ok = true;
  for (JxsPacketPlan p; pkt.next(p);) {
    pkt.write_payload(p, src.data(), buf.data());
    JxsRxPacketInfo info;
    if (!depkt.parse(buf.data(), p.payload_len, info) || info.packet_counter != count) {
      ok = false;
      break;
    }
    ++count;
  }
  check(count == 5000, "5000 packets produced");
  check(ok, "{SEP,P} stays dense past the 2048-packet P wrap");
}

// The frame counter advances per frame (mod 32) and the sequence number is continuous across the
// frame boundary — a receiver uses both to detect a dropped frame boundary.
void test_frame_counter() {
  std::printf("----- frame counter + cross-frame sequence -----\n");
  const uint32_t max_payload = kJxsHeaderOctets + 64;
  std::vector<uint8_t> src(200, 0x5a);
  JxsPacketizer pkt(max_payload);
  JxsDepacketizer depkt;
  std::vector<uint8_t> buf(max_payload + 64);

  bool f_ok = true, seq_ok = true;
  uint16_t prev_seq = 0;
  bool have_prev = false;
  for (uint32_t frame = 0; frame < 40; ++frame) {  // past the mod-32 wrap
    pkt.start_frame(frame, static_cast<uint32_t>(src.size()));
    for (JxsPacketPlan p; pkt.next(p);) {
      pkt.write_payload(p, src.data(), buf.data());
      JxsRxPacketInfo info;
      if (!depkt.parse(buf.data(), p.payload_len, info)) {
        f_ok = false;
        break;
      }
      if (info.frame_counter != (frame & 31u)) f_ok = false;
      if (have_prev && info.sequence != static_cast<uint16_t>(prev_seq + 1)) seq_ok = false;
      prev_seq = info.sequence;
      have_prev = true;
    }
  }
  check(f_ok, "frame counter == frame number mod 32");
  check(seq_ok, "RTP sequence is continuous across frame boundaries");
}

// Malformed / unsupported payloads must be rejected rather than mis-parsed into codestream bytes.
void test_rejects() {
  std::printf("----- defensive parsing -----\n");
  JxsDepacketizer depkt;
  JxsRxPacketInfo info;
  uint8_t buf[64] = {};

  check(!depkt.parse(buf, 15, info), "rejects a payload shorter than RTP + JXS headers");

  buf[0] = 0x80;  // V=2, no CSRC, no extension
  buf[1] = 96;
  check(depkt.parse(buf, 16, info), "accepts a header-only (zero-length fragment) packet");
  check(info.data_len == 0, "header-only packet carries no codestream bytes");

  buf[0] = 0x40;  // version 1
  check(!depkt.parse(buf, 32, info), "rejects a non-RTPv2 packet");

  buf[0] = 0x81;  // CC=1: a CSRC would shift the payload header
  check(!depkt.parse(buf, 32, info), "rejects a packet carrying a CSRC list");

  buf[0] = 0x90;  // X=1: a header extension would shift the payload header
  check(!depkt.parse(buf, 32, info), "rejects a packet carrying a header extension");
}

// seq16_extend must unwrap the 16-bit sequence monotonically (JPEG XS has no RFC 4175 ESN).
void test_seq_extend() {
  std::printf("----- 16-bit sequence extension -----\n");
  uint32_t s = seq16_extend(0, 65530, false);
  check(s == 65530, "first packet seeds the state verbatim");
  for (int i = 0; i < 10; ++i) s = seq16_extend(s, static_cast<uint16_t>(65531 + i), true);
  check(s == 65540, "extends monotonically across the 16-bit wrap");
  const uint32_t back = seq16_extend(s, static_cast<uint16_t>(s - 2), true);
  check(back == s - 2, "a reordered packet resolves backwards, not a cycle forward");
}

// RFC 9134 codestream mode carries the ISO/IEC 21122-3 video support boxes ahead of the SOC, so the
// reassembled packetization unit is a picture SEGMENT, not a bare codestream. These cases are the
// exact box chain a RED V-Raptor sends (captured 2026-07-26 from a 2160p59.94 jxsv flow) plus the
// malformed shapes that must not send the walk off the end of the buffer.
void test_codestream_offset() {
  std::printf("----- picture-segment box prefix -----\n");

  // Bare codestream: already at SOC.
  const uint8_t bare[] = {0xff, 0x10, 0xff, 0x50, 0x00, 0x02};
  check(jxs_codestream_offset(bare, sizeof(bare)) == 0, "bare codestream needs no skip");

  // The RED prefix: 'jpvs' superbox (42 B) = 'jpvi' (22 B) + 'jxpl' (12 B), then 'colr' (18 B).
  std::vector<uint8_t> red = {
      0x00, 0x00, 0x00, 0x2a, 'j',  'p',  'v',  's',   // jpvs, 42
      0x00, 0x00, 0x00, 0x16, 'j',  'p',  'v',  'i',   //   jpvi, 22
      0x00, 0x00, 0x00, 0xc6, 0x02, 0x00, 0x00, 0x3c, 0x80, 0x90, 0x10, 0x14, 0x34, 0x20,
      0x00, 0x00, 0x00, 0x0c, 'j',  'x',  'p',  'l',   //   jxpl, 12
      0x4a, 0x40, 0x24, 0x06,                          //   Ppih=0x4a40 Plev=0x2406
      0x00, 0x00, 0x00, 0x12, 'c',  'o',  'l',  'r',   // colr, 18
      0x05, 0x00, 0x00, 0x00, 0x01, 0x00, 0x0e, 0x00, 0x01, 0x00,
      0xff, 0x10, 0xff, 0x50};                         // SOC at 60
  check(red.size() == 64, "RED fixture is 60 bytes of boxes + 4 of codestream");
  check(jxs_codestream_offset(red.data(), red.size()) == 60, "skips the RED box chain to SOC");

  // A single box, and a chain whose SOC sits at the very end of the buffer.
  std::vector<uint8_t> one = {0x00, 0x00, 0x00, 0x0a, 'c', 'o', 'l', 'r', 0x00, 0x00, 0xff, 0x10};
  check(jxs_codestream_offset(one.data(), one.size()) == 10, "skips a single box");

  std::printf("----- malformed prefixes are handed to the decoder unchanged -----\n");
  // Box size smaller than its own header.
  const uint8_t tiny[] = {0x00, 0x00, 0x00, 0x04, 'c', 'o', 'l', 'r', 0xff, 0x10};
  check(jxs_codestream_offset(tiny, sizeof(tiny)) == 0, "rejects a box smaller than its header");
  // Box size running past the buffer.
  const uint8_t over[] = {0x00, 0x00, 0xff, 0x00, 'c', 'o', 'l', 'r', 0xff, 0x10};
  check(jxs_codestream_offset(over, sizeof(over)) == 0, "rejects a box running past the buffer");
  // ISOBMFF size 0 ("to end of data") and size 1 ("64-bit size") are not picture segments.
  const uint8_t zero[] = {0x00, 0x00, 0x00, 0x00, 'c', 'o', 'l', 'r', 0xff, 0x10};
  check(jxs_codestream_offset(zero, sizeof(zero)) == 0, "rejects size-0 box");
  const uint8_t ext[] = {0x00, 0x00, 0x00, 0x01, 'c', 'o', 'l', 'r', 0xff, 0x10};
  check(jxs_codestream_offset(ext, sizeof(ext)) == 0, "rejects 64-bit extended size");
  // Never walks off a short buffer.
  check(jxs_codestream_offset(nullptr, 0) == 0, "null is safe");
  const uint8_t stub[] = {0x00};
  check(jxs_codestream_offset(stub, 1) == 0, "1-byte buffer is safe");
  const uint8_t hdr_only[] = {0x00, 0x00, 0x00, 0x20, 'j', 'p', 'v', 's'};
  check(jxs_codestream_offset(hdr_only, sizeof(hdr_only)) == 0, "header with no body is safe");
  // A chain that never reaches SOC must terminate, not spin.
  std::vector<uint8_t> endless(4096, 0);
  for (size_t i = 0; i + 8 <= endless.size(); i += 8) endless[i + 3] = 8;  // 8-byte empty boxes
  check(jxs_codestream_offset(endless.data(), endless.size()) == 0, "bounded when SOC never appears");
}

}  // namespace

int main() {
  // 1080p and 2160p 4:2:2 10-bit at a typical ST 2110-22 rate, plus edge sizes. The 1420/1300-octet
  // budgets are the ones the TX operator actually uses for raw / IP10.
  test_codestream("2160p ~6 bpp", 3840u * 2160u * 6u / 8u, 1420);
  test_codestream("1080p ~4 bpp", 1920u * 1080u * 4u / 8u, 1300);
  test_codestream("exact multiple of the fragment size", (1420u - kJxsHeaderOctets) * 4u, 1420);
  test_codestream("single-packet codestream", 100, 1420);
  test_codestream("one byte", 1, 1420);
  test_counter_overrun();
  test_frame_counter();
  test_rejects();
  test_seq_extend();
  test_codestream_offset();

  if (g_failures == 0) {
    std::printf("\nAll JPEG XS RTP framing checks passed.\n");
    return 0;
  }
  std::printf("\n%d check(s) FAILED.\n", g_failures);
  return 1;
}
