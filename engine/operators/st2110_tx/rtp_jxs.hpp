// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// ST 2110-22 / RFC 9134 JPEG XS packetization (pure; no DPDK / Holoscan deps — fully unit-testable).
//
// The raw sibling of this module (rtp_st2110.hpp) walks a PIXEL RASTER: RFC 4175 packets carry
// line/offset Sample Row Descriptors, so any packet can be placed in the frame buffer independently.
// A JPEG XS frame has no raster on the wire — it is ONE variable-length codestream (SOC..EOC) that
// gets sliced into MTU-sized fragments and reassembled by concatenation, in order. That difference
// drives every design choice here:
//   * a packet's position is implicit (its packet counter), not addressed — a lost packet corrupts
//     the codestream, so the receiver drops the whole frame instead of decoding a hole;
//   * frame length varies per frame (rate control), so packets-per-frame is a per-frame quantity,
//     not a format constant;
//   * there are no lines, so there is no line-aligned/gapped pacing to do — the TX paces evenly.
//
// Payload format (RFC 9134 §4), codestream packetization mode (K=0), one 4-byte header per packet
// between the RTP header and the codestream fragment:
//
//    0                   1                   2                   3
//    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//   |T|K|L| I |    F    |        SEP        |         P             |
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//
//   T   (1)  transmission mode: 1 = sequential (packets sent in codestream order) — what we do.
//   K   (1)  packetization mode: 0 = codestream, 1 = slice. We implement/accept codestream only.
//   L   (1)  last packet of the packetization unit. In codestream mode RFC 9134 requires L == M,
//            so it mirrors the RTP marker bit.
//   I   (2)  interlace: 00 = progressive (the only mode this pipeline carries).
//   F   (5)  frame counter mod 32 — constant across a frame, +1 per frame.
//   SEP (11) in codestream mode this is the high half of the packet counter: it resets when P
//            resets and increments when P overruns (RFC 9134 §4.3), i.e. {SEP,P} is one 22-bit
//            packet index within the frame.
//   P   (11) packet counter mod 2048 within the packetization unit, 0-based.
//
// Walk model mirrors Packetizer: start_frame(rtp_ts, codestream_bytes) then next() until false.
#pragma once

#include <cstddef>
#include <cstdint>

namespace spark::st2110 {

// One JPEG XS media packet's plan. Allocation-free, like PacketPlan.
struct JxsPacketPlan {
  uint16_t sequence = 0;        // RTP sequence number (16-bit; JPEG XS has no RFC 4175 ESN)
  uint32_t rtp_timestamp = 0;   // 90 kHz media clock (same for every packet of a frame)
  bool marker = false;          // RTP M bit == the header's L bit on the last packet of the frame
  uint32_t offset = 0;          // byte offset of this fragment within the codestream
  uint32_t length = 0;          // codestream bytes carried by this packet
  uint32_t payload_len = 0;     // total UDP payload bytes = 12 (RTP) + 4 (JXS header) + length
  uint8_t frame_counter = 0;    // F, 0..31
  uint32_t packet_counter = 0;  // 22-bit {SEP, P}: this packet's index within the frame
};

// Bytes of RTP + JPEG XS payload header ahead of every codestream fragment.
inline constexpr uint32_t kJxsHeaderOctets = 12u + 4u;

class JxsPacketizer {
 public:
  // max_payload_octets: UDP payload budget per packet (RTP header + JXS header + codestream bytes).
  JxsPacketizer(uint32_t max_payload_octets, uint8_t payload_type = 96, uint32_t ssrc = 0)
      : max_payload_(max_payload_octets), pt_(payload_type), ssrc_(ssrc) {}

  // Begin a frame: its RTP media timestamp and the exact codestream length to fragment. Advances
  // the frame counter. codestream_bytes == 0 produces no packets (next() returns false at once).
  void start_frame(uint32_t rtp_timestamp, uint32_t codestream_bytes);

  // Fill `out` with the next packet's plan and advance. Returns false when the codestream is done.
  bool next(JxsPacketPlan& out);

  // Serialize a plan into `dst` (caller guarantees capacity >= out.payload_len), copying the
  // fragment from `codestream` (which must hold at least plan.offset + plan.length bytes).
  void write_payload(const JxsPacketPlan& plan, const uint8_t* codestream, uint8_t* dst) const;

  // Packets a codestream of `codestream_bytes` will produce at the current payload budget. Varies
  // per frame with the codestream length, so the TX recomputes its pacing gap each frame.
  uint32_t packets_for(uint32_t codestream_bytes) const;

  // Codestream bytes carried per packet (the payload budget minus the two headers).
  uint32_t fragment_octets() const {
    return max_payload_ > kJxsHeaderOctets ? max_payload_ - kJxsHeaderOctets : 1u;
  }

  uint32_t ssrc() const { return ssrc_; }
  void set_ssrc(uint32_t ssrc) { ssrc_ = ssrc; }

 private:
  uint32_t max_payload_;
  uint8_t pt_;
  uint32_t ssrc_;

  uint16_t seq_ = 0;         // persistent across frames (RTP requirement)
  uint8_t frame_ = 31;       // pre-increment: the first start_frame() emits F = 0
  uint32_t rtp_ts_ = 0;      // this frame's media timestamp (stamped on every packet)
  uint32_t size_ = 0;        // current codestream length
  uint32_t cur_ = 0;         // walk cursor within the codestream
  uint32_t pkt_ = 0;         // packet index within the frame ({SEP,P})
  bool active_ = false;
};

// --- Receive side: parse an ST 2110-22 / RFC 9134 UDP payload -------------------------------------

struct JxsRxPacketInfo {
  uint16_t sequence = 0;
  uint32_t rtp_timestamp = 0;
  bool marker = false;
  bool last = false;            // L bit (== marker in codestream mode)
  bool slice_mode = false;      // K bit: true = slice packetization, which we do not reassemble
  bool progressive = true;      // I == 00
  uint8_t frame_counter = 0;    // F
  uint32_t packet_counter = 0;  // {SEP, P}
  uint32_t data_offset = 0;     // byte offset within the payload where codestream bytes begin (16)
  uint32_t data_len = 0;        // codestream bytes in this packet
};

class JxsDepacketizer {
 public:
  // Parse one UDP payload (RTP + JPEG XS payload header). Returns false if malformed or truncated
  // (defensive against the open wire). A well-formed slice-mode packet parses with slice_mode set
  // so the caller can report it rather than silently mis-assemble.
  bool parse(const uint8_t* payload, uint32_t len, JxsRxPacketInfo& info) const;
};

// Offset of the JPEG XS codestream (SOC, 0xff10) within a reassembled packetization unit.
//
// RFC 9134 §4.2: in codestream packetization mode "the packetization unit SHALL be the entire JPEG
// XS picture segment (i.e., codestream preceded by boxes)" — so a compliant sender puts the ISO/IEC
// 21122-3 video support boxes in front of every frame, and what comes off the wire is NOT what a
// codestream decoder can be handed directly. A RED V-Raptor, for instance, prefixes 60 bytes:
//
//   00 00 00 2a 'jpvs'   video support superbox (42 B) = 'jpvi' (22 B) + 'jxpl' (12 B)
//   00 00 00 12 'colr'   colour specification (18 B)
//   ff 10 ...            SOC — the codestream the decoder wants
//
// Returns the byte offset of SOC, walking the ISOBMFF box chain (4-byte big-endian size + 4-byte
// type) rather than scanning for the marker, so codestream bytes can never be mistaken for a header.
// Returns 0 for a bare codestream (already at SOC) and for anything malformed — letting the decoder
// reject a bad stream with its own error rather than second-guessing it here.
size_t jxs_codestream_offset(const uint8_t* data, size_t len);

// Extend a 16-bit RTP sequence number to a monotonic 32-bit counter. RFC 4175 carries a 32-bit
// sequence explicitly (ESN); JPEG XS does not, so loss accounting has to unwrap it. `state` is the
// caller's last extended value; reordering inside half the 16-bit space resolves backwards rather
// than jumping a whole cycle forward.
inline uint32_t seq16_extend(uint32_t state, uint16_t seq, bool have_state) {
  if (!have_state) return seq;
  const uint16_t prev = static_cast<uint16_t>(state);
  const int16_t delta = static_cast<int16_t>(seq - prev);  // wrap-safe signed distance
  return static_cast<uint32_t>(static_cast<int64_t>(state) + delta);
}

}  // namespace spark::st2110
