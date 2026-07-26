// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

#include "rtp_jxs.hpp"

#include <cstring>

namespace spark::st2110 {
namespace {

// JPEG XS payload header field placement within the big-endian 32-bit word (RFC 9134 §4).
constexpr uint32_t kBitT = 1u << 31;   // transmission mode: sequential
constexpr uint32_t kBitK = 1u << 30;   // packetization mode: 1 = slice (we always send 0)
constexpr uint32_t kBitL = 1u << 29;   // last packet of the packetization unit
constexpr int kShiftI = 27;            // interlace, 2 bits
constexpr int kShiftF = 22;            // frame counter, 5 bits
constexpr int kShiftSep = 11;          // SEP counter, 11 bits
constexpr uint32_t kMaskI = 0x3u;
constexpr uint32_t kMaskF = 0x1fu;
constexpr uint32_t kMask11 = 0x7ffu;

inline void put_be16(uint8_t* p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v >> 8);
  p[1] = static_cast<uint8_t>(v & 0xff);
}

inline void put_be32(uint8_t* p, uint32_t v) {
  p[0] = static_cast<uint8_t>(v >> 24);
  p[1] = static_cast<uint8_t>(v >> 16);
  p[2] = static_cast<uint8_t>(v >> 8);
  p[3] = static_cast<uint8_t>(v & 0xff);
}

inline uint32_t get_be32(const uint8_t* p) {
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

}  // namespace

void JxsPacketizer::start_frame(uint32_t rtp_timestamp, uint32_t codestream_bytes) {
  rtp_ts_ = rtp_timestamp;
  size_ = codestream_bytes;
  cur_ = 0;
  pkt_ = 0;
  frame_ = static_cast<uint8_t>((frame_ + 1u) & kMaskF);
  active_ = codestream_bytes > 0;
}

bool JxsPacketizer::next(JxsPacketPlan& out) {
  if (!active_) return false;
  const uint32_t frag = fragment_octets();
  const uint32_t left = size_ - cur_;
  const uint32_t take = left < frag ? left : frag;

  out.sequence = seq_++;
  out.rtp_timestamp = rtp_ts_;
  out.offset = cur_;
  out.length = take;
  out.payload_len = kJxsHeaderOctets + take;
  out.frame_counter = frame_;
  out.packet_counter = pkt_++;

  cur_ += take;
  // Codestream mode: L == M, both set on the packet that carries the last codestream byte.
  out.marker = cur_ >= size_;
  if (out.marker) active_ = false;
  return true;
}

void JxsPacketizer::write_payload(const JxsPacketPlan& p, const uint8_t* codestream,
                                  uint8_t* dst) const {
  uint8_t* w = dst;

  // --- RTP header (RFC 3550), 12 bytes ---
  w[0] = 0x80;  // V=2, P=0, X=0, CC=0
  w[1] = static_cast<uint8_t>((p.marker ? 0x80 : 0x00) | (pt_ & 0x7f));
  put_be16(w + 2, p.sequence);
  put_be32(w + 4, p.rtp_timestamp);
  put_be32(w + 8, ssrc_);
  w += 12;

  // --- JPEG XS payload header (RFC 9134), 4 bytes ---
  // T=1 (sequential), K=0 (codestream mode), I=00 (progressive). {SEP,P} is the 22-bit packet
  // index: SEP is the high 11 bits, so it advances exactly when P overruns, as the RFC requires.
  uint32_t hdr = kBitT;
  if (p.marker) hdr |= kBitL;
  hdr |= (static_cast<uint32_t>(p.frame_counter) & kMaskF) << kShiftF;
  hdr |= ((p.packet_counter >> kShiftSep) & kMask11) << kShiftSep;
  hdr |= p.packet_counter & kMask11;
  put_be32(w, hdr);
  w += 4;

  // --- codestream fragment ---
  std::memcpy(w, codestream + p.offset, p.length);
}

uint32_t JxsPacketizer::packets_for(uint32_t codestream_bytes) const {
  const uint32_t frag = fragment_octets();
  return (codestream_bytes + frag - 1u) / frag;
}

size_t jxs_codestream_offset(const uint8_t* data, size_t len) {
  if (data == nullptr || len < 2) return 0;
  // Bound the walk: the video support boxes are small metadata. A stream whose "boxes" run past this
  // is not one we should keep parsing — hand it to the decoder unmodified and let it say so.
  constexpr size_t kMaxPrefix = 4096;
  constexpr int kMaxBoxes = 16;
  size_t off = 0;
  for (int i = 0; i < kMaxBoxes; ++i) {
    if (off + 2 > len) return 0;
    if (data[off] == 0xff && data[off + 1] == 0x10) return off;  // SOC: the codestream starts here
    if (off + 8 > len || off >= kMaxPrefix) return 0;            // no room for another box header
    const size_t size = (static_cast<size_t>(data[off]) << 24) |
                        (static_cast<size_t>(data[off + 1]) << 16) |
                        (static_cast<size_t>(data[off + 2]) << 8) |
                        static_cast<size_t>(data[off + 3]);
    // size 0 ("to end of data") and size 1 ("64-bit extended size") are legal ISOBMFF but would mean
    // no codestream follows in the first case and an unbounded header in the second; neither is a
    // JPEG XS picture segment, so treat both as malformed rather than guessing.
    if (size < 8 || size > len - off) return 0;
    off += size;
  }
  return 0;
}

bool JxsDepacketizer::parse(const uint8_t* p, uint32_t len, JxsRxPacketInfo& info) const {
  if (len < kJxsHeaderOctets) return false;      // RTP(12) + JXS payload header(4)
  if ((p[0] & 0xC0) != 0x80) return false;       // RTP version 2
  // CSRC list / header extension would shift the payload header; padding (P=1) would count its pad
  // octets into the codestream fragment. This pipeline's senders emit none of the three, and
  // silently mis-parsing any of them turns header/pad bytes into codestream bytes and corrupts a
  // whole frame — reject, so the RX drops the frame cleanly instead of feeding the decoder garbage.
  if ((p[0] & 0x3f) != 0) return false;

  info.marker = (p[1] & 0x80) != 0;
  info.sequence = static_cast<uint16_t>((p[2] << 8) | p[3]);
  info.rtp_timestamp = get_be32(p + 4);

  const uint32_t hdr = get_be32(p + 12);
  info.last = (hdr & kBitL) != 0;
  info.slice_mode = (hdr & kBitK) != 0;
  info.progressive = ((hdr >> kShiftI) & kMaskI) == 0;
  info.frame_counter = static_cast<uint8_t>((hdr >> kShiftF) & kMaskF);
  info.packet_counter = (((hdr >> kShiftSep) & kMask11) << kShiftSep) | (hdr & kMask11);
  info.data_offset = kJxsHeaderOctets;
  info.data_len = len - kJxsHeaderOctets;
  return true;
}

}  // namespace spark::st2110
