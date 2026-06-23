#include "rtp_st2110.hpp"

#include <algorithm>
#include <cstring>

namespace spark::st2110 {
namespace {

struct Cursor {
  uint32_t line;
  uint32_t octet;  // octet offset within `line` (pgroup-aligned)
};

// Plan one packet starting at `c`, greedily filling SRDs up to the payload budget, advancing `c`.
// Returns false when there is no data left (c already past the last line). Fills srd[]/nsrd/
// payload_len; sequence/timestamp/marker are stamped by the caller.
bool plan_one(const VideoFormat& fmt, uint32_t max_payload, Cursor& c, PacketPlan& out) {
  if (c.line >= fmt.height) return false;
  const uint32_t opl = fmt.octets_per_line();
  const uint32_t pg = fmt.octets_per_pgroup();

  // Budget for SRD headers + pixel data = UDP payload - 12 (RTP) - 2 (ESN).
  int budget = static_cast<int>(max_payload) - 12 - 2;
  out.nsrd = 0;
  while (c.line < fmt.height && out.nsrd < PacketPlan::kMaxSrd) {
    const int after_hdr = budget - 6;  // reserve this SRD's 6-byte header
    if (after_hdr < static_cast<int>(pg)) break;
    const uint32_t left_in_line = opl - c.octet;
    uint32_t seg = std::min<uint32_t>(left_in_line, static_cast<uint32_t>(after_hdr));
    seg = (seg / pg) * pg;  // pgroup-align the segment
    if (seg == 0) break;

    Srd& s = out.srd[out.nsrd++];
    s.length = static_cast<uint16_t>(seg);
    s.line_no = static_cast<uint16_t>(c.line);
    s.offset_pixels = static_cast<uint16_t>((c.octet / pg) * VideoFormat::kPixelsPerPgroup);

    budget -= static_cast<int>(6 + seg);
    c.octet += seg;
    if (c.octet >= opl) {  // segment reached end of line -> wrap to next line
      c.octet = 0;
      ++c.line;
    }
  }

  uint32_t data = 0;
  for (int i = 0; i < out.nsrd; ++i) data += out.srd[i].length;
  out.payload_len = 12u + 2u + 6u * static_cast<uint32_t>(out.nsrd) + data;
  return out.nsrd > 0;
}

inline void put_be16(uint8_t* p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v >> 8);
  p[1] = static_cast<uint8_t>(v & 0xff);
}

}  // namespace

void Packetizer::start_frame(uint32_t rtp_timestamp) {
  cur_line_ = 0;
  cur_octet_ = 0;
  rtp_ts_ = rtp_timestamp;
  active_ = true;
}

bool Packetizer::next(PacketPlan& out) {
  if (!active_) return false;
  Cursor c{cur_line_, cur_octet_};
  if (!plan_one(fmt_, max_payload_, c, out)) {
    active_ = false;
    return false;
  }
  out.sequence = seq_++;
  out.rtp_timestamp = rtp_ts_;
  cur_line_ = c.line;
  cur_octet_ = c.octet;
  out.marker = (cur_line_ >= fmt_.height);  // RTP M bit on the final packet of the frame
  if (out.marker) active_ = false;
  return true;
}

void Packetizer::write_payload(const PacketPlan& p, const uint8_t* frame, uint8_t* dst) const {
  uint8_t* w = dst;

  // --- RTP header (RFC 3550), 12 bytes ---
  w[0] = 0x80;  // V=2, P=0, X=0, CC=0
  w[1] = static_cast<uint8_t>((p.marker ? 0x80 : 0x00) | (pt_ & 0x7f));
  put_be16(w + 2, static_cast<uint16_t>(p.sequence & 0xffff));  // low 16 bits of the sequence
  w[4] = static_cast<uint8_t>((p.rtp_timestamp >> 24) & 0xff);
  w[5] = static_cast<uint8_t>((p.rtp_timestamp >> 16) & 0xff);
  w[6] = static_cast<uint8_t>((p.rtp_timestamp >> 8) & 0xff);
  w[7] = static_cast<uint8_t>(p.rtp_timestamp & 0xff);
  w[8] = static_cast<uint8_t>((ssrc_ >> 24) & 0xff);
  w[9] = static_cast<uint8_t>((ssrc_ >> 16) & 0xff);
  w[10] = static_cast<uint8_t>((ssrc_ >> 8) & 0xff);
  w[11] = static_cast<uint8_t>(ssrc_ & 0xff);
  w += 12;

  // --- RFC 4175 payload header: Extended Sequence Number (high 16 bits of the 32-bit sequence) ---
  put_be16(w, static_cast<uint16_t>((p.sequence >> 16) & 0xffff));
  w += 2;

  // --- Sample Row Data headers: Length(16) | Field|LineNo(16) | Continuation|Offset(16) ---
  // Line numbers are 0-based here; verify against the target receiver before production interop.
  for (int i = 0; i < p.nsrd; ++i) {
    const Srd& s = p.srd[i];
    const bool cont = (i + 1) < p.nsrd;  // continuation bit set on every SRD but the last
    put_be16(w + 0, s.length);
    put_be16(w + 2, static_cast<uint16_t>(s.line_no & 0x7fff));  // Field bit 0 (progressive)
    put_be16(w + 4, static_cast<uint16_t>((cont ? 0x8000 : 0x0000) | (s.offset_pixels & 0x7fff)));
    w += 6;
  }

  // --- Pixel octets, in SRD order ---
  for (int i = 0; i < p.nsrd; ++i) {
    const Srd& s = p.srd[i];
    std::memcpy(w, frame + fmt_.byte_offset(s.line_no, s.offset_pixels), s.length);
    w += s.length;
  }
}

uint32_t Packetizer::packets_per_frame() const {
  Cursor c{0, 0};
  PacketPlan tmp;
  uint32_t n = 0;
  while (plan_one(fmt_, max_payload_, c, tmp)) ++n;
  return n;
}

bool Depacketizer::parse(const uint8_t* p, uint32_t len, RxPacketInfo& info) const {
  if (len < 12 + 2 + 6) return false;     // RTP(12) + ESN(2) + >=1 SRD(6)
  if ((p[0] & 0xC0) != 0x80) return false;  // RTP version 2

  info.marker = (p[1] & 0x80) != 0;
  const uint16_t seq_lo = static_cast<uint16_t>((p[2] << 8) | p[3]);
  info.rtp_timestamp = (static_cast<uint32_t>(p[4]) << 24) | (static_cast<uint32_t>(p[5]) << 16) |
                       (static_cast<uint32_t>(p[6]) << 8) | p[7];
  const uint16_t esn = static_cast<uint16_t>((p[12] << 8) | p[13]);
  info.sequence = (static_cast<uint32_t>(esn) << 16) | seq_lo;

  uint32_t off = 14;
  info.nsrd = 0;
  for (;;) {
    if (off + 6 > len) return false;
    if (info.nsrd >= PacketPlan::kMaxSrd) return false;
    const uint16_t length = static_cast<uint16_t>((p[off] << 8) | p[off + 1]);
    const uint16_t line = static_cast<uint16_t>(((p[off + 2] << 8) | p[off + 3]) & 0x7fff);
    const uint16_t f3 = static_cast<uint16_t>((p[off + 4] << 8) | p[off + 5]);
    const bool cont = (f3 & 0x8000) != 0;
    Srd& s = info.srd[info.nsrd++];
    s.length = length;
    s.line_no = line;
    s.offset_pixels = static_cast<uint16_t>(f3 & 0x7fff);
    off += 6;
    if (!cont) break;
  }
  info.data_offset = off;

  uint32_t data = 0;
  for (int i = 0; i < info.nsrd; ++i) {
    const Srd& s = info.srd[i];
    if (s.length % fmt_.octets_per_pgroup() != 0) return false;
    if (s.line_no >= fmt_.height) return false;
    if (fmt_.byte_offset(s.line_no, s.offset_pixels) + s.length > fmt_.octets_per_frame())
      return false;
    data += s.length;
  }
  return info.data_offset + data <= len;
}

void Depacketizer::scatter(const RxPacketInfo& info, const uint8_t* p, uint8_t* frame) const {
  uint32_t off = info.data_offset;
  for (int i = 0; i < info.nsrd; ++i) {
    const Srd& s = info.srd[i];
    std::memcpy(frame + fmt_.byte_offset(s.line_no, s.offset_pixels), p + off, s.length);
    off += s.length;
  }
}

}  // namespace spark::st2110
