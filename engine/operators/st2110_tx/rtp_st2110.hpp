// ST 2110-20 / RFC 4175 packetization (pure; no DPDK / Holoscan deps — fully unit-testable).
//
// Produces the UDP *payload* of each media packet: a 12-byte RTP header (RFC 3550) followed by the
// RFC 4175 payload header (2-byte Extended Sequence Number + one or more 6-byte Sample Row Data
// headers) and then the packed pixel octets. The Ethernet/IPv4/UDP encapsulation and the per-packet
// hardware send-timestamp are the transport backend's job (see tx_backend.hpp) — this module owns
// only the media framing, so it stays identical whether TX goes out raw DPDK or advanced_network.
//
// Walk model: a Packetizer holds the persistent 32-bit RTP sequence counter across frames. Per frame
// you call start_frame(rtp_ts) then next() repeatedly until it returns false; each call fills a
// PacketPlan describing one packet (no heap allocation). write_payload() then serializes a plan's
// bytes. Splitting plan/serialize lets the operator size/alloc a transport buffer before filling it.
#pragma once

#include <cstdint>

#include "st2110_format.hpp"

namespace spark::st2110 {

// One Sample Row Data segment: a contiguous run of pixels within a single scan line.
struct Srd {
  uint16_t length = 0;         // octets of pixel data in this segment (multiple of kOctetsPerPgroup)
  uint16_t line_no = 0;        // 0-based scan line
  uint16_t offset_pixels = 0;  // pgroup-aligned pixel offset of the segment start within the line
};

// One media packet's plan. A packet spans at most a couple of lines at our payload sizes (payload <
// one line), so kMaxSrd=3 is comfortably sufficient and keeps PacketPlan allocation-free.
struct PacketPlan {
  static constexpr int kMaxSrd = 3;
  uint32_t sequence = 0;       // 32-bit; low 16 -> RTP header, high 16 -> RFC 4175 ESN
  uint32_t rtp_timestamp = 0;  // 90 kHz media clock (same value for every packet of a frame)
  bool marker = false;         // RTP M bit: set on the last packet of the frame
  uint32_t payload_len = 0;    // total UDP payload bytes for this packet
  Srd srd[kMaxSrd];
  int nsrd = 0;
};

// Convert a PTP/wall-clock nanosecond timestamp to the 32-bit 90 kHz RTP media timestamp
// (ST 2110-10). Split into seconds/sub-seconds to avoid 64-bit overflow before the mod-2^32 wrap.
inline uint32_t rtp_timestamp_90k(uint64_t ns) {
  constexpr uint64_t kNsPerSec = 1000000000ULL;
  constexpr uint64_t kRate = 90000ULL;
  uint64_t whole = (ns / kNsPerSec) * kRate;
  uint64_t frac = (ns % kNsPerSec) * kRate / kNsPerSec;
  return static_cast<uint32_t>(whole + frac);
}

class Packetizer {
 public:
  // max_payload_octets: the UDP payload budget per packet (RTP + RFC 4175 headers + pixel data).
  // ~1420 keeps the on-wire L2 frame near 1438 B (matching the spike) inside a 1500 MTU.
  Packetizer(VideoFormat fmt, uint32_t max_payload_octets, uint8_t payload_type = 96,
             uint32_t ssrc = 0)
      : fmt_(fmt), max_payload_(max_payload_octets), pt_(payload_type), ssrc_(ssrc) {}

  // Begin a frame with its RTP media timestamp. Resets the per-frame line/offset cursor.
  void start_frame(uint32_t rtp_timestamp);

  // Fill `out` with the next packet's plan and advance. Returns false when the frame is exhausted.
  bool next(PacketPlan& out);

  // Serialize a plan into `dst` (caller guarantees capacity >= out.payload_len), copying pixel
  // octets from the packed `frame_data` buffer (size == fmt.octets_per_frame()).
  void write_payload(const PacketPlan& plan, const uint8_t* frame_data, uint8_t* dst) const;

  // Exact packet count for one frame at the current payload budget (for stats / pre-sizing rings).
  uint32_t packets_per_frame() const;

  const VideoFormat& format() const { return fmt_; }
  uint32_t ssrc() const { return ssrc_; }
  void set_ssrc(uint32_t ssrc) { ssrc_ = ssrc; }

 private:
  VideoFormat fmt_;
  uint32_t max_payload_;
  uint8_t pt_;
  uint32_t ssrc_;

  uint32_t seq_ = 0;          // persistent across frames (RTP requirement)
  uint32_t cur_line_ = 0;     // per-frame walk cursor
  uint32_t cur_octet_ = 0;    // octet offset within cur_line_ (pgroup-aligned)
  uint32_t rtp_ts_ = 0;
  bool active_ = false;
};

}  // namespace spark::st2110
