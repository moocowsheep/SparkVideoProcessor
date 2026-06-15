// ST 2110-20 video format + frame message types (pure; no DPDK / Holoscan deps).
//
// Scope for M1 v1: YCbCr 4:2:2 10-bit — the SMPTE ST 2110-20 broadcast baseline and the format the
// gate-4 spike paced (spike/st2110_loopback_gate4.sh). Other samplings (RGB, 4:4:4, 8/12-bit) are a
// later addition; isolating the pgroup geometry here is what makes that a localized change.
//
// RFC 4175 "pgroup": the smallest set of pixels whose sample bits pack to a whole number of octets.
// For 4:2:2 10-bit one pgroup = 2 pixels = {Cb0,Y0,Cr0,Y1} = 4 samples x 10 bits = 40 bits = 5 octets.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace spark::st2110 {

enum class Sampling {
  YCbCr422_10,  // 2 pixels/pgroup, 5 octets/pgroup (v1 target)
};

struct VideoFormat {
  uint32_t width = 0;
  uint32_t height = 0;
  double fps = 0.0;                       // e.g. 59.94 (60000/1001)
  Sampling sampling = Sampling::YCbCr422_10;

  // --- RFC 4175 pgroup geometry (4:2:2 10-bit) ---
  static constexpr uint32_t kPixelsPerPgroup = 2;
  static constexpr uint32_t kOctetsPerPgroup = 5;

  uint32_t pgroups_per_line() const { return width / kPixelsPerPgroup; }
  uint32_t octets_per_line() const { return pgroups_per_line() * kOctetsPerPgroup; }
  uint64_t octets_per_frame() const {
    return static_cast<uint64_t>(octets_per_line()) * height;
  }
  // Byte offset, within a packed frame buffer, of a pixel position on a given line.
  // offset_pixels must be pgroup-aligned (a multiple of kPixelsPerPgroup).
  uint64_t byte_offset(uint32_t line, uint32_t offset_pixels) const {
    return static_cast<uint64_t>(line) * octets_per_line() +
           (offset_pixels / kPixelsPerPgroup) * kOctetsPerPgroup;
  }
};

// Standard profiles, kept in sync with the gate-4 spike's rate table.
inline VideoFormat profile_1080p() { return VideoFormat{1920, 1080, 60000.0 / 1001.0}; }  // ~2.97 Gbps
inline VideoFormat profile_2160p() { return VideoFormat{3840, 2160, 60000.0 / 1001.0}; }  // ~11.9 Gbps

// Frame message passed between operators. v1 scaffold carries a host buffer of packed ST 2110-20
// octets; the production path replaces `data` with a zero-copy GPU tensor (the operator's compute()
// is the only place that touches it, so that swap is localized). `capture_ts_ns` is the PTP capture
// time that becomes the RTP media timestamp (ST 2110-10, 90 kHz).
struct VideoFrame {
  std::shared_ptr<std::vector<uint8_t>> data;  // size == format.octets_per_frame()
  VideoFormat format;
  uint64_t capture_ts_ns = 0;
  uint64_t frame_number = 0;
};

}  // namespace spark::st2110
