// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// ST 2110-30 audio (RFC 3190 L16/L24 PCM) framing + lip-sync delay — pure, no DPDK/runtime deps, so it
// unit-tests without a NIC. The processor passes audio through unchanged but DELAYED to match the
// video processing latency (FRC/resize add frames of delay; lip-sync needs the audio held the same).
//
// 2110-30 is far simpler than 2110-20: each RTP packet carries `samples_per_packet` audio samples for
// every channel, interleaved big-endian, with no payload header beyond the 12-byte RTP header. The RTP
// clock is the audio sample rate (e.g. 48 kHz), so the timestamp advances by samples_per_packet/pkt.
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

namespace spark::st2110 {

enum class AudioEncoding { L16, L24 };

struct AudioFormat {
  uint32_t sample_rate = 48000;
  uint16_t channels = 2;
  AudioEncoding encoding = AudioEncoding::L24;
  double packet_time_ms = 1.0;  // ST 2110-30: 1 ms (<=8 ch) or 125 us; sets samples per packet

  uint16_t bytes_per_sample() const { return encoding == AudioEncoding::L24 ? 3 : 2; }
  // one sample across all channels (the indivisible interleave unit on the wire)
  uint32_t channel_group_bytes() const { return uint32_t(channels) * bytes_per_sample(); }
  uint32_t samples_per_packet() const {
    return uint32_t(double(sample_rate) * packet_time_ms / 1000.0 + 0.5);
  }
  uint32_t packet_payload_bytes() const { return samples_per_packet() * channel_group_bytes(); }
};

inline AudioFormat audio_stereo_l24() { return AudioFormat{48000, 2, AudioEncoding::L24, 1.0}; }

// One packet's worth of interleaved PCM (kept in wire byte order for a zero-conversion passthrough),
// with the PTP capture time and the sample-clock RTP timestamp of its first sample.
struct AudioBlock {
  std::shared_ptr<std::vector<uint8_t>> pcm;  // size == format.packet_payload_bytes()
  AudioFormat format;
  uint64_t capture_ts_ns = 0;
  uint32_t rtp_timestamp = 0;
};

// 12-byte RTP header (RFC 3550), no CSRC/extension. Audio leaves the marker bit clear.
inline void write_rtp_header(uint8_t* p, uint8_t pt, bool marker, uint16_t seq, uint32_t ts, uint32_t ssrc) {
  p[0] = 0x80;  // V=2, P=0, X=0, CC=0
  p[1] = uint8_t((marker ? 0x80 : 0x00) | (pt & 0x7f));
  p[2] = uint8_t(seq >> 8);  p[3] = uint8_t(seq);
  p[4] = uint8_t(ts >> 24);  p[5] = uint8_t(ts >> 16);  p[6] = uint8_t(ts >> 8);  p[7] = uint8_t(ts);
  p[8] = uint8_t(ssrc >> 24); p[9] = uint8_t(ssrc >> 16); p[10] = uint8_t(ssrc >> 8); p[11] = uint8_t(ssrc);
}

// Sample-clock RTP timestamp from a PTP nanosecond time (ST 2110-10/-30, clock = sample_rate).
inline uint32_t rtp_timestamp_audio(uint64_t ns, uint32_t sample_rate) {
  constexpr uint64_t kNsPerSec = 1000000000ULL;
  const uint64_t whole = (ns / kNsPerSec) * sample_rate;
  const uint64_t frac = (ns % kNsPerSec) * sample_rate / kNsPerSec;
  return uint32_t(whole + frac);
}

constexpr uint32_t kRtpHeaderBytes = 12;

// Serialize one ST 2110-30 packet (RTP header + interleaved PCM payload).
class AudioPacketizer {
 public:
  AudioPacketizer(AudioFormat fmt, uint8_t payload_type = 97, uint32_t ssrc = 0)
      : fmt_(fmt), pt_(payload_type), ssrc_(ssrc) {}

  // Write one packet into `dst` (capacity >= 12 + packet_payload_bytes()); `pcm` holds exactly
  // packet_payload_bytes() interleaved bytes. Returns the total UDP-payload length.
  uint32_t write_packet(uint32_t rtp_timestamp, const uint8_t* pcm, uint8_t* dst);

  const AudioFormat& format() const { return fmt_; }
  uint32_t ssrc() const { return ssrc_; }
  void set_ssrc(uint32_t ssrc) { ssrc_ = ssrc; }

 private:
  AudioFormat fmt_;
  uint8_t pt_;
  uint32_t ssrc_;
  uint16_t seq_ = 0;  // persistent across packets (RTP requirement)
};

struct AudioRxPacketInfo {
  uint16_t sequence = 0;
  uint32_t rtp_timestamp = 0;
  bool marker = false;
  uint32_t data_offset = 0;  // = kRtpHeaderBytes
  uint32_t pcm_len = 0;      // payload bytes after the RTP header
};

// Parse one ST 2110-30 UDP payload back into header info (the PCM begins at data_offset).
class AudioDepacketizer {
 public:
  explicit AudioDepacketizer(AudioFormat fmt) : fmt_(fmt) {}
  bool parse(const uint8_t* payload, uint32_t len, AudioRxPacketInfo& info) const;
  const AudioFormat& format() const { return fmt_; }

 private:
  AudioFormat fmt_;
};

// Fixed-length lip-sync delay: holds `delay_blocks` packets, releasing the oldest as new ones arrive.
// delay seconds = delay_blocks * packet_time_ms / 1000.
class AudioDelayLine {
 public:
  explicit AudioDelayLine(size_t delay_blocks) : delay_blocks_(delay_blocks) {}

  // Push an input block; if the line is full, set `out` to the released (delayed) block and return
  // true, else return false (still filling). delay_blocks == 0 passes through immediately.
  bool push(AudioBlock in, AudioBlock& out) {
    if (delay_blocks_ == 0) { out = std::move(in); return true; }
    q_.push_back(std::move(in));
    if (q_.size() <= delay_blocks_) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
  }
  size_t delay_blocks() const { return delay_blocks_; }
  size_t buffered() const { return q_.size(); }

 private:
  size_t delay_blocks_;
  std::deque<AudioBlock> q_;
};

}  // namespace spark::st2110
