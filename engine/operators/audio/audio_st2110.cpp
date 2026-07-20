// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// ST 2110-30 audio packetizer/depacketizer implementation (pure; see audio_st2110.hpp).
#include "audio_st2110.hpp"

#include <cstring>

namespace spark::st2110 {

uint32_t AudioPacketizer::write_packet(uint32_t rtp_timestamp, const uint8_t* pcm, uint8_t* dst) {
  const uint32_t payload = fmt_.packet_payload_bytes();
  write_rtp_header(dst, pt_, /*marker=*/false, seq_++, rtp_timestamp, ssrc_);
  std::memcpy(dst + kRtpHeaderBytes, pcm, payload);  // interleaved PCM, already wire byte order
  return kRtpHeaderBytes + payload;
}

bool AudioDepacketizer::parse(const uint8_t* payload, uint32_t len, AudioRxPacketInfo& info) const {
  if (len < kRtpHeaderBytes) return false;
  if ((payload[0] >> 6) != 2) return false;  // RTP version 2
  const uint8_t cc = payload[0] & 0x0f;       // CSRC count
  const uint32_t header = kRtpHeaderBytes + uint32_t(cc) * 4;
  if (len < header) return false;
  info.marker = (payload[1] & 0x80) != 0;
  info.sequence = uint16_t((payload[2] << 8) | payload[3]);
  info.rtp_timestamp = (uint32_t(payload[4]) << 24) | (uint32_t(payload[5]) << 16) |
                       (uint32_t(payload[6]) << 8) | uint32_t(payload[7]);
  info.data_offset = header;
  info.pcm_len = len - header;
  // The payload must be a whole number of channel-groups (defensive against a malformed wire packet).
  const uint32_t group = fmt_.channel_group_bytes();
  if (group == 0 || info.pcm_len % group != 0) return false;
  return true;
}

}  // namespace spark::st2110
