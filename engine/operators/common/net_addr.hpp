// ST 2110 multicast addressing — pure, header-only, no DPDK (so it unit-tests without a NIC/root).
//
// Two jobs the M6 multicast path needs and that are easy to get subtly wrong:
//   * RFC 1112 IPv4-multicast -> Ethernet MAC mapping (01:00:5e + low 23 bits), for TX egress.
//   * Building an IGMPv3 Membership Report so a switch with IGMP snooping forwards the group to us
//     (a DPDK app owns the port, so there is no kernel IGMP stack to do this for us).
// All integers here are HOST byte order unless a name ends in _be; buffers are written big-endian.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace spark::net {

// IPv4 multicast is 224.0.0.0/4.
inline bool ipv4_is_multicast(uint32_t ip_host) { return (ip_host & 0xf0000000u) == 0xe0000000u; }

// RFC 1112 §6.4: low 23 bits of the group address map into 01:00:5e:00:00:00. (The 24→23-bit loss
// means 32 group addresses share one MAC — callers still IP-filter, so this is fine.)
inline void multicast_mac(uint32_t group_host, uint8_t mac[6]) {
  mac[0] = 0x01; mac[1] = 0x00; mac[2] = 0x5e;
  mac[3] = static_cast<uint8_t>((group_host >> 16) & 0x7f);
  mac[4] = static_cast<uint8_t>((group_host >> 8) & 0xff);
  mac[5] = static_cast<uint8_t>(group_host & 0xff);
}

// Internet (one's-complement) checksum over a big-endian byte buffer; result is host order.
inline uint16_t inet_checksum(const uint8_t* data, size_t len) {
  uint32_t sum = 0;
  for (size_t i = 0; i + 1 < len; i += 2) sum += (uint32_t(data[i]) << 8) | data[i + 1];
  if (len & 1) sum += uint32_t(data[len - 1]) << 8;
  while (sum >> 16) sum = (sum & 0xffff) + (sum >> 16);
  return static_cast<uint16_t>(~sum & 0xffff);
}

namespace detail {
inline void put_be16(uint8_t* p, uint16_t v) { p[0] = uint8_t(v >> 8); p[1] = uint8_t(v); }
inline void put_be32(uint8_t* p, uint32_t v) {
  p[0] = uint8_t(v >> 24); p[1] = uint8_t(v >> 16); p[2] = uint8_t(v >> 8); p[3] = uint8_t(v);
}
}  // namespace detail

// IGMPv3 group-record types (RFC 3376 §4.2.12).
enum : uint8_t { kIgmpChangeToInclude = 3, kIgmpChangeToExclude = 4 };

// Build a complete Ethernet/IPv4(+Router-Alert)/IGMPv3 Membership Report that joins `group_host`.
// If `source_host != 0` it joins source-specific (INCLUDE{source}, for SSM/ST 2110 source-filter);
// otherwise any-source (EXCLUDE{}). Writes the frame into `buf` and returns its length (before the
// caller pads to the 60-byte Ethernet minimum). `buf` must hold >= 58 bytes.
inline size_t build_igmpv3_join(uint8_t* buf, const uint8_t src_mac[6], uint32_t iface_ip_host,
                                uint32_t group_host, uint32_t source_host) {
  using namespace detail;
  constexpr uint32_t kReportDst = 0xe0000016u;  // 224.0.0.22, the all-IGMPv3-routers group
  const bool ssm = source_host != 0;
  const uint16_t nsrc = ssm ? 1 : 0;
  const uint16_t igmp_len = 8 + 8 + 4 * nsrc;  // report header + one group record (+ sources)

  // Ethernet (14)
  multicast_mac(kReportDst, buf);
  std::memcpy(buf + 6, src_mac, 6);
  put_be16(buf + 12, 0x0800);  // IPv4

  // IPv4 header with the Router-Alert option -> IHL = 6 (24 bytes)
  uint8_t* ip = buf + 14;
  ip[0] = 0x46;                          // version 4, IHL 6
  ip[1] = 0xc0;                          // DSCP CS6 (internetwork control)
  put_be16(ip + 2, uint16_t(24 + igmp_len));
  put_be16(ip + 4, 0);                   // identification
  put_be16(ip + 6, 0x4000);              // don't fragment
  ip[8] = 1;                             // TTL 1 (link-local control traffic)
  ip[9] = 2;                             // protocol = IGMP
  put_be16(ip + 10, 0);                  // header checksum (filled below)
  put_be32(ip + 12, iface_ip_host);      // source (0.0.0.0 is valid before an IP is assigned)
  put_be32(ip + 16, kReportDst);
  ip[20] = 0x94; ip[21] = 0x04; ip[22] = 0x00; ip[23] = 0x00;  // IP Router Alert (RFC 2113)
  put_be16(ip + 10, inet_checksum(ip, 24));

  // IGMPv3 Membership Report (type 0x22)
  uint8_t* ig = ip + 24;
  ig[0] = 0x22; ig[1] = 0;     // type, reserved
  put_be16(ig + 2, 0);         // checksum (filled below)
  put_be16(ig + 4, 0);         // reserved
  put_be16(ig + 6, 1);         // number of group records
  uint8_t* rec = ig + 8;
  rec[0] = ssm ? kIgmpChangeToInclude : kIgmpChangeToExclude;
  rec[1] = 0;                  // aux data length
  put_be16(rec + 2, nsrc);     // number of sources
  put_be32(rec + 4, group_host);
  if (ssm) put_be32(rec + 8, source_host);
  put_be16(ig + 2, inet_checksum(ig, igmp_len));

  return 14u + 24u + igmp_len;
}

}  // namespace spark::net
