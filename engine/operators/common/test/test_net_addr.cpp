// Unit test for the ST 2110 multicast addressing helpers (net_addr.hpp). Pure CPU, no NIC/root.
// Covers the RFC 1112 MAC mapping (incl. the 23-bit overlap) and the IGMPv3 join frame layout +
// checksums (IP header and IGMP message must each checksum to zero when verified).
#include <cstdint>
#include <cstdio>

#include "../net_addr.hpp"

using namespace spark::net;

namespace {
int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
}
constexpr uint32_t ip(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  return (uint32_t(a) << 24) | (uint32_t(b) << 16) | (uint32_t(c) << 8) | d;
}
bool mac_eq(const uint8_t m[6], uint8_t a, uint8_t b, uint8_t c, uint8_t d, uint8_t e, uint8_t f) {
  return m[0] == a && m[1] == b && m[2] == c && m[3] == d && m[4] == e && m[5] == f;
}
}  // namespace

int main() {
  std::printf("----- ipv4_is_multicast -----\n");
  check(ipv4_is_multicast(ip(224, 0, 0, 1)), "224.0.0.1 is multicast");
  check(ipv4_is_multicast(ip(239, 100, 0, 10)), "239.100.0.10 is multicast");
  check(ipv4_is_multicast(ip(239, 255, 255, 255)), "239.255.255.255 is multicast");
  check(!ipv4_is_multicast(ip(192, 168, 1, 1)), "192.168.1.1 is not multicast");
  check(!ipv4_is_multicast(ip(223, 255, 255, 255)), "223.255.255.255 is not multicast");

  std::printf("----- multicast_mac (RFC 1112) -----\n");
  uint8_t m[6];
  multicast_mac(ip(239, 100, 0, 10), m);
  check(mac_eq(m, 0x01, 0x00, 0x5e, 0x64, 0x00, 0x0a), "239.100.0.10 -> 01:00:5e:64:00:0a");
  multicast_mac(ip(224, 0, 0, 22), m);
  check(mac_eq(m, 0x01, 0x00, 0x5e, 0x00, 0x00, 0x16), "224.0.0.22 -> 01:00:5e:00:00:16");
  // the high bit of the second octet is dropped: 239.128.0.10 and 239.0.0.10 share a MAC
  uint8_t m2[6];
  multicast_mac(ip(239, 128, 0, 10), m);
  multicast_mac(ip(239, 0, 0, 10), m2);
  check(mac_eq(m, m2[0], m2[1], m2[2], m2[3], m2[4], m2[5]), "23-bit overlap: 239.128.0.10 == 239.0.0.10 MAC");

  std::printf("----- IGMPv3 join (any-source / EXCLUDE) -----\n");
  uint8_t buf[128] = {};
  const uint8_t src_mac[6] = {0x30, 0xc5, 0x99, 0x3e, 0x9d, 0x2f};
  size_t n = build_igmpv3_join(buf, src_mac, ip(192, 168, 18, 101), ip(239, 100, 0, 10), 0);
  check(n == 14 + 24 + 16, "ASM frame length = 54");
  check(buf[12] == 0x08 && buf[13] == 0x00, "ethertype IPv4");
  check(mac_eq(buf, 0x01, 0x00, 0x5e, 0x00, 0x00, 0x16), "dst MAC = report group 224.0.0.22");
  const uint8_t* iph = buf + 14;
  check(iph[0] == 0x46, "IPv4 IHL=6 (router-alert option)");
  check(iph[9] == 2, "IP protocol = IGMP(2)");
  check(iph[8] == 1, "IP TTL = 1");
  check(inet_checksum(iph, 24) == 0, "IP header checksum verifies to 0");
  const uint8_t* igmp = iph + 24;
  check(igmp[0] == 0x22, "IGMPv3 membership report type 0x22");
  check(igmp[7] == 1, "one group record");
  check(igmp[8] == kIgmpChangeToExclude, "record type = CHANGE_TO_EXCLUDE (ASM)");
  check(igmp[10] == 0 && igmp[11] == 0, "zero sources");
  check(inet_checksum(igmp, 16) == 0, "IGMP checksum verifies to 0");

  std::printf("----- IGMPv3 join (source-specific / INCLUDE) -----\n");
  n = build_igmpv3_join(buf, src_mac, ip(192, 168, 18, 101), ip(239, 100, 0, 10), ip(192, 168, 18, 50));
  check(n == 14 + 24 + 20, "SSM frame length = 58");
  igmp = buf + 14 + 24;
  check(igmp[8] == kIgmpChangeToInclude, "record type = CHANGE_TO_INCLUDE (SSM)");
  check(igmp[11] == 1, "one source");
  check(igmp[16] == 192 && igmp[17] == 168 && igmp[18] == 18 && igmp[19] == 50, "source = 192.168.18.50");
  check(inet_checksum(igmp, 20) == 0, "IGMP checksum verifies to 0 (SSM)");

  std::printf(g_failures ? "\nFAILED (%d)\n" : "\nPASS\n", g_failures);
  return g_failures ? 1 : 0;
}
