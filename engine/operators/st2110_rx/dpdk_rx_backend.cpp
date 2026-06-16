// Raw DPDK + mlx5 implementation of ISt2110RxBackend. Receives ST 2110-20 UDP/IP frames on the RX
// port, returns the RTP payload (zero-copy, pointing into the mbuf) plus the NIC HW RX timestamp.
// All DPDK headers confined to this TU. Two-process loopback partner of dpdk_tx_backend.cpp.
//
// STATUS: compiles against DPDK 23.11; runtime bring-up marked  // BRINGUP (clock units, rx ts flag).
#include <arpa/inet.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <rte_dev.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_mbuf.h>
#include <rte_mbuf_dyn.h>
#include <rte_udp.h>

#include "../common/dpdk_eal.hpp"
#include "../common/net_addr.hpp"
#include "rx_backend.hpp"

namespace spark::net {
namespace {

// Find the DPDK port whose PCI BDF matches `pci` (rte_device::name is the BDF for PCI devices).
uint16_t find_port_by_pci(const std::string& pci) {
  uint16_t p;
  RTE_ETH_FOREACH_DEV(p) {
    rte_eth_dev_info di{};
    if (rte_eth_dev_info_get(p, &di) != 0 || !di.device) continue;
    const char* name = rte_dev_name(di.device);  // BDF for PCI devices (rte_device is opaque)
    if (name && pci == name) return p;
  }
  return RTE_MAX_ETHPORTS;
}

constexpr uint16_t kRxBurst = 256;
constexpr uint32_t kMbufCount = 16384;

[[noreturn]] void die(const std::string& msg) {
  throw std::runtime_error("dpdk_rx_backend: " + msg);
}

class DpdkRxBackend final : public ISt2110RxBackend {
 public:
  void init(const RxBackendConfig& cfg) override {
    cfg_ = cfg;
    udp_port_be_ = rte_cpu_to_be_16(cfg_.udp_port);
    parse_addrs();
    if (cfg_.manage_eal)
      own_eal_init();
    else if (!DpdkEal::instance().initialized())
      die("manage_eal=false but shared EAL is not initialized (call DpdkEal::init first)");
    create_pool();
    port_ = find_port_by_pci(cfg_.pci_addr);
    if (port_ == RTE_MAX_ETHPORTS) die("no DPDK port matches PCI " + cfg_.pci_addr);
    setup_port();
    lookup_rx_timestamp();
    if (rte_eth_dev_start(port_) < 0) die("rte_eth_dev_start failed");
    if (group_be_) {
      rte_eth_allmulticast_enable(port_);  // accept the group's multicast MAC (no promiscuous flood)
      send_igmp_join();                    // make the switch's IGMP snooping forward the group to us
    } else {
      rte_eth_promiscuous_enable(port_);   // legacy loopback: accept all, filter by dst port only
    }
    std::printf("[st2110_rx] port %u up: udp_port=%u group=%s rxd=%u rx_timestamp=%d\n", port_,
                cfg_.udp_port, cfg_.mcast_group.empty() ? "(none)" : cfg_.mcast_group.c_str(),
                cfg_.rxd, have_ts_);
  }

  uint16_t receive(RxPacket* out, uint16_t max) override {
    rte_mbuf* bufs[kRxBurst];
    const uint16_t want = max < kRxBurst ? max : kRxBurst;
    const uint16_t n = rte_eth_rx_burst(port_, 0, bufs, want);
    raw_received_ += n;
    uint16_t k = 0;
    for (uint16_t i = 0; i < n; ++i) {
      rte_mbuf* m = bufs[i];
      RxPacket pkt;
      if (parse_udp(m, pkt)) {
        out[k++] = pkt;
      } else {
        rte_pktmbuf_free(m);  // not our UDP stream — drop now
      }
    }
    return k;
  }

  void release(RxPacket* pkts, uint16_t n) override {
    for (uint16_t i = 0; i < n; ++i)
      if (pkts[i].opaque) rte_pktmbuf_free(static_cast<rte_mbuf*>(pkts[i].opaque));
  }

  uint64_t now_ns() override {
    uint64_t clk = 0;
    // BRINGUP: same realtime-clock-as-ns assumption as the TX backend; PHC-shared with the rx ts.
    if (rte_eth_read_clock(port_, &clk) != 0) die("rte_eth_read_clock unsupported on this port");
    return clk;
  }

  RxStats stats() override {
    RxStats s{};
    rte_eth_stats es{};
    if (rte_eth_stats_get(port_, &es) == 0) {
      s.rx_packets = es.ipackets;
      s.rx_bytes = es.ibytes;
      s.rx_missed = es.imissed;
      s.rx_nombuf = es.rx_nombuf;
    }
    s.raw_received = raw_received_;
    return s;
  }

  void shutdown() override {
    if (port_ != RTE_MAX_ETHPORTS) {
      rte_eth_dev_stop(port_);
      rte_eth_dev_close(port_);
      port_ = RTE_MAX_ETHPORTS;
    }
    if (eal_inited_) {
      rte_eal_cleanup();
      eal_inited_ = false;
    }
  }

  ~DpdkRxBackend() override { shutdown(); }

 private:
  // Validate Eth(IPv4)/UDP and (dst port) and fill pkt with the UDP payload + HW rx timestamp.
  bool parse_udp(rte_mbuf* m, RxPacket& pkt) {
    if (m->data_len < sizeof(rte_ether_hdr) + sizeof(rte_ipv4_hdr) + sizeof(rte_udp_hdr))
      return false;
    auto* eth = rte_pktmbuf_mtod(m, rte_ether_hdr*);
    if (eth->ether_type != rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4)) return false;
    auto* ip = reinterpret_cast<rte_ipv4_hdr*>(reinterpret_cast<uint8_t*>(eth) +
                                               sizeof(rte_ether_hdr));
    if (ip->next_proto_id != IPPROTO_UDP) return false;
    const uint8_t ihl = (ip->version_ihl & 0x0f) * 4;
    auto* udp = reinterpret_cast<rte_udp_hdr*>(reinterpret_cast<uint8_t*>(ip) + ihl);
    if (udp->dst_port != udp_port_be_) return false;
    if (group_be_ && ip->dst_addr != group_be_) return false;      // only our multicast group
    if (ssm_src_be_ && ip->src_addr != ssm_src_be_) return false;  // ST 2110 source-specific filter

    const uint16_t dgram = rte_be_to_cpu_16(udp->dgram_len);
    if (dgram < sizeof(rte_udp_hdr)) return false;
    pkt.payload = reinterpret_cast<uint8_t*>(udp) + sizeof(rte_udp_hdr);
    pkt.len = dgram - sizeof(rte_udp_hdr);
    pkt.opaque = m;
    if (have_ts_ && (m->ol_flags & rx_ts_flag_)) {
      pkt.hw_timestamp_ns = *RTE_MBUF_DYNFIELD(m, ts_field_off_, uint64_t*);
      pkt.has_timestamp = true;
    }
    return true;
  }

  // Parse the (optional) multicast group / SSM source / interface IP from the config (network order,
  // except iface which IGMP wants in host order). 0 means "unset".
  void parse_addrs() {
    if (!cfg_.mcast_group.empty() && inet_pton(AF_INET, cfg_.mcast_group.c_str(), &group_be_) != 1)
      die("bad mcast_group " + cfg_.mcast_group);
    if (!cfg_.src_ip.empty() && inet_pton(AF_INET, cfg_.src_ip.c_str(), &ssm_src_be_) != 1)
      die("bad src_ip " + cfg_.src_ip);
    if (!cfg_.iface_ip.empty()) {
      uint32_t be = 0;
      if (inet_pton(AF_INET, cfg_.iface_ip.c_str(), &be) != 1) die("bad iface_ip " + cfg_.iface_ip);
      iface_ip_host_ = rte_be_to_cpu_32(be);
    }
  }

  // Emit an IGMPv3 Membership Report out the RX port so the fabric forwards the group to us. Sent
  // once at join; a production node also answers periodic general queries (follow-on).
  void send_igmp_join() {
    rte_ether_addr mac{};
    rte_eth_macaddr_get(port_, &mac);
    rte_mbuf* m = rte_pktmbuf_alloc(pool_);
    if (!m) { std::printf("[st2110_rx] WARNING: IGMP join skipped (mbuf alloc failed)\n"); return; }
    uint8_t* p = rte_pktmbuf_mtod(m, uint8_t*);
    size_t len = spark::net::build_igmpv3_join(p, mac.addr_bytes, iface_ip_host_,
                                               rte_be_to_cpu_32(group_be_),
                                               ssm_src_be_ ? rte_be_to_cpu_32(ssm_src_be_) : 0);
    if (len < 60) { std::memset(p + len, 0, 60 - len); len = 60; }  // pad to the Ethernet minimum
    m->data_len = static_cast<uint16_t>(len);
    m->pkt_len = static_cast<uint32_t>(len);
    if (rte_eth_tx_burst(port_, 0, &m, 1) != 1) {
      rte_pktmbuf_free(m);
      std::printf("[st2110_rx] WARNING: IGMP join report not sent (tx_burst=0)\n");
    } else {
      std::printf("[st2110_rx] IGMP join sent for %s%s%s\n", cfg_.mcast_group.c_str(),
                  cfg_.src_ip.empty() ? "" : " src ", cfg_.src_ip.c_str());
    }
  }

  void own_eal_init() {
    std::vector<std::string> args = {"spark_rx", "-l",           cfg_.eal_core_list,
                                     "-a",       cfg_.pci_addr,  "--file-prefix",
                                     cfg_.file_prefix};
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    if (rte_eal_init(static_cast<int>(argv.size()), argv.data()) < 0)
      die("rte_eal_init failed (root? hugepages? PCI " + cfg_.pci_addr + "?)");
    eal_inited_ = true;
  }

  void create_pool() {
    pool_ = rte_pktmbuf_pool_create("spark_rx_pool", kMbufCount, 256, 0,
                                    RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if (!pool_) die("rte_pktmbuf_pool_create failed");
  }

  void setup_port() {
    rte_eth_dev_info dev_info{};
    if (rte_eth_dev_info_get(port_, &dev_info) != 0) die("rte_eth_dev_info_get failed");

    rte_eth_conf port_conf{};
    if (dev_info.rx_offload_capa & RTE_ETH_RX_OFFLOAD_TIMESTAMP)
      port_conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_TIMESTAMP;
    // A TX queue is configured only to emit IGMP membership reports for the joined group.
    const uint16_t nb_tx_q = group_be_ ? 1 : 0;
    if (rte_eth_dev_configure(port_, 1, nb_tx_q, &port_conf) < 0) die("rte_eth_dev_configure failed");

    uint16_t nb_rxd = cfg_.rxd, nb_txd = group_be_ ? 64 : 0;
    if (rte_eth_dev_adjust_nb_rx_tx_desc(port_, &nb_rxd, &nb_txd) < 0)
      die("adjust_nb_rx_tx_desc failed");
    cfg_.rxd = nb_rxd;

    rte_eth_rxconf rxconf = dev_info.default_rxconf;
    rxconf.offloads = port_conf.rxmode.offloads;
    if (rte_eth_rx_queue_setup(port_, 0, nb_rxd, rte_eth_dev_socket_id(port_), &rxconf, pool_) < 0)
      die("rte_eth_rx_queue_setup failed");
    if (nb_tx_q &&
        rte_eth_tx_queue_setup(port_, 0, nb_txd, rte_eth_dev_socket_id(port_), nullptr) < 0)
      die("rte_eth_tx_queue_setup (IGMP) failed");
  }

  void lookup_rx_timestamp() {
    ts_field_off_ = rte_mbuf_dynfield_lookup(RTE_MBUF_DYNFIELD_TIMESTAMP_NAME, nullptr);
    const int flag_bit = rte_mbuf_dynflag_lookup(RTE_MBUF_DYNFLAG_RX_TIMESTAMP_NAME, nullptr);
    if (ts_field_off_ < 0 || flag_bit < 0) {
      std::printf("[st2110_rx] WARNING: rx timestamp dynfield/flag absent — ingest latency disabled\n");
      have_ts_ = false;
      return;
    }
    rx_ts_flag_ = 1ULL << flag_bit;
    have_ts_ = true;
  }

  RxBackendConfig cfg_;
  bool eal_inited_ = false;
  uint16_t port_ = RTE_MAX_ETHPORTS;
  rte_mempool* pool_ = nullptr;
  uint16_t udp_port_be_ = 0;
  uint32_t group_be_ = 0, ssm_src_be_ = 0;  // network order; 0 = unset (legacy promiscuous path)
  uint32_t iface_ip_host_ = 0;              // host order; IGMP report source address

  int ts_field_off_ = -1;
  uint64_t rx_ts_flag_ = 0;
  bool have_ts_ = false;
  uint64_t raw_received_ = 0;
};

}  // namespace

std::unique_ptr<ISt2110RxBackend> make_dpdk_rx_backend() {
  return std::make_unique<DpdkRxBackend>();
}

}  // namespace spark::net
