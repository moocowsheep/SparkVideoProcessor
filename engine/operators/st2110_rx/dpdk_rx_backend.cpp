// Raw DPDK + mlx5 implementation of ISt2110RxBackend. Receives ST 2110-20 UDP/IP frames on the RX
// port, returns the RTP payload (zero-copy, pointing into the mbuf) plus the NIC HW RX timestamp.
// All DPDK headers confined to this TU. Two-process loopback partner of dpdk_tx_backend.cpp.
//
// STATUS: compiles against DPDK 23.11; runtime bring-up marked  // BRINGUP (clock units, rx ts flag).
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_mbuf.h>
#include <rte_mbuf_dyn.h>
#include <rte_udp.h>

#include "rx_backend.hpp"

namespace spark::net {
namespace {

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
    init_eal();
    pick_port();
    setup_port();
    lookup_rx_timestamp();
    if (rte_eth_dev_start(port_) < 0) die("rte_eth_dev_start failed");
    rte_eth_promiscuous_enable(port_);  // switched fabric may flood; accept all then filter by port
    std::printf("[st2110_rx] port %u up: udp_port=%u rxd=%u rx_timestamp=%d\n", port_, cfg_.udp_port,
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

  void init_eal() {
    std::vector<std::string> args = {"spark_rx", "-l",           cfg_.eal_core_list,
                                     "-a",       cfg_.pci_addr,  "--file-prefix",
                                     cfg_.file_prefix};
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    if (rte_eal_init(static_cast<int>(argv.size()), argv.data()) < 0)
      die("rte_eal_init failed (root? hugepages? PCI " + cfg_.pci_addr + "?)");
    eal_inited_ = true;
    pool_ = rte_pktmbuf_pool_create("spark_rx_pool", kMbufCount, 256, 0,
                                    RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if (!pool_) die("rte_pktmbuf_pool_create failed");
  }

  void pick_port() {
    uint16_t p;
    RTE_ETH_FOREACH_DEV(p) {
      port_ = p;
      return;
    }
    die("no DPDK eth port found (check -a " + cfg_.pci_addr + ")");
  }

  void setup_port() {
    rte_eth_dev_info dev_info{};
    if (rte_eth_dev_info_get(port_, &dev_info) != 0) die("rte_eth_dev_info_get failed");

    rte_eth_conf port_conf{};
    if (dev_info.rx_offload_capa & RTE_ETH_RX_OFFLOAD_TIMESTAMP)
      port_conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_TIMESTAMP;
    if (rte_eth_dev_configure(port_, 1, 0, &port_conf) < 0) die("rte_eth_dev_configure failed");

    uint16_t nb_rxd = cfg_.rxd, nb_txd = 0;
    if (rte_eth_dev_adjust_nb_rx_tx_desc(port_, &nb_rxd, &nb_txd) < 0)
      die("adjust_nb_rx_tx_desc failed");
    cfg_.rxd = nb_rxd;

    rte_eth_rxconf rxconf = dev_info.default_rxconf;
    rxconf.offloads = port_conf.rxmode.offloads;
    if (rte_eth_rx_queue_setup(port_, 0, nb_rxd, rte_eth_dev_socket_id(port_), &rxconf, pool_) < 0)
      die("rte_eth_rx_queue_setup failed");
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
