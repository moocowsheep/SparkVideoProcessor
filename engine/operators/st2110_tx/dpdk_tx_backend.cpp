// Raw DPDK + mlx5 tx_pp implementation of ISt2110TxBackend — the production-bound IO grown directly
// from spike/st2110_loopback_gate4.sh (same mechanism: send_on_timestamp Tx offload + per-packet HW
// send-scheduling on the CX-7). All DPDK headers are confined to this translation unit.
//
// Gate-4 finding baked in: pacing precision is excellent (jitter <=18 ns, sync_lost 0) but the
// schedule horizon (txd * gap) must stay inside the NIC's tx_pp window, and the *generator* must
// feed at the media rate — which the operator does (one packet per ST 2110-21 slot), unlike the
// open-loop testpmd probe. See docs/M1-gate4-pacing.md.
//
// STATUS: compiles against DPDK 23.11; runtime bring-up (EAL/port/clock units) is the on-box step.
// Spots that may need a tweak against live hardware are marked  // BRINGUP.
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
#include "tx_backend.hpp"

namespace spark::net {
namespace {

// Find the DPDK port whose PCI BDF matches `pci` (rte_device::name is the BDF for PCI devices).
// Needed once a process owns more than one port (shared EAL) — "first port" is no longer unique.
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

constexpr uint32_t kL2L3L4Hdr = sizeof(rte_ether_hdr) + sizeof(rte_ipv4_hdr) + sizeof(rte_udp_hdr);
constexpr uint16_t kBurst = 32;        // packets queued to the NIC per tx_burst
constexpr uint32_t kMbufCount = 16384;

[[noreturn]] void die(const std::string& msg) {
  throw std::runtime_error("dpdk_tx_backend: " + msg);
}

class DpdkTxBackend final : public ISt2110TxBackend {
 public:
  void init(const TxBackendConfig& cfg) override {
    cfg_ = cfg;
    if (cfg_.manage_eal)
      own_eal_init();  // standalone: this backend runs rte_eal_init for its single port
    else if (!DpdkEal::instance().initialized())
      die("manage_eal=false but shared EAL is not initialized (call DpdkEal::init first)");
    create_pool_and_addrs();
    port_ = find_port_by_pci(cfg_.pci_addr);
    if (port_ == RTE_MAX_ETHPORTS) die("no DPDK port matches PCI " + cfg_.pci_addr);
    setup_port();
    lookup_timestamp_dynfield();
    if (rte_eth_dev_start(port_) < 0) die("rte_eth_dev_start failed");
    rte_ether_addr mac{};
    rte_eth_macaddr_get(port_, &mac);
    std::memcpy(src_mac_, mac.addr_bytes, 6);
    std::printf("[st2110_tx] port %u up: src_mac %02x:%02x:%02x:%02x:%02x:%02x, "
                "tx_pp=%uns pacing=%d txd=%u\n",
                port_, src_mac_[0], src_mac_[1], src_mac_[2], src_mac_[3], src_mac_[4], src_mac_[5],
                cfg_.tx_pp_ns, cfg_.pacing && have_ts_, cfg_.txd);
  }

  TxBuf reserve_packet(uint32_t payload_len) override {
    rte_mbuf* m = rte_pktmbuf_alloc(pool_);
    if (!m) die("mbuf pool exhausted (slow consumer / NIC backpressure)");
    const uint32_t total = kL2L3L4Hdr + payload_len;
    m->data_len = static_cast<uint16_t>(total);
    m->pkt_len = total;
    uint8_t* p = rte_pktmbuf_mtod(m, uint8_t*);

    auto* eth = reinterpret_cast<rte_ether_hdr*>(p);
    std::memcpy(eth->dst_addr.addr_bytes, dst_mac_, 6);
    std::memcpy(eth->src_addr.addr_bytes, src_mac_, 6);
    eth->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);

    auto* ip = reinterpret_cast<rte_ipv4_hdr*>(p + sizeof(rte_ether_hdr));
    ip->version_ihl = 0x45;
    ip->type_of_service = 0xb8;  // DSCP EF (46) — ST 2110 media is expedited-forwarding class
    ip->total_length = rte_cpu_to_be_16(static_cast<uint16_t>(sizeof(rte_ipv4_hdr) +
                                                              sizeof(rte_udp_hdr) + payload_len));
    ip->packet_id = 0;
    ip->fragment_offset = 0;
    ip->time_to_live = 64;
    ip->next_proto_id = IPPROTO_UDP;
    ip->src_addr = src_ip_be_;
    ip->dst_addr = dst_ip_be_;
    ip->hdr_checksum = 0;
    ip->hdr_checksum = rte_ipv4_cksum(ip);

    auto* udp = reinterpret_cast<rte_udp_hdr*>(p + sizeof(rte_ether_hdr) + sizeof(rte_ipv4_hdr));
    udp->src_port = udp_port_be_;
    udp->dst_port = udp_port_be_;
    udp->dgram_len = rte_cpu_to_be_16(static_cast<uint16_t>(sizeof(rte_udp_hdr) + payload_len));
    udp->dgram_cksum = 0;  // UDP checksum optional over IPv4

    return TxBuf{p + kL2L3L4Hdr, payload_len, m};
  }

  void submit(const TxBuf& buf, uint64_t send_ts_ns) override {
    auto* m = static_cast<rte_mbuf*>(buf.opaque);
    if (cfg_.pacing && have_ts_) {
      *RTE_MBUF_DYNFIELD(m, ts_field_off_, uint64_t*) = send_ts_ns;
      m->ol_flags |= ts_flag_;
    }
    pending_[npending_++] = m;
    if (npending_ == kBurst) drain_pending();
  }

  void flush() override { drain_pending(); }

  uint64_t now_ns() override {
    uint64_t clk = 0;
    // BRINGUP: with REAL_TIME_CLOCK_ENABLE the mlx5 device clock is the realtime PHC in ns, so we
    // treat rte_eth_read_clock as ns. now_ns() and the send-timestamp dynfield share this base, so
    // pacing is self-consistent regardless of absolute offset; verify units against the NIC.
    if (rte_eth_read_clock(port_, &clk) != 0) die("rte_eth_read_clock unsupported on this port");
    return clk;
  }

  TxStats stats() override {
    TxStats s{};
    rte_eth_stats es{};
    if (rte_eth_stats_get(port_, &es) == 0) {
      s.tx_packets = es.opackets;
      s.tx_bytes = es.obytes;
    }
    const int n = rte_eth_xstats_get_names(port_, nullptr, 0);
    if (n <= 0) return s;
    std::vector<rte_eth_xstat_name> names(n);
    std::vector<rte_eth_xstat> vals(n);
    if (rte_eth_xstats_get_names(port_, names.data(), n) != n) return s;
    if (rte_eth_xstats_get(port_, vals.data(), n) != n) return s;
    auto find = [&](const char* key) -> uint64_t {
      for (int i = 0; i < n; ++i)
        if (std::strstr(names[i].name, key)) return vals[i].value;
      return 0;
    };
    s.future_errors = find("tx_pp_timestamp_future_errors");
    s.past_errors = find("tx_pp_timestamp_past_errors");
    s.jitter_ns = find("tx_pp_jitter");
    s.wander_ns = find("tx_pp_wander");
    s.sync_lost = find("tx_pp_sync_lost");
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

  ~DpdkTxBackend() override { shutdown(); }

 private:
  void own_eal_init() {
    std::string allow = cfg_.pci_addr;
    if (cfg_.pacing) allow += ",tx_pp=" + std::to_string(cfg_.tx_pp_ns);
    std::vector<std::string> args = {"spark_tx",      "-l", cfg_.eal_core_list,
                                     "-a",            allow, "--file-prefix",
                                     cfg_.file_prefix};
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    if (rte_eal_init(static_cast<int>(argv.size()), argv.data()) < 0)
      die("rte_eal_init failed (root? hugepages? PCI " + cfg_.pci_addr + "?)");
    eal_inited_ = true;
  }

  void create_pool_and_addrs() {
    pool_ = rte_pktmbuf_pool_create("spark_tx_pool", kMbufCount, 256, 0,
                                    RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if (!pool_) die("rte_pktmbuf_pool_create failed");
    if (inet_pton(AF_INET, cfg_.src_ip.c_str(), &src_ip_be_) != 1) die("bad src_ip");
    if (inet_pton(AF_INET, cfg_.dst_ip.c_str(), &dst_ip_be_) != 1) die("bad dst_ip");
    udp_port_be_ = rte_cpu_to_be_16(cfg_.udp_port);
    // dst MAC: if none was given (all-zero) and dst_ip is a multicast group, derive it (RFC 1112) —
    // the NMOS egress path just sets the group. An explicit MAC wins (gate-4 loopback to a known RX
    // port uses a unicast MAC even though dst_ip is a placeholder group).
    bool mac_set = false;
    for (uint8_t b : cfg_.dst_mac) mac_set |= (b != 0);
    const uint32_t dst_ip_host = rte_be_to_cpu_32(dst_ip_be_);
    if (!mac_set && spark::net::ipv4_is_multicast(dst_ip_host))
      spark::net::multicast_mac(dst_ip_host, dst_mac_);
    else
      std::memcpy(dst_mac_, cfg_.dst_mac.data(), 6);
  }

  void setup_port() {
    rte_eth_dev_info dev_info{};
    if (rte_eth_dev_info_get(port_, &dev_info) != 0) die("rte_eth_dev_info_get failed");

    rte_eth_conf port_conf{};
    if (cfg_.pacing) {
      if (!(dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_SEND_ON_TIMESTAMP))
        die("NIC lacks SEND_ON_TIMESTAMP offload (REAL_TIME_CLOCK_ENABLE=1 set? see M0 findings)");
      port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_SEND_ON_TIMESTAMP;
    }
    if (rte_eth_dev_configure(port_, 0, 1, &port_conf) < 0) die("rte_eth_dev_configure failed");

    uint16_t nb_rxd = 0, nb_txd = cfg_.txd;
    if (rte_eth_dev_adjust_nb_rx_tx_desc(port_, &nb_rxd, &nb_txd) < 0)
      die("adjust_nb_rx_tx_desc failed");
    cfg_.txd = nb_txd;  // mlx5 may clamp; reflect the effective horizon (gate-4 relevance)

    rte_eth_txconf txconf = dev_info.default_txconf;
    txconf.offloads = port_conf.txmode.offloads;
    if (rte_eth_tx_queue_setup(port_, 0, nb_txd, rte_eth_dev_socket_id(port_), &txconf) < 0)
      die("rte_eth_tx_queue_setup failed");
  }

  void lookup_timestamp_dynfield() {
    if (!cfg_.pacing) return;
    ts_field_off_ = rte_mbuf_dynfield_lookup(RTE_MBUF_DYNFIELD_TIMESTAMP_NAME, nullptr);
    const int flag_bit = rte_mbuf_dynflag_lookup(RTE_MBUF_DYNFLAG_TX_TIMESTAMP_NAME, nullptr);
    if (ts_field_off_ < 0 || flag_bit < 0) {
      std::printf("[st2110_tx] WARNING: tx timestamp dynfield/flag absent — sending UNPACED\n");
      have_ts_ = false;
      return;
    }
    ts_flag_ = 1ULL << flag_bit;
    have_ts_ = true;
  }

  void drain_pending() {
    uint16_t sent = 0;
    while (sent < npending_) {
      const uint16_t n = rte_eth_tx_burst(port_, 0, &pending_[sent], npending_ - sent);
      if (n == 0) {
        // Ring full: NIC is holding scheduled packets. Brief retry; this is normal backpressure.
        if (++spin_ > 1000000) {
          for (uint16_t i = sent; i < npending_; ++i) rte_pktmbuf_free(pending_[i]);
          dropped_ += npending_ - sent;
          break;
        }
        continue;
      }
      spin_ = 0;
      sent += n;
    }
    npending_ = 0;
  }

  TxBackendConfig cfg_;
  bool eal_inited_ = false;
  uint16_t port_ = RTE_MAX_ETHPORTS;
  rte_mempool* pool_ = nullptr;

  uint8_t src_mac_[6] = {};
  uint8_t dst_mac_[6] = {};  // resolved at init: derived from a multicast group, else cfg_.dst_mac
  uint32_t src_ip_be_ = 0, dst_ip_be_ = 0;
  uint16_t udp_port_be_ = 0;

  int ts_field_off_ = -1;
  uint64_t ts_flag_ = 0;
  bool have_ts_ = false;

  rte_mbuf* pending_[kBurst] = {};
  uint16_t npending_ = 0;
  uint64_t spin_ = 0, dropped_ = 0;
};

}  // namespace

std::unique_ptr<ISt2110TxBackend> make_dpdk_tx_backend() {
  return std::make_unique<DpdkTxBackend>();
}

}  // namespace spark::net
