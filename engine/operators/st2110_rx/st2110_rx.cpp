#include "st2110_rx.hpp"

#include <chrono>

namespace spark::ops {

void St2110RxOp::setup(holoscan::OperatorSpec& spec) {
  // Terminal sink for the loopback smoke: no input/output ports; driven by a CountCondition(1).
  spec.param(pci_addr_, "pci_addr", "RX PCI", "CX-7 RX port BDF", std::string("0002:01:00.1"));
  spec.param(udp_port_, "udp_port", "UDP port", "RTP destination port to accept", uint32_t(20000));
  spec.param(profile_, "profile", "Profile", "1080p|2160p (frame geometry)", std::string("1080p"));
  spec.param(rxd_, "rxd", "RX descriptors", "RX ring depth", uint32_t(4096));
  spec.param(eal_cores_, "eal_cores", "EAL cores", "DPDK lcore list", std::string("2,3"));
  spec.param(run_seconds_, "run_seconds", "Run seconds", "poll duration", 12.0);
  spec.param(manage_eal_, "manage_eal", "Manage EAL",
             "true: own rte_eal_init; false: shared DpdkEal already up", true);
}

void St2110RxOp::start() {
  fmt_ = (profile_.get() == "2160p") ? spark::st2110::profile_2160p()
                                     : spark::st2110::profile_1080p();
  depkt_ = std::make_unique<spark::st2110::Depacketizer>(fmt_);
  frame_buf_.assign(fmt_.octets_per_frame(), 0);

  backend_ = spark::net::make_dpdk_rx_backend();
  spark::net::RxBackendConfig cfg;
  cfg.pci_addr = pci_addr_.get();
  cfg.udp_port = static_cast<uint16_t>(udp_port_.get());
  cfg.rxd = static_cast<uint16_t>(rxd_.get());
  cfg.eal_core_list = eal_cores_.get();
  cfg.manage_eal = manage_eal_.get();
  backend_->init(cfg);
  HOLOSCAN_LOG_INFO("st2110_rx started: RX {} udp:{} profile={}", cfg.pci_addr, cfg.udp_port,
                    profile_.get());
}

void St2110RxOp::compute(holoscan::InputContext&, holoscan::OutputContext&,
                         holoscan::ExecutionContext&) {
  using clock = std::chrono::steady_clock;
  const auto deadline = clock::now() + std::chrono::duration<double>(run_seconds_.get());
  HOLOSCAN_LOG_INFO("st2110_rx: polling for {}s ...", run_seconds_.get());

  spark::net::RxPacket pkts[256];
  while (clock::now() < deadline) {
    const uint16_t n = backend_->receive(pkts, 256);
    if (n == 0) continue;
    const uint64_t now = backend_->now_ns();  // one clock read per burst (latency basis)
    for (uint16_t i = 0; i < n; ++i) {
      spark::st2110::RxPacketInfo info;
      if (!depkt_->parse(pkts[i].payload, pkts[i].len, info)) {
        ++bad_;
        continue;
      }
      if (have_last_) {
        const uint32_t gap = info.sequence - (last_seq_ + 1);  // uint32 wrap-safe
        if (gap != 0 && gap < 0x80000000u) lost_ += gap;       // forward gap = loss; ignore reorder
      }
      last_seq_ = info.sequence;
      have_last_ = true;
      depkt_->scatter(info, pkts[i].payload, frame_buf_.data());
      if (pkts[i].has_timestamp && now >= pkts[i].hw_timestamp_ns) {
        const uint64_t lat = now - pkts[i].hw_timestamp_ns;
        lat_sum_ += lat;
        ++lat_cnt_;
        if (lat < lat_min_) lat_min_ = lat;
        if (lat > lat_max_) lat_max_ = lat;
      }
      ++packets_;
      if (info.marker) ++frames_;
    }
    backend_->release(pkts, n);
  }
  print_stats();
}

void St2110RxOp::print_stats() {
  if (stats_printed_) return;
  stats_printed_ = true;
  const auto rs = backend_ ? backend_->stats() : spark::net::RxStats{};
  const uint64_t avg = lat_cnt_ ? lat_sum_ / lat_cnt_ : 0;
  HOLOSCAN_LOG_INFO(
      "st2110_rx stats: frames={} matched_pkts={} lost={} bad={} | nic_ipackets={} raw_burst={} "
      "hw_missed={} nombuf={} | ingest latency min/avg/max = {}/{}/{} ns ({} samples)",
      frames_, packets_, lost_, bad_, rs.rx_packets, rs.raw_received, rs.rx_missed, rs.rx_nombuf,
      (lat_min_ == UINT64_MAX ? 0 : lat_min_), avg, lat_max_, lat_cnt_);
}

void St2110RxOp::stop() {
  print_stats();  // backup if compute() exited early
  if (backend_) {
    backend_->shutdown();
    backend_.reset();
  }
}

}  // namespace spark::ops
