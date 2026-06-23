#include "st2110_rx.hpp"

#include <chrono>

namespace spark::ops {

void St2110RxOp::setup(holoscan::OperatorSpec& spec) {
  // Output used only in source mode (emit_frames=true); harmlessly unconnected in sink mode.
  spec.output<spark::st2110::VideoFrame>("frame");
  spec.param(pci_addr_, "pci_addr", "RX PCI", "CX-7 RX port BDF", std::string("0002:01:00.1"));
  spec.param(udp_port_, "udp_port", "UDP port", "RTP destination port to accept", uint32_t(20000));
  spec.param(profile_, "profile", "Profile", "1080p|2160p (frame geometry)", std::string("1080p"));
  spec.param(mcast_group_, "mcast_group", "Multicast group",
             "ST 2110 group to IGMP-join + filter (NMOS); empty = legacy promiscuous", std::string(""));
  spec.param(src_ip_, "src_ip", "SSM source", "source-specific filter (sender IP)", std::string(""));
  spec.param(iface_ip_, "iface_ip", "Interface IP", "local media IP (IGMP report source)", std::string(""));
  spec.param(in_width_, "in_width", "Input width", "override profile geometry (0 = profile)", uint32_t(0));
  spec.param(in_height_, "in_height", "Input height", "override profile geometry (0 = profile)", uint32_t(0));
  spec.param(in_fps_, "in_fps", "Input fps", "source frame rate (0 = profile)", 0.0);
  spec.param(rxd_, "rxd", "RX descriptors", "RX ring depth", uint32_t(4096));
  spec.param(eal_cores_, "eal_cores", "EAL cores", "DPDK lcore list", std::string("2,3"));
  spec.param(run_seconds_, "run_seconds", "Run seconds", "poll duration (sink mode)", 12.0);
  spec.param(manage_eal_, "manage_eal", "Manage EAL",
             "true: own rte_eal_init; false: shared DpdkEal already up", true);
  spec.param(emit_frames_, "emit_frames", "Emit frames",
             "true: source mode (one VideoFrame per compute); false: terminal sink", false);
  spec.param(ip10_, "ip10", "IP10 source",
             "source is Blackmagic IP10 (8-bit 4:2:2 pgroups, decoded to 10-bit downstream)", false);
}

void St2110RxOp::start() {
  fmt_ = (profile_.get() == "2160p") ? spark::st2110::profile_2160p()
                                     : spark::st2110::profile_1080p();
  // Adopt the source's actual geometry/rate from the SDP (NMOS) when provided; the depacketizer
  // geometry must match the wire, and the rate flows downstream for correct TX pacing.
  if (in_width_.get() > 0) fmt_.width = in_width_.get();
  if (in_height_.get() > 0) fmt_.height = in_height_.get();
  if (in_fps_.get() > 0.0) fmt_.fps = in_fps_.get();
  // IP10 source: the wire carries 8-bit codeword pgroups (4 octets/pgroup), so the depacketizer
  // geometry + reassembly buffer must be 8-bit; UnpackOp IP10-decodes them back to 10-bit planar.
  if (ip10_.get()) fmt_.sampling = spark::st2110::Sampling::YCbCr422_8;
  depkt_ = std::make_unique<spark::st2110::Depacketizer>(fmt_);
  frame_buf_.assign(fmt_.octets_per_frame(), 0);

  backend_ = spark::net::make_dpdk_rx_backend();
  spark::net::RxBackendConfig cfg;
  cfg.pci_addr = pci_addr_.get();
  cfg.udp_port = static_cast<uint16_t>(udp_port_.get());
  cfg.rxd = static_cast<uint16_t>(rxd_.get());
  cfg.eal_core_list = eal_cores_.get();
  cfg.manage_eal = manage_eal_.get();
  cfg.mcast_group = mcast_group_.get();
  cfg.src_ip = src_ip_.get();
  cfg.iface_ip = iface_ip_.get();
  backend_->init(cfg);
  HOLOSCAN_LOG_INFO("st2110_rx started: RX {} udp:{} group={} profile={} emit_frames={}{}", cfg.pci_addr,
                    cfg.udp_port, cfg.mcast_group.empty() ? "(none)" : cfg.mcast_group, profile_.get(),
                    emit_frames_.get(), ip10_.get() ? " IP10" : "");

  // Source mode: a dedicated thread drains the NIC continuously (see poll_loop). compute() only
  // pops finished frames, so NIC polling never stalls while TX paces the previous frame.
  if (emit_frames_.get()) poll_thread_ = std::thread(&St2110RxOp::poll_loop, this);
}

void St2110RxOp::account(const spark::st2110::RxPacketInfo& info, const spark::net::RxPacket& pkt,
                         uint64_t now_ns) {
  if (have_last_) {
    const uint32_t gap = info.sequence - (last_seq_ + 1);  // uint32 wrap-safe
    if (gap != 0 && gap < 0x80000000u) lost_ += gap;       // forward gap = loss; ignore reorder
  }
  last_seq_ = info.sequence;
  have_last_ = true;
  if (pkt.has_timestamp && now_ns >= pkt.hw_timestamp_ns) {
    const uint64_t lat = now_ns - pkt.hw_timestamp_ns;
    lat_sum_ += lat;
    ++lat_cnt_;
    if (lat < lat_min_) lat_min_ = lat;
    if (lat > lat_max_) lat_max_ = lat;
  }
  ++packets_;
}

std::shared_ptr<std::vector<uint8_t>> St2110RxOp::next_buffer() {
  if (buf_ring_.empty()) {
    buf_ring_.resize(8);
    for (auto& b : buf_ring_) b = std::make_shared<std::vector<uint8_t>>(fmt_.octets_per_frame());
  }
  auto b = buf_ring_[buf_idx_];
  buf_idx_ = (buf_idx_ + 1) % buf_ring_.size();
  return b;
}

void St2110RxOp::compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
                         holoscan::ExecutionContext&) {
  if (emit_frames_.get())
    compute_emit_one(op_output);
  else
    compute_sink();
}

void St2110RxOp::compute_sink() {
  using clock = std::chrono::steady_clock;
  const auto deadline = clock::now() + std::chrono::duration<double>(run_seconds_.get());
  HOLOSCAN_LOG_INFO("st2110_rx: polling for {}s ...", run_seconds_.get());

  spark::net::RxPacket pkts[256];
  while (clock::now() < deadline) {
    emit_live();  // 1 Hz live stats (throttled)
    const uint16_t n = backend_->receive(pkts, 256);
    if (n == 0) continue;
    const uint64_t now = backend_->now_ns();
    for (uint16_t i = 0; i < n; ++i) {
      spark::st2110::RxPacketInfo info;
      if (!depkt_->parse(pkts[i].payload, pkts[i].len, info)) {
        ++bad_;
        continue;
      }
      account(info, pkts[i], now);
      depkt_->scatter(info, pkts[i].payload, frame_buf_.data());
      if (info.marker) ++frames_;
    }
    backend_->release(pkts, n);
  }
  print_stats();
}

// Dedicated NIC-drain thread (emit mode). Continuously receives bursts and reassembles frames,
// pushing each completed frame onto a bounded queue. Because it never blocks on the Holoscan emit
// path, the NIC ring stays drained even while TX paces the previous frame (the fix for the HW-drop /
// latency seen when polling only inside compute()). On queue-full it drops the frame but keeps
// draining (whole-frame drop beats HW packet loss).
void St2110RxOp::poll_loop() {
  constexpr size_t kMaxQ = 8;
  cur_buf_ = next_buffer();
  cur_first_ = true;
  spark::net::RxPacket pkts[256];
  while (!stop_poll_.load(std::memory_order_relaxed)) {
    const uint16_t n = backend_->receive(pkts, 256);
    if (n == 0) continue;
    const uint64_t now = backend_->now_ns();
    for (uint16_t i = 0; i < n; ++i) {
      spark::st2110::RxPacketInfo info;
      if (!depkt_->parse(pkts[i].payload, pkts[i].len, info)) {
        ++bad_;
        continue;
      }
      account(info, pkts[i], now);
      if (cur_first_) {
        cur_ts_ = info.rtp_timestamp;
        cur_first_ = false;
      }
      depkt_->scatter(info, pkts[i].payload, cur_buf_->data());
      if (info.marker) {  // frame complete (a burst spans < one frame, so at most once)
        spark::st2110::VideoFrame f;
        f.data = cur_buf_;
        f.format = fmt_;
        f.capture_ts_ns = static_cast<uint64_t>(cur_ts_) * 1000000000ULL / 90000ULL;
        f.frame_number = frames_++;
        {
          std::lock_guard<std::mutex> lk(q_mu_);
          if (frame_q_.size() < kMaxQ)
            frame_q_.push_back(std::move(f));
          else
            ++q_dropped_;
        }
        q_cv_.notify_one();
        cur_buf_ = next_buffer();
        cur_first_ = true;
      }
    }
    backend_->release(pkts, n);
  }
}

// Source mode: pop one finished frame from the poll thread's queue and emit it.
void St2110RxOp::compute_emit_one(holoscan::OutputContext& op_output) {
  emit_live();  // 1 Hz live stats (throttled)
  std::unique_lock<std::mutex> lk(q_mu_);
  if (!q_cv_.wait_for(lk, std::chrono::seconds(2),
                      [&] { return !frame_q_.empty() || stop_poll_.load(); }))
    return;  // timed out (upstream stopped) — emit nothing this tick
  if (frame_q_.empty()) return;
  auto f = std::move(frame_q_.front());
  frame_q_.pop_front();
  lk.unlock();
  op_output.emit(f, "frame");
}

void St2110RxOp::emit_live(bool force) {
  const double t =
      std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if (!force && t - last_live_s_ < 1.0) return;
  last_live_s_ = t;
  const uint64_t avg_us = lat_cnt_ ? (lat_sum_ / lat_cnt_) / 1000 : 0;  // ns -> us
  // Unique tokens so the control daemon can grab each field unambiguously (latest-wins).
  HOLOSCAN_LOG_INFO("spark_live rx_frames={} rx_packets={} rx_lost={} rx_latency_us={}", frames_,
                    packets_, lost_, avg_us);
}

void St2110RxOp::print_stats() {
  if (stats_printed_) return;
  stats_printed_ = true;
  const auto rs = backend_ ? backend_->stats() : spark::net::RxStats{};
  const uint64_t avg = lat_cnt_ ? lat_sum_ / lat_cnt_ : 0;
  HOLOSCAN_LOG_INFO(
      "st2110_rx stats: frames={} matched_pkts={} lost={} bad={} q_dropped={} | nic_ipackets={} "
      "raw_burst={} hw_missed={} nombuf={} | ingest latency min/avg/max = {}/{}/{} ns ({} samples)",
      frames_, packets_, lost_, bad_, q_dropped_, rs.rx_packets, rs.raw_received, rs.rx_missed,
      rs.rx_nombuf, (lat_min_ == UINT64_MAX ? 0 : lat_min_), avg, lat_max_, lat_cnt_);
}

void St2110RxOp::stop() {
  // Stop + join the poll thread BEFORE shutting the backend down (it owns the NIC the thread polls).
  stop_poll_.store(true);
  q_cv_.notify_all();
  if (poll_thread_.joinable()) poll_thread_.join();
  emit_live(true);  // final live snapshot so the daemon captures end-of-run values
  print_stats();
  if (backend_) {
    backend_->shutdown();
    backend_.reset();
  }
}

}  // namespace spark::ops
