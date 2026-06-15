#include "st2110_tx.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <thread>

namespace spark::ops {
namespace {

std::array<uint8_t, 6> parse_mac(const std::string& s) {
  std::array<uint8_t, 6> m{};
  std::sscanf(s.c_str(), "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx", &m[0], &m[1], &m[2], &m[3], &m[4], &m[5]);
  return m;
}

}  // namespace

void St2110TxOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<spark::st2110::VideoFrame>("frame");

  spec.param(pci_addr_, "pci_addr", "TX PCI", "CX-7 TX port BDF", std::string("0002:01:00.0"));
  spec.param(dst_mac_, "dst_mac", "Dest MAC", "Destination MAC (gate-4 switched fabric: the RX port MAC)",
             std::string("00:00:5e:00:53:30"));  // enP2p1s0f1np1 / 0002:01:00.1 from detect_loopback
  spec.param(src_ip_, "src_ip", "Source IP", "ST 2110-10 source address", std::string("192.168.50.10"));
  spec.param(dst_ip_, "dst_ip", "Dest IP", "ST 2110 multicast group (loopback may use RX unicast)",
             std::string("239.0.0.1"));
  spec.param(udp_port_, "udp_port", "UDP port", "RTP destination port", uint32_t(20000));
  spec.param(payload_size_, "payload_size", "Payload octets", "UDP payload budget per packet",
             uint32_t(1420));
  spec.param(tx_pp_ns_, "tx_pp_ns", "tx_pp ns", "mlx5 tx_pp clock granularity", uint32_t(500));
  spec.param(txd_, "txd", "TX descriptors", "TX ring depth (pacing horizon = txd * gap)",
             uint32_t(2048));
  spec.param(pacing_horizon_ns_, "pacing_horizon_ns", "Pacing horizon",
             "Max schedule lead over the NIC clock (keep < tx_pp window; see M1-gate4-pacing)",
             uint32_t(50000));
  spec.param(ssrc_, "ssrc", "RTP SSRC", "RTP synchronization source id", uint32_t(0x53504b31));
  spec.param(eal_cores_, "eal_cores", "EAL cores", "DPDK lcore list", std::string("0,1"));
  spec.param(pacing_, "pacing", "Enable pacing", "tx_pp hardware send-scheduling", true);
  spec.param(manage_eal_, "manage_eal", "Manage EAL",
             "true: own rte_eal_init; false: shared DpdkEal already up", true);
  spec.param(warmup_ms_, "warmup_ms", "Warmup ms",
             "one-time delay before the first frame (let an RX peer start polling first)", 0u);
}

void St2110TxOp::start() {
  backend_ = spark::net::make_dpdk_tx_backend();

  spark::net::TxBackendConfig cfg;
  cfg.pci_addr = pci_addr_.get();
  cfg.dst_mac = parse_mac(dst_mac_.get());
  cfg.src_ip = src_ip_.get();
  cfg.dst_ip = dst_ip_.get();
  cfg.udp_port = static_cast<uint16_t>(udp_port_.get());
  cfg.tx_pp_ns = tx_pp_ns_.get();
  cfg.txd = static_cast<uint16_t>(txd_.get());
  cfg.eal_core_list = eal_cores_.get();
  cfg.pacing = pacing_.get();
  cfg.manage_eal = manage_eal_.get();
  backend_->init(cfg);

  HOLOSCAN_LOG_INFO("st2110_tx started: TX {} -> {} pacing={}", cfg.pci_addr, dst_mac_.get(),
                    cfg.pacing);
}

void St2110TxOp::emit_live(bool force) {
  const double t =
      std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if (!force && t - last_live_s_ < 1.0) return;
  last_live_s_ = t;
  const spark::net::TxStats s = backend_ ? backend_->stats() : spark::net::TxStats{};
  HOLOSCAN_LOG_INFO("spark_live tx_frames={} tx_packets={} tx_future_err={} tx_past_err={}",
                    frames_sent_, packets_sent_, s.future_errors, s.past_errors);
}

void St2110TxOp::ensure_pacer(const spark::st2110::VideoFormat& fmt) {
  if (pktz_) return;
  pktz_ = std::make_unique<spark::st2110::Packetizer>(fmt, payload_size_.get(), /*pt=*/96,
                                                      ssrc_.get());
  const uint32_t ppf = pktz_->packets_per_frame();
  frame_interval_ns_ = static_cast<uint64_t>(1e9 / fmt.fps);
  gap_ns_ = ppf ? frame_interval_ns_ / ppf : 0;
  HOLOSCAN_LOG_INFO("st2110_tx: {}x{}@{:.3f}fps -> {} pkts/frame, gap {} ns (~{} pps)", fmt.width,
                    fmt.height, fmt.fps, ppf, gap_ns_,
                    static_cast<uint64_t>(ppf * fmt.fps));
}

void St2110TxOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext&,
                         holoscan::ExecutionContext&) {
  auto maybe = op_input.receive<spark::st2110::VideoFrame>("frame");
  if (!maybe) return;
  auto& frame = maybe.value();
  if (!frame.data || frame.data->size() != frame.format.octets_per_frame()) {
    HOLOSCAN_LOG_ERROR("st2110_tx: frame buffer size mismatch — dropping frame {}",
                       frame.frame_number);
    return;
  }
  ensure_pacer(frame.format);
  emit_live();  // 1 Hz live stats (throttled)

  if (!warmed_) {  // one-time: give a co-located/peer RX time to enter its poll loop before flooding
    warmed_ = true;
    if (warmup_ms_.get() > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(warmup_ms_.get()));
  }

  pktz_->start_frame(spark::st2110::rtp_timestamp_90k(frame.capture_ts_ns));

  // (Re)anchor the schedule: on the first frame, or if we have fallen behind the NIC clock, restart
  // the schedule one horizon ahead so the very first packet is schedulable (not already in the past).
  const uint64_t now = backend_->now_ns();
  if (schedule_base_ns_ == 0 || now + frame_interval_ns_ > schedule_base_ns_ + frame_interval_ns_) {
    if (schedule_base_ns_ == 0 || now > schedule_base_ns_)
      schedule_base_ns_ = now + pacing_horizon_ns_;
  }
  const uint64_t base = schedule_base_ns_;
  const bool pace = pacing_.get();

  uint32_t i = 0;
  for (spark::st2110::PacketPlan p; pktz_->next(p); ++i) {
    const uint64_t send_ts = base + static_cast<uint64_t>(i) * gap_ns_;
    // Closed-loop throttle (the gate-4 lesson): keep the in-flight schedule within the tx_pp window
    // so packets never become future_errors. A real-time sender is *meant* to wait here. Guarded by
    // an independent wall-clock cap so a now_ns() unit mismatch (// BRINGUP) surfaces as future_errors
    // in the xstats rather than hanging a root process.
    if (pace && throttle_enabled_) {
      const auto t0 = std::chrono::steady_clock::now();
      while (send_ts > backend_->now_ns() + pacing_horizon_ns_) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(50)) {
          HOLOSCAN_LOG_WARN(
              "st2110_tx: pacing throttle hit 50ms cap — disabling self-throttle (check now_ns() "
              "units / pacing_horizon_ns); NIC tx_pp still active");
          throttle_enabled_ = false;
          break;
        }
      }
    }
    spark::net::TxBuf buf = backend_->reserve_packet(p.payload_len);
    pktz_->write_payload(p, frame.data->data(), buf.payload);
    backend_->submit(buf, send_ts);
    ++packets_sent_;
  }
  backend_->flush();

  schedule_base_ns_ = base + frame_interval_ns_;  // next frame begins exactly one interval later
  ++frames_sent_;
}

void St2110TxOp::stop() {
  if (!backend_) return;
  emit_live(true);  // final live snapshot for the daemon
  const auto s = backend_->stats();
  HOLOSCAN_LOG_INFO(
      "st2110_tx stopped: frames={} packets={} | tx_pp jitter={}ns wander={}ns sync_lost={} "
      "future_err={} past_err={}",
      frames_sent_, packets_sent_, s.jitter_ns, s.wander_ns, s.sync_lost, s.future_errors,
      s.past_errors);
  backend_->shutdown();
  backend_.reset();
}

}  // namespace spark::ops
