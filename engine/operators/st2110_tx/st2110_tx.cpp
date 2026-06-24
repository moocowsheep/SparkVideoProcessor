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
  // TX ring depth: the HW tx_pp shock-absorber. The NIC paces packets out of this ring precisely on
  // their stamped send time, so a deep ring lets software run *ahead* and submit jitter-free — an OS
  // preemption or GPU stall shorter than the ring's drain time (txd * gap) is invisible on the wire.
  // 8192 * ~2250ns (2160p29.97) ≈ 18ms of buffer. (mlx5 may clamp; effective value is logged at init.)
  spec.param(txd_, "txd", "TX descriptors", "TX ring depth — HW pacing buffer (txd * gap of wire time)",
             uint32_t(8192));
  // How far ahead of the NIC clock software keeps the schedule. Must stay (a) under the ring's wire
  // time txd*gap so submissions don't block, and (b) under the tx_pp future window (else future_err).
  // Sized as a jitter shock-absorber: 8ms rides any normal scheduling stall, so past_err stays ~0 and
  // the wire never gaps — the gate-4 "don't flood" lesson was about the *window*, not keeping it tiny.
  spec.param(pacing_horizon_ns_, "pacing_horizon_ns", "Pacing horizon",
             "Schedule lead over the NIC clock — jitter buffer (keep < txd*gap and < tx_pp window)",
             uint32_t(8000000));
  spec.param(reanchor_lead_ns_, "reanchor_lead_ns", "Re-anchor lead",
             "schedule base lead over NIC clock on (re)anchor; 0 = use pacing_horizon_ns", uint32_t(0));
  spec.param(ssrc_, "ssrc", "RTP SSRC", "RTP synchronization source id", uint32_t(0x53504b31));
  spec.param(eal_cores_, "eal_cores", "EAL cores", "DPDK lcore list", std::string("0,1"));
  spec.param(pacing_, "pacing", "Enable pacing", "tx_pp hardware send-scheduling", true);
  // Spread each frame's packets over fill×(frame interval) instead of the whole interval. <1.0 finishes
  // the frame early so its tail reaches a narrow (2110TPN) receiver before its display deadline — fixes
  // bottom-of-frame breakup at high rates (e.g. 2160p59.94 IP10). Keep fill above (avg rate / link rate)
  // so the higher instantaneous rate stays under the receiver's link (e.g. 8.5/10G ⇒ fill ≥ ~0.9).
  spec.param(pacing_fill_, "pacing_fill", "Pacing fill", "fraction of the frame interval to pace over (<=1.0)", 1.0);
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
  HOLOSCAN_LOG_INFO("spark_live tx_frames={} tx_packets={} tx_future_err={} tx_past_err={} tx_reanchors={}",
                    frames_sent_, packets_sent_, s.future_errors, s.past_errors, reanchors_);
}

void St2110TxOp::ensure_pacer(const spark::st2110::VideoFormat& fmt) {
  if (pktz_) return;
  pktz_ = std::make_unique<spark::st2110::Packetizer>(fmt, payload_size_.get(), /*pt=*/96,
                                                      ssrc_.get());
  const uint32_t ppf = pktz_->packets_per_frame();
  frame_interval_ns_ = static_cast<uint64_t>(1e9 / fmt.fps);
  // ST 2110-21 narrow pacing: spread the frame's packets over the ACTIVE period (T_frame × active/total
  // lines), the rate a narrow receiver drains at. Pacing faster (the whole frame, or less) sends ahead
  // of that drain and overflows its small buffer (tail drop); slower delivers the bottom late. `fill` is
  // a fine-tune on top (default 1.0 = exact active-period rate).
  const double active = fmt.active_ratio();
  double fill = pacing_fill_.get();
  if (!(fill > 0.0) || fill > 1.0) fill = 1.0;
  gap_ns_ = ppf ? static_cast<uint64_t>(frame_interval_ns_ * active * fill) / ppf : 0;
  HOLOSCAN_LOG_INFO("st2110_tx: {}x{}@{:.3f}fps -> {} pkts/frame, gap {} ns (~{} pps, active {:.3f} fill {:.2f})",
                    fmt.width, fmt.height, fmt.fps, ppf, gap_ns_,
                    static_cast<uint64_t>(ppf * fmt.fps), active, fill);

  // Gapped (2110TPN) pacing for IP10: deliver each line's packets bunched early in its line slot, then
  // idle — exactly how the Blackmagic reference paces (measured: ~6 pkts/line at ~1082 ns spacing, then
  // a ~2 us gap to the next line). Even pacing puts a line's LAST packet at the slot end, too late for
  // the receiver's per-line drain -> tail-of-line drop, which IP10's per-line context then smears into
  // colored line corruption. Raw 10-bit stays on the even gap_ns_ schedule (validated).
  pace_gapped_ = (fmt.sampling == spark::st2110::Sampling::YCbCr422_8) && fmt.height > 0;
  if (pace_gapped_) {
    t_line_ns_ = static_cast<uint64_t>(frame_interval_ns_ * active * fill) / fmt.height;
    const uint32_t pix_per_pkt = payload_size_.get() > 20 ? payload_size_.get() - 20 : 1;
    const uint32_t ppl = (fmt.octets_per_line() + pix_per_pkt - 1) / pix_per_pkt;  // packets per line
    // Spread the line's packets across ~73% of the slot (the measured BMD burst fraction), leaving the
    // tail of the slot idle. ppl-1 gaps for ppl packets.
    intra_gap_ns_ = (ppl > 1) ? (t_line_ns_ * 73) / (100 * (ppl - 1)) : gap_ns_;
    HOLOSCAN_LOG_INFO("st2110_tx: gapped pacing — T_line {} ns, {} pkts/line, intra-gap {} ns", t_line_ns_,
                      ppl, intra_gap_ns_);
  }
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

  const uint64_t now = backend_->now_ns();
  const uint64_t tgt = reanchor_lead_ns_.get();
  if (tgt > 0 && frame.capture_ts_ns != 0) {
    // GENLOCK (IP10): anchor the send base to the SOURCE's frame timing (capture_ts, from the sender's RTP
    // clock, plumbed through the GPU stages) plus a calibrated constant offset. capture_ts advances at the
    // source rate and is jitter-free — unlike the compute-time NIC clock, whose pipeline jitter forced
    // every free-running-grid scheme into a no-win: hard corrections showed as dips to black, smooth ones
    // let the lead erode into unpaced past-error bursts (colored lines). With GM-lock the source-clock vs
    // NIC-clock offset is constant, so one calibration holds and the base tracks the source smoothly: no
    // jumps (no dips) and a steady lead (no past-errors). Recalibrate only if a latency spike or RTP wrap
    // pushes the lead out of a safe band (rare, single-frame).
    // Constant offset => base = capture_ts + offset is PERFECTLY source-locked and smooth (no per-frame
    // step), so the only thing that can disturb the receiver is a re-calibration. Keep the offset fixed
    // and re-calibrate ONLY when the lead leaves a wide band: below tgt/4 (a real latency spike about to
    // past-error -> colored lines) or above 2 frames (RTP-clock wrap / sustained queue growth). With a
    // generous tgt the band is rarely tripped, so corrections (and thus dips) are rare while past-errors
    // stay at zero.
    if (genlock_offset_ == 0) genlock_offset_ = (now + tgt) - frame.capture_ts_ns;
    int64_t gbase = static_cast<int64_t>(frame.capture_ts_ns) + static_cast<int64_t>(genlock_offset_);
    const int64_t lead = gbase - static_cast<int64_t>(now);
    if (lead < static_cast<int64_t>(tgt) / 4 || lead > 2 * static_cast<int64_t>(frame_interval_ns_)) {
      genlock_offset_ = (now + tgt) - frame.capture_ts_ns;  // re-center the lead (rare)
      gbase = static_cast<int64_t>(now) + static_cast<int64_t>(tgt);
      ++reanchors_;
    }
    schedule_base_ns_ = static_cast<uint64_t>(gbase);
  } else if (tgt > 0) {
    // Fallback (capture_ts unavailable): smooth bounded-lead servo on the compute clock — hold the
    // schedule ~tgt ahead, nudging a fraction of the error per frame; hard-resync only on a gross stall.
    if (schedule_base_ns_ == 0) {
      schedule_base_ns_ = now + tgt;
    } else {
      const int64_t err = static_cast<int64_t>(now + tgt) - static_cast<int64_t>(schedule_base_ns_);
      if (err > static_cast<int64_t>(frame_interval_ns_) ||
          err < -static_cast<int64_t>(frame_interval_ns_)) {
        schedule_base_ns_ = now + tgt;
        ++reanchors_;
      } else {
        schedule_base_ns_ += err / 8;
      }
    }
  } else {
    // Legacy fixed grid (raw path): re-anchor only on a genuine >1-frame stall.
    if (schedule_base_ns_ == 0 || now > schedule_base_ns_ + frame_interval_ns_) {
      if (schedule_base_ns_ != 0) ++reanchors_;
      schedule_base_ns_ = now + pacing_horizon_ns_;
    }
  }
  const uint64_t base = schedule_base_ns_;
  const bool pace = pacing_.get();

  // RTP media timestamp from a MONOTONIC clock anchored once to the GM-locked egress and advanced by
  // exactly one frame interval per frame. (frame.capture_ts_ns doesn't survive the GPU path — planar
  // GpuFrames carry no timestamp, so it arrives 0; and a processor's output media time is the locked
  // send time anyway.) Decoupling it from schedule_base_ns_ matters: the pacer may re-anchor base
  // (when it slips), and a timestamp that jumped with it would break a downstream receiver's clock
  // recovery — the monotonic clock never jumps, so BMD-class receivers hold a solid lock.
  if (media_ts_ns_ == 0) media_ts_ns_ = base;
  pktz_->start_frame(spark::st2110::rtp_timestamp_90k(media_ts_ns_));
  media_ts_ns_ += frame_interval_ns_;

  uint32_t i = 0;
  uint32_t pace_line = 0xffffffffu, pace_intra = 0;
  for (spark::st2110::PacketPlan p; pktz_->next(p); ++i) {
    uint64_t send_ts;
    if (pace_gapped_) {
      // Per-line slot: line L's packets fire at L*T_line + k*intra_gap (k = packet index within line L),
      // an early burst then idle — matching the BMD reference. Lines arrive in order, so send_ts is
      // monotonic and the closed-loop throttle below still applies cleanly.
      const uint32_t line = (p.nsrd > 0) ? p.srd[0].line_no : 0;
      if (line != pace_line) {
        pace_line = line;
        pace_intra = 0;
      } else {
        ++pace_intra;
      }
      send_ts = base + static_cast<uint64_t>(line) * t_line_ns_ +
                static_cast<uint64_t>(pace_intra) * intra_gap_ns_;
    } else {
      send_ts = base + static_cast<uint64_t>(i) * gap_ns_;
    }
    if (send_ts < now) send_ts = now;  // grid briefly behind the NIC clock: send now, never past-stamp
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
      "st2110_tx stopped: frames={} packets={} reanchors={} | tx_pp jitter={}ns wander={}ns "
      "sync_lost={} future_err={} past_err={}",
      frames_sent_, packets_sent_, reanchors_, s.jitter_ns, s.wander_ns, s.sync_lost, s.future_errors,
      s.past_errors);
  backend_->shutdown();
  backend_.reset();
}

}  // namespace spark::ops
