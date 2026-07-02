#include "st2110_tx.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
             std::string("30:c5:99:3e:9d:30"));  // enP2p1s0f1np1 / 0002:01:00.1 from detect_loopback
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
  spec.param(tx_wide_, "tx_wide", "Wide shaping",
             "ST 2110-21 Wide (2110TPW): pace evenly over the full frame instead of Narrow active-period gapped",
             false);
  spec.param(manage_eal_, "manage_eal", "Manage EAL",
             "true: own rte_eal_init; false: shared DpdkEal already up", true);
  spec.param(warmup_ms_, "warmup_ms", "Warmup ms",
             "one-time delay before the first frame (let an RX peer start polling first)", 0u);
  // Latency trim rate. The genlock offset is calibrated on the FIRST frame through a cold pipeline
  // (CUDA pools, NVOF init, startup queue backlog), so without trim the stream carries that transient
  // forever — anywhere up to the 2-frame re-anchor band (~33 ms at 59.94) above the achievable floor.
  // The servo shaves at most this many ns per frame off the base whenever the lead sits above target,
  // and the RTP media clock slews with it, so wire time and timestamps stay glued — a bounded ~300 ppm
  // (at 5 us/frame, 59.94) rate offset while converging, no jumps. 0 disables (exact legacy timing).
  spec.param(trim_ns_, "trim_ns", "Trim ns/frame",
             "max ns/frame to slew the send base (and media clock) toward the target lead; 0 = off",
             uint32_t(5000));
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
  // tx_e2e_us: capture -> first-bit-out latency estimate, i.e. the genlock offset modulo the 32-bit
  // 90 kHz RTP wrap (~13.25 h). Exact when the source stamps RTP on the PTP epoch (ST 2110-10 — the
  // BMDs do); for a free-running source the absolute value is meaningless but its CHANGES are real.
  // tx_lead_us is always real: schedule base over the NIC clock at compute time (target = reanchor
  // lead; anything persistently above target is trimmable latency). tx_trim_ms totals what the servo
  // has reclaimed.
  constexpr uint64_t kRtpWrapNs = 4294967296ULL * 100000ULL / 9ULL;  // 2^32 ticks @90 kHz, in ns
  const uint64_t e2e_us = genlock_offset_ ? (genlock_offset_ % kRtpWrapNs) / 1000 : 0;
  HOLOSCAN_LOG_INFO(
      "spark_live tx_frames={} tx_packets={} tx_future_err={} tx_past_err={} tx_reanchors={} "
      "tx_skipped={} tx_lead_us={} tx_e2e_us={} tx_trim_ms={}",
      frames_sent_, packets_sent_, s.future_errors, s.past_errors, reanchors_, skipped_late_,
      last_lead_ns_ / 1000, e2e_us, trim_total_ns_ / 1000000);
}

void St2110TxOp::ensure_pacer(const spark::st2110::VideoFormat& fmt) {
  if (pktz_) return;
  pktz_ = std::make_unique<spark::st2110::Packetizer>(fmt, payload_size_.get(), /*pt=*/96,
                                                      ssrc_.get());
  const uint32_t ppf = pktz_->packets_per_frame();
  frame_interval_ns_ = static_cast<uint64_t>(1e9 / fmt.fps);
  const bool wide = tx_wide_.get();
  // ST 2110-21 pacing window. Narrow (2110TPN) spreads the frame's packets over the ACTIVE period only
  // (T_frame × active/total lines) — the rate a narrow receiver drains at; pacing faster overflows its
  // small buffer (tail drop), slower delivers the bottom late. Wide (2110TPW) spreads them EVENLY over
  // the FULL frame interval: the lowest peak rate, and the receiver's larger wide buffer absorbs the
  // tx_pp/pipeline jitter that narrow's tight buffer turned into dips to black. `fill` fine-tunes on top.
  const double active = wide ? 1.0 : fmt.active_ratio();
  double fill = pacing_fill_.get();
  if (!(fill > 0.0) || fill > 1.0) fill = 1.0;
  gap_ns_ = ppf ? static_cast<uint64_t>(frame_interval_ns_ * active * fill) / ppf : 0;
  HOLOSCAN_LOG_INFO("st2110_tx: {}x{}@{:.3f}fps -> {} pkts/frame, gap {} ns (~{} pps, {} active {:.3f} fill {:.2f})",
                    fmt.width, fmt.height, fmt.fps, ppf, gap_ns_,
                    static_cast<uint64_t>(ppf * fmt.fps), wide ? "WIDE" : "NARROW", active, fill);

  // Gapped (2110TPN) pacing for IP10: deliver each line's packets bunched early in its line slot, then
  // idle — exactly how the Blackmagic reference paces (measured: ~6 pkts/line at ~1082 ns spacing, then
  // a ~2 us gap to the next line). This is the validated IP10 path: per-PACKET send timestamps with an
  // intra-line gap. Raw 2160p does NOT use it — at 15120 pkts/frame the per-packet send-on-timestamp WQEs
  // cost ~25ms/frame (capping compute() at the frame budget). Raw 2160p (narrow AND wide) instead uses
  // pace_line_burst_ below (one timestamp per line, rest back-to-back ≈ the BMD's ~1082ns intra-line).
  const bool ip10_sampling = (fmt.sampling == spark::st2110::Sampling::YCbCr422_8);
  // IP10 at >=~50fps can't pay per-packet send-on-timestamp: ~1.7us of mlx5 WQE build x 12960
  // pkts/frame (2160p) is ~22ms against the 16.7ms 59.94 budget — TX caps ~45fps. Use per-line-burst
  // instead: one stamp/line and the NIC sends the line's remaining packets back-to-back, which at 10G
  // is ~1.06us/pkt for IP10's 1300-byte payloads — wire-identical to the validated ~1082ns intra-line
  // gap. Lower rates keep the hardware-validated per-packet gapped path (2160p29.97 IP10). The lead
  // no longer drifts into ring overflow (the M8 trim servo holds it at target), which is what blocked
  // this before. SPARK_TX_IP10_BURST=1/0 forces it on/off for A/B on the rig.
  bool ip10_burst = ip10_sampling && frame_interval_ns_ < 20000000;
  if (const char* e = std::getenv("SPARK_TX_IP10_BURST")) ip10_burst = ip10_sampling && e[0] == '1';
  pace_gapped_ = !wide && ip10_sampling && !ip10_burst && fmt.height > 0;
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

  // Per-line-burst pacing for RAW 2160p (narrow AND wide): HW-timestamp only the FIRST packet of each line
  // (the NIC waits to the line's slot, then sends that line's ~7 packets back-to-back at ~line rate, close
  // to the BMD reference's ~1082ns intra-line spacing). Per-PACKET send-on-timestamp costs ~1.7us/pkt of
  // mlx5 WQE build (~25ms for a 15120-pkt 2160p frame) — that alone capped compute() at the frame budget,
  // so the RX frame queue overflowed and BD2 dipped. One stamp/line cuts that ~7x. The per-line slot
  // (T_line = frame_interval * active * fill / height) sets the average pace: narrow uses active=0.96
  // (finishes within the active period, matching the BMD), wide uses 1.0; tx_fill tunes BD2 buffer
  // headroom on top. IP10 <=~50fps keeps its validated per-packet gapped path (pace_gapped_);
  // IP10 at higher rates joins this path (ip10_burst above) — per-packet WQEs don't fit the budget.
  pace_line_burst_ = ((!ip10_sampling && fmt.height >= 2160) || ip10_burst) && fmt.height > 0;
  if (pace_line_burst_) {
    t_line_ns_ = static_cast<uint64_t>(frame_interval_ns_ * active * fill) / fmt.height;
    HOLOSCAN_LOG_INFO("st2110_tx: per-line-burst pacing — T_line {} ns ({} active {:.3f} fill {:.2f}, "
                      "1 HW timestamp/line{})", t_line_ns_, wide ? "WIDE" : "NARROW", active, fill,
                      ip10_burst ? ", IP10" : "");
  }

  // Effective throttle horizon. The compute() spin runs until the last packet is within `horizon` of the
  // clock, i.e. compute wall ~ lead + T_active - horizon. The 8ms default is right for 59.94 (T_active
  // 16ms), but at lower frame rates T_active grows (32ms at 29.97), so an 8ms horizon makes the spin eat
  // almost the whole frame budget and the TX falls behind (dropped frames / wrong cadence). Scale the
  // horizon up with the frame interval — submit further ahead, spin less — so compute() keeps real slack.
  // Capped at 17ms (within the tx_pp future window) and never below the configured value, so 59.94 is
  // unchanged. The genlock send-timestamps still pace the wire, so the reduced spin doesn't burst.
  eff_horizon_ns_ = pacing_horizon_ns_.get();
  // Low frame rates (<=~50fps) have a long active period (32ms at 29.97), so the default 8ms horizon
  // leaves the throttle spinning most of the frame and compute() can't return inside the budget -> TX
  // falls behind -> queue overflow -> black. Submit nearly the whole active period ahead so the NIC
  // tx_pp hardware-paces it and compute() returns with slack; the genlock send-stamps still pace the
  // wire, so the reduced spin doesn't burst. 59.94/60 (frame_interval < 20ms) keep the validated horizon.
  if (frame_interval_ns_ > 20000000) {
    const uint64_t big = frame_interval_ns_ - 4000000;  // ~T_active minus a small submit margin
    if (big > eff_horizon_ns_) eff_horizon_ns_ = big;
  } else if (pace_line_burst_) {
    // 59.94/60 line-burst: FRC pair-burst frames have bases 16.7ms apart, and each compute() blocks
    // until T_active - horizon past its base — at the 8ms horizon a pair is busy ~32.7ms of its
    // 33.4ms period (<1ms slack). Jitter then accumulates into queue growth -> RX drops -> capture_ts
    // jumps -> genlock re-anchor oscillation (~34fps drain, seen on the rig 2026-07-02). Submit
    // further ahead (same trick as the low-rate branch): ~12.7ms horizon cuts pair busy to ~28ms,
    // and in-flight (~10k pkts at 59.94) still fits txd 16384. The NIC send-stamps pace the wire,
    // so the earlier hand-off doesn't burst.
    const uint64_t big = frame_interval_ns_ - 4000000;
    if (big > eff_horizon_ns_) eff_horizon_ns_ = big;
  }
  HOLOSCAN_LOG_INFO("st2110_tx: throttle horizon {} ns (configured {} ns)", eff_horizon_ns_,
                    pacing_horizon_ns_.get());
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
    int64_t lead = gbase - static_cast<int64_t>(now);
    // Skip-rate window (two rotating 64-frame buckets) for the chronic-clipping guard below.
    // NOTE (rig lessons, 2026-07-02): lead-based trim servos CANNOT work here. While any upstream
    // queue stands, a frame arrives at TX the moment TX finishes the previous one, so the measured
    // lead is set by the pipeline cycle time and is INVARIANT to the offset: a min-lead shave gate
    // reads noise and never fires, and a reverse trim (raise on low min-lead) only slows TX, grows
    // the queues by exactly the raise, and ratchets forever (-51/-77/-123ms observed). The only
    // signals that truly measure the schedule vs the floor are SKIPS (below) — so latency trim is
    // skip-gated (AIMD), and the offset is otherwise left alone.
    const bool shallow = lead > -static_cast<int64_t>(frame_interval_ns_);
    if (++win_count_ >= 64) {
      win_skips_prev_ = win_skips_;
      win_skips_ = 0;
      win_count_ = 0;
    }
    // Chronic-clipping guard: shallow-late skips are the designed drain mechanism for a standing
    // queue (each one sheds a queue slot; occasional repeats on air while converging). But if a
    // SUSTAINED fraction of frames (>25% of the last 64-128) is being clipped, the latency floor
    // has genuinely risen — fall through to the re-anchor below and accept it ONCE (a step), which
    // is what the removed reverse-trim servo got wrong: it ratcheted the offset continuously
    // because a standing queue is indistinguishable per-frame from a higher floor (-77ms trim seen
    // on the rig 2026-07-02).
    const bool chronic_clip = (win_skips_ + win_skips_prev_) > 32;
    if (lead < static_cast<int64_t>(tgt) / 4 && skip_streak_ < 32 && !chronic_clip) {
      // Late frame (lead below the safe-send floor): skip it UNSENT, offset untouched. Re-anchoring
      // here — the old behavior — bakes the lateness into e2e latency (offset grows by the backlog,
      // and the trim servo takes minutes to shed it) while the queue behind stays full, so upstream
      // keeps dropping and the capture gaps punch idle holes in the schedule: the ~34-44fps limit
      // cycle + past-error bursts seen on the rig 2026-07-02. Skipping purges a standing backlog at
      // compute speed and fresh frames come back inside the band at the ORIGINAL calibrated latency.
      // The receiver sees a brief frame gap (repeat), not a base jump (dip) or thin-lead colored
      // lines. The streak cap handles a genuinely higher new latency floor: if ~32 consecutive frames
      // are all late this is not a purgeable backlog, so fall through once to the re-anchor below and
      // accept the new offset.
      ++skip_streak_;
      ++skipped_late_;
      if (shallow) ++win_skips_;
      // AIMD floor probe (see trim below): a skip after a long clean run means the shave just
      // touched the true latency floor. MULTIPLICATIVE decrease: give back HALF of everything
      // shaved since the last probe (floor at 4ms), so the resting margin self-scales to the
      // pipeline's real jitter (~70ms observed at 4K60 FRC — a fixed 4ms back-off sat inside the
      // jitter band and skip-stormed). Each probe also doubles the re-arm delay, so probes decay
      // geometrically once settled. A skip during a purge/burst (short clean run) is backlog, not
      // a probe: no back-off, no ratchet.
      if (trim_ns_.get() > 0 && frames_since_skip_ > trim_arm_frames_) {
        // Tail-referenced back-off: the probe fires on the jitter TAIL (the worst frame of the
        // window), so restore THAT frame's would-be lead to ~2x target — parking the tail ~tgt
        // above the skip floor and the typical frame well clear. A flat few-ms back-off sat inside
        // the jitter band and churned thin-lead past-errors between probes (42/s observed).
        const int64_t back =
            std::max<int64_t>({2 * static_cast<int64_t>(tgt) - lead,
                               shaved_since_probe_ / 2, int64_t(4000000)});
        genlock_offset_ += static_cast<uint64_t>(back);
        trim_total_ns_ -= back;
        shaved_since_probe_ = 0;
        trim_arm_frames_ = std::min<uint32_t>(trim_arm_frames_ * 2, 8192);
        HOLOSCAN_LOG_INFO(
            "st2110_tx: trim probe hit the floor (lead {} us) — backing off {} us, re-arm {} frames",
            lead / 1000, back / 1000, trim_arm_frames_);
      }
      frames_since_skip_ = 0;
      if ((skipped_late_ & 31) == 1)
        HOLOSCAN_LOG_INFO("st2110_tx: skipping late frame (lead {} us) — purging backlog ({} skipped)",
                          lead / 1000, skipped_late_);
      if (media_ts_ns_ != 0) media_ts_ns_ += frame_interval_ns_;
      return;
    }
    if (lead < static_cast<int64_t>(tgt) / 4 || lead > 2 * static_cast<int64_t>(frame_interval_ns_)) {
      genlock_offset_ = (now + tgt) - frame.capture_ts_ns;  // re-center the lead (rare)
      gbase = static_cast<int64_t>(now) + static_cast<int64_t>(tgt);
      HOLOSCAN_LOG_INFO("st2110_tx: genlock re-anchor #{} — lead was {} us, re-centered to {} us{}",
                        reanchors_ + 1, lead / 1000, static_cast<int64_t>(tgt) / 1000,
                        chronic_clip ? " (chronic clipping: latency floor rose)" : "");
      lead = static_cast<int64_t>(tgt);
      ++reanchors_;
      win_skips_ = win_skips_prev_ = 0;  // fresh offset: don't re-trigger on the old window
      win_count_ = 0;
      frames_since_skip_ = 0;    // and re-approach the floor gently from the new offset
      trim_arm_frames_ = 512;
      shaved_since_probe_ = 0;
    } else if (trim_ns_.get() > 0 && frames_since_skip_ > trim_arm_frames_) {
      // Skip-gated latency trim (AIMD): after a clean run (starts at ~512 sends = ~8.5s, doubles
      // per floor probe), shave the offset a bounded step per frame. Unlike a lead-gated shave
      // this genuinely drains the standing queues — every departure moves earlier, so queue wait
      // shrinks 1:1 — and it keeps shaving until the schedule actually probes the pipeline floor,
      // which announces itself as ONE skipped frame; the skip branch then gives back half of what
      // was shaved (jitter-scaled margin) and doubles the re-arm delay. Immune to the
      // offset-invariance trap that sank the min-lead servos (see the window comment above).
      genlock_offset_ -= trim_ns_.get();
      gbase -= static_cast<int64_t>(trim_ns_.get());
      lead -= static_cast<int64_t>(trim_ns_.get());
      trim_total_ns_ += static_cast<int64_t>(trim_ns_.get());
      shaved_since_probe_ += static_cast<int64_t>(trim_ns_.get());
    }
    ++frames_since_skip_;
    skip_streak_ = 0;  // this frame sends — any late streak is over
    schedule_base_ns_ = static_cast<uint64_t>(gbase);
    last_lead_ns_ = lead;
    genlocked_ = true;
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
    last_lead_ns_ = static_cast<int64_t>(schedule_base_ns_) - static_cast<int64_t>(now);
    genlocked_ = false;
  } else {
    // Legacy fixed grid (raw path): re-anchor only on a genuine >1-frame stall.
    if (schedule_base_ns_ == 0 || now > schedule_base_ns_ + frame_interval_ns_) {
      if (schedule_base_ns_ != 0) ++reanchors_;
      schedule_base_ns_ = now + pacing_horizon_ns_;
    }
    last_lead_ns_ = static_cast<int64_t>(schedule_base_ns_) - static_cast<int64_t>(now);
    genlocked_ = false;
  }
  const uint64_t base = schedule_base_ns_;
  const bool pace = pacing_.get();

  // RTP media timestamp from a MONOTONIC clock anchored once to the GM-locked egress. Decoupling it
  // from schedule_base_ns_ matters: the pacer may re-anchor base (jump), and a timestamp that jumped
  // with it would break a downstream receiver's clock recovery — the monotonic clock never jumps, so
  // BMD-class receivers hold a solid lock. It does SLEW (bounded, below) toward base in genlock mode
  // so the two stay glued across trims and re-anchors without ever stepping.
  if (media_ts_ns_ == 0) media_ts_ns_ = base;
  pktz_->start_frame(spark::st2110::rtp_timestamp_90k(media_ts_ns_));
  // Advance the media clock one frame — plus, in genlock mode, a bounded slew toward the wire base.
  // Without the slew, any base re-anchor or trim leaves the RTP timestamps permanently offset from
  // the wire schedule, and a timestamp-driven receiver keeps playing at the OLD latency: the trimmed
  // wire time buys nothing. Slewing at the same trim rate keeps ts == schedule (both receiver models
  // see the latency drop) while staying smooth and monotonic (|slew| < frame interval by orders).
  int64_t media_adj = 0;
  if (genlocked_ && trim_ns_.get() > 0) {
    const int64_t skew = static_cast<int64_t>(base) - static_cast<int64_t>(media_ts_ns_);
    const int64_t lim = static_cast<int64_t>(trim_ns_.get());
    media_adj = skew > lim ? lim : (skew < -lim ? -lim : skew);
  }
  media_ts_ns_ += frame_interval_ns_ + media_adj;

  uint32_t i = 0;
  uint32_t pace_line = 0xffffffffu, pace_intra = 0;
  for (spark::st2110::PacketPlan p; pktz_->next(p); ++i) {
    uint64_t send_ts;
    if (pace_line_burst_) {
      // Per-line burst: HW-timestamp the FIRST packet of each line at its slot (base + line*T_line); the
      // rest of the line send back-to-back via the send_ts=0 sentinel (no per-packet timestamp WQE). The
      // NIC waits to the slot, then bursts the line at line rate. ~7x fewer send-on-timestamp WQEs.
      const uint32_t line = (p.nsrd > 0) ? p.srd[0].line_no : 0;
      if (line != pace_line) {
        pace_line = line;
        send_ts = base + static_cast<uint64_t>(line) * t_line_ns_;
        if (send_ts < now) send_ts = now;  // grid briefly behind the NIC clock: send now, never past-stamp
      } else {
        send_ts = 0;  // back-to-back after this line's first packet (no HW timestamp)
      }
    } else if (pace_gapped_) {
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
      if (send_ts < now) send_ts = now;  // grid briefly behind the NIC clock: send now, never past-stamp
    } else {
      send_ts = base + static_cast<uint64_t>(i) * gap_ns_;
      if (send_ts < now) send_ts = now;  // grid briefly behind the NIC clock: send now, never past-stamp
    }
    // Closed-loop throttle (the gate-4 lesson): keep the in-flight schedule within the tx_pp window
    // so packets never become future_errors. A real-time sender is *meant* to wait here. Guarded by
    // an independent wall-clock cap so a now_ns() unit mismatch (// BRINGUP) surfaces as future_errors
    // in the xstats rather than hanging a root process.
    if (pace && throttle_enabled_) {
      const auto t0 = std::chrono::steady_clock::now();
      while (send_ts > backend_->now_ns() + eff_horizon_ns_) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(50)) {
          HOLOSCAN_LOG_WARN(
              "st2110_tx: pacing throttle hit 50ms cap — disabling self-throttle (check now_ns() "
              "units / pacing_horizon_ns); NIC tx_pp still active");
          throttle_enabled_ = false;
          break;
        }
        // Per-line-burst path: sleep the bulk of the wait instead of busy-spinning. The throttle's only
        // job is to not submit too far ahead of the NIC clock; the actual wire timing is the per-packet
        // send timestamp the NIC paces on, NOT when software submits. So a sleep-based (less precise)
        // submit is fine given the large horizon — and it stops the throttle pegging a core, which under
        // load starved the RX poll thread (rx packet loss). IP10/others keep the validated tight spin.
        if (pace_line_burst_) {
          const int64_t wait_ns =
              static_cast<int64_t>(send_ts) - static_cast<int64_t>(backend_->now_ns() + eff_horizon_ns_);
          if (wait_ns > 200000) std::this_thread::sleep_for(std::chrono::nanoseconds(wait_ns - 100000));
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
      "st2110_tx stopped: frames={} packets={} reanchors={} skipped_late={} | tx_pp jitter={}ns "
      "wander={}ns sync_lost={} future_err={} past_err={} | final lead {} us, latency trimmed {} ms",
      frames_sent_, packets_sent_, reanchors_, skipped_late_, s.jitter_ns, s.wander_ns, s.sync_lost,
      s.future_errors, s.past_errors, last_lead_ns_ / 1000, trim_total_ns_ / 1000000);
  backend_->shutdown();
  backend_.reset();
}

}  // namespace spark::ops
