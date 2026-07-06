// Full M1/M2 processing pipeline in one process:
//   st2110_rx -> unpack -> resize -> pack -> st2110_tx
// Receives an ST 2110-20 stream, unpacks to the GPU, NPP-resizes (default 1080p->2160p), repacks, and
// re-transmits tx_pp-paced — the transport pass-through with real GPU processing in the middle. Shares
// one EAL across both ports. Validate with a generator feeding the RX port (see st2110_passthrough).
//
//   sudo -n SPARK_PROFILE=1080p SPARK_OUT_W=3840 SPARK_OUT_H=2160 ./engine/build/st2110_pipeline
#include <sys/mman.h>
#include <sys/resource.h>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "operators/codec/codec_ops.hpp"
#include "operators/common/dpdk_eal.hpp"
#include "operators/filters/filters.hpp"
#include "operators/frc/frc.hpp"
#include "operators/resize/resize.hpp"
#include "operators/st2110_rx/st2110_rx.hpp"
#include "operators/st2110_tx/st2110_tx.hpp"

namespace spark {

class St2110Pipeline : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto env = [](const char* k, const char* d) {
      const char* v = std::getenv(k);
      return std::string(v ? v : d);
    };
    const std::string profile = env("SPARK_PROFILE", "1080p");  // input resolution
    // "auto": supersampling for downscales (anti-aliased), cubic for upscales, passthrough at 1:1.
    const std::string interp = env("SPARK_INTERP", "auto");
    // FRC mode: 0 = off, 1 = retime (1:1 motion-comp), 2 = up-convert (real + mid -> 2x rate),
    // 3 = uniform-grid up-convert (2x rate on a rigid nominal grid, phase from true capture times —
    // absorbs erratic source timing; see frc_grid.hpp).
    const std::string frc_mode = env("SPARK_FRC", "1");
    const bool with_frc = frc_mode != "0";
    const bool frc_2x = frc_mode == "2" || frc_mode == "3";
    const bool frc_uniform = frc_mode == "3";
    // NVOF flow grid: 1|2|4 px per output vector (hardware only supports those three; anything
    // else falls back to 4 in NvofFlow). Default is resolution-aware — resolved after the input
    // dims are parsed below; SPARK_FRC_GRID forces a value.
    const uint32_t frc_grid_env = static_cast<uint32_t>(std::atoll(env("SPARK_FRC_GRID", "0").c_str()));
    const int64_t frames = std::atoll(env("SPARK_FRAMES", "300").c_str());
    const uint32_t ow = static_cast<uint32_t>(std::atoll(env("SPARK_OUT_W", "3840").c_str()));
    const uint32_t oh = static_cast<uint32_t>(std::atoll(env("SPARK_OUT_H", "2160").c_str()));
    // Blackmagic IP10 (10:8) output: required for Blackmagic receivers to take 2160p59.94/60 over 10G
    // (uncompressed 4K60 ~12 Gbps won't fit). Off => uncompressed RFC 4175 (unchanged default path).
    const bool ip10 = env("SPARK_IP10", "0") != "0";
    // IP10 on the INPUT side: the source (e.g. a Blackmagic 2160p60 sender) is IP10-coded, so the RX
    // depacketizes 8-bit pgroups and the unpack stage IP10-decodes them back to 10-bit. Independent of
    // the output codec — Spark can receive IP10 and send raw, or vice versa.
    const bool in_ip10 = env("SPARK_IN_IP10", "0") != "0";
    const std::string rx_pci = env("SPARK_RX_PCI", "0000:01:00.1");
    const std::string tx_pci = env("SPARK_TX_PCI", "0002:01:00.0");
    const std::string dst_mac = env("SPARK_DST_MAC", "00:00:5e:00:53:30");
    // NMOS / network-layer source + sink (M6). When SPARK_RX_MCAST is set the RX joins that group
    // (IGMPv3) and filters by group/source; when SPARK_TX_MCAST is set the TX egresses to that group
    // with an RFC-1112-derived MAC (dst_mac forced to zero so the backend derives it).
    const std::string rx_mcast = env("SPARK_RX_MCAST", "");
    const std::string rx_src = env("SPARK_RX_SRC", "");
    const std::string rx_iface = env("SPARK_RX_IFACE", "");
    uint32_t rx_port = static_cast<uint32_t>(std::atoll(env("SPARK_RX_PORT", "0").c_str()));
    if (rx_port == 0) rx_port = 20000;  // proto default 0 / legacy runs -> the standard 2110 port
    const std::string tx_mcast = env("SPARK_TX_MCAST", "");
    uint32_t tx_port = static_cast<uint32_t>(std::atoll(env("SPARK_TX_PORT", "0").c_str()));
    if (tx_port == 0) tx_port = 20000;
    const bool tx_multicast = !tx_mcast.empty();
    const std::string tx_src = env("SPARK_TX_SRC", "192.168.50.10");  // egress source IP (SDP source-filter)
    // Pacing fill: fine-tune on top of the ST 2110-21 active-period rate (the TX paces over T_active by
    // default now). 1.0 = exact narrow rate; lower only if a receiver needs the tail even earlier.
    const double tx_fill = std::atof(env("SPARK_TX_FILL", "1.0").c_str());
    // ST 2110-21 sender compliance: "narrow" (default, 2110TPN) paces over the active period with
    // per-line gapped bursts; "wide" (2110TPW) paces evenly over the full frame. Set SPARK_TX_TP=wide
    // for receivers (e.g. BiDirect-2) whose wide buffer tolerates looser timing. The NMOS node reads the
    // SAME env var for the SDP's TP= field, so set it for both processes to keep wire + SDP in sync.
    const std::string tx_tp = env("SPARK_TX_TP", "narrow");
    const bool tx_wide = tx_tp == "wide" || tx_tp == "W" || tx_tp == "2110TPW";
    // Latency knobs (docs/M8-latency.md). SPARK_TX_LEAD_NS: how far ahead of the NIC clock the genlock
    // schedules each frame — every ns of lead is a ns of end-to-end latency, but it's also the pipeline
    // jitter shock-absorber; lower it only while watching tx_past_err stay 0. SPARK_TX_TRIM_NS: max
    // ns/frame the AIMD skip-gated shave (st2110_tx) slews the send base + RTP clock down while
    // probing the latency floor (0 = off). The 2026-07-02 stall episodes that made any near-floor
    // schedule skip-storm turned out to be driven by the burst-deep inter-op queues themselves;
    // with per-hop capacities (pipeline_queue_cap) the chain is stall-free and the shave converges.
    // 20us/frame = 0.12% momentary media-clock skew while converging (~1.2ms/s).
    const uint32_t tx_lead =
        static_cast<uint32_t>(std::atoll(env("SPARK_TX_LEAD_NS", "8000000").c_str()));
    const uint32_t tx_trim =
        static_cast<uint32_t>(std::atoll(env("SPARK_TX_TRIM_NS", "20000").c_str()));
    // FIXED end-to-end latency (M10, docs/M10-audio-fixed-latency.md): wire time = capture + L for
    // BOTH essences. Default ON at 105 ms — deterministic beats minimal (the servo's moving latency
    // made lip-sync unsolvable). Sizing (2026-07-03 probe ladder): chain floor ~73 ms (L=65 fails
    // ~13%, lead ≈ −8 ms) + the residual host stall class (~17 ms bursts, ~1-2 per 10 min — bit an
    // L=85 run; L=80's clean pass predates a stall window) + margin. Below L≈78 the RX L-store
    // drains fully and tx_margin_us becomes a true headroom signal; at higher L the arrival lead
    // pins at ~13.4 ms and margin is meaningless. SPARK_LATENCY_MS=0 restores the servo.
    const double latency_ms = std::atof(env("SPARK_LATENCY_MS", "105").c_str());
    const uint64_t latency_ns =
        latency_ms > 0 ? static_cast<uint64_t>(latency_ms * 1e6) : uint64_t(0);
    const int64_t av_offset_ns =
        static_cast<int64_t>(std::atof(env("SPARK_AV_OFFSET_MS", "0").c_str()) * 1e6);
    // Companion ST 2110-30 audio (relayed bit-transparent, re-timed to capture + L). Active only
    // when BOTH sides are routed and the schedule is fixed — lip-sync is undefined under the servo.
    const std::string rx_a_mcast = env("SPARK_RX_AUDIO_MCAST", "");
    const std::string rx_a_src = env("SPARK_RX_AUDIO_SRC", "");
    uint32_t rx_a_port = static_cast<uint32_t>(std::atoll(env("SPARK_RX_AUDIO_PORT", "0").c_str()));
    if (rx_a_port == 0) rx_a_port = 5004;
    const std::string tx_a_mcast = env("SPARK_TX_AUDIO_MCAST", "");
    uint32_t tx_a_port = static_cast<uint32_t>(std::atoll(env("SPARK_TX_AUDIO_PORT", "0").c_str()));
    if (tx_a_port == 0) tx_a_port = 5004;
    const uint32_t audio_rate =
        static_cast<uint32_t>(std::atoll(env("SPARK_AUDIO_RATE", "48000").c_str()));
    bool audio = !rx_a_mcast.empty() && !tx_a_mcast.empty();
    if (!rx_a_mcast.empty() && tx_a_mcast.empty())
      HOLOSCAN_LOG_WARN("audio: RX group set but no SPARK_TX_AUDIO_MCAST — audio disabled");
    if (audio && latency_ns == 0) {
      HOLOSCAN_LOG_WARN("audio: requires the fixed-latency schedule (SPARK_LATENCY_MS>0) — disabled");
      audio = false;
    }
    // Source format from the SDP (SPARK_IN_*; the NMOS bridge fills these from the sender's fmtp). The
    // real input rate must reach the TX pacer — FRC here is 1:1, so the output rate == the input rate.
    auto parse_rate = [](const std::string& s) -> double {
      const auto slash = s.find('/');
      if (slash == std::string::npos) return std::atof(s.c_str());
      const double den = std::atof(s.substr(slash + 1).c_str());
      return den != 0.0 ? std::atof(s.substr(0, slash).c_str()) / den : 0.0;
    };
    const uint32_t in_w = static_cast<uint32_t>(std::atoll(env("SPARK_IN_W", "0").c_str()));
    const uint32_t in_h = static_cast<uint32_t>(std::atoll(env("SPARK_IN_H", "0").c_str()));
    const double in_fps = parse_rate(env("SPARK_IN_FPS", ""));
    const double base_fps = in_fps > 0.0 ? in_fps : 60000.0 / 1001.0;
    // Up-convert doubles the media rate; the TX pacer + RTP media clock track out_fps (e.g. 30->60).
    const double out_fps = frc_2x ? base_fps * 2.0 : base_fps;
    // Flow runs at INPUT resolution (frc sits before scale in the chain). At <=1080p, grid 1 is
    // realtime with the same TX margin as grid 4 and a clearly better picture (on-air A/B
    // 2026-07-06: 1 >> 2 >> 4); larger formats keep 4 for headroom. Unknown dims -> conservative 4.
    const uint32_t frc_grid = frc_grid_env != 0
        ? frc_grid_env
        : (in_w > 0 && in_h > 0 && in_w <= 1920 && in_h <= 1080 ? 1u : 4u);

    auto& eal = spark::net::DpdkEal::instance();
    eal.add_device(tx_pci, "tx_pp=500");
    eal.add_device(rx_pci, "");
    eal.init("0-11", "spark_pipe");

    // --- modular filter chain (M9) ---
    // Optional GPU stages between unpack and pack, each enabled by its own parameters:
    //   sharpen: luma unsharp amount (SPARK_SHARPEN, 0 = off)
    //   procamp: brightness/contrast/saturation/hue (SPARK_PA_BRIGHT/CONTRAST/SAT/HUE; neutral = off)
    // Default order frc,scale,sharpen,procamp — flow estimation at native input resolution, sharpen
    // at the delivery resolution, levels trimmed last. SPARK_FILTERS overrides the set/order
    // explicitly (comma list of frc|scale|sharpen|procamp; unpack/pack/tx stay implicit).
    const double sharpen_amt = std::atof(env("SPARK_SHARPEN", "0").c_str());
    const double pa_bright = std::atof(env("SPARK_PA_BRIGHT", "0").c_str());
    // 0/negative contrast+saturation read as "unset" -> neutral: a proto3/JSON config that omits the
    // field arrives as 0, and silently forcing every frame to black would be a rude default.
    double pa_contrast = std::atof(env("SPARK_PA_CONTRAST", "1").c_str());
    if (pa_contrast <= 0.0) pa_contrast = 1.0;
    double pa_sat = std::atof(env("SPARK_PA_SAT", "1").c_str());
    if (pa_sat <= 0.0) pa_sat = 1.0;
    const double pa_hue = std::atof(env("SPARK_PA_HUE", "0").c_str());
    const bool with_procamp =
        pa_bright != 0.0 || pa_contrast != 1.0 || pa_sat != 1.0 || pa_hue != 0.0;
    std::string filters = env("SPARK_FILTERS", "");
    if (filters.empty()) {
      filters = with_frc ? "frc,scale" : "scale";
      if (sharpen_amt > 0.0) filters += ",sharpen";
      if (with_procamp) filters += ",procamp";
    }

    // Announce which op receives FRC's multi-frame burst (the only hop that needs burst-deep input
    // capacity; see pipeline_queue_cap). Must happen BEFORE any make_operator — setup() runs there.
    {
      std::string sink;
      if (with_frc) {
        std::stringstream pre(filters);
        std::map<std::string, int> seen_pre;
        bool after_frc = false;
        for (std::string tok; std::getline(pre, tok, ',');) {
          if (tok != "frc" && tok != "scale" && tok != "sharpen" && tok != "procamp") continue;
          const int nth = seen_pre[tok]++;
          const std::string name = nth ? tok + std::to_string(nth + 1) : tok;
          if (after_frc) {
            sink = name;
            break;
          }
          if (tok == "frc") after_frc = true;
        }
        if (after_frc && sink.empty()) sink = "pack";  // frc is the last filter -> pack takes the burst
      }
      setenv("SPARK_BURST_SINK", sink.c_str(), 1);
    }
    // Fixed-latency frame store: the standing ~L worth of source frames waits in the RX frame
    // queue (the NIC's tx_pp window holds only ~ms), so size it from L + the source rate. Servo
    // mode keeps the legacy depth 8.
    const uint32_t q_depth =
        latency_ns > 0
            ? std::max<uint32_t>(8, static_cast<uint32_t>(latency_ms * base_fps / 1000.0) + 4)
            : 8;
    auto rx = make_operator<ops::St2110RxOp>("st2110_rx", Arg("pci_addr", rx_pci),
                                             Arg("profile", profile), Arg("manage_eal", false),
                                             Arg("emit_frames", true), Arg("udp_port", rx_port),
                                             Arg("mcast_group", rx_mcast), Arg("src_ip", rx_src),
                                             Arg("iface_ip", rx_iface), Arg("in_width", in_w),
                                             Arg("in_height", in_h), Arg("in_fps", in_fps),
                                             Arg("ip10", in_ip10), Arg("frame_q_depth", q_depth),
                                             Arg("audio_mcast", audio ? rx_a_mcast : std::string("")),
                                             Arg("audio_src", rx_a_src),
                                             Arg("audio_port", audio ? rx_a_port : uint32_t(0)),
                                             Arg("audio_rate", audio_rate));
    // frames <= 0 => run until stopped (proto contract: 0 = unbounded, e.g. a live NMOS feed);
    // > 0 => bounded run via CountCondition. Without this guard frames=0 made CountCondition(0)
    // gate the RX to zero compute() calls, so the graph emitted nothing and exited at startup.
    if (frames > 0) rx->add_arg(make_condition<CountCondition>(frames));
    auto unpack = make_operator<ops::UnpackOp>("unpack");
    auto pack = make_operator<ops::PackOp>("pack", Arg("out_fps", out_fps), Arg("ip10", ip10));
    // Multicast egress: pass the group as dst_ip and zero the MAC so the backend derives it (RFC 1112).
    auto tx = make_operator<ops::St2110TxOp>(
        "st2110_tx", Arg("pci_addr", tx_pci), Arg("manage_eal", false), Arg("udp_port", tx_port),
        Arg("src_ip", tx_src), Arg("dst_ip", tx_multicast ? tx_mcast : std::string("239.0.0.1")),
        Arg("dst_mac", tx_multicast ? std::string("00:00:00:00:00:00") : dst_mac),
        Arg("pacing_fill", tx_fill),
        // IP10: 1300-octet UDP payload -> 1280 pixel octets -> exactly 6 packets per 2160p line, matching
        // the Blackmagic reference (line-aligned). Raw keeps the ~1420 budget.
        Arg("payload_size", ip10 ? uint32_t(1300) : uint32_t(1420)),
        // IP10 HW pacing: a 2160p frame is ~12960 packets, so give the ring room for a whole frame and a
        // horizon that lets compute() submit it all at once with only a small schedule lead. The NIC
        // tx_pp then HW-paces the frame while compute() returns in ~ms — ~a frame of slack absorbs
        // pipeline jitter so the cadence grid never lags (no past-errors / re-anchors / dropped frames).
        // IP10 pacing. The throttle spin (horizon 8ms) is kept: it both HW-paces (submits each packet
        // ~8ms ahead for tx_pp) AND backpressures the compute() rate to one frame per interval, so the tx
        // never drains the RX queue in bursts (which collapses frame spacing). The bounded-lead re-anchor
        // (reanchor_lead) then holds the schedule a stable 8ms ahead of the NIC clock: the spin keeps the
        // lead steady frame-to-frame, so the band only trips on real drift — no per-frame slip (drops) and
        // no eroded lead (unpaced past-error bursts -> colored lines). txd gives the 8ms horizon headroom.
        // Raw 2160p is even denser than IP10 (line-aligned => ~15120 pkts/frame), so it needs a full-frame
        // ring too. Without it the TX ring (8192) fills mid-frame and the backend busy-spins on a full ring
        // (drain_pending), capping throughput below the source rate -> RX frame-queue overflow -> dips to
        // black. 16384 is the mlx5 hardware max (WQEBB limit; 32768 is rejected at queue setup) and still
        // holds the whole 15120-pkt frame: compute() dumps it and returns, the NIC tx_pp HW-paces it out,
        // and the scaled throttle horizon keeps frame-overlap ring occupancy under 16384. 1080p raw keeps
        // the small ring (its frame easily fits 8192).
        Arg("txd", ip10 ? uint32_t(16384) : (oh >= 2160 ? uint32_t(16384) : uint32_t(8192))),
        // Genlock (BOTH codecs): anchor the send base to the source's clean capture_ts with an 8ms lead,
        // so the base is source-locked and smooth (no dips/drops from chasing the jittery local clock); it
        // only re-centers on a real >6ms latency spike. horizon 8ms gives the throttle strong backpressure
        // (shallow RX queue -> ~no drops). IP10 adds gapped/line-aligned packetization on top; raw keeps
        // even ST 2110-21 narrow pacing (no per-line context, so even is correct there).
        Arg("pacing_horizon_ns", std::max(tx_lead, uint32_t(8000000))),
        Arg("reanchor_lead_ns", tx_lead), Arg("trim_ns", tx_trim),
        // Wide (2110TPW) when SPARK_TX_TP=wide: even full-frame pacing, no per-line gaps. The receiver's
        // wide buffer absorbs tx_pp jitter (fixes the narrow dips); keep the SDP TP= in sync via the NMOS node.
        Arg("tx_wide", tx_wide),
        // M10: fixed capture->wire latency + the audio relay slaved to the same constant.
        Arg("latency_ns", latency_ns),
        Arg("audio_dst_ip", audio ? tx_a_mcast : std::string("")),
        Arg("audio_port", audio ? tx_a_port : uint32_t(0)), Arg("audio_rate", audio_rate),
        Arg("av_offset_ns", av_offset_ns));
    // Compose the GpuFrame chain unpack -> [filters...] -> pack from the `filters` token list. Every
    // filter is a GpuFrame->GpuFrame operator (own stream, ready-event ordered, out-of-place pool),
    // so any subset in any order wires the same way. Default order rationale: FRC BEFORE scale so
    // optical flow + interpolation run at NATIVE input resolution (pixel displacements stay inside
    // the NVOFA search range — they'd double on 2160p-upscaled frames and exceed it -> torn warps —
    // and the flow field is 1/4 the size); sharpen AFTER scale so it counters interpolation softness
    // at the delivery resolution; procamp last as the final levels trim.
    std::vector<std::shared_ptr<Operator>> chain{unpack};
    std::string composed = "rx -> unpack";
    std::stringstream toks(filters);
    std::map<std::string, int> seen;  // operator names must be unique; "procamp,procamp" is legal
    for (std::string tok; std::getline(toks, tok, ',');) {
      if (tok.empty()) continue;
      const int nth = seen[tok]++;
      const std::string name = nth ? tok + std::to_string(nth + 1) : tok;
      if (tok == "frc") {
        if (!with_frc) {  // frc listed but SPARK_FRC=0: mode governs behavior, so skip it
          HOLOSCAN_LOG_WARN("chain: 'frc' listed in SPARK_FILTERS but SPARK_FRC=0 — skipping");
          continue;
        }
        // motion-compensated interpolation (OFA): rate_mult 2 inserts a real+mid pair -> 2x rate;
        // mode 3 hands FRC the nominal output interval and it emits on that rigid grid instead
        // (phase from true capture times — erratic source timing becomes phase error, not cadence).
        chain.push_back(make_operator<ops::FrcOp>(
            name, Arg("rate_mult", frc_2x ? 2u : 1u), Arg("grid_size", frc_grid),
            Arg("out_interval_ns",
                frc_uniform ? static_cast<uint64_t>(1e9 / out_fps + 0.5) : uint64_t(0))));
      } else if (tok == "scale") {
        chain.push_back(make_operator<ops::ResizeOp>(name, Arg("out_width", ow),
                                                     Arg("out_height", oh), Arg("interp", interp)));
      } else if (tok == "sharpen") {
        chain.push_back(make_operator<ops::SharpenOp>(name, Arg("amount", sharpen_amt)));
      } else if (tok == "procamp") {
        chain.push_back(make_operator<ops::ProcAmpOp>(
            name, Arg("brightness", pa_bright), Arg("contrast", pa_contrast),
            Arg("saturation", pa_sat), Arg("hue_deg", pa_hue)));
      } else {
        HOLOSCAN_LOG_WARN("chain: unknown filter '{}' in SPARK_FILTERS — skipping", tok);
        continue;
      }
      composed += " -> " + name;
    }
    chain.push_back(pack);
    add_flow(rx, unpack);
    for (size_t i = 0; i + 1 < chain.size(); ++i) add_flow(chain[i], chain[i + 1]);
    add_flow(pack, tx);
    HOLOSCAN_LOG_INFO("chain: {} -> pack -> tx", composed);
  }
};

}  // namespace spark

int main() {
  HOLOSCAN_LOG_INFO("ST 2110 pipeline: rx -> unpack -> resize -> pack -> tx (1080p->2160p).");
  // Shield the realtime pipeline from host CPU contention. The residual ~300ms pipe stalls were
  // reproduced ON DEMAND with 16 CPU burners (GPU fence 1ms — pure CFS scheduling starvation of
  // the RX poll / scheduler worker threads) and never occur on a quiet host. Linux nice is
  // per-thread and INHERITED at thread creation, so set it first thing in main: every EAL lcore,
  // Holoscan worker, and RX poll thread created below runs at this priority. -15 outweighs
  // default-nice work ~29:1 without the starve-the-kernel risks of SCHED_FIFO. SPARK_NICE
  // overrides (0 disables).
  {
    const char* n = std::getenv("SPARK_NICE");
    const int prio = n ? std::atoi(n) : -15;
    if (prio != 0) {
      if (setpriority(PRIO_PROCESS, 0, prio) == 0)
        HOLOSCAN_LOG_INFO("pipeline nice set to {} (inherited by all engine threads)", prio);
      else
        HOLOSCAN_LOG_WARN("setpriority({}) failed (not root?) — vulnerable to host CPU contention",
                          prio);
    }
  }
  // The perf autopsy of the induced ~250ms stall (2026-07-02) showed the worker thread inside
  // cuLibraryLoadData (CUDA lazily loading an NPP resize kernel mid-run) plus kernel page-fault
  // storms — host MEMORY pressure, not timeslice starvation (idle CPU existed; nice didn't help).
  // Two shields, both env-overridable:
  //  - eager CUDA module loading: every kernel loads at init instead of on first use, so the
  //    driver never takes the module-load path mid-frame (costs startup time only);
  //  - mlockall ONFAULT: engine + driver pages stay resident once touched, so external memory
  //    churn can't evict them into reload/major-fault stalls.
  if (!std::getenv("SPARK_CUDA_LAZY")) setenv("CUDA_MODULE_LOADING", "EAGER", 0);
  if (!std::getenv("SPARK_NO_MLOCK")) {
    if (mlockall(MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT) == 0)
      HOLOSCAN_LOG_INFO("mlockall(ONFAULT) — engine pages pinned resident");
    else
      HOLOSCAN_LOG_WARN("mlockall failed — vulnerable to host memory pressure (page-fault stalls)");
  }
  auto app = holoscan::make_application<spark::St2110Pipeline>();
  // max run time (ms). Default 0 = run until stopped (a live feed must not self-terminate); set
  // SPARK_MAX_MS>0 to bound a test run. Holoscan needs a finite value, so 0 maps to ~1 week.
  const char* ms = std::getenv("SPARK_MAX_MS");
  const int64_t max_ms = (ms && std::atoll(ms) > 0) ? std::atoll(ms) : 7LL * 24 * 3600 * 1000;
  // Scheduler (SPARK_SCHED): "event" (default) = EventBasedScheduler — operators are dispatched the
  // moment an upstream emit readies them. "mts" = the legacy MultiThreadScheduler, whose polling
  // thread sleeps check_recession_period_ms (default 5 ms!) whenever a pass finds nothing ready, so
  // EVERY operator hop can eat up to 5 ms of pure scheduler latency — across this graph's 5 hops
  // that's the single largest avoidable latency term. Keep "mts" as the fallback escape hatch.
  const char* sched = std::getenv("SPARK_SCHED");
  if (sched && std::string(sched) == "mts") {
    HOLOSCAN_LOG_INFO("scheduler: MultiThreadScheduler (SPARK_SCHED=mts fallback)");
    app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
        "mts", holoscan::Arg("worker_thread_number", static_cast<int64_t>(6)),
        holoscan::Arg("stop_on_deadlock", true),
        holoscan::Arg("stop_on_deadlock_timeout", static_cast<int64_t>(3000)),
        holoscan::Arg("max_duration_ms", max_ms)));
  } else {
    HOLOSCAN_LOG_INFO("scheduler: EventBasedScheduler (low-latency; SPARK_SCHED=mts to fall back)");
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
        "ebs", holoscan::Arg("worker_thread_number", static_cast<int64_t>(6)),
        holoscan::Arg("stop_on_deadlock", true),
        holoscan::Arg("stop_on_deadlock_timeout", static_cast<int64_t>(3000)),
        holoscan::Arg("max_duration_ms", max_ms)));
  }
  app->run();
  return 0;
}
