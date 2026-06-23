// Full M1/M2 processing pipeline in one process:
//   st2110_rx -> unpack -> resize -> pack -> st2110_tx
// Receives an ST 2110-20 stream, unpacks to the GPU, NPP-resizes (default 1080p->2160p), repacks, and
// re-transmits tx_pp-paced — the transport pass-through with real GPU processing in the middle. Shares
// one EAL across both ports. Validate with a generator feeding the RX port (see st2110_passthrough).
//
//   sudo -n SPARK_PROFILE=1080p SPARK_OUT_W=3840 SPARK_OUT_H=2160 ./engine/build/st2110_pipeline
#include <cstdlib>

#include <holoscan/holoscan.hpp>

#include "operators/codec/codec_ops.hpp"
#include "operators/common/dpdk_eal.hpp"
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
    const std::string interp = env("SPARK_INTERP", "cubic");
    // FRC mode: 0 = off, 1 = retime (1:1 motion-comp), 2 = up-convert (real + mid -> 2x rate).
    const std::string frc_mode = env("SPARK_FRC", "1");
    const bool with_frc = frc_mode != "0";
    const bool frc_2x = frc_mode == "2";
    const uint32_t frc_grid = static_cast<uint32_t>(std::atoll(env("SPARK_FRC_GRID", "1").c_str()));
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
    // Pacing fill: finish each frame within fill×interval so the tail reaches a narrow (2110TPN) receiver
    // before its display deadline. 0.9 default fixes bottom-of-frame breakup at 2160p59.94 IP10 on a 10G BMD.
    const double tx_fill = std::atof(env("SPARK_TX_FILL", "0.9").c_str());
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

    auto& eal = spark::net::DpdkEal::instance();
    eal.add_device(tx_pci, "tx_pp=500");
    eal.add_device(rx_pci, "");
    eal.init("0-11", "spark_pipe");

    auto rx = make_operator<ops::St2110RxOp>("st2110_rx", Arg("pci_addr", rx_pci),
                                             Arg("profile", profile), Arg("manage_eal", false),
                                             Arg("emit_frames", true), Arg("udp_port", rx_port),
                                             Arg("mcast_group", rx_mcast), Arg("src_ip", rx_src),
                                             Arg("iface_ip", rx_iface), Arg("in_width", in_w),
                                             Arg("in_height", in_h), Arg("in_fps", in_fps),
                                             Arg("ip10", in_ip10));
    // frames <= 0 => run until stopped (proto contract: 0 = unbounded, e.g. a live NMOS feed);
    // > 0 => bounded run via CountCondition. Without this guard frames=0 made CountCondition(0)
    // gate the RX to zero compute() calls, so the graph emitted nothing and exited at startup.
    if (frames > 0) rx->add_arg(make_condition<CountCondition>(frames));
    auto unpack = make_operator<ops::UnpackOp>("unpack");
    auto resize = make_operator<ops::ResizeOp>("resize", Arg("out_width", ow), Arg("out_height", oh),
                                               Arg("interp", interp));
    auto pack = make_operator<ops::PackOp>("pack", Arg("out_fps", out_fps), Arg("ip10", ip10));
    // Multicast egress: pass the group as dst_ip and zero the MAC so the backend derives it (RFC 1112).
    auto tx = make_operator<ops::St2110TxOp>(
        "st2110_tx", Arg("pci_addr", tx_pci), Arg("manage_eal", false), Arg("udp_port", tx_port),
        Arg("src_ip", tx_src), Arg("dst_ip", tx_multicast ? tx_mcast : std::string("239.0.0.1")),
        Arg("dst_mac", tx_multicast ? std::string("00:00:00:00:00:00") : dst_mac),
        Arg("pacing_fill", tx_fill));
    add_flow(rx, unpack);
    // FRC runs BEFORE resize so optical flow + interpolation happen at NATIVE input resolution: pixel
    // displacements stay inside the NVOFA search range (they would double on 2160p-upscaled frames and
    // exceed it -> torn warps) and the flow field is 1/4 the size (less GPU, lower jitter). The resize
    // then upscales the real + mid frames identically, so there's no sharpness flicker between them.
    if (with_frc) {
      // motion-compensated interpolation (OFA): rate_mult 2 inserts a real+mid pair -> 2x output rate.
      auto frc = make_operator<ops::FrcOp>("frc", Arg("rate_mult", frc_2x ? 2u : 1u),
                                           Arg("grid_size", frc_grid));
      add_flow(unpack, frc);
      add_flow(frc, resize);
    } else {
      add_flow(unpack, resize);
    }
    add_flow(resize, pack);
    add_flow(pack, tx);
  }
};

}  // namespace spark

int main() {
  HOLOSCAN_LOG_INFO("ST 2110 pipeline: rx -> unpack -> resize -> pack -> tx (1080p->2160p).");
  auto app = holoscan::make_application<spark::St2110Pipeline>();
  // max run time (ms). Default 0 = run until stopped (a live feed must not self-terminate); set
  // SPARK_MAX_MS>0 to bound a test run. Holoscan needs a finite value, so 0 maps to ~1 week.
  const char* ms = std::getenv("SPARK_MAX_MS");
  const int64_t max_ms = (ms && std::atoll(ms) > 0) ? std::atoll(ms) : 7LL * 24 * 3600 * 1000;
  app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
      "mts", holoscan::Arg("worker_thread_number", static_cast<int64_t>(6)),
      holoscan::Arg("stop_on_deadlock", true),
      holoscan::Arg("stop_on_deadlock_timeout", static_cast<int64_t>(3000)),
      holoscan::Arg("max_duration_ms", max_ms)));
  app->run();
  return 0;
}
