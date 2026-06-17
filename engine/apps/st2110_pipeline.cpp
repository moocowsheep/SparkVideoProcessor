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
    const bool with_frc = env("SPARK_FRC", "1") != "0";  // include motion-comp FRC stage
    const int64_t frames = std::atoll(env("SPARK_FRAMES", "300").c_str());
    const uint32_t ow = static_cast<uint32_t>(std::atoll(env("SPARK_OUT_W", "3840").c_str()));
    const uint32_t oh = static_cast<uint32_t>(std::atoll(env("SPARK_OUT_H", "2160").c_str()));
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
    const double out_fps = in_fps > 0.0 ? in_fps : 60000.0 / 1001.0;

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
                                             make_condition<CountCondition>(frames));
    auto unpack = make_operator<ops::UnpackOp>("unpack");
    auto resize = make_operator<ops::ResizeOp>("resize", Arg("out_width", ow), Arg("out_height", oh),
                                               Arg("interp", interp));
    auto pack = make_operator<ops::PackOp>("pack", Arg("out_fps", out_fps));
    // Multicast egress: pass the group as dst_ip and zero the MAC so the backend derives it (RFC 1112).
    auto tx = make_operator<ops::St2110TxOp>(
        "st2110_tx", Arg("pci_addr", tx_pci), Arg("manage_eal", false), Arg("udp_port", tx_port),
        Arg("src_ip", tx_src), Arg("dst_ip", tx_multicast ? tx_mcast : std::string("239.0.0.1")),
        Arg("dst_mac", tx_multicast ? std::string("00:00:00:00:00:00") : dst_mac));
    add_flow(rx, unpack);
    add_flow(unpack, resize);
    if (with_frc) {
      auto frc = make_operator<ops::FrcOp>("frc");  // motion-compensated interpolation (OFA)
      add_flow(resize, frc);
      add_flow(frc, pack);
    } else {
      add_flow(resize, pack);
    }
    add_flow(pack, tx);
  }
};

}  // namespace spark

int main() {
  HOLOSCAN_LOG_INFO("ST 2110 pipeline: rx -> unpack -> resize -> pack -> tx (1080p->2160p).");
  auto app = holoscan::make_application<spark::St2110Pipeline>();
  app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
      "mts", holoscan::Arg("worker_thread_number", static_cast<int64_t>(6)),
      holoscan::Arg("stop_on_deadlock", true),
      holoscan::Arg("stop_on_deadlock_timeout", static_cast<int64_t>(3000)),
      holoscan::Arg("max_duration_ms", static_cast<int64_t>(120000))));
  app->run();
  return 0;
}
