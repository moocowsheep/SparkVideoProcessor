// ST 2110 RX bring-up app: a single st2110_rx operator that receives the loopback stream and reports
// loss + zero-copy ingest latency. Run as the RX half of the two-process loopback (mirrors gate-4):
//
//   T=~/holoscan-sdk/install-cu13-aarch64-dgpu
//   sudo -n SPARK_PROFILE=2160p SPARK_SECONDS=12 ./engine/build/st2110_rx_smoke &   # start RX first
//   sudo -n SPARK_PROFILE=2160p ./engine/build/st2110_tx_smoke                      # then TX
//
// Defaults: RX on 0002:01:00.1 (the gate-4 RX port), udp:20000. SPARK_PROFILE / SPARK_SECONDS env.
// Multicast (NMOS/real source): set SPARK_RX_MCAST (+ SPARK_RX_SRC for SSM, SPARK_RX_PORT,
// SPARK_RX_IFACE) to IGMP-join a group and filter to it — e.g. point it at a real 2110-20 sender.
#include <cstdint>
#include <cstdlib>

#include <holoscan/holoscan.hpp>

#include "operators/frame_sink/frame_sink.hpp"
#include "operators/st2110_rx/st2110_rx.hpp"

namespace spark {

class St2110RxSmoke : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto env = [](const char* k, const char* d) {
      const char* v = std::getenv(k);
      return std::string(v ? v : d);
    };
    const char* secs = std::getenv("SPARK_SECONDS");
    const char* port = std::getenv("SPARK_RX_PORT");
    auto rx = make_operator<ops::St2110RxOp>(
        "st2110_rx", Arg("profile", env("SPARK_PROFILE", "1080p")),
        Arg("pci_addr", env("SPARK_RX_PCI", "0002:01:00.1")),
        Arg("udp_port", port ? static_cast<uint32_t>(std::atoi(port)) : 20000u),
        Arg("mcast_group", env("SPARK_RX_MCAST", "")), Arg("src_ip", env("SPARK_RX_SRC", "")),
        Arg("iface_ip", env("SPARK_RX_IFACE", "")),
        Arg("run_seconds", secs ? std::atof(secs) : 12.0), make_condition<CountCondition>(1));
    auto sink = make_operator<ops::FrameSinkOp>("frame_sink");  // sink-mode rx never emits; wire anyway
    add_flow(rx, sink);
  }
};

}  // namespace spark

int main() {
  HOLOSCAN_LOG_INFO("ST 2110 RX smoke: st2110_rx (loopback receiver, loss + ingest latency).");
  auto app = holoscan::make_application<spark::St2110RxSmoke>();
  app->run();
  return 0;
}
