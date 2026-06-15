// ST 2110 RX bring-up app: a single st2110_rx operator that receives the loopback stream and reports
// loss + zero-copy ingest latency. Run as the RX half of the two-process loopback (mirrors gate-4):
//
//   T=~/holoscan-sdk/install-cu13-aarch64-dgpu
//   sudo -n SPARK_PROFILE=2160p SPARK_SECONDS=12 ./engine/build/st2110_rx_smoke &   # start RX first
//   sudo -n SPARK_PROFILE=2160p ./engine/build/st2110_tx_smoke                      # then TX
//
// Defaults: RX on 0002:01:00.1 (the gate-4 RX port), udp:20000. SPARK_PROFILE / SPARK_SECONDS env.
#include <cstdlib>

#include <holoscan/holoscan.hpp>

#include "operators/frame_sink/frame_sink.hpp"
#include "operators/st2110_rx/st2110_rx.hpp"

namespace spark {

class St2110RxSmoke : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    const char* prof = std::getenv("SPARK_PROFILE");
    const char* secs = std::getenv("SPARK_SECONDS");
    const char* pci = std::getenv("SPARK_RX_PCI");
    auto rx = make_operator<ops::St2110RxOp>(
        "st2110_rx", Arg("profile", std::string(prof ? prof : "1080p")),
        Arg("pci_addr", std::string(pci ? pci : "0002:01:00.1")),
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
