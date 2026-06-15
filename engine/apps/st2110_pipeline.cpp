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
    const int64_t frames = std::atoll(env("SPARK_FRAMES", "300").c_str());
    const uint32_t ow = static_cast<uint32_t>(std::atoll(env("SPARK_OUT_W", "3840").c_str()));
    const uint32_t oh = static_cast<uint32_t>(std::atoll(env("SPARK_OUT_H", "2160").c_str()));
    const std::string rx_pci = env("SPARK_RX_PCI", "0000:01:00.1");
    const std::string tx_pci = env("SPARK_TX_PCI", "0002:01:00.0");
    const std::string dst_mac = env("SPARK_DST_MAC", "30:c5:99:3e:9d:30");

    auto& eal = spark::net::DpdkEal::instance();
    eal.add_device(tx_pci, "tx_pp=500");
    eal.add_device(rx_pci, "");
    eal.init("0-11", "spark_pipe");

    auto rx = make_operator<ops::St2110RxOp>("st2110_rx", Arg("pci_addr", rx_pci),
                                             Arg("profile", profile), Arg("manage_eal", false),
                                             Arg("emit_frames", true),
                                             make_condition<CountCondition>(frames));
    auto unpack = make_operator<ops::UnpackOp>("unpack");
    auto resize = make_operator<ops::ResizeOp>("resize", Arg("out_width", ow), Arg("out_height", oh),
                                               Arg("interp", interp));
    auto pack = make_operator<ops::PackOp>("pack");
    auto tx = make_operator<ops::St2110TxOp>("st2110_tx", Arg("pci_addr", tx_pci),
                                             Arg("dst_mac", dst_mac), Arg("manage_eal", false));
    add_flow(rx, unpack);
    add_flow(unpack, resize);
    add_flow(resize, pack);
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
