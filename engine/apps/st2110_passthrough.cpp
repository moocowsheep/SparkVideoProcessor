// ST 2110 pass-through: st2110_rx -> st2110_tx in ONE process (shared EAL across two ports). RX
// receives a stream on one port, reassembles each frame, and TX re-transmits it paced on another —
// the M1 "first deliverable" graph (processing operators graft between rx and tx later).
//
// Topology (4-port switched fabric): a separate generator feeds the RX port; this process forwards to
// a third port. Default: RX 0000:01:00.1 (port B) -> TX 0002:01:00.0 (port C) -> dst 0002:01:00.1
// (port D). Validate the forwarding by this process's own rx (frames in, loss) + tx (frames out,
// future_err) stats; an st2110_rx_smoke on port D confirms end-to-end.
//
//   sudo -n SPARK_PROFILE=2160p ./engine/build/st2110_passthrough        # the forwarder
//   sudo -n SPARK_TX_PCI=0000:01:00.0 SPARK_DST_MAC=00:00:5e:00:53:2c \   # the generator -> port B
//           SPARK_WARMUP_MS=800 SPARK_PROFILE=2160p ./engine/build/st2110_tx_smoke
#include <cstdlib>

#include <holoscan/holoscan.hpp>

#include "operators/common/dpdk_eal.hpp"
#include "operators/st2110_rx/st2110_rx.hpp"
#include "operators/st2110_tx/st2110_tx.hpp"

namespace spark {

class St2110Passthrough : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto env = [](const char* k, const char* d) {
      const char* v = std::getenv(k);
      return std::string(v ? v : d);
    };
    const std::string profile = env("SPARK_PROFILE", "1080p");
    const int64_t frames = std::atoll(env("SPARK_FRAMES", "300").c_str());
    const std::string rx_pci = env("SPARK_RX_PCI", "0000:01:00.1");           // port B (in)
    const std::string tx_pci = env("SPARK_TX_PCI", "0002:01:00.0");           // port C (out)
    const std::string dst_mac = env("SPARK_DST_MAC", "00:00:5e:00:53:30");    // port D (0002:01:00.1)

    auto& eal = spark::net::DpdkEal::instance();
    eal.add_device(tx_pci, "tx_pp=500");
    eal.add_device(rx_pci, "");
    eal.init("0-11", "spark_pass");

    // RX in source mode: each compute() reassembles one frame and emits it.
    // RX in source mode: a dedicated thread drains the NIC continuously into a frame queue, so the
    // ring stays drained while TX paces the prior frame (no HW drops). Default rxd suffices.
    auto rx = make_operator<ops::St2110RxOp>(
        "st2110_rx", Arg("pci_addr", rx_pci), Arg("profile", profile), Arg("manage_eal", false),
        Arg("emit_frames", true), make_condition<CountCondition>(frames));
    // TX consumes each frame and paces it out (no warmup: it only fires once RX delivers a frame).
    auto tx = make_operator<ops::St2110TxOp>("st2110_tx", Arg("pci_addr", tx_pci),
                                             Arg("dst_mac", dst_mac), Arg("manage_eal", false));
    add_flow(rx, tx);
  }
};

}  // namespace spark

int main() {
  HOLOSCAN_LOG_INFO("ST 2110 pass-through: st2110_rx -> st2110_tx (shared EAL, frame forwarding).");
  auto app = holoscan::make_application<spark::St2110Passthrough>();
  // Multi-thread: RX assembles frame N+1 while TX paces frame N (pipelined at the media rate).
  app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
      "mts", holoscan::Arg("worker_thread_number", static_cast<int64_t>(4)),
      holoscan::Arg("stop_on_deadlock", true),
      holoscan::Arg("stop_on_deadlock_timeout", static_cast<int64_t>(3000)),
      holoscan::Arg("max_duration_ms", static_cast<int64_t>(120000))));
  app->run();
  return 0;
}
