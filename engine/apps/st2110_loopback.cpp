// One-process ST 2110 loopback: test_pattern -> st2110_tx (port .0) AND st2110_rx (port .1), sharing
// ONE EAL across both CX-7 ports. Self-contained (no second process): TX generates, the switched
// fabric returns it to the RX port, RX measures loss + ingest latency.
//
// Proves the shared-EAL refactor (one rte_eal_init, both ports, backends attach by PCI with
// manage_eal=false) AND clean line-rate co-location of the TX pacer + RX poll in one process:
// at 2160p/12G this gets lost=0, hw_missed=0, ~2 µs steady-state ingest latency — matching the
// two-process runs. Two fixes got there: (1) DpdkEal widens thread affinity after rte_eal_init (else
// Holoscan workers inherit the single main-lcore mask and every spin serializes on one core); (2) a
// one-time TX warmup so RX enters its poll loop before TX floods (a startup race, not steady-state).
// These are the prerequisites for the single-process rx->tx pass-through (frame forwarding next).
//   sudo -n SPARK_PROFILE=2160p ./engine/build/st2110_loopback   # SPARK_PROFILE/SPARK_FRAMES/SPARK_SECONDS
#include <cstdlib>

#include <holoscan/holoscan.hpp>

#include "operators/common/dpdk_eal.hpp"
#include "operators/frame_sink/frame_sink.hpp"
#include "operators/st2110_rx/st2110_rx.hpp"
#include "operators/st2110_tx/st2110_tx.hpp"
#include "operators/test_pattern/test_pattern.hpp"

namespace spark {

class St2110Loopback : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    const char* prof = std::getenv("SPARK_PROFILE");
    const char* fr = std::getenv("SPARK_FRAMES");
    const char* sc = std::getenv("SPARK_SECONDS");
    const std::string profile = prof ? prof : "1080p";
    const int64_t frames = fr ? std::atoll(fr) : 300;
    const double seconds = sc ? std::atof(sc) : 12.0;

    // One EAL for BOTH ports (tx_pp devarg on the TX port). compose() runs before any operator
    // start(), so EAL is up before the backends (manage_eal=false) attach to their ports.
    auto& eal = spark::net::DpdkEal::instance();
    eal.add_device("0002:01:00.0", "tx_pp=500");  // TX
    eal.add_device("0002:01:00.1", "");           // RX
    // Wide lcore set: rte_eal_init pins the process affinity mask to these cores and Holoscan's
    // worker threads inherit it, so the busy-spin TX throttle + busy-poll RX need room (4 cores
    // starved them -> loss + ms-scale latency in one process; two-process runs had the box each).
    eal.init("0-11", "spark_loop");

    // TX branch: synthetic frames -> paced TX out .0, addressed to the RX port (.1) MAC.
    auto src = make_operator<ops::TestPatternOp>("test_pattern", Arg("profile", profile),
                                                 make_condition<CountCondition>(frames));
    auto tx = make_operator<ops::St2110TxOp>(
        "st2110_tx", Arg("pci_addr", std::string("0002:01:00.0")),
        Arg("dst_mac", std::string("30:c5:99:3e:9d:30")), Arg("manage_eal", false),
        Arg("warmup_ms", 500u));  // let st2110_rx enter its poll loop before we flood
    add_flow(src, tx);

    // RX branch: receive on .1, measure loss + zero-copy ingest latency.
    auto rx = make_operator<ops::St2110RxOp>(
        "st2110_rx", Arg("pci_addr", std::string("0002:01:00.1")), Arg("profile", profile),
        Arg("run_seconds", seconds), Arg("manage_eal", false), make_condition<CountCondition>(1));
    auto sink = make_operator<ops::FrameSinkOp>("frame_sink");  // sink-mode rx never emits; wire anyway
    add_flow(rx, sink);
  }
};

}  // namespace spark

int main() {
  HOLOSCAN_LOG_INFO("ST 2110 one-process loopback: test_pattern->st2110_tx(.0) + st2110_rx(.1).");
  auto app = holoscan::make_application<spark::St2110Loopback>();
  // Multi-thread scheduler: st2110_rx blocks in its poll loop, so it needs its own worker thread
  // concurrent with the TX branch (test_pattern -> st2110_tx).
  app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
      "mts", holoscan::Arg("worker_thread_number", static_cast<int64_t>(4)),
      holoscan::Arg("stop_on_deadlock", true),
      holoscan::Arg("max_duration_ms", static_cast<int64_t>(30000))));
  app->run();
  return 0;
}
