// ST 2110 TX bring-up app: test_pattern -> st2110_tx (tx_pp paced).
//
// This is the gate-4 follow-on the spike pointed to — a *real* media generator (one packet per
// ST 2110-21 slot, self-throttled to the tx_pp window) instead of testpmd's open-loop txonly. Run on
// the box, as root, with the CX-7 cabled (see docs/M1-gate4-pacing.md):
//
//   T=~/holoscan-sdk/install-cu13-aarch64-dgpu
//   sudo -E LD_LIBRARY_PATH="$T/lib" ./engine/build/st2110_tx_smoke
//
// Defaults target the gate-4 loopback pair (TX 0002:01:00.0 -> RX MAC 00:00:5e:00:53:30). On the RX
// port, count packets with the spike's rxonly testpmd (or, later, st2110_rx). Success = sustained pps
// == target with future_err/past_err ~0 and tx_pp jitter in the tens of ns — the proof testpmd could
// not give.
#include <holoscan/holoscan.hpp>

#include "operators/st2110_tx/st2110_tx.hpp"
#include "operators/test_pattern/test_pattern.hpp"

namespace spark {

class St2110TxSmoke : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto src = make_operator<ops::TestPatternOp>(
        "test_pattern", Arg("profile", std::string("1080p")),
        make_condition<CountCondition>(300));  // ~5 s at 59.94 fps
    auto tx = make_operator<ops::St2110TxOp>("st2110_tx");
    add_flow(src, tx);
  }
};

}  // namespace spark

int main() {
  HOLOSCAN_LOG_INFO("ST 2110 TX smoke: test_pattern -> st2110_tx (tx_pp paced).");
  auto app = holoscan::make_application<spark::St2110TxSmoke>();
  app->run();
  return 0;
}
