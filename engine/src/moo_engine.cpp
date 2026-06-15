// Moo Video Processor — engine entry point.
//
// Milestone status: M1 skeleton. This currently runs a PLACEHOLDER ping graph to validate the
// Holoscan C++ runtime + our build chain on the DGX Spark (GB10 / CUDA 13). The real low-latency
// pipeline replaces the placeholder once the ConnectX-7 + DPDK ST 2110 IO are available (M0 gates 1/2/4):
//
//     st2110_rx -> unpack -> resize (NPP) -> frc (Optical Flow/FRUC) -> pack -> st2110_tx
//
// Each stage is a Holoscan operator passing zero-copy GPU tensors; the graph is assembled by a
// config-driven pipeline builder (engine/pipelines/) so modules can be added without engine surgery.
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>

namespace moo {

class MooEngine : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // --- PLACEHOLDER GRAPH (M1 build/runtime validation) ---
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(10));
    auto rx = make_operator<ops::PingRxOp>("rx");
    add_flow(tx, rx);

    // --- TARGET GRAPH (wired by the config-driven pipeline builder; see engine/operators/) ---
    // auto src   = make_operator<ops::St2110RxOp>("st2110_rx", from_config("rx"));
    // auto unpack= make_operator<ops::UnpackOp>("unpack");
    // auto scale = make_operator<ops::ResizeOp>("resize", from_config("resize"));
    // auto frc   = make_operator<ops::FrcOp>("frc", from_config("frc"));
    // auto pack  = make_operator<ops::PackOp>("pack");
    // auto sink  = make_operator<ops::St2110TxOp>("st2110_tx", from_config("tx"));
    // add_flow(src, unpack); add_flow(unpack, scale); add_flow(scale, frc);
    // add_flow(frc, pack);   add_flow(pack, sink);
  }
};

}  // namespace moo

int main() {
  HOLOSCAN_LOG_INFO("Moo engine starting — placeholder ping graph (M1 skeleton).");
  auto app = holoscan::make_application<moo::MooEngine>();
  app->run();
  return 0;
}
