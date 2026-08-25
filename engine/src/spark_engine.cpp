// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Spark Video Processor — engine entry point.
//
// Milestone status: M1 skeleton. This currently runs a PLACEHOLDER ping graph to validate the
// spark::rt runtime + our build chain (it used holoscan::ops::ping_tx/ping_rx for the same purpose
// before the SDK was dropped; the ping operators are local now, so nothing outside the engine is
// needed to prove the graph executes). The real low-latency pipeline replaces the placeholder:
//
//     st2110_rx -> unpack -> resize (NPP) -> frc (Optical Flow/FRUC) -> pack -> st2110_tx
//
// Each stage is a spark::rt operator passing zero-copy GPU frames; the graph is assembled by a
// config-driven pipeline builder (engine/pipelines/) so modules can be added without engine surgery.
#include <cstdint>

#include "runtime/runtime.hpp"

namespace spark {
namespace ops {

// Minimal ping pair — the runtime's own build/execution self-test (emit N ints, count them).
class PingTxOp : public spark::rt::Operator {
 public:
  SPARK_OPERATOR_FORWARD_ARGS(PingTxOp)
  void setup(spark::rt::OperatorSpec& spec) override { spec.output<int64_t>("out"); }
  void compute(spark::rt::InputContext&, spark::rt::OutputContext& op_output,
               spark::rt::ExecutionContext&) override {
    op_output.emit<int64_t>(++n_, "out");
  }

 private:
  int64_t n_ = 0;
};

class PingRxOp : public spark::rt::Operator {
 public:
  SPARK_OPERATOR_FORWARD_ARGS(PingRxOp)
  void setup(spark::rt::OperatorSpec& spec) override { spec.input<int64_t>("in"); }
  void compute(spark::rt::InputContext& op_input, spark::rt::OutputContext&,
               spark::rt::ExecutionContext&) override {
    if (auto v = op_input.receive<int64_t>("in")) SPARK_LOG_INFO("Rx message value: {}", *v);
  }
};

}  // namespace ops

class SparkEngine : public spark::rt::Application {
 public:
  void compose() override {
    using namespace spark::rt;

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

}  // namespace spark

int main() {
  SPARK_LOG_INFO("Spark engine starting — placeholder ping graph (M1 skeleton, spark::rt).");
  auto app = spark::rt::make_application<spark::SparkEngine>();
  app->run();
  return 0;
}
