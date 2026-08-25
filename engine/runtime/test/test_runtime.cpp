// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// spark::rt runtime unit test — the dataflow guarantees the video operators rely on. Pure host
// code, no CUDA, no NIC, no root.
//
// What is checked, and why each matters to the pipeline:
//   1. end-to-end delivery + CountCondition          — a bounded run ends by itself and loses nothing
//   2. multi-emit per compute (the FrcOp burst)      — 2:1 up-conversion must not drop the second frame
//   3. backpressure through a capacity-1 queue       — a slow consumer throttles the producer instead
//                                                      of overflowing (what kDownstreamMessageAffordable
//                                                      approximated under Holoscan)
//   4. Args override parameter defaults
//   5. an unconnected output port is a safe no-op    — st2110_rx in sink mode declares one
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "runtime/runtime.hpp"

using namespace spark::rt;

namespace {

std::atomic<int> g_sink_count{0};
std::atomic<long long> g_sink_sum{0};
std::atomic<int> g_max_in_flight{0};
std::atomic<int> g_in_flight{0};

// Emits `start + i` on each tick.
class Source : public Operator {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output<int>("out");
    spec.param(start_, "start", "Start", "first value emitted", 0);
  }
  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    const int v = start_.get() + n_++;
    g_in_flight.fetch_add(1);
    op_output.emit<int>(v, "out");
  }

 private:
  Parameter<int> start_;
  int n_ = 0;
};

// The FrcOp shape: one input, two outputs per compute.
class Doubler : public Operator {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out").condition(ConditionType::kDownstreamMessageAffordable,
                                      Arg("min_size", uint64_t(2)));
  }
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto v = op_input.receive<int>("in");
    if (!v) return;
    op_output.emit<int>(*v, "out");
    op_output.emit<int>(*v, "out");
  }
};

// Deliberately slow, behind a capacity-1 queue: the producer must block rather than run ahead.
class SlowSink : public Operator {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input<int>("in").connector(IOSpec::ConnectorType::kDoubleBuffer,
                                    Arg("capacity", uint64_t(1)));
  }
  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto v = op_input.receive<int>("in");
    if (!v) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const int now = g_in_flight.fetch_sub(1) - 1;
    int prev = g_max_in_flight.load();
    while (now + 1 > prev && !g_max_in_flight.compare_exchange_weak(prev, now + 1)) {
    }
    g_sink_sum.fetch_add(*v);
    g_sink_count.fetch_add(1);
  }
};

// Declares an output nobody consumes (st2110_rx sink mode) — emitting must not hang or crash.
class Orphan : public Operator {
 public:
  void setup(OperatorSpec& spec) override { spec.output<int>("frame"); }
  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    op_output.emit<int>(1, "frame");
    ++ticks;
  }
  int ticks = 0;
};

class ChainApp : public Application {
 public:
  void compose() override {
    auto src = make_operator<Source>("src", Arg("start", 1), make_condition<CountCondition>(10));
    auto dbl = make_operator<Doubler>("dbl");
    auto sink = make_operator<SlowSink>("sink");
    add_flow(src, dbl);
    add_flow(dbl, sink);
  }
};

class OrphanApp : public Application {
 public:
  std::shared_ptr<Orphan> orphan;
  void compose() override {
    orphan = make_operator<Orphan>("orphan", make_condition<CountCondition>(5));
  }
};

int failures = 0;
void check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

}  // namespace

int main() {
  std::printf("spark::rt runtime\n");

  {
    auto app = make_application<ChainApp>();
    app->run();
    // 10 source ticks, each doubled: 20 messages, sum = 2 * (1+...+10) = 110.
    check(g_sink_count.load() == 20, "CountCondition(10) x 2 emits => 20 messages delivered");
    check(g_sink_sum.load() == 110, "every message arrives intact (sum == 110)");
    // capacity-1 edge into a 2 ms sink: the producer cannot get more than a couple of frames ahead.
    check(g_max_in_flight.load() <= 4, "bounded queue backpressures the producer");
    std::printf("  (peak in flight: %d)\n", g_max_in_flight.load());
  }

  {
    auto app = make_application<OrphanApp>();
    app->run();
    check(app->orphan->ticks == 5, "operator with an unconnected output still ticks, emit is a no-op");
  }

  std::printf(failures ? "[FAIL] spark::rt\n" : "[PASS] spark::rt\n");
  return failures ? 1 : 0;
}
