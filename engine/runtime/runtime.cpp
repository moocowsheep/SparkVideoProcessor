// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// The spark::rt executor: thread per operator, bounded blocking queue per edge.
#include "runtime/runtime.hpp"

#include <csignal>
#include <deque>
#include <thread>

namespace spark::rt {
namespace {

// SIGINT/SIGTERM handling. A live video pipeline has no natural end, so Ctrl-C is the normal way it
// stops; the handler only sets a flag, and the executor polls it (async-signal-safe).
std::atomic<bool>* g_stop_flag = nullptr;

void on_signal(int) {
  if (g_stop_flag) g_stop_flag->store(true, std::memory_order_relaxed);
}

}  // namespace

class Executor {
 public:
  explicit Executor(Application& app) : app_(app) {}

  void run() {
    build();
    start_threads();
    wait();
    shutdown();
  }

 private:
  struct Node {
    std::shared_ptr<Operator> op;
    EdgeQueue* input = nullptr;              // null = source (nothing upstream)
    std::vector<EdgeQueue*> outputs;         // empty = sink (nothing downstream)
    const char* in_port = nullptr;
    const char* out_port = nullptr;
    std::string in_port_name, out_port_name;
  };

  void build() {
    // setup() -> apply Args -> initialize(), in that order: setup() declares the parameters that
    // the Args then fill, and initialize() may read them.
    for (auto& op : app_.operators_) {
      op->setup(op->spec_);
      op->spec_.apply_args(op->args_, op->name());
      op->initialize();
    }

    // One queue per edge, sized to the larger of the two declared capacities so both the producer's
    // and the consumer's stated needs are met (FrcOp sizes its output to its emit burst; the filter
    // operators size their input).
    for (const auto& e : app_.edges_) {
      const size_t out_cap = port_capacity(e.from->spec_.outputs());
      const size_t in_cap = port_capacity(e.to->spec_.inputs());
      queues_.push_back(std::make_unique<EdgeQueue>(out_cap > in_cap ? out_cap : in_cap));
      node_for(e.from).outputs.push_back(queues_.back().get());
      Node& to = node_for(e.to);
      if (to.input)
        throw std::runtime_error(e.to->name() +
                                 ": spark::rt is a linear-pipeline runtime — an operator may have "
                                 "at most one upstream");
      to.input = queues_.back().get();
    }

    for (auto& op : app_.operators_) {
      Node& n = node_for(op);
      if (!op->spec_.inputs().empty()) {
        n.in_port_name = op->spec_.inputs().front().name();
        n.in_port = n.in_port_name.c_str();
      }
      if (!op->spec_.outputs().empty()) {
        n.out_port_name = op->spec_.outputs().front().name();
        n.out_port = n.out_port_name.c_str();
      }
    }
  }

  static size_t port_capacity(const std::vector<IOSpec>& ports) {
    return ports.empty() ? 1 : ports.front().capacity();
  }

  Node& node_for(const std::shared_ptr<Operator>& op) {
    for (auto& n : nodes_)
      if (n.op == op) return n;
    nodes_.push_back(Node{op, nullptr, {}, nullptr, nullptr, {}, {}});
    return nodes_.back();
  }

  void start_threads() {
    g_stop_flag = &stop_;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    for (auto& n : nodes_) threads_.emplace_back([this, &n] { run_node(n); });
  }

  // One operator's whole life: start() -> compute() until end-of-stream/count/stop -> stop().
  void run_node(Node& n) {
    try {
      n.op->start();
    } catch (const std::exception& e) {
      SPARK_LOG_ERROR("{}: start() failed: {}", n.op->name(), e.what());
      stop_.store(true, std::memory_order_relaxed);
      for (auto* q : n.outputs) q->close();
      return;
    }

    ExecutionContext ctx;
    uint64_t ticks = 0;
    while (!stop_.load(std::memory_order_relaxed)) {
      InputContext in;
      if (n.input) {
        auto msg = n.input->pop();
        if (!msg) break;  // upstream closed and drained, or stopping
        in.port_ = n.in_port;
        in.has_message_ = true;
        in.message_ = std::move(*msg);
      }
      OutputContext out;
      out.port_ = n.out_port;
      out.queues_ = n.outputs;
      try {
        n.op->compute(in, out, ctx);
      } catch (const std::exception& e) {
        SPARK_LOG_ERROR("{}: compute() threw: {} — stopping the pipeline", n.op->name(), e.what());
        stop_.store(true, std::memory_order_relaxed);
        break;
      }
      if (n.op->count_ && ++ticks >= n.op->count_) break;
    }

    // Closing (not stopping) the downstream queues lets the rest of the chain drain what is still
    // in flight and exit in order — the clean end of a bounded run.
    for (auto* q : n.outputs) q->close();
    try {
      n.op->stop();
    } catch (const std::exception& e) {
      SPARK_LOG_ERROR("{}: stop() threw: {}", n.op->name(), e.what());
    }
    done_.fetch_add(1, std::memory_order_relaxed);
  }

  void wait() {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(app_.max_duration_ms_ > 0 ? app_.max_duration_ms_
                                                                             : 0);
    const bool bounded = app_.max_duration_ms_ > 0;
    while (done_.load(std::memory_order_relaxed) < nodes_.size()) {
      if (stop_.load(std::memory_order_relaxed)) break;
      if (bounded && std::chrono::steady_clock::now() >= deadline) {
        SPARK_LOG_INFO("max_duration_ms reached — stopping");
        stop_.store(true, std::memory_order_relaxed);
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

  void shutdown() {
    stop_.store(true, std::memory_order_relaxed);
    for (auto& q : queues_) q->stop();
    for (auto& t : threads_)
      if (t.joinable()) t.join();
    g_stop_flag = nullptr;
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);
  }

  Application& app_;
  // deque, not vector: build() and the operator threads hold Node& across insertions.
  std::deque<Node> nodes_;
  std::vector<std::unique_ptr<EdgeQueue>> queues_;
  std::vector<std::thread> threads_;
  std::atomic<bool> stop_{false};
  std::atomic<size_t> done_{0};
};

void Application::run() {
  compose();
  if (operators_.empty()) {
    SPARK_LOG_WARN("application composed no operators — nothing to run");
    return;
  }
  Executor exec(*this);
  exec.run();
}

}  // namespace spark::rt
