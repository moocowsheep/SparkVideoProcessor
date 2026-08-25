// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// spark::rt — the engine's dataflow runtime (the Holoscan SDK replacement).
//
// WHY THIS EXISTS. The engine used Holoscan for exactly one thing: an operator graph. It never used
// Holoscan's tensors, allocators, HoloInfer, Holoviz, or advanced_network — networking is raw
// DPDK/mlx5, resize is NPP, FRC is NVOF, super-resolution is hand-written CUDA. In exchange for the
// graph it cost an hour-long containerized SDK build, a CMake >= 3.30.4 floor, a CUDA-13 coupling
// through GXF, and a ~GB install tree. Every graph the engine builds is a STRICT LINEAR CHAIN
// (rx -> unpack -> [filters] -> pack -> tx), single input and single output per operator, no
// fan-out, no fan-in, no cycles. This is that pipeline, and nothing more.
//
// EXECUTION MODEL: one thread per operator, bounded blocking queue per edge. This is why it can be
// both smaller and lower-latency than what it replaces:
//   * no scheduler poll. Holoscan's MultiThreadScheduler slept check_recession_period_ms (5 ms by
//     default) whenever a pass found nothing ready, up to 5 ms of pure dispatch latency PER HOP —
//     the largest avoidable latency term across this graph's 5 hops (see docs/M8-latency.md). A
//     blocked thread wakes on the condvar the instant a frame is pushed.
//   * backpressure IS the bounded queue. A full queue blocks the producer, which is what
//     ConditionType::kDownstreamMessageAffordable was configured to approximate; FrcOp's 2-frames-
//     per-compute burst simply blocks between emits instead of overflowing a transmitter.
//   * "message available" IS the blocking pop, replacing kMessageAvailable.
// Because the graph is acyclic and linear, blocking-on-full cannot deadlock: the downstream end is
// always draining or shutting down.
//
// The API deliberately mirrors the Holoscan shapes the operators were written against (setup/
// start/compute/stop, spec.param, spec.input/output, receive/emit, Parameter<T>, Arg), so the
// operator bodies — the code that carries the validated video behavior — port by namespace swap
// rather than by rewrite.
#pragma once

#include <any>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "runtime/log.hpp"

namespace spark::rt {

// ---------------------------------------------------------------------------------------------
// Arg — a named scalar handed to an operator at construction, or to a connector/condition.
// ---------------------------------------------------------------------------------------------
// A tagged scalar rather than std::variant<...>: Args are a startup-only concern, and GCC 13
// mis-analyses the inlined destructor of a variant holding an alternative it cannot prove
// (-Wfree-nonheap-object on every setup() that builds one). This is smaller and warning-clean.
class ArgValue {
 public:
  enum class Kind { kBool, kInt, kUint, kDouble, kString };

  ArgValue() = default;
  explicit ArgValue(bool v) : kind_(Kind::kBool), i_(v ? 1 : 0) {}
  explicit ArgValue(int64_t v) : kind_(Kind::kInt), i_(v) {}
  explicit ArgValue(uint64_t v) : kind_(Kind::kUint), u_(v) {}
  explicit ArgValue(double v) : kind_(Kind::kDouble), d_(v) {}
  explicit ArgValue(std::string v) : kind_(Kind::kString), s_(std::move(v)) {}

  Kind kind() const { return kind_; }
  bool is_string() const { return kind_ == Kind::kString; }
  const std::string& string_value() const { return s_; }

  // Numeric value as T, whatever arithmetic alternative is held.
  template <class T>
  T as() const {
    switch (kind_) {
      case Kind::kBool:
      case Kind::kInt:
        return static_cast<T>(i_);
      case Kind::kUint:
        return static_cast<T>(u_);
      case Kind::kDouble:
        return static_cast<T>(d_);
      case Kind::kString:
        break;
    }
    return T{};
  }

 private:
  Kind kind_ = Kind::kInt;
  int64_t i_ = 0;
  uint64_t u_ = 0;
  double d_ = 0.0;
  std::string s_;
};

class Arg {
 public:
  Arg(std::string name, bool v) : name_(std::move(name)), value_(v) {}
  Arg(std::string name, double v) : name_(std::move(name)), value_(v) {}
  Arg(std::string name, float v) : name_(std::move(name)), value_(static_cast<double>(v)) {}
  Arg(std::string name, std::string v) : name_(std::move(name)), value_(std::move(v)) {}
  Arg(std::string name, const char* v) : name_(std::move(name)), value_(std::string(v)) {}
  template <class T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int> = 0>
  Arg(std::string name, T v)
      : name_(std::move(name)),
        value_(std::is_signed_v<T> ? ArgValue(static_cast<int64_t>(v))
                                   : ArgValue(static_cast<uint64_t>(v))) {}

  const std::string& name() const { return name_; }
  const ArgValue& value() const { return value_; }

 private:
  std::string name_;
  ArgValue value_;
};

namespace detail {

// Arithmetic Args convert across types (an Arg("capacity", 4u) fills a Parameter<uint64_t>);
// strings only fill strings. Anything else is a wiring bug and throws at startup, not mid-frame.
template <class T>
bool assign_arg(T& dst, const ArgValue& v) {
  if constexpr (std::is_same_v<T, std::string>) {
    if (!v.is_string()) return false;
    dst = v.string_value();
    return true;
  } else if constexpr (std::is_arithmetic_v<T>) {
    if (v.is_string()) return false;
    dst = v.template as<T>();
    return true;
  } else {
    (void)dst;
    (void)v;
    return false;
  }
}

}  // namespace detail

// ---------------------------------------------------------------------------------------------
// Parameter<T> — declared in setup() via spec.param(), filled from Args at initialize().
// ---------------------------------------------------------------------------------------------
template <class T>
class Parameter {
 public:
  Parameter() = default;
  const T& get() const { return value_; }
  operator const T&() const { return value_; }  // NOLINT(google-explicit-constructor)

  void set_default(T v) { value_ = std::move(v); }
  bool set_from(const ArgValue& v) { return detail::assign_arg(value_, v); }

 private:
  T value_{};
};

// ---------------------------------------------------------------------------------------------
// Port specification. connector()/condition() keep the Holoscan call shape; capacity is the one
// setting with teeth here (it sizes the edge queue, i.e. the latency/backpressure knob).
// ---------------------------------------------------------------------------------------------
enum class ConditionType { kNone, kMessageAvailable, kDownstreamMessageAffordable };

class IOSpec {
 public:
  enum class ConnectorType { kDefault, kDoubleBuffer };

  // Variadic to match the Holoscan call shape: .connector(kDoubleBuffer, Arg("capacity", n), ...)
  template <class... Args>
  IOSpec& connector(ConnectorType type, const Args&... args) {
    (void)type;  // kDoubleBuffer vs kDefault: both are a bounded queue here.
    (apply_connector_arg(args), ...);
    if (capacity_ == 0) capacity_ = 1;
    return *this;
  }
  template <class... Args>
  IOSpec& condition(ConditionType type, const Args&... args) {
    condition_ = type;
    (apply_condition_arg(args), ...);
    return *this;
  }

  const std::string& name() const { return name_; }
  size_t capacity() const { return capacity_; }
  size_t min_size() const { return min_size_; }
  ConditionType condition_type() const { return condition_; }

 private:
  friend class OperatorSpec;
  void apply_connector_arg(const Arg& arg) {
    if (arg.name() == "capacity") detail::assign_arg(capacity_, arg.value());
    // "policy" (2 = fault-on-overflow) has no analogue: a bounded blocking queue never overflows,
    // it backpressures, so there is no drop/fault decision left to configure.
  }
  void apply_condition_arg(const Arg& arg) {
    if (arg.name() == "min_size") detail::assign_arg(min_size_, arg.value());
  }

  std::string name_;
  size_t capacity_ = 1;
  size_t min_size_ = 1;
  ConditionType condition_ = ConditionType::kNone;
};

class OperatorSpec {
 public:
  template <class T>
  IOSpec& input(const std::string& name) {
    return add_port(inputs_, name);
  }
  template <class T>
  IOSpec& output(const std::string& name) {
    return add_port(outputs_, name);
  }

  // spec.param(member, name[, label[, description[, default]]]). Label and description are kept for
  // call-site compatibility and self-documentation; nothing introspects them (the web UI's filter
  // catalog is hand-authored JSON in control/daemon.cpp, not generated from these).
  template <class T>
  void param(Parameter<T>& p, const char* name) {
    bind(p, name);
  }
  template <class T>
  void param(Parameter<T>& p, const char* name, const char*) {
    bind(p, name);
  }
  template <class T>
  void param(Parameter<T>& p, const char* name, const char*, const char*) {
    bind(p, name);
  }
  template <class T, class D>
  void param(Parameter<T>& p, const char* name, const char*, const char*, D&& default_value) {
    p.set_default(static_cast<T>(std::forward<D>(default_value)));
    bind(p, name);
  }

  const std::vector<IOSpec>& inputs() const { return inputs_; }
  const std::vector<IOSpec>& outputs() const { return outputs_; }

  // Applied by the runtime once all Args are known.
  void apply_args(const std::vector<Arg>& args, const std::string& op_name) const {
    for (const auto& arg : args) {
      auto it = setters_.find(arg.name());
      if (it == setters_.end()) {
        SPARK_LOG_WARN("{}: unknown parameter '{}' — ignored", op_name, arg.name());
        continue;
      }
      if (!it->second(arg.value()))
        throw std::runtime_error(op_name + ": parameter '" + arg.name() + "' has the wrong type");
    }
  }

 private:
  template <class T>
  void bind(Parameter<T>& p, const char* name) {
    setters_[name] = [&p](const ArgValue& v) { return p.set_from(v); };
  }
  static IOSpec& add_port(std::vector<IOSpec>& ports, const std::string& name) {
    for (auto& p : ports)
      if (p.name_ == name) return p;
    ports.emplace_back();
    ports.back().name_ = name;
    return ports.back();
  }

  std::vector<IOSpec> inputs_;
  std::vector<IOSpec> outputs_;
  mutable std::unordered_map<std::string, std::function<bool(const ArgValue&)>> setters_;
};


// ---------------------------------------------------------------------------------------------
// Edge queue — bounded, blocking both ways. This single class provides what Holoscan spread across
// the double-buffer transmitter/receiver, kMessageAvailable and kDownstreamMessageAffordable.
// ---------------------------------------------------------------------------------------------
class EdgeQueue {
 public:
  explicit EdgeQueue(size_t capacity) : capacity_(capacity ? capacity : 1) {}

  // Blocks while full. Returns false if the pipeline is stopping (message dropped, by design:
  // shutdown must not hang on a downstream that has already exited).
  bool push(std::any msg) {
    std::unique_lock<std::mutex> lk(m_);
    not_full_.wait(lk, [this] { return q_.size() < capacity_ || stopping_; });
    if (stopping_) return false;
    q_.push_back(std::move(msg));
    lk.unlock();
    not_empty_.notify_one();
    return true;
  }

  // Blocks while empty. Returns nullopt on end-of-stream or stop.
  std::optional<std::any> pop() {
    std::unique_lock<std::mutex> lk(m_);
    not_empty_.wait(lk, [this] { return !q_.empty() || closed_ || stopping_; });
    if (q_.empty()) return std::nullopt;  // closed or stopping, and drained
    std::any msg = std::move(q_.front());
    q_.pop_front();
    lk.unlock();
    not_full_.notify_one();
    return msg;
  }

  // Upstream is done: let the consumer drain what is queued, then see end-of-stream.
  void close() {
    {
      std::lock_guard<std::mutex> lk(m_);
      closed_ = true;
    }
    not_empty_.notify_all();
  }

  // Hard stop: wake everyone, drop what is in flight.
  void stop() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stopping_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

 private:
  const size_t capacity_;
  std::mutex m_;
  std::condition_variable not_full_, not_empty_;
  std::deque<std::any> q_;
  bool closed_ = false;
  bool stopping_ = false;
};

// ---------------------------------------------------------------------------------------------
// compute() contexts.
// ---------------------------------------------------------------------------------------------
class ExecutionContext {};  // no per-execution services are used by any operator; kept for signature parity

class InputContext {
 public:
  // The message for this tick was dequeued by the runtime before compute() was called.
  template <class T>
  std::optional<T> receive(const char* port) {
    if (!has_message_ || (port_ && port && std::string(port_) != port)) return std::nullopt;
    const T* typed = std::any_cast<T>(&message_);
    if (!typed) {
      SPARK_LOG_ERROR("receive('{}'): message type does not match the declared port type", port);
      return std::nullopt;
    }
    return *typed;
  }

 private:
  friend class Executor;
  const char* port_ = nullptr;
  bool has_message_ = false;
  std::any message_;
};

class OutputContext {
 public:
  // Blocks if the downstream queue is full — that block IS the pipeline's backpressure.
  template <class T>
  void emit(T value, const char* port) {
    if (port_ && port && std::string(port_) != port) {
      SPARK_LOG_ERROR("emit('{}'): no such output port", port);
      return;
    }
    if (queues_.empty()) return;  // declared but unconnected (e.g. st2110_rx in sink mode)
    for (size_t i = 0; i + 1 < queues_.size(); ++i) queues_[i]->push(std::any(value));
    queues_.back()->push(std::any(std::move(value)));
  }

 private:
  friend class Executor;
  const char* port_ = nullptr;
  std::vector<EdgeQueue*> queues_;
};

// ---------------------------------------------------------------------------------------------
// Conditions. CountCondition is the only one applied at the operator level (bounded test runs).
// ---------------------------------------------------------------------------------------------
struct CountCondition {
  uint64_t count = 0;
};

// ---------------------------------------------------------------------------------------------
// Operator.
// ---------------------------------------------------------------------------------------------
class Operator {
 public:
  virtual ~Operator() = default;

  virtual void setup(OperatorSpec& spec) { (void)spec; }
  virtual void initialize() {}
  virtual void start() {}
  virtual void compute(InputContext& op_input, OutputContext& op_output,
                       ExecutionContext& context) = 0;
  virtual void stop() {}

  const std::string& name() const { return name_; }

  void add_arg(const Arg& arg) { args_.push_back(arg); }
  void add_arg(const CountCondition& c) { count_ = c.count; }

 private:
  friend class Application;
  friend class Executor;
  std::string name_;
  std::vector<Arg> args_;
  OperatorSpec spec_;
  uint64_t count_ = 0;  // 0 = run until end-of-stream / stop
};

// Holoscan required a forwarding constructor macro so the framework could pass Args through the
// operator's constructor. Here the runtime applies Args after construction, so this is a no-op kept
// only so operator headers read the same.
#define SPARK_OPERATOR_FORWARD_ARGS(cls)

// ---------------------------------------------------------------------------------------------
// Application.
// ---------------------------------------------------------------------------------------------
class Application {
 public:
  virtual ~Application() = default;
  virtual void compose() = 0;

  template <class T, class... Args>
  std::shared_ptr<T> make_operator(const std::string& name, Args&&... args) {
    auto op = std::make_shared<T>();
    op->name_ = name;
    (apply_ctor_arg(*op, std::forward<Args>(args)), ...);
    operators_.push_back(op);
    return op;
  }

  template <class C, class A>
  C make_condition(A&& count) {
    return C{static_cast<uint64_t>(count)};
  }

  void add_flow(const std::shared_ptr<Operator>& from, const std::shared_ptr<Operator>& to) {
    edges_.push_back({from, to});
  }

  // Bound run length in milliseconds; 0 (default) runs until the graph ends or SIGINT.
  void max_duration_ms(int64_t ms) { max_duration_ms_ = ms; }

  void run();

 private:
  friend class Executor;
  struct Edge {
    std::shared_ptr<Operator> from, to;
  };

  template <class Op>
  static void apply_ctor_arg(Op& op, const Arg& a) {
    op.add_arg(a);
  }
  template <class Op>
  static void apply_ctor_arg(Op& op, const CountCondition& c) {
    op.add_arg(c);
  }

  std::vector<std::shared_ptr<Operator>> operators_;
  std::vector<Edge> edges_;
  int64_t max_duration_ms_ = 0;
};

template <class App>
std::shared_ptr<App> make_application() {
  return std::make_shared<App>();
}

}  // namespace spark::rt
