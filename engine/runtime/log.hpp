// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Minimal brace-style logging — the SPARK_LOG_* replacement for HOLOSCAN_LOG_*.
//
// The engine's 88 log sites use exactly four format specs: {}, {:.1f}, {:.2f}, {:.3f}. That is far
// less than fmt provides, and pulling fmt/spdlog back in would reintroduce a third-party dependency
// this runtime exists to shed — so this implements just those, on top of <sstream>.
//
// Deliberate fmt compatibility: bool prints true/false (not 1/0) so log text is unchanged from the
// Holoscan era. Unmatched '{' is emitted literally rather than throwing: a log line must never take
// down a live video pipeline.
#pragma once

#include <cstdio>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

namespace spark::rt {

enum class LogLevel { kInfo, kWarn, kError };

namespace detail {

inline void append_value(std::ostringstream& os, const std::string& spec, bool v) {
  (void)spec;
  os << (v ? "true" : "false");
}

template <class T>
void append_value(std::ostringstream& os, const std::string& spec, const T& v) {
  // spec is the text between ':' and '}' — only ".<n>f" is understood, anything else prints plain.
  if (spec.size() >= 3 && spec[0] == '.' && spec.back() == 'f') {
    const int prec = std::atoi(spec.c_str() + 1);
    const auto flags = os.flags();
    const auto old_prec = os.precision();
    os << std::fixed << std::setprecision(prec) << v;
    os.flags(flags);
    os.precision(old_prec);
    return;
  }
  os << v;
}

inline void format_into(std::ostringstream& os, const char* fmt) {
  // No arguments left: copy the remainder verbatim (including any stray braces).
  os << fmt;
}

template <class T, class... Rest>
void format_into(std::ostringstream& os, const char* fmt, const T& value, const Rest&... rest) {
  for (const char* p = fmt; *p; ++p) {
    if (*p != '{') {
      os << *p;
      continue;
    }
    const char* close = p + 1;
    while (*close && *close != '}') ++close;
    if (!*close) {  // unterminated '{' — emit literally, nothing left to substitute
      os << p;
      return;
    }
    std::string spec(p + 1, close);
    if (!spec.empty() && spec[0] == ':') spec.erase(0, 1);
    append_value(os, spec, value);
    format_into(os, close + 1, rest...);
    return;
  }
}

inline std::mutex& log_mutex() {
  static std::mutex m;
  return m;
}

template <class... Args>
void log(LogLevel level, const char* fmt, const Args&... args) {
  std::ostringstream os;
  format_into(os, fmt, args...);
  const char* tag = level == LogLevel::kInfo ? "info" : (level == LogLevel::kWarn ? "warn" : "error");
  const std::string line = os.str();
  // One locked fwrite per line: operators log from their own threads, and interleaved partial lines
  // are worse than the lock (which is off the frame path — logging is per-event, not per-packet).
  std::lock_guard<std::mutex> lk(log_mutex());
  std::fprintf(level == LogLevel::kError ? stderr : stdout, "[%s] %s\n", tag, line.c_str());
  std::fflush(level == LogLevel::kError ? stderr : stdout);
}

}  // namespace detail
}  // namespace spark::rt

#define SPARK_LOG_INFO(...) ::spark::rt::detail::log(::spark::rt::LogLevel::kInfo, __VA_ARGS__)
#define SPARK_LOG_WARN(...) ::spark::rt::detail::log(::spark::rt::LogLevel::kWarn, __VA_ARGS__)
#define SPARK_LOG_ERROR(...) ::spark::rt::detail::log(::spark::rt::LogLevel::kError, __VA_ARGS__)
