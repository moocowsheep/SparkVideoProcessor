// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// AudioBridge — in-process handoff of received ST 2110-30 packets from the video RX op's poll
// thread (which drains the shared DPDK queue and classifies audio by flow) to the TX op's audio
// relay thread (which re-times each packet to capture_ts + L and submits it tx_pp-paced).
//
// A process-wide singleton, same precedent as DpdkEal::instance(): the RX and TX operators are
// composed independently by the app and share no object, but there is exactly one audio path per
// engine process. Bounded; push drops (counted) rather than blocks — the RX poll thread must never
// stall on a slow consumer.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace spark::st2110 {

class AudioBridge {
 public:
  struct Pkt {
    std::vector<uint8_t> rtp;  // full RTP packet (header + PCM payload), verbatim from the wire
    uint64_t capture_ns = 0;   // absolute capture time (RTP ts unwrapped against the PHC arrival)
    uint32_t ts_delta = 0;     // mod-2^32 correction latched by RX when the sender's stamp epoch
                               // is broken (capture_ns already includes it); TX must add it to the
                               // outgoing RTP ts so the emitted stamp matches the send schedule
  };

  static AudioBridge& instance() {
    static AudioBridge b;
    return b;
  }

  bool push(Pkt&& p) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (q_.size() >= kCap) {
        ++dropped_;
        return false;
      }
      q_.push_back(std::move(p));
      ++pushed_;
    }
    cv_.notify_one();
    return true;
  }

  bool pop(Pkt& out, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!cv_.wait_for(lk, timeout, [&] { return !q_.empty(); })) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
  }

  uint64_t pushed() const {
    std::lock_guard<std::mutex> lk(mu_);
    return pushed_;
  }
  uint64_t dropped() const {
    std::lock_guard<std::mutex> lk(mu_);
    return dropped_;
  }

 private:
  // Must hold the full L + audio-delay backlog: audio_loop holds each packet until
  // capture + L + Da, so a class-A 125 us-ptime stream (8000 pkt/s) at the panel's 1000 ms max
  // delay plus L~105 ms queues ~8840 packets here. 16384 covers ~2 s of 125 us ptime (worst case
  // ~25 MB at max RTP size, and a deque only allocates what the backlog actually uses).
  static constexpr size_t kCap = 16384;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Pkt> q_;
  uint64_t pushed_ = 0;
  uint64_t dropped_ = 0;
};

}  // namespace spark::st2110
