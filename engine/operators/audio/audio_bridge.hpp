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
  // ~4 s of 1 ms-ptime audio: far above any sane latency L, small enough to bound memory.
  static constexpr size_t kCap = 4096;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Pkt> q_;
  uint64_t pushed_ = 0;
  uint64_t dropped_ = 0;
};

}  // namespace spark::st2110
