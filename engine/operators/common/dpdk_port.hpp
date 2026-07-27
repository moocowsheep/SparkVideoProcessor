// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Per-port ownership: lets the ST 2110 RX and TX backends share ONE physical NIC port.
//
// DPDK wants a port's whole shape — queue counts and offloads — declared in a single
// rte_eth_dev_configure() before rte_eth_dev_start(), and a started port refuses to be
// reconfigured. The backends init independently, each from its own Holoscan operator's start(), so
// neither of them can own that call once both sit on the same BDF: whichever ran second would take
// -EBUSY off configure and die. Hence a third party owning the port lifecycle.
//
// The app declares the plan up front — DpdkEal::add_device() tags each BDF with the role that will
// attach to it — and the backends then attach with what they need. The port is configured and
// started exactly ONCE, when the last planned role has attached; each backend's port-dependent work
// (queue indices, flow rules, IGMP joins, the PHC clock) runs from the on_started callback it hands
// the broker, not inline in its init().
//
// A port with a single planned role — every config that predates this, plus the standalone smokes,
// which never touch DpdkEal at all — is brought up inside its one attach(), so the callback fires
// synchronously and the sequence is exactly what each backend used to do for itself.
//
// Why sharing is safe at all: the RX side asks for flow isolation, so the NIC delivers only
// explicitly created flows to DPDK and the kernel netdev on the same port function keeps SSH, ARP
// and PTP. Isolation is ingress-only — it does not touch egress, so the TX role is unaffected.
#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>

struct rte_mempool;  // forward-declared: no rte_* header leaks out of this one

namespace spark::net {

enum class PortRole { kRx, kTx };

constexpr uint16_t kNoQueue = 0xFFFF;  // "this role has no queue of that kind"
constexpr uint16_t kNoPort = 0xFFFF;

// Descriptor counts for the two small side queues. The IGMP queue carries a membership report every
// 30 s; the audio queue ~1000 pps of 2110-30 (512 descriptors is ~half a second in flight).
constexpr uint16_t kIgmpTxd = 64;
constexpr uint16_t kAudioTxd = 512;

// The queue map of one port. Computed at bring-up, when every planned role's request is known.
struct PortLayout {
  uint16_t nb_rxq = 0, nb_txq = 0;
  uint16_t rx_queue = kNoQueue;        // RX role: the media queue (video + companion audio)
  uint16_t rx_igmp_queue = kNoQueue;   // RX role: TX queue for its own membership reports
  uint16_t tx_video_queue = kNoQueue;  // TX role: the paced media queue
  uint16_t tx_audio_queue = kNoQueue;  // TX role: companion ST 2110-30 egress
};

// Queue indices are handed out in a fixed order — TX video, TX audio, RX IGMP — so the solo cases
// reproduce exactly the layout each backend used to pick for itself (TX: 0 video, 1 audio; RX:
// 0 IGMP) and a shared port simply appends: 0 video, 1 audio, 2 IGMP. Pure, so the off-by-one risk
// here is unit-testable without a NIC (see test/test_dpdk_port.cpp).
constexpr PortLayout plan_queues(bool has_rx, bool has_tx, bool want_igmp, bool want_audio) {
  PortLayout l;
  if (has_rx) {
    l.rx_queue = 0;
    l.nb_rxq = 1;
  }
  uint16_t q = 0;
  if (has_tx) {
    l.tx_video_queue = q++;
    if (want_audio) l.tx_audio_queue = q++;
  }
  if (has_rx && want_igmp) l.rx_igmp_queue = q++;
  l.nb_txq = q;
  return l;
}

// What one role needs from the port. The broker merges the planned roles' requests into the single
// rte_eth_dev_configure() that brings the port up.
struct PortRequest {
  PortRole role = PortRole::kRx;
  uint16_t rxd = 0;                // RX ring depth (kRx)
  uint16_t txd = 0;                // ring depth of the paced media queue (kTx)
  bool rx_timestamp = false;       // RX HW timestamp offload — the ingest-latency basis (kRx)
  bool send_on_timestamp = false;  // tx_pp send scheduling; fatal if the NIC lacks it (kTx)
  bool isolate = false;            // rte_flow_isolate() before configure (kRx)
  bool igmp_tx = false;            // kRx also needs a TX queue, for IGMP membership reports
  bool audio_tx = false;           // kTx also needs a queue for the companion 2110-30 egress
  rte_mempool* rx_pool = nullptr;  // pool backing the RX queue (kRx; required)
};

// What a role got, delivered to its on_started callback once the port is live.
struct PortLease {
  uint16_t port_id = kNoPort;
  uint16_t rx_queue = kNoQueue;
  uint16_t tx_queue = kNoQueue;        // kTx: the media queue. kRx: the IGMP queue.
  uint16_t audio_tx_queue = kNoQueue;  // kTx only
  uint16_t rxd = 0, txd = 0;           // effective depths after rte_eth_dev_adjust_nb_rx_tx_desc
  bool audio_ok = false;               // the companion audio queue came up (kTx)
  bool isolated = false;               // rte_flow_isolate() succeeded
  bool shared = false;                 // the other role shares this port
};

class DpdkPorts {
 public:
  using StartedFn = std::function<void(const PortLease&)>;

  static DpdkPorts& instance();

  // App-side plan, one call per backend that will attach (made for you by DpdkEal::add_device).
  // Two different roles on one BDF = a shared port. Must precede attach().
  void plan(const std::string& pci, PortRole role);

  // Backend-side. Declares `req` and registers `on_started`, which runs — with the final lease —
  // the moment the port is live: inside this call when this was the last planned role to arrive,
  // otherwise from the peer's attach(). Throws std::runtime_error if the port cannot be brought up.
  void attach(const std::string& pci, const PortRequest& req, StartedFn on_started);

  // Drop a role's claim. The port is stopped and closed when the last holder lets go.
  void release(const std::string& pci, PortRole role);

 private:
  DpdkPorts() = default;

  struct PortState {
    std::string pci;
    uint16_t port_id = kNoPort;
    bool planned[2] = {false, false};   // indexed by role: 0 = kRx, 1 = kTx
    bool attached[2] = {false, false};
    bool held[2] = {false, false};      // still using the port (cleared by release())
    PortRequest req[2];
    StartedFn cb[2];
    PortRole order[2] = {PortRole::kRx, PortRole::kRx};  // attach order; callbacks fire in it
    int norder = 0;
    bool up = false;
    bool isolated = false;
  };

  PortState& state_for(const std::string& pci);
  PortState* find(const std::string& pci);
  void bring_up(PortState& p);

  mutable std::mutex mu_;
  // deque, not vector: attach() holds a reference across bring_up(), so the elements must not move.
  std::deque<PortState> ports_;
};

}  // namespace spark::net
