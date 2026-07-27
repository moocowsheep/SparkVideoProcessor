// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

#include "dpdk_port.hpp"

#include <cstdio>
#include <stdexcept>

#include <rte_dev.h>
#include <rte_ethdev.h>
#include <rte_flow.h>

namespace spark::net {
namespace {

int idx(PortRole r) { return r == PortRole::kRx ? 0 : 1; }
const char* rname(PortRole r) { return r == PortRole::kRx ? "rx" : "tx"; }

[[noreturn]] void die(const std::string& msg) { throw std::runtime_error("dpdk_port: " + msg); }

// Find the DPDK port whose PCI BDF matches `pci` (rte_device::name is the BDF for PCI devices).
// Lives here now: both backends used to carry a private copy, and neither resolves its own port.
uint16_t find_port_by_pci(const std::string& pci) {
  uint16_t p;
  RTE_ETH_FOREACH_DEV(p) {
    rte_eth_dev_info di{};
    if (rte_eth_dev_info_get(p, &di) != 0 || !di.device) continue;
    const char* name = rte_dev_name(di.device);  // BDF for PCI devices (rte_device is opaque)
    if (name && pci == name) return p;
  }
  return kNoPort;
}

// The result of one configure + queue-setup attempt.
struct Shape {
  PortLayout lay;
  uint16_t nb_rxd = 0, nb_txd = 0;  // effective depths of the media queues
};

}  // namespace

DpdkPorts& DpdkPorts::instance() {
  static DpdkPorts ports;
  return ports;
}

DpdkPorts::PortState* DpdkPorts::find(const std::string& pci) {
  for (auto& p : ports_)
    if (p.pci == pci) return &p;
  return nullptr;
}

DpdkPorts::PortState& DpdkPorts::state_for(const std::string& pci) {
  if (PortState* p = find(pci)) return *p;
  ports_.push_back(PortState{});
  ports_.back().pci = pci;
  return ports_.back();
}

void DpdkPorts::plan(const std::string& pci, PortRole role) {
  std::lock_guard<std::mutex> lk(mu_);
  PortState& p = state_for(pci);
  if (p.planned[idx(role)])
    std::printf("[dpdk_port] WARN: %s planned twice for the %s role — one backend per role per "
                "port is the contract (the second will be refused)\n",
                pci.c_str(), rname(role));
  p.planned[idx(role)] = true;
}

void DpdkPorts::attach(const std::string& pci, const PortRequest& req, StartedFn on_started) {
  std::lock_guard<std::mutex> lk(mu_);
  PortState& p = state_for(pci);
  const int r = idx(req.role);

  // No plan at all: a standalone smoke that owns EAL itself and never called DpdkEal. Treat the
  // arriving role as the port's only one, so the port comes up inside this attach().
  if (!p.planned[0] && !p.planned[1]) p.planned[r] = true;
  if (!p.planned[r]) {
    // Plan and graph disagree. Trust the backend that actually showed up — a port whose planned
    // peer never attaches would otherwise sit unconfigured forever, silently deaf.
    std::printf("[dpdk_port] WARN: %s attached to %s but the app never planned that role\n",
                rname(req.role), pci.c_str());
    p.planned[r] = true;
  }
  if (p.attached[r]) die(std::string("two ") + rname(req.role) + " backends on " + pci);
  if (p.up) die(std::string(rname(req.role)) + " attached to " + pci + " after it was started");

  const uint16_t port_id = find_port_by_pci(pci);
  if (port_id == kNoPort) die("no DPDK port matches PCI " + pci);
  p.port_id = port_id;
  p.req[r] = req;
  p.cb[r] = std::move(on_started);
  p.attached[r] = true;
  p.held[r] = true;
  p.order[p.norder++] = req.role;

  const int peer = r ^ 1;
  if (p.planned[peer] && !p.attached[peer]) {
    std::printf("[dpdk_port] %s (port %u): %s attached; holding the port until the %s role does\n",
                pci.c_str(), port_id, rname(req.role),
                rname(peer == 0 ? PortRole::kRx : PortRole::kTx));
    return;
  }
  bring_up(p);
}

void DpdkPorts::bring_up(PortState& p) {
  const PortRequest* rx = p.planned[0] ? &p.req[0] : nullptr;
  const PortRequest* tx = p.planned[1] ? &p.req[1] : nullptr;
  const bool shared = rx && tx;

  // Isolation must be requested BEFORE configure. It is what makes a shared port safe: the NIC then
  // delivers only explicitly created flows to DPDK, and the kernel netdev on this same port function
  // keeps everything else (SSH, ARP, ptp4l). See install_flow_rule() in dpdk_rx_backend.cpp.
  if (rx && rx->isolate) {
    rte_flow_error ferr{};
    p.isolated = rte_flow_isolate(p.port_id, 1, &ferr) == 0;
    if (!p.isolated)
      std::printf("[dpdk_port] WARN: flow isolation unavailable on %s (%s) — legacy steering will "
                  "disrupt kernel traffic (SSH/PTP) sharing this port\n",
                  p.pci.c_str(), ferr.message ? ferr.message : "?");
  }

  rte_eth_dev_info di{};
  if (rte_eth_dev_info_get(p.port_id, &di) != 0) die("rte_eth_dev_info_get failed on " + p.pci);

  rte_eth_conf conf{};
  if (rx && rx->rx_timestamp && (di.rx_offload_capa & RTE_ETH_RX_OFFLOAD_TIMESTAMP))
    conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_TIMESTAMP;
  if (tx && tx->send_on_timestamp) {
    if (!(di.tx_offload_capa & RTE_ETH_TX_OFFLOAD_SEND_ON_TIMESTAMP))
      die("NIC lacks SEND_ON_TIMESTAMP offload (REAL_TIME_CLOCK_ENABLE=1 set? see M0 findings)");
    conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_SEND_ON_TIMESTAMP;
  }

  // One configure + queue-setup attempt for a given audio decision. Returns false (without
  // throwing) when the NIC rejects the shape, so the caller can retry a smaller one.
  auto try_shape = [&](bool want_audio, Shape& out) -> bool {
    out.lay = plan_queues(rx != nullptr, tx != nullptr, rx && rx->igmp_tx, want_audio);
    if (rte_eth_dev_configure(p.port_id, out.lay.nb_rxq, out.lay.nb_txq, &conf) < 0) return false;

    out.nb_rxd = rx ? rx->rxd : 0;
    out.nb_txd = tx ? tx->txd : 0;
    if (rte_eth_dev_adjust_nb_rx_tx_desc(p.port_id, &out.nb_rxd, &out.nb_txd) < 0) return false;
    // The side queues carry their own fixed depth, clamped to the NIC's limits the same way.
    auto adjust_tx = [&](uint16_t want) {
      uint16_t nrx = 0, ntx = want;
      rte_eth_dev_adjust_nb_rx_tx_desc(p.port_id, &nrx, &ntx);
      return ntx;
    };

    if (out.lay.rx_queue != kNoQueue) {
      if (!rx->rx_pool) die("the rx role attached to " + p.pci + " without a mempool");
      rte_eth_rxconf rxconf = di.default_rxconf;
      rxconf.offloads = conf.rxmode.offloads;
      if (rte_eth_rx_queue_setup(p.port_id, out.lay.rx_queue, out.nb_rxd,
                                 rte_eth_dev_socket_id(p.port_id), &rxconf, rx->rx_pool) < 0)
        return false;
    }
    // Every TX queue inherits the port's send-scheduling offload, the small side queues included.
    // That is safe: mlx5 schedules only mbufs actually carrying the TX-timestamp dynflag, and the
    // IGMP reports never set it — they go out immediately.
    rte_eth_txconf txconf = di.default_txconf;
    txconf.offloads = conf.txmode.offloads;
    auto setup_tx = [&](uint16_t q, uint16_t nd) {
      return q == kNoQueue ||
             rte_eth_tx_queue_setup(p.port_id, q, nd, rte_eth_dev_socket_id(p.port_id), &txconf) >= 0;
    };
    return setup_tx(out.lay.tx_video_queue, out.nb_txd) &&
           setup_tx(out.lay.tx_audio_queue, adjust_tx(kAudioTxd)) &&
           setup_tx(out.lay.rx_igmp_queue, adjust_tx(kIgmpTxd));
  };

  Shape shape;
  const bool want_audio = tx && tx->audio_tx;
  if (!try_shape(want_audio, shape)) {
    // The companion audio queue is the one optional part of the shape, so a solo TX port drops it
    // and keeps the validated video path — the fallback the TX backend used to make for itself.
    // A SHARED port does not: its peer role has already been told audio is coming (audio_ready()
    // answers with the configured intent while the port waits), and a relay thread quietly writing
    // into a queue that never came up is worse than refusing to start.
    if (!want_audio || shared) die("the NIC rejected the port shape on " + p.pci);
    std::printf("[dpdk_port] WARN: %s rejected the audio TX queue — audio channel disabled\n",
                p.pci.c_str());
    if (!try_shape(false, shape)) die("the NIC rejected the port shape on " + p.pci);
  }

  if (rte_eth_dev_start(p.port_id) < 0) die("rte_eth_dev_start failed on " + p.pci);
  p.up = true;

  std::printf("[dpdk_port] %s (port %u) up: roles=%s%s rxq=%u txq=%u rxd=%u txd=%u isolated=%d\n",
              p.pci.c_str(), p.port_id, rx ? "rx" : "", tx ? "tx" : "", shape.lay.nb_rxq,
              shape.lay.nb_txq, shape.nb_rxd, shape.nb_txd, p.isolated);

  // Hand each role its lease, in attach order, now that the port is live.
  for (int i = 0; i < p.norder; ++i) {
    const PortRole role = p.order[i];
    PortLease l;
    l.port_id = p.port_id;
    l.isolated = p.isolated;
    l.shared = shared;
    if (role == PortRole::kRx) {
      l.rx_queue = shape.lay.rx_queue;
      l.tx_queue = shape.lay.rx_igmp_queue;
      l.rxd = shape.nb_rxd;
    } else {
      l.tx_queue = shape.lay.tx_video_queue;
      l.audio_tx_queue = shape.lay.tx_audio_queue;
      l.audio_ok = shape.lay.tx_audio_queue != kNoQueue;
      l.txd = shape.nb_txd;
    }
    if (p.cb[idx(role)]) p.cb[idx(role)](l);
  }
}

void DpdkPorts::release(const std::string& pci, PortRole role) {
  std::lock_guard<std::mutex> lk(mu_);
  PortState* p = find(pci);
  if (!p) return;
  p->held[idx(role)] = false;
  p->cb[idx(role)] = nullptr;
  if (p->held[0] || p->held[1]) return;  // the peer still owns the port

  if (p->up) {
    if (p->isolated) {
      rte_flow_error e{};
      rte_flow_flush(p->port_id, &e);
    }
    rte_eth_dev_stop(p->port_id);
    rte_eth_dev_close(p->port_id);
    p->up = false;
  }
  p->attached[0] = p->attached[1] = false;
  p->norder = 0;
  p->port_id = kNoPort;
}

}  // namespace spark::net
