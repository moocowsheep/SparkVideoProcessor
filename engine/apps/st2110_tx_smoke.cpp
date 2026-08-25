// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// ST 2110 TX bring-up app: test_pattern -> st2110_tx (tx_pp paced).
//
// This is the gate-4 follow-on the spike pointed to — a *real* media generator (one packet per
// ST 2110-21 slot, self-throttled to the tx_pp window) instead of testpmd's open-loop txonly. Run on
// the box, as root, with the CX-7 cabled (see docs/M1-gate4-pacing.md):
//
//   sudo -E ./engine/build/st2110_tx_smoke
//
// Ports are DISCOVERED, not hardcoded (operators/common/nic_ports.hpp): TX defaults to the first
// linked-up ConnectX port and the destination MAC to its sibling port on the same card — the
// loopback pair on the DGX Spark (0002:01:00.x) and on an x86_64 box (e.g. 0000:82:00.x) alike.
// Override either with SPARK_TX_PCI / SPARK_DST_MAC. On the RX port, count packets with the spike's
// rxonly testpmd (or st2110_rx_smoke). Success = sustained pps == target with future_err/past_err ~0
// and tx_pp jitter in the tens of ns — the proof testpmd could not give.
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "runtime/runtime.hpp"

#include "operators/common/nic_ports.hpp"
#include "operators/st2110_tx/st2110_tx.hpp"
#include "operators/test_pattern/test_pattern.hpp"

namespace spark {

// Env overrides (via the SETENV allowlist): SPARK_PROFILE=1080p|2160p, SPARK_FRAMES=<n>,
// SPARK_TX_PCI=<bdf>, SPARK_DST_MAC=<mac>, SPARK_WARMUP_MS=<ms>. Unset ports/MAC are discovered
// (see above); profile 1080p, 300 frames.
class St2110TxSmoke : public spark::rt::Application {
 public:
  void compose() override {
    using namespace spark::rt;
    auto env = [](const char* k, const char* d) { const char* v = std::getenv(k); return std::string(v ? v : d); };
    const std::string profile = env("SPARK_PROFILE", "1080p");
    const int64_t frames = std::atoll(env("SPARK_FRAMES", "300").c_str());
    const uint32_t warmup = static_cast<uint32_t>(std::atoll(env("SPARK_WARMUP_MS", "0").c_str()));

    // Discover the local ConnectX pair, then let the env override either half. An explicit
    // SPARK_TX_PCI re-anchors the pair, so the derived dst MAC is that card's sibling port.
    const auto ports = net::connectx_ports();
    auto tx_port = net::default_tx_port(ports);
    const std::string tx_pci = env("SPARK_TX_PCI", tx_port.bdf.c_str());
    for (const auto& p : ports)
      if (p.bdf == tx_pci) tx_port = p;
    const auto rx_port = net::default_rx_port(ports, tx_port);
    const std::string dst_mac = env("SPARK_DST_MAC", rx_port.mac.c_str());
    if (tx_pci.empty())
      throw std::runtime_error("no ConnectX (15b3) port found — cable the NIC or set SPARK_TX_PCI");
    if (dst_mac.empty())
      throw std::runtime_error("no second ConnectX port to send to — set SPARK_DST_MAC explicitly");
    const bool derived_mac = !rx_port.mac.empty() && dst_mac == rx_port.mac;
    SPARK_LOG_INFO("TX {} ({}) -> dst MAC {} ({})", tx_pci, tx_port.iface, dst_mac,
                      derived_mac ? (rx_port.bdf + (rx_port.carrier ? "" : ", LINK DOWN"))
                                  : std::string("explicit"));

    auto src = make_operator<ops::TestPatternOp>("test_pattern", Arg("profile", profile),
                                                 make_condition<CountCondition>(frames));
    auto tx = make_operator<ops::St2110TxOp>(
        "st2110_tx", Arg("pci_addr", tx_pci), Arg("dst_mac", dst_mac), Arg("warmup_ms", warmup));
    add_flow(src, tx);
  }
};

}  // namespace spark

int main() {
  SPARK_LOG_INFO("ST 2110 TX smoke: test_pattern -> st2110_tx (tx_pp paced).");
  auto app = spark::rt::make_application<spark::St2110TxSmoke>();
  app->run();
  return 0;
}
