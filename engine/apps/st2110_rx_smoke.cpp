// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// ST 2110 RX bring-up app: a single st2110_rx operator that receives the loopback stream and reports
// loss + zero-copy ingest latency. Run as the RX half of the two-process loopback (mirrors gate-4):
//
//   sudo -n SPARK_PROFILE=2160p SPARK_SECONDS=12 ./engine/build/st2110_rx_smoke &   # start RX first
//   sudo -n SPARK_PROFILE=2160p ./engine/build/st2110_tx_smoke                      # then TX
//
// The RX port is DISCOVERED, not hardcoded (operators/common/nic_ports.hpp): it is the sibling of
// the port st2110_tx_smoke picks for TX, so the two halves land on the same loopback pair on either
// host (Spark 0002:01:00.x, x86_64 e.g. 0000:82:00.x). Override with SPARK_RX_PCI. Default
// udp:20000. SPARK_PROFILE / SPARK_SECONDS env.
// Multicast (NMOS/real source): set SPARK_RX_MCAST (+ SPARK_RX_SRC for SSM, SPARK_RX_PORT,
// SPARK_RX_IFACE) to IGMP-join a group and filter to it — e.g. point it at a real 2110-20 sender.
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "runtime/runtime.hpp"

#include "operators/common/nic_ports.hpp"
#include "operators/frame_sink/frame_sink.hpp"
#include "operators/st2110_rx/st2110_rx.hpp"

namespace spark {

class St2110RxSmoke : public spark::rt::Application {
 public:
  void compose() override {
    using namespace spark::rt;
    auto env = [](const char* k, const char* d) {
      const char* v = std::getenv(k);
      return std::string(v ? v : d);
    };
    const char* secs = std::getenv("SPARK_SECONDS");
    const char* port = std::getenv("SPARK_RX_PORT");

    // Mirror st2110_tx_smoke's discovery: it transmits from the default TX port, so we receive on
    // the sibling it targets.
    const auto ports = net::connectx_ports();
    auto rx_port = net::default_rx_port(ports, net::default_tx_port(ports));
    const std::string rx_pci = env("SPARK_RX_PCI", rx_port.bdf.c_str());
    if (rx_pci.empty())
      throw std::runtime_error("no second ConnectX (15b3) port found — set SPARK_RX_PCI");
    for (const auto& p : ports)
      if (p.bdf == rx_pci) rx_port = p;   // an explicit SPARK_RX_PCI may not be the discovered one
    SPARK_LOG_INFO("RX {} ({}){}", rx_pci, rx_port.iface,
                      rx_port.bdf == rx_pci && !rx_port.carrier ? " — LINK DOWN" : "");

    auto rx = make_operator<ops::St2110RxOp>(
        "st2110_rx", Arg("profile", env("SPARK_PROFILE", "1080p")),
        Arg("pci_addr", rx_pci),
        Arg("udp_port", port ? static_cast<uint32_t>(std::atoi(port)) : 20000u),
        Arg("mcast_group", env("SPARK_RX_MCAST", "")), Arg("src_ip", env("SPARK_RX_SRC", "")),
        Arg("iface_ip", env("SPARK_RX_IFACE", "")),
        Arg("run_seconds", secs ? std::atof(secs) : 12.0), make_condition<CountCondition>(1));
    auto sink = make_operator<ops::FrameSinkOp>("frame_sink");  // sink-mode rx never emits; wire anyway
    add_flow(rx, sink);
  }
};

}  // namespace spark

int main() {
  SPARK_LOG_INFO("ST 2110 RX smoke: st2110_rx (loopback receiver, loss + ingest latency).");
  auto app = spark::rt::make_application<spark::St2110RxSmoke>();
  app->run();
  return 0;
}
