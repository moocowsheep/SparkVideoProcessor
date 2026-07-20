// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// ST 2110-30 audio passthrough: audio_rx -> audio_delay -> audio_tx, in one process sharing one EAL
// across the two CX-7 ports. Receives an audio multicast (IGMP-joined), holds it by a lip-sync delay
// matched to the video path's latency, and re-transmits to the output group (tx_pp-paced).
//
// Runs as its own process today (owns the ports). Folding audio into st2110_pipeline so one process
// carries video+audio on the SAME port needs the shared-port RX demux (see docs/M6-nmos.md) — a
// follow-on; the framing/operators/delay here are complete.
//
//   sudo -n SPARK_RX_AUDIO_MCAST=239.100.0.20 SPARK_TX_AUDIO_MCAST=239.100.1.20 \
//           SPARK_AUDIO_DELAY_MS=40 ./engine/build/audio_passthrough
#include <cstdint>
#include <cstdlib>

#include <holoscan/holoscan.hpp>

#include "operators/audio/audio_delay.hpp"
#include "operators/audio/audio_rx.hpp"
#include "operators/audio/audio_tx.hpp"
#include "operators/common/dpdk_eal.hpp"

namespace spark {

class AudioPassthrough : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto env = [](const char* k, const char* d) {
      const char* v = std::getenv(k);
      return std::string(v ? v : d);
    };
    auto envu = [&](const char* k, uint32_t d) {
      const char* v = std::getenv(k);
      return v && *v ? static_cast<uint32_t>(std::atoll(v)) : d;
    };

    const std::string rx_pci = env("SPARK_RX_PCI", "0000:01:00.1");
    const std::string tx_pci = env("SPARK_TX_PCI", "0002:01:00.0");
    const std::string rx_group = env("SPARK_RX_AUDIO_MCAST", "");
    const std::string rx_src = env("SPARK_RX_AUDIO_SRC", "");
    const std::string rx_iface = env("SPARK_RX_IFACE", "");
    const std::string tx_group = env("SPARK_TX_AUDIO_MCAST", "239.100.1.20");
    uint32_t rx_port = envu("SPARK_RX_AUDIO_PORT", 0);
    if (rx_port == 0) rx_port = 20010;
    uint32_t tx_port = envu("SPARK_TX_AUDIO_PORT", 0);
    if (tx_port == 0) tx_port = 20010;
    const uint32_t channels = envu("SPARK_AUDIO_CH", 2);
    const uint32_t depth = envu("SPARK_AUDIO_DEPTH", 24);
    const double ptime = std::atof(env("SPARK_AUDIO_PTIME", "1.0").c_str());
    const double delay_ms = std::atof(env("SPARK_AUDIO_DELAY_MS", "0").c_str());
    const int64_t packets = static_cast<int64_t>(envu("SPARK_AUDIO_PACKETS", 60000));  // ~60 s @ 1 ms

    auto& eal = spark::net::DpdkEal::instance();
    eal.add_device(tx_pci, "tx_pp=500");
    eal.add_device(rx_pci, "");
    eal.init("0-11", "spark_audio");

    auto rx = make_operator<ops::AudioRxOp>(
        "audio_rx", Arg("pci_addr", rx_pci), Arg("udp_port", rx_port), Arg("mcast_group", rx_group),
        Arg("src_ip", rx_src), Arg("iface_ip", rx_iface), Arg("channels", channels),
        Arg("bit_depth", depth), Arg("packet_time_ms", ptime), Arg("manage_eal", false),
        make_condition<CountCondition>(packets));
    auto delay = make_operator<ops::AudioDelayOp>("audio_delay", Arg("delay_ms", delay_ms));
    auto tx = make_operator<ops::AudioTxOp>(
        "audio_tx", Arg("pci_addr", tx_pci), Arg("dst_ip", tx_group), Arg("udp_port", tx_port),
        Arg("dst_mac", std::string("00:00:00:00:00:00")), Arg("manage_eal", false));

    add_flow(rx, delay);
    add_flow(delay, tx);
  }
};

}  // namespace spark

int main() {
  HOLOSCAN_LOG_INFO("ST 2110-30 audio passthrough: audio_rx -> delay -> audio_tx.");
  auto app = holoscan::make_application<spark::AudioPassthrough>();
  app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
      "mts", holoscan::Arg("worker_thread_number", static_cast<int64_t>(3)),
      holoscan::Arg("stop_on_deadlock", true),
      holoscan::Arg("stop_on_deadlock_timeout", static_cast<int64_t>(3000)),
      holoscan::Arg("max_duration_ms", static_cast<int64_t>(120000))));
  app->run();
  return 0;
}
