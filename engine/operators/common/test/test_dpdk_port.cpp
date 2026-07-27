// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Unit test: the shared-port queue map (dpdk_port.hpp plan_queues). Pure CPU, no NIC/DPDK/root.
//
// The properties the RX+TX-on-one-port path stands on:
//  1. the solo cases reproduce exactly the layout each backend used to choose for itself, so
//     every pre-existing two-port config keeps its queue indices;
//  2. a shared port gives each role its own TX queue — the RX role's IGMP reports never land on
//     the paced media queue, whose descriptors are the pacing horizon;
//  3. queue indices are dense and below the configured count (DPDK rejects a setup outside it);
//  4. a queue is configured if and only if some role was given it.
#include <cstdio>
#include <cstdlib>

#include "../dpdk_port.hpp"

using spark::net::kNoQueue;
using spark::net::plan_queues;
using spark::net::PortLayout;

// assert() vanishes under Release/-DNDEBUG — use an explicit check like the other test binaries.
#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

namespace {

// Every TX index handed out must be distinct and inside [0, nb_txq); same for the single RX queue.
void check_wellformed(const PortLayout& l, const char* what) {
  const uint16_t txq[3] = {l.tx_video_queue, l.tx_audio_queue, l.rx_igmp_queue};
  int used = 0;
  for (int i = 0; i < 3; ++i) {
    if (txq[i] == kNoQueue) continue;
    ++used;
    if (txq[i] >= l.nb_txq) {
      std::fprintf(stderr, "FAIL %s: tx queue %u outside the configured %u\n", what, txq[i],
                   l.nb_txq);
      std::exit(1);
    }
    for (int j = i + 1; j < 3; ++j)
      if (txq[i] == txq[j]) {
        std::fprintf(stderr, "FAIL %s: two roles share tx queue %u\n", what, txq[i]);
        std::exit(1);
      }
  }
  // No queue is configured that nobody was given: an unset queue fails rte_eth_dev_start.
  if (used != l.nb_txq) {
    std::fprintf(stderr, "FAIL %s: %u tx queues configured but %d handed out\n", what, l.nb_txq,
                 used);
    std::exit(1);
  }
  if (l.rx_queue == kNoQueue)
    CHECK(l.nb_rxq == 0);
  else
    CHECK(l.rx_queue < l.nb_rxq);
}

}  // namespace

int main() {
  // 1. solo TX, no audio — what dpdk_tx_backend used to configure: one queue, index 0.
  {
    const PortLayout l = plan_queues(/*rx=*/false, /*tx=*/true, /*igmp=*/false, /*audio=*/false);
    CHECK(l.nb_rxq == 0 && l.nb_txq == 1);
    CHECK(l.tx_video_queue == 0);
    CHECK(l.tx_audio_queue == kNoQueue && l.rx_igmp_queue == kNoQueue && l.rx_queue == kNoQueue);
    check_wellformed(l, "solo tx");
  }
  // 1b. solo TX with the companion audio channel: video 0, audio 1.
  {
    const PortLayout l = plan_queues(false, true, false, true);
    CHECK(l.nb_txq == 2 && l.tx_video_queue == 0 && l.tx_audio_queue == 1);
    check_wellformed(l, "solo tx + audio");
  }
  // 1c. solo RX joined to a group — one RX queue plus a TX queue at index 0 for IGMP.
  {
    const PortLayout l = plan_queues(true, false, true, false);
    CHECK(l.nb_rxq == 1 && l.rx_queue == 0);
    CHECK(l.nb_txq == 1 && l.rx_igmp_queue == 0);
    CHECK(l.tx_video_queue == kNoQueue && l.tx_audio_queue == kNoQueue);
    check_wellformed(l, "solo rx");
  }
  // 1d. solo RX, no group (the legacy promiscuous loopback): no TX queue at all.
  {
    const PortLayout l = plan_queues(true, false, false, false);
    CHECK(l.nb_rxq == 1 && l.nb_txq == 0 && l.rx_igmp_queue == kNoQueue);
    check_wellformed(l, "solo rx, no group");
  }

  // 2. shared port, everything on: RX media queue 0; TX video 0, TX audio 1, RX IGMP 2.
  {
    const PortLayout l = plan_queues(true, true, true, true);
    CHECK(l.nb_rxq == 1 && l.rx_queue == 0);
    CHECK(l.nb_txq == 3);
    CHECK(l.tx_video_queue == 0 && l.tx_audio_queue == 1 && l.rx_igmp_queue == 2);
    // The property that matters: IGMP never rides the paced media queue.
    CHECK(l.rx_igmp_queue != l.tx_video_queue);
    check_wellformed(l, "shared, all on");
  }
  // 2b. shared without audio — IGMP closes up to index 1 rather than leaving a hole.
  {
    const PortLayout l = plan_queues(true, true, true, false);
    CHECK(l.nb_txq == 2 && l.tx_video_queue == 0 && l.rx_igmp_queue == 1);
    CHECK(l.tx_audio_queue == kNoQueue);
    check_wellformed(l, "shared, no audio");
  }
  // 2c. shared with audio but no group joined (RX filtering by dst port only).
  {
    const PortLayout l = plan_queues(true, true, false, true);
    CHECK(l.nb_txq == 2 && l.tx_video_queue == 0 && l.tx_audio_queue == 1);
    CHECK(l.rx_igmp_queue == kNoQueue);
    check_wellformed(l, "shared, no group");
  }

  // 3. exhaustive well-formedness over every combination, including the degenerate no-role port.
  for (int m = 0; m < 16; ++m) {
    const bool rx = m & 1, tx = m & 2, igmp = m & 4, audio = m & 8;
    const PortLayout l = plan_queues(rx, tx, igmp, audio);
    check_wellformed(l, "combination");
    // A role only ever gets the queues its own kind uses.
    CHECK((l.rx_queue != kNoQueue) == rx);
    CHECK((l.tx_video_queue != kNoQueue) == tx);
    CHECK((l.tx_audio_queue != kNoQueue) == (tx && audio));
    CHECK((l.rx_igmp_queue != kNoQueue) == (rx && igmp));
  }

  std::printf("test_dpdk_port: OK\n");
  return 0;
}
