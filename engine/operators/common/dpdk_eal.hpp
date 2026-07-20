// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Process-wide DPDK EAL ownership. rte_eal_init() may run only ONCE per process, so when several
// backends live in one process (the rx -> tx pass-through, or the one-process loopback), they cannot
// each init EAL. Instead the app registers every port with add_device() and calls init() once; the
// backends are then created with manage_eal=false and only attach to their port.
//
// Standalone smokes (st2110_tx_smoke / st2110_rx_smoke) don't touch this — their single backend keeps
// managing EAL itself (manage_eal=true), which is the simpler one-port path.
#pragma once

#include <string>
#include <vector>

namespace spark::net {

class DpdkEal {
 public:
  static DpdkEal& instance();

  // Register a port to allowlist, e.g. add_device("0002:01:00.0", "tx_pp=500") or
  // add_device("0002:01:00.1", ""). Must be called before init().
  void add_device(const std::string& pci, const std::string& devargs = "");

  // Run rte_eal_init() with all registered devices + cores. Idempotent: only the first call inits;
  // later calls are no-ops (a warning if devices were added after init). Throws on EAL failure.
  void init(const std::string& core_list = "0,1,2,3", const std::string& file_prefix = "spark");

  bool initialized() const { return inited_; }

 private:
  DpdkEal() = default;
  std::vector<std::string> allow_;  // each entry is a full "-a" value: "pci[,devargs]"
  bool inited_ = false;
};

}  // namespace spark::net
