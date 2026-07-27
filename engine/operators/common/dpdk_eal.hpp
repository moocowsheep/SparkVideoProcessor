// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Process-wide DPDK EAL ownership. rte_eal_init() may run only ONCE per process, so when several
// backends live in one process (the rx -> tx pass-through, or the one-process loopback), they cannot
// each init EAL. Instead the app registers every port with add_device() and calls init() once; the
// backends are then created with manage_eal=false and only attach to their port.
//
// add_device() is also where the app declares WHICH ROLE will attach to each BDF. One call per
// backend: naming the same BDF twice, once per role, is how a shared RX+TX port is declared, and
// DpdkPorts turns that plan into a single configure/start (see dpdk_port.hpp). The `-a` allowlist
// is deduplicated, so a shared port is still passed to EAL once, with the union of its devargs.
//
// Standalone smokes (st2110_tx_smoke / st2110_rx_smoke) don't touch this — their single backend keeps
// managing EAL itself (manage_eal=true), which is the simpler one-port path.
#pragma once

#include <string>
#include <vector>

#include "dpdk_port.hpp"

namespace spark::net {

class DpdkEal {
 public:
  static DpdkEal& instance();

  // Register a port to allowlist and declare the role that will attach to it, e.g.
  // add_device("0002:01:00.0", PortRole::kTx, "tx_pp=500"). Must be called before init(). Calling
  // it twice for one BDF with different roles declares a shared port; the devargs are merged.
  void add_device(const std::string& pci, PortRole role, const std::string& devargs = "");

  // Run rte_eal_init() with all registered devices + cores. Idempotent: only the first call inits;
  // later calls are no-ops (a warning if devices were added after init). Throws on EAL failure.
  void init(const std::string& core_list = "0,1,2,3", const std::string& file_prefix = "spark");

  bool initialized() const { return inited_; }

 private:
  DpdkEal() = default;
  struct Device {
    std::string pci;
    std::string devargs;  // merged across roles; "" = none
  };
  std::vector<Device> allow_;
  bool inited_ = false;
};

}  // namespace spark::net
