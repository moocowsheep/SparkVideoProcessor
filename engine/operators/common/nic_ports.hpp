// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// ConnectX port discovery — pure, header-only, sysfs-only (no DPDK, no root, no NIC required).
//
// The bring-up apps used to hardcode the DGX Spark's gate-4 BDFs (TX 0002:01:00.0 -> RX
// 0002:01:00.1). Those are meaningless on the x86_64 host, where the card is an ordinary PCIe
// device and enumerates wherever the slot puts it (e.g. 0000:82:00.0/.1). This walks
// /sys/bus/pci/devices for vendor 15b3 (Mellanox/NVIDIA) and reports each port's BDF, kernel
// interface, MAC and carrier, so a default can be *derived* on whatever box is running.
//
// Explicit env/params always win; this only supplies the default. Selection policy is
// carrier-first, then sibling-port (the two ports of one dual-port card are the usual loopback
// pair), which lands on the cabled pair on both supported hosts.
#pragma once

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace spark::net {

struct NicPort {
  std::string bdf;      // "0000:82:00.0"
  std::string iface;    // "enp130s0f0np0" (empty if the kernel driver isn't bound)
  std::string mac;      // "c4:70:bd:dc:9b:b0" (empty if no iface)
  bool carrier = false; // link up (a down port still works for a DPDK bind, just carries nothing)
};

namespace detail {
inline std::string read_trimmed(const std::filesystem::path& p) {
  std::ifstream f(p);
  std::string s;
  if (f) std::getline(f, s);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
  return s;
}
}  // namespace detail

// Every ConnectX port on the bus, sorted by BDF (so the order is stable across boots).
inline std::vector<NicPort> connectx_ports() {
  namespace fs = std::filesystem;
  std::vector<NicPort> ports;
  std::error_code ec;
  for (const auto& dev : fs::directory_iterator("/sys/bus/pci/devices", ec)) {
    if (detail::read_trimmed(dev.path() / "vendor") != "0x15b3") continue;
    NicPort p;
    p.bdf = dev.path().filename().string();
    for (const auto& n : fs::directory_iterator(dev.path() / "net", ec)) {
      p.iface = n.path().filename().string();
      p.mac = detail::read_trimmed(n.path() / "address");
      p.carrier = detail::read_trimmed(n.path() / "carrier") == "1";
      break;  // one netdev per PCI function
    }
    ports.push_back(std::move(p));
  }
  std::sort(ports.begin(), ports.end(), [](const NicPort& a, const NicPort& b) { return a.bdf < b.bdf; });
  return ports;
}

// Default TX port: the first port with link up, else the first port present, else "".
inline NicPort default_tx_port(const std::vector<NicPort>& ports) {
  for (const auto& p : ports)
    if (p.carrier) return p;
  return ports.empty() ? NicPort{} : ports.front();
}

// Default RX port, given the TX pick: prefer the sibling function on the same card (the usual
// loopback pair — 0000:82:00.0 <-> .1), preferring one with carrier; else any other port with
// carrier; else any other port at all. Returns an empty port if the box has only one.
inline NicPort default_rx_port(const std::vector<NicPort>& ports, const NicPort& tx) {
  const std::string card = tx.bdf.substr(0, tx.bdf.rfind('.'));  // strip the function digit
  const NicPort* sibling = nullptr;
  const NicPort* other_up = nullptr;
  const NicPort* other = nullptr;
  for (const auto& p : ports) {
    if (p.bdf == tx.bdf) continue;
    if (!card.empty() && p.bdf.compare(0, card.size(), card) == 0 && (!sibling || p.carrier)) sibling = &p;
    if (p.carrier && !other_up) other_up = &p;
    if (!other) other = &p;
  }
  const NicPort* pick = sibling ? sibling : (other_up ? other_up : other);
  return pick ? *pick : NicPort{};
}

}  // namespace spark::net
