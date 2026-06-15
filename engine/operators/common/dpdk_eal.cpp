#include "dpdk_eal.hpp"

#include <cstdio>
#include <stdexcept>

#include <rte_eal.h>

namespace spark::net {

DpdkEal& DpdkEal::instance() {
  static DpdkEal eal;
  return eal;
}

void DpdkEal::add_device(const std::string& pci, const std::string& devargs) {
  if (inited_) {
    std::printf("[dpdk_eal] WARNING: add_device(%s) after init() — ignored\n", pci.c_str());
    return;
  }
  allow_.push_back(devargs.empty() ? pci : pci + "," + devargs);
}

void DpdkEal::init(const std::string& core_list, const std::string& file_prefix) {
  if (inited_) return;  // idempotent: first caller wins

  std::vector<std::string> args = {"spark", "-l", core_list, "--file-prefix", file_prefix};
  for (auto& a : allow_) {
    args.push_back("-a");
    args.push_back(a);
  }
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());

  if (rte_eal_init(static_cast<int>(argv.size()), argv.data()) < 0)
    throw std::runtime_error("DpdkEal::init: rte_eal_init failed (root? hugepages? device args?)");
  inited_ = true;
  std::printf("[dpdk_eal] EAL up: %zu device(s), cores=%s, prefix=%s\n", allow_.size(),
              core_list.c_str(), file_prefix.c_str());
}

}  // namespace spark::net
