// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

#define _GNU_SOURCE
#include "dpdk_eal.hpp"

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

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

  // rte_eal_init pins THIS (main) thread to the single main lcore. Threads created afterward
  // (the runtime's operator threads) inherit that one-core affinity mask, so every busy-spin operator
  // serializes onto one core -> ~ms preemption when >1 spin (the one-process co-location symptom;
  // standalone smokes have a single hot operator and were unaffected). Widen the mask back to all
  // online CPUs so workers spread across cores.
  cpu_set_t set;
  CPU_ZERO(&set);
  const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  for (long i = 0; i < ncpu; ++i) CPU_SET(i, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
    std::printf("[dpdk_eal] WARNING: could not widen thread affinity post-EAL\n");

  std::printf("[dpdk_eal] EAL up: %zu device(s), cores=%s, prefix=%s, affinity widened to %ld CPUs\n",
              allow_.size(), core_list.c_str(), file_prefix.c_str(), ncpu);
}

}  // namespace spark::net
