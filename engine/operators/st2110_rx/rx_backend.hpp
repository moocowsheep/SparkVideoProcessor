// Transport backend interface for ST 2110 RX (keeps DPDK out of the operator's translation unit).
// Mirrors tx_backend.hpp: the St2110RxOp talks only to this vtable, so a raw-DPDK/mlx5 implementation
// (v1) can later be replaced by an advanced_network-backed one without touching the operator. All
// rte_* headers stay inside dpdk_rx_backend.cpp.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace spark::net {

struct RxBackendConfig {
  std::string pci_addr;       // CX-7 RX port, e.g. "0002:01:00.1"
  uint16_t udp_port = 20000;  // accept only UDP packets to this destination port
  uint16_t rxd = 4096;        // RX ring depth
  uint16_t mtu = 1500;
  std::string eal_core_list = "2,3";
  std::string file_prefix = "spark_rx";  // distinct from the TX process (two-process loopback)
};

// One received media packet. `payload` points at the UDP payload (the RTP/RFC 4175 bytes) inside the
// mbuf — no copy. `hw_timestamp_ns` is the NIC's HW RX timestamp (PHC ns), the basis for ingest
// latency. The mbuf is owned by the backend until release().
struct RxPacket {
  const uint8_t* payload = nullptr;
  uint32_t len = 0;
  uint64_t hw_timestamp_ns = 0;
  bool has_timestamp = false;
  void* opaque = nullptr;  // the rte_mbuf*
};

struct RxStats {
  uint64_t rx_packets = 0;   // ipackets: frames the NIC RX port delivered (any kind)
  uint64_t rx_bytes = 0;
  uint64_t rx_missed = 0;    // imissed: dropped by HW for lack of a descriptor
  uint64_t rx_nombuf = 0;    // mbuf allocation failures
  uint64_t raw_received = 0;  // mbufs returned by rx_burst before the UDP/dst-port filter
};

class ISt2110RxBackend {
 public:
  virtual ~ISt2110RxBackend() = default;

  // Bring up EAL + the port (RX timestamp offload). Throws std::runtime_error on failure.
  virtual void init(const RxBackendConfig& cfg) = 0;

  // Poll up to `max` UDP/dst-port-matching packets into `out`; returns the count (0 if none ready).
  // Non-matching frames are dropped internally. Caller must release() what it gets back.
  virtual uint16_t receive(RxPacket* out, uint16_t max) = 0;
  virtual void release(RxPacket* pkts, uint16_t n) = 0;

  virtual uint64_t now_ns() = 0;  // NIC PHC — same time base as RxPacket::hw_timestamp_ns
  virtual RxStats stats() = 0;
  virtual void shutdown() = 0;
};

std::unique_ptr<ISt2110RxBackend> make_dpdk_rx_backend();

}  // namespace spark::net
