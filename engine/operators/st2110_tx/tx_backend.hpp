// Transport backend interface for ST 2110 TX (keeps DPDK out of the operator's translation unit).
//
// The St2110TxOp talks only to this interface, so the IO plane is swappable: the v1 implementation
// is raw DPDK + mlx5 tx_pp (make_dpdk_tx_backend(), grown straight from the gate-4 spike), and the
// production path can later add an advanced_network-backed implementation behind the same vtable
// without touching the operator or the framing code. All DPDK/rte_* headers stay inside
// dpdk_tx_backend.cpp.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace spark::net {

struct TxBackendConfig {
  std::string pci_addr;                  // CX-7 TX port, e.g. "0002:01:00.0"
  std::array<uint8_t, 6> dst_mac{};      // dest MAC (gate-4 switched fabric: the RX port's MAC)
  std::string src_ip = "192.168.50.10";  // ST 2110-10 source
  std::string dst_ip = "239.0.0.1";      // ST 2110 multicast group (loopback may use RX unicast)
  uint16_t udp_port = 20000;
  uint32_t tx_pp_ns = 500;               // mlx5 tx_pp clock granularity (M0/spike used 500)
  uint16_t txd = 2048;                   // TX ring depth; pacing horizon = txd * gap (gate-4 finding)
  uint16_t mtu = 1500;
  std::string eal_core_list = "0,1";
  std::string file_prefix = "spark_tx";  // distinct per process (matches the spike's --file-prefix)
  bool pacing = true;                    // enable tx_pp HW send-scheduling
  bool manage_eal = true;  // true: this backend owns rte_eal_init (standalone). false: shared EAL
                           // already up (DpdkEal) — just attach to the port (multi-backend process).
  // Companion ST 2110-30 audio egress on the SAME port (M10): a second header template and a
  // dedicated TX queue (index 1), so the audio relay thread submits without touching the video
  // queue's state. Both queues share the port's tx_pp clock — audio packets get the same
  // send-on-timestamp pacing on the same PHC. Empty dst = audio disabled (unchanged behavior).
  std::string audio_dst_ip;
  uint16_t audio_udp_port = 0;
};

// HW pacing counters mirrored from the mlx5 tx_pp xstats — the same metrics the gate-4 spike read.
struct TxStats {
  uint64_t tx_packets = 0;
  uint64_t tx_bytes = 0;
  uint64_t future_errors = 0;  // schedule horizon overran the tx_pp window (see gate-4 findings)
  uint64_t past_errors = 0;    // packet handed over after its scheduled send time
  uint64_t jitter_ns = 0;      // tx_pp clock jitter
  uint64_t wander_ns = 0;      // tx_pp clock wander
  uint64_t sync_lost = 0;      // tx_pp clock lost lock to the PHC (pacing unusable if > 0)
};

// A reserved transport packet. The backend has already written Eth/IPv4/UDP headers; the caller
// fills `payload` with the RTP/RFC 4175 bytes (exactly `payload_cap` of them) then submit()s it.
struct TxBuf {
  uint8_t* payload = nullptr;
  uint32_t payload_cap = 0;
  void* opaque = nullptr;  // backend-private handle (the rte_mbuf*)
};

class ISt2110TxBackend {
 public:
  virtual ~ISt2110TxBackend() = default;

  // Bring up EAL + the port (tx_pp pacing, send_on_timestamp offload). Throws std::runtime_error
  // on failure. Must run as root with hugepages (as the gate-4 spike documents).
  virtual void init(const TxBackendConfig& cfg) = 0;

  // Reserve a packet sized for `payload_len` UDP-payload bytes, headers pre-filled.
  virtual TxBuf reserve_packet(uint32_t payload_len) = 0;

  // Queue a filled packet to be transmitted by the NIC at PHC time `send_ts_ns`.
  virtual void submit(const TxBuf& buf, uint64_t send_ts_ns) = 0;

  // Drain any buffered packets to the NIC (call at end of frame / burst).
  virtual void flush() = 0;

  // --- companion audio channel (config.audio_dst_ip; see TxBackendConfig) ---------------------
  // Thread contract: reserve_audio/submit_audio are called ONLY from the audio relay thread and use
  // a dedicated TX queue, so they never contend with the video compute() thread's reserve/submit.
  virtual bool audio_ready() = 0;  // audio channel configured AND its queue came up
  virtual TxBuf reserve_audio(uint32_t payload_len) = 0;
  virtual void submit_audio(const TxBuf& buf, uint64_t send_ts_ns) = 0;  // bursts immediately

  virtual TxStats stats() = 0;
  virtual uint64_t now_ns() = 0;  // read the NIC PHC — the pacing time base for send_ts_ns
  virtual void shutdown() = 0;
};

std::unique_ptr<ISt2110TxBackend> make_dpdk_tx_backend();

}  // namespace spark::net
