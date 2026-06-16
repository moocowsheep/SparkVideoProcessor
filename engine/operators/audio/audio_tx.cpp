#include "audio_tx.hpp"

#include <array>
#include <cstdio>

namespace spark::ops {
namespace {
std::array<uint8_t, 6> parse_mac(const std::string& s) {
  std::array<uint8_t, 6> m{};
  std::sscanf(s.c_str(), "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx", &m[0], &m[1], &m[2], &m[3], &m[4], &m[5]);
  return m;
}
}  // namespace

void AudioTxOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<spark::st2110::AudioBlock>("audio");
  spec.param(pci_addr_, "pci_addr", "TX PCI", "CX-7 TX port BDF", std::string("0002:01:00.0"));
  spec.param(dst_mac_, "dst_mac", "Dest MAC", "egress MAC; 00:..:00 -> derive from multicast group",
             std::string("00:00:00:00:00:00"));
  spec.param(src_ip_, "src_ip", "Source IP", "ST 2110-10 source address", std::string("192.168.50.10"));
  spec.param(dst_ip_, "dst_ip", "Dest IP", "ST 2110-30 multicast group", std::string("239.100.0.20"));
  spec.param(udp_port_, "udp_port", "UDP port", "RTP destination port", uint32_t(20010));
  spec.param(ssrc_, "ssrc", "RTP SSRC", "RTP synchronization source id", uint32_t(0x53504b32));
  spec.param(tx_pp_ns_, "tx_pp_ns", "tx_pp ns", "mlx5 tx_pp clock granularity", uint32_t(500));
  spec.param(txd_, "txd", "TX descriptors", "TX ring depth", uint32_t(2048));
  spec.param(pacing_horizon_ns_, "pacing_horizon_ns", "Pacing horizon", "max schedule lead", uint32_t(2000000));
  spec.param(pacing_, "pacing", "Enable pacing", "tx_pp hardware send-scheduling", true);
  spec.param(manage_eal_, "manage_eal", "Manage EAL", "true: own rte_eal_init; false: shared DpdkEal", true);
}

void AudioTxOp::start() {
  backend_ = spark::net::make_dpdk_tx_backend();
  spark::net::TxBackendConfig cfg;
  cfg.pci_addr = pci_addr_.get();
  cfg.dst_mac = parse_mac(dst_mac_.get());
  cfg.src_ip = src_ip_.get();
  cfg.dst_ip = dst_ip_.get();
  cfg.udp_port = static_cast<uint16_t>(udp_port_.get());
  cfg.tx_pp_ns = tx_pp_ns_.get();
  cfg.txd = static_cast<uint16_t>(txd_.get());
  cfg.pacing = pacing_.get();
  cfg.manage_eal = manage_eal_.get();
  cfg.file_prefix = "spark_audio_tx";
  cfg.eal_core_list = "0,1";
  backend_->init(cfg);
  HOLOSCAN_LOG_INFO("audio_tx started: TX {} -> {} pacing={}", cfg.pci_addr, cfg.dst_ip, cfg.pacing);
}

void AudioTxOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext&,
                        holoscan::ExecutionContext&) {
  auto maybe = op_input.receive<spark::st2110::AudioBlock>("audio");
  if (!maybe) return;
  auto& blk = maybe.value();
  if (!blk.pcm) return;

  if (!pktz_) {
    pktz_ = std::make_unique<spark::st2110::AudioPacketizer>(blk.format, /*pt=*/97, ssrc_.get());
    packet_time_ns_ = static_cast<uint64_t>(blk.format.packet_time_ms * 1e6);
  }

  // Evenly space packets at the packet time; re-anchor one horizon ahead if we have fallen behind.
  const uint64_t now = backend_->now_ns();
  if (schedule_base_ns_ == 0 || now > schedule_base_ns_)
    schedule_base_ns_ = now + pacing_horizon_ns_;
  const uint64_t send_ts = schedule_base_ns_;
  schedule_base_ns_ += packet_time_ns_;

  const uint32_t payload_len = spark::st2110::kRtpHeaderBytes + blk.format.packet_payload_bytes();
  spark::net::TxBuf buf = backend_->reserve_packet(payload_len);
  // preserve the original sampling instant (RTP timestamp) for downstream A/V sync after the delay
  pktz_->write_packet(blk.rtp_timestamp, blk.pcm->data(), buf.payload);
  backend_->submit(buf, send_ts);
  backend_->flush();
  ++packets_sent_;
}

void AudioTxOp::stop() {
  if (!backend_) return;
  const auto s = backend_->stats();
  HOLOSCAN_LOG_INFO("audio_tx stopped: packets={} | tx_pp future_err={} past_err={} sync_lost={}",
                    packets_sent_, s.future_errors, s.past_errors, s.sync_lost);
  backend_->shutdown();
  backend_.reset();
}

}  // namespace spark::ops
