#include "audio_rx.hpp"

#include <chrono>

namespace spark::ops {

void AudioRxOp::setup(holoscan::OperatorSpec& spec) {
  spec.output<spark::st2110::AudioBlock>("audio");
  spec.param(pci_addr_, "pci_addr", "RX PCI", "CX-7 RX port BDF", std::string("0000:01:00.1"));
  spec.param(udp_port_, "udp_port", "UDP port", "ST 2110-30 RTP destination port", uint32_t(20010));
  spec.param(mcast_group_, "mcast_group", "Multicast group", "audio group to join (NMOS)", std::string(""));
  spec.param(src_ip_, "src_ip", "SSM source", "source-specific filter", std::string(""));
  spec.param(iface_ip_, "iface_ip", "Interface IP", "local media IP (IGMP report source)", std::string(""));
  spec.param(channels_, "channels", "Channels", "audio channel count", uint32_t(2));
  spec.param(bit_depth_, "bit_depth", "Bit depth", "16 (L16) or 24 (L24)", uint32_t(24));
  spec.param(packet_time_ms_, "packet_time_ms", "Packet time", "ms per RTP packet (1 or 0.125)", 1.0);
  spec.param(manage_eal_, "manage_eal", "Manage EAL", "true: own rte_eal_init; false: shared DpdkEal", true);
}

void AudioRxOp::start() {
  fmt_.sample_rate = 48000;
  fmt_.channels = static_cast<uint16_t>(channels_.get());
  fmt_.encoding = bit_depth_.get() == 16 ? spark::st2110::AudioEncoding::L16
                                         : spark::st2110::AudioEncoding::L24;
  fmt_.packet_time_ms = packet_time_ms_.get();
  depkt_ = std::make_unique<spark::st2110::AudioDepacketizer>(fmt_);

  backend_ = spark::net::make_dpdk_rx_backend();
  spark::net::RxBackendConfig cfg;
  cfg.pci_addr = pci_addr_.get();
  cfg.udp_port = static_cast<uint16_t>(udp_port_.get());
  cfg.manage_eal = manage_eal_.get();
  cfg.file_prefix = "spark_audio_rx";
  cfg.mcast_group = mcast_group_.get();
  cfg.src_ip = src_ip_.get();
  cfg.iface_ip = iface_ip_.get();
  backend_->init(cfg);
  HOLOSCAN_LOG_INFO("audio_rx started: RX {} udp:{} group={} {}ch/{}bit @{}ms", cfg.pci_addr,
                    cfg.udp_port, cfg.mcast_group.empty() ? "(none)" : cfg.mcast_group,
                    fmt_.channels, bit_depth_.get(), fmt_.packet_time_ms);
  poll_thread_ = std::thread(&AudioRxOp::poll_loop, this);
}

std::shared_ptr<std::vector<uint8_t>> AudioRxOp::next_buffer() {
  if (buf_ring_.empty()) {
    buf_ring_.resize(64);  // small, frequent packets — a deep ring absorbs poll/emit jitter
    for (auto& b : buf_ring_) b = std::make_shared<std::vector<uint8_t>>(fmt_.packet_payload_bytes());
  }
  auto b = buf_ring_[buf_idx_];
  buf_idx_ = (buf_idx_ + 1) % buf_ring_.size();
  return b;
}

void AudioRxOp::poll_loop() {
  constexpr size_t kMaxQ = 256;
  spark::net::RxPacket pkts[256];
  while (!stop_poll_.load(std::memory_order_relaxed)) {
    const uint16_t n = backend_->receive(pkts, 256);
    if (n == 0) continue;
    const uint64_t now = backend_->now_ns();
    for (uint16_t i = 0; i < n; ++i) {
      spark::st2110::AudioRxPacketInfo info;
      if (!depkt_->parse(pkts[i].payload, pkts[i].len, info)) continue;
      auto buf = next_buffer();
      const uint32_t copy = info.pcm_len < buf->size() ? info.pcm_len : (uint32_t)buf->size();
      std::memcpy(buf->data(), pkts[i].payload + info.data_offset, copy);
      spark::st2110::AudioBlock blk;
      blk.pcm = buf;
      blk.format = fmt_;
      blk.rtp_timestamp = info.rtp_timestamp;
      blk.capture_ts_ns = pkts[i].has_timestamp ? pkts[i].hw_timestamp_ns : now;
      ++packets_;
      {
        std::lock_guard<std::mutex> lk(q_mu_);
        if (q_.size() < kMaxQ) q_.push_back(std::move(blk));
        else ++dropped_;
      }
      q_cv_.notify_one();
    }
    backend_->release(pkts, n);
  }
}

void AudioRxOp::compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
                        holoscan::ExecutionContext&) {
  std::unique_lock<std::mutex> lk(q_mu_);
  if (!q_cv_.wait_for(lk, std::chrono::seconds(2), [&] { return !q_.empty() || stop_poll_.load(); }))
    return;
  if (q_.empty()) return;
  auto blk = std::move(q_.front());
  q_.pop_front();
  lk.unlock();
  op_output.emit(blk, "audio");
}

void AudioRxOp::stop() {
  stop_poll_.store(true);
  q_cv_.notify_all();
  if (poll_thread_.joinable()) poll_thread_.join();
  HOLOSCAN_LOG_INFO("audio_rx stopped: packets={} dropped={}", packets_, dropped_);
  if (backend_) { backend_->shutdown(); backend_.reset(); }
}

}  // namespace spark::ops
