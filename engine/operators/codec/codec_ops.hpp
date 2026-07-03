// UnpackOp / PackOp — bridge the network frame format (packed ST 2110-20 octets, host) and the GPU
// processing format (planar GpuFrame). They wrap the RFC 4175 CUDA codec (pixel_codec) so the graph
// reads:  st2110_rx -> [unpack] -> resize/frc -> [pack] -> st2110_tx.
//
// unpack: VideoFrame (host packed) -> GpuFrame (device planar 16u). pack: the inverse.
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "../resize/gpu_frame.hpp"
#include "../st2110_tx/st2110_format.hpp"

namespace spark::ops {

class UnpackOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(UnpackOp)
  UnpackOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;

 private:
  void ensure(uint32_t width, uint32_t height, bool ip10);
  void drain_inflight(bool wait);  // release RX buffers whose async GPU read has completed
  cudaStream_t stream_ = nullptr;  // own CUDA stream (H2D + unpack kernel), pipelined vs other ops
  uint8_t* dpacked_ = nullptr;  // device staging for the packed frame (copy path only)
  size_t dpacked_bytes_ = 0;
  bool zerocopy_ = false;  // GB10 coherent memory: kernel reads the RX host buffer directly (no H2D)
  std::vector<spark::gpu::GpuFramePtr> pool_;  // output ring
  size_t idx_ = 0;
  // Zero-copy lifetime: the unpack kernel reads the RX ring buffer ASYNCHRONOUSLY, but the RX reuses
  // any buffer whose use_count drops to 1 — so hold the shared_ptr here until the read completed.
  std::deque<std::pair<cudaEvent_t, std::shared_ptr<std::vector<uint8_t>>>> inflight_;
  std::vector<cudaEvent_t> ev_pool_;  // recycled completion events for inflight_
};

class PackOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PackOp)
  PackOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;

 private:
  void ensure(uint32_t width, uint32_t height);
  holoscan::Parameter<double> out_fps_;  // output RTP media rate (drives TX pacing); from the source SDP
  holoscan::Parameter<bool> ip10_;  // Blackmagic IP10 10:8 output (8-bit pgroups) instead of raw 10-bit
  cudaStream_t stream_ = nullptr;  // own CUDA stream (pack kernel + D2H), pipelined vs other ops
  uint8_t* dpacked_ = nullptr;  // device staging for the packed frame (copy path only)
  size_t dpacked_bytes_ = 0;
  bool zerocopy_ = false;  // GB10 coherent memory: kernel writes the TX host buffer directly (no D2H)
  spark::st2110::VideoFormat fmt_{};
  std::vector<std::shared_ptr<std::vector<uint8_t>>> host_pool_;  // output ring (host packed)
  size_t idx_ = 0;
  // unpack->pack pipeline latency (ns), from GpuFrame::t_ingest_ns; logged 1 Hz + a final summary.
  double last_live_s_ = 0;
  double last_stall_s_ = 0;  // rate limit for the "pipe stall" localization warn
  uint64_t lat_min_ = UINT64_MAX, lat_max_ = 0, lat_sum_ = 0, lat_n_ = 0;
};

}  // namespace spark::ops
