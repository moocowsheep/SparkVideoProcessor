// UnpackOp / PackOp — bridge the network frame format (packed ST 2110-20 octets, host) and the GPU
// processing format (planar GpuFrame). They wrap the RFC 4175 CUDA codec (pixel_codec) so the graph
// reads:  st2110_rx -> [unpack] -> resize/frc -> [pack] -> st2110_tx.
//
// unpack: VideoFrame (host packed) -> GpuFrame (device planar 16u). pack: the inverse.
#pragma once

#include <cstdint>
#include <memory>
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
  void ensure(uint32_t width, uint32_t height);
  uint8_t* dpacked_ = nullptr;  // device staging for the packed frame
  size_t dpacked_bytes_ = 0;
  std::vector<spark::gpu::GpuFramePtr> pool_;  // output ring
  size_t idx_ = 0;
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
  uint8_t* dpacked_ = nullptr;  // device staging for the packed frame
  size_t dpacked_bytes_ = 0;
  spark::st2110::VideoFormat fmt_{};
  std::vector<std::shared_ptr<std::vector<uint8_t>>> host_pool_;  // output ring (host packed)
  size_t idx_ = 0;
};

}  // namespace spark::ops
