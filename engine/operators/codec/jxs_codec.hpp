// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// JxsDecodeOp / JxsEncodeOp — the ST 2110-22 (JPEG XS) counterparts of UnpackOp / PackOp.
//
// They sit at exactly the same place in the graph and speak exactly the same message types, so the
// pipeline app swaps them in per side without touching anything between them:
//
//   raw in :  st2110_rx -> [unpack]     -> filters -> [pack]       -> st2110_tx  (RFC 4175)
//   jxs in :  st2110_rx -> [jxs_decode] -> filters -> [jxs_encode] -> st2110_tx  (RFC 9134)
//
// and either side can be chosen independently — decode JPEG XS in and send uncompressed out, or the
// reverse, or transcode both ways. The codec is MooCUDAJXS (CUDA JPEG XS, zero-copy device planes),
// driven through its asynchronous frame API: one persistent handle per operator owns all workspace,
// and frame submission allocates nothing and — on the decode side — never synchronizes.
//
// Format: GpuFrame is planar 10-bit YCbCr 4:2:2 in uint16, which is MooCUDAJXS's YUV422P at
// bit_depth 10 with no conversion — Y is (w x h), Cb/Cr are (w/2 x h), all tightly pitched.
//
// LICENSING: JPEG XS (ISO/IEC 21122) is a standardized codec that may be covered by patents. You are
// solely responsible for determining if your use of jpeg-xs requires any additional licenses. The
// developer of this project is not responsible for obtaining any such licenses, nor liable for any
// licensing fees due, in connection with your use of jpeg-xs. See NOTICE.
#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/holoscan.hpp>

#include "../resize/gpu_frame.hpp"
#include "../st2110_tx/st2110_format.hpp"

namespace spark::ops {

// True when the engine was built against MooCUDAJXS. False builds still expose both operators (the
// app wires them the same way); they fail loudly at start() instead of silently sending nothing.
bool jxs_available();

// Codec build/version identification for the log line and telemetry ("unavailable" when not built).
const char* jxs_version();

// ---- JxsDecodeOp: VideoFrame (host JPEG XS codestream) -> GpuFrame (device planar 10-bit) ----
class JxsDecodeOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(JxsDecodeOp)
  JxsDecodeOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;

 private:
  // Release finished decodes, recycling their slots and the RX buffers they pinned. wait=true (stop
  // path) blocks on stragglers so no buffer outlives the operator while kernels still read it.
  void drain_inflight(bool wait);
  // Worst-case precinct columns (PIH Cw) the decoder plans workspace for. A codestream with more
  // columns than this is rejected rather than decoded, so it bounds interop, not just memory.
  holoscan::Parameter<uint32_t> max_precinct_columns_;
  struct Impl;
  // Created in start(): Holoscan builds operators through the HOLOSCAN_OPERATOR_FORWARD_ARGS
  // constructor, which never runs a body of ours, so this cannot be a constructor-time allocation.
  // shared_ptr, not unique_ptr: that same forwarding constructor is instantiated in the pipeline
  // app, where Impl is incomplete, and unique_ptr's deleter would need the full type there.
  // shared_ptr type-erases its deleter at make_shared time — inside this operator's own .cpp.
  std::shared_ptr<Impl> impl_;
};

// ---- JxsEncodeOp: GpuFrame (device planar 10-bit) -> VideoFrame (host JPEG XS codestream) ----
class JxsEncodeOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(JxsEncodeOp)
  JxsEncodeOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;

 private:
  holoscan::Parameter<double> out_fps_;        // output RTP media rate (drives TX pacing)
  holoscan::Parameter<double> bits_per_pixel_; // constant-rate target; sets the codestream size
  // JPEG XS wavelet decomposition + slice height. The standard defaults (5/2/16) are what the codec
  // is exercised with; exposed because a receiver may require a specific slice geometry.
  holoscan::Parameter<uint32_t> horizontal_decomposition_;
  holoscan::Parameter<uint32_t> vertical_decomposition_;
  holoscan::Parameter<uint32_t> precincts_per_slice_;
  // PIH Cw. 0 = one full-width precinct column.
  holoscan::Parameter<uint32_t> precinct_width_;
  // PIH Ppih/Plev, written verbatim into the codestream header. 0 = unrestricted, which is the
  // honest default: MooCUDAJXS does not claim ISO Part 4 conformance, so stamping a standardized
  // profile/level would assert something unverified. Set them when a receiver demands them.
  holoscan::Parameter<uint32_t> profile_;
  holoscan::Parameter<uint32_t> level_;
  struct Impl;
  std::shared_ptr<Impl> impl_;  // created in start() — see the note on JxsDecodeOp::impl_
};

}  // namespace spark::ops
