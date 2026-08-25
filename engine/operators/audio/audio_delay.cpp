// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

#include "audio_delay.hpp"

#include <cmath>

namespace spark::ops {

void AudioDelayOp::setup(spark::rt::OperatorSpec& spec) {
  spec.input<spark::st2110::AudioBlock>("in");
  spec.output<spark::st2110::AudioBlock>("out");
  spec.param(delay_ms_, "delay_ms", "Lip-sync delay",
             "audio delay in ms; set >= the video processing latency", 0.0);
}

void AudioDelayOp::compute(spark::rt::InputContext& op_input, spark::rt::OutputContext& op_output,
                           spark::rt::ExecutionContext&) {
  auto maybe = op_input.receive<spark::st2110::AudioBlock>("in");
  if (!maybe) return;
  auto& blk = maybe.value();

  if (!line_) {  // size the delay (in packets) from the first block's packet time
    const double ptime = blk.format.packet_time_ms > 0 ? blk.format.packet_time_ms : 1.0;
    const size_t blocks = static_cast<size_t>(std::lround(delay_ms_.get() / ptime));
    line_ = std::make_unique<spark::st2110::AudioDelayLine>(blocks);
    SPARK_LOG_INFO("audio_delay: {} ms -> {} packets of lip-sync delay", delay_ms_.get(), blocks);
  }

  spark::st2110::AudioBlock out;
  if (line_->push(std::move(blk), out)) op_output.emit(out, "out");
}

}  // namespace spark::ops
