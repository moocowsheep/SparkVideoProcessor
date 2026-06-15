#include "test_pattern.hpp"

namespace spark::ops {

void TestPatternOp::setup(holoscan::OperatorSpec& spec) {
  spec.output<spark::st2110::VideoFrame>("frame");
  spec.param(profile_, "profile", "Video profile", "1080p | 2160p", std::string("1080p"));
}

void TestPatternOp::start() {
  fmt_ = (profile_.get() == "2160p") ? spark::st2110::profile_2160p()
                                     : spark::st2110::profile_1080p();
  frame_interval_ns_ = static_cast<uint64_t>(1e9 / fmt_.fps);
  base_ts_ns_ = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());

  // Fill a deterministic test pattern once (packed 4:2:2 10-bit octets). Content is a per-line
  // gradient — enough for a receiver to sanity-check; it is not a colorimetrically correct bar.
  buffer_ = std::make_shared<std::vector<uint8_t>>(fmt_.octets_per_frame());
  const uint32_t opl = fmt_.octets_per_line();
  for (uint32_t line = 0; line < fmt_.height; ++line) {
    uint8_t* row = buffer_->data() + static_cast<uint64_t>(line) * opl;
    const uint8_t v = static_cast<uint8_t>((line * 255u) / (fmt_.height ? fmt_.height : 1));
    for (uint32_t o = 0; o < opl; ++o) row[o] = static_cast<uint8_t>(v ^ (o & 0xff));
  }
  HOLOSCAN_LOG_INFO("test_pattern: {}x{} profile={} ({} octets/frame)", fmt_.width, fmt_.height,
                    profile_.get(), fmt_.octets_per_frame());
}

void TestPatternOp::compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
                            holoscan::ExecutionContext&) {
  spark::st2110::VideoFrame frame;
  frame.data = buffer_;  // reused buffer; St2110TxOp copies octets into packets
  frame.format = fmt_;
  frame.frame_number = frame_number_;
  frame.capture_ts_ns = base_ts_ns_ + frame_number_ * frame_interval_ns_;
  ++frame_number_;
  op_output.emit(frame, "frame");
}

}  // namespace spark::ops
