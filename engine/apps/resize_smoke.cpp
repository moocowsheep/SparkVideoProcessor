// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Resize bring-up/benchmark: test_gpu_source -> resize (NPP) -> sink. Pure GPU — no NIC, no root.
//   ./engine/build/resize_smoke                       # 1080p -> 2160p cubic, 300 frames
//   SPARK_PROFILE=1080p SPARK_OUT_W=3840 SPARK_OUT_H=2160 SPARK_INTERP=lanczos ./engine/build/resize_smoke
#include <cstdlib>

#include "runtime/runtime.hpp"

#include "operators/resize/resize.hpp"

namespace spark {

class ResizeSmoke : public spark::rt::Application {
 public:
  void compose() override {
    using namespace spark::rt;
    auto env = [](const char* k, const char* d) {
      const char* v = std::getenv(k);
      return std::string(v ? v : d);
    };
    const std::string in_profile = env("SPARK_PROFILE", "1080p");
    const std::string interp = env("SPARK_INTERP", "cubic");
    const int64_t frames = std::atoll(env("SPARK_FRAMES", "300").c_str());
    const uint32_t ow = static_cast<uint32_t>(std::atoll(env("SPARK_OUT_W", "3840").c_str()));
    const uint32_t oh = static_cast<uint32_t>(std::atoll(env("SPARK_OUT_H", "2160").c_str()));

    auto src = make_operator<ops::TestGpuSourceOp>("src", Arg("profile", in_profile),
                                                   make_condition<CountCondition>(frames));
    auto rz = make_operator<ops::ResizeOp>("resize", Arg("out_width", ow), Arg("out_height", oh),
                                           Arg("interp", interp), Arg("measure", true));
    auto sink = make_operator<ops::GpuFrameSinkOp>("sink");
    add_flow(src, rz);
    add_flow(rz, sink);
  }
};

}  // namespace spark

int main() {
  SPARK_LOG_INFO("Resize smoke: test_gpu_source -> resize (NPP) -> sink.");
  auto app = spark::rt::make_application<spark::ResizeSmoke>();
  app->run();
  return 0;
}
