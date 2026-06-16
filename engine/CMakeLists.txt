# Spark Video Processor — engine (C++/CUDA Holoscan pipeline).
# Build against the source-built Holoscan SDK install tree (see docs/SETUP.md step 3):
#   T=~/holoscan-sdk/install-cu13-aarch64-dgpu
#   cmake -G Ninja -S engine -B engine/build -DCMAKE_PREFIX_PATH="$T"
#   cmake --build engine/build
#   ctest --test-dir engine/build                          # framing unit test (no NIC/root)
#   LD_LIBRARY_PATH="$T/lib" ./engine/build/spark_engine    # placeholder ping graph
#   sudo -E LD_LIBRARY_PATH="$T/lib" ./engine/build/st2110_tx_smoke   # gate-4 follow-on (needs CX-7)
cmake_minimum_required(VERSION 3.24)  # CMAKE_CUDA_ARCHITECTURES native
project(spark_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
# GB10 (Grace Blackwell) is compute capability 12.1 = sm_121. Set explicitly: enabling the CUDA
# language pre-defines CMAKE_CUDA_ARCHITECTURES to a conservative default (sm_75 here), whose PTX
# won't run on Blackwell ("unsupported toolchain"). Override the cache var for the build.
set(CMAKE_CUDA_ARCHITECTURES 121 CACHE STRING "CUDA archs (GB10 = 121)" FORCE)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(holoscan REQUIRED CONFIG)
# Bake the Holoscan lib dir into executable rpath so binaries run without LD_LIBRARY_PATH
# (holoscan_DIR = <install>/lib/cmake/holoscan, so ../.. is the lib dir holding gxf/rmm/etc.).
get_filename_component(HOLOSCAN_LIB_DIR "${holoscan_DIR}/../.." ABSOLUTE)
list(APPEND CMAKE_BUILD_RPATH "${HOLOSCAN_LIB_DIR}")

find_package(PkgConfig REQUIRED)
pkg_check_modules(DPDK REQUIRED IMPORTED_TARGET libdpdk)  # DPDK 23.11 mlx5 (tx_pp); PMDs dlopen'd by EAL
find_package(CUDAToolkit REQUIRED)  # cudart + NPP (resize); NPP host API, no CUDA language needed

# --- ST 2110-20 framing core (pure C++; no DPDK / Holoscan) ---------------------------------------
add_library(spark_st2110_core STATIC
  operators/st2110_tx/rtp_st2110.cpp
)
target_include_directories(spark_st2110_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# --- ST 2110-30 audio framing core (pure C++; no DPDK / Holoscan) ---------------------------------
add_library(spark_audio_core STATIC
  operators/audio/audio_st2110.cpp
)
target_include_directories(spark_audio_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# --- ST 2110 transport backends + shared EAL (all rte_* confined here) ----------------------------
add_library(spark_st2110_net STATIC
  operators/common/dpdk_eal.cpp
  operators/st2110_tx/dpdk_tx_backend.cpp
  operators/st2110_rx/dpdk_rx_backend.cpp
)
target_include_directories(spark_st2110_net PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spark_st2110_net PUBLIC PkgConfig::DPDK)

# --- GPU processing: resize (NPP); host-callable, no .cu / CUDA language --------------------------
add_library(spark_resize STATIC
  operators/resize/gpu_frame.cpp
  operators/resize/resize.cpp
)
target_include_directories(spark_resize PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spark_resize PUBLIC holoscan::core CUDA::cudart CUDA::nppig CUDA::nppc)

# --- RFC 4175 pixel codec (CUDA kernels) + unpack/pack operators ----------------------------------
add_library(spark_codec STATIC
  operators/codec/pixel_codec.cu
  operators/codec/codec_ops.cpp
)
target_include_directories(spark_codec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spark_codec PUBLIC holoscan::core spark_resize CUDA::cudart)

# --- FRC: NVOF optical flow + motion-compensated interpolation (CUDA) -----------------------------
# Raw NVOF (libnvidia-opticalflow, driver API) — NvOFFRUC isn't available on aarch64, so FRC is
# hand-rolled on the flow field. SDK headers are vendored under spike/third_party (M0).
find_library(NVOF_LIB nvidia-opticalflow PATHS /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu REQUIRED)
add_library(spark_frc STATIC
  operators/frc/nvof_flow.cpp
  operators/frc/frc_kernels.cu
  operators/frc/frc.cpp
)
target_include_directories(spark_frc
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../spike/third_party/NVIDIAOpticalFlowSDK)
target_link_libraries(spark_frc PUBLIC holoscan::core spark_resize CUDA::cudart CUDA::cuda_driver ${NVOF_LIB})

# --- Holoscan operators ---------------------------------------------------------------------------
add_library(spark_operators STATIC
  operators/st2110_tx/st2110_tx.cpp
  operators/st2110_rx/st2110_rx.cpp
  operators/test_pattern/test_pattern.cpp
)
target_include_directories(spark_operators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spark_operators PUBLIC holoscan::core spark_st2110_core spark_st2110_net)

# --- ST 2110-30 audio Holoscan operators (rx -> lip-sync delay -> tx) ------------------------------
add_library(spark_audio_operators STATIC
  operators/audio/audio_rx.cpp
  operators/audio/audio_delay.cpp
  operators/audio/audio_tx.cpp
)
target_include_directories(spark_audio_operators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spark_audio_operators PUBLIC holoscan::core spark_audio_core spark_st2110_net)

# --- placeholder engine (kept as Holoscan build/runtime validation; M1 skeleton) ------------------
add_executable(spark_engine src/spark_engine.cpp)
target_link_libraries(spark_engine
  PRIVATE
    holoscan::core
    holoscan::ops::ping_tx
    holoscan::ops::ping_rx
)

# --- ST 2110 TX / RX bring-up apps (gate-4 follow-on; two-process loopback) ------------------------
add_executable(st2110_tx_smoke apps/st2110_tx_smoke.cpp)
target_link_libraries(st2110_tx_smoke PRIVATE spark_operators holoscan::core)

add_executable(st2110_rx_smoke apps/st2110_rx_smoke.cpp)
target_link_libraries(st2110_rx_smoke PRIVATE spark_operators holoscan::core)

# One-process loopback (shared EAL across both ports; multi-thread scheduler)
add_executable(st2110_loopback apps/st2110_loopback.cpp)
target_link_libraries(st2110_loopback PRIVATE spark_operators holoscan::core)

# rx -> tx pass-through (shared EAL; frame forwarding) — the M1 first-deliverable graph
add_executable(st2110_passthrough apps/st2110_passthrough.cpp)
target_link_libraries(st2110_passthrough PRIVATE spark_operators holoscan::core)

# ST 2110-30 audio passthrough: audio_rx -> lip-sync delay -> audio_tx (shared EAL)
add_executable(audio_passthrough apps/audio_passthrough.cpp)
target_link_libraries(audio_passthrough PRIVATE spark_audio_operators holoscan::core)

# resize benchmark/bring-up (NPP; pure GPU, no NIC/root)
add_executable(resize_smoke apps/resize_smoke.cpp)
target_link_libraries(resize_smoke PRIVATE spark_resize holoscan::core)

# FRC validation (NVOF flow + motion-comp interpolation; pure GPU, no NIC/root)
add_executable(frc_smoke apps/frc_smoke.cpp)
target_link_libraries(frc_smoke PRIVATE spark_frc)

# full processing pipeline: rx -> unpack -> resize -> frc -> pack -> tx (shared EAL)
add_executable(st2110_pipeline apps/st2110_pipeline.cpp)
target_link_libraries(st2110_pipeline
  PRIVATE spark_operators spark_codec spark_resize spark_frc holoscan::core)

# --- tests (framing round-trip; pure CPU, runs without NIC/root) -----------------------------------
enable_testing()
add_executable(test_rtp_st2110 operators/st2110_tx/test/test_rtp_st2110.cpp)
target_link_libraries(test_rtp_st2110 PRIVATE spark_st2110_core)
add_test(NAME rtp_st2110 COMMAND test_rtp_st2110)

add_executable(test_pixel_codec operators/codec/test/test_pixel_codec.cpp)
target_link_libraries(test_pixel_codec PRIVATE spark_codec)
add_test(NAME pixel_codec COMMAND test_pixel_codec)

# ST 2110 multicast addressing (RFC 1112 MAC + IGMPv3 join); header-only, no DPDK
add_executable(test_net_addr operators/common/test/test_net_addr.cpp)
add_test(NAME net_addr COMMAND test_net_addr)

# ST 2110-30 audio framing round-trip + lip-sync delay (pure CPU, no NIC)
add_executable(test_audio_st2110 operators/audio/test/test_audio_st2110.cpp)
target_link_libraries(test_audio_st2110 PRIVATE spark_audio_core)
add_test(NAME audio_st2110 COMMAND test_audio_st2110)
