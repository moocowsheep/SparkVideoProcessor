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

# --- JPEG XS codec (MooCUDAJXS) -------------------------------------------------------------------
# ST 2110-22 encode/decode is OPTIONAL: the codec lives in its own repository and is not vendored
# here. Point -DSPARK_JXS_DIR at a checkout, or leave it beside this repo / under third_party and it
# is found automatically. Without it the engine still builds and runs — the JPEG XS operators exist
# but refuse to start, so only a run that actually asked for JPEG XS is affected.
#
# LICENSING: JPEG XS (ISO/IEC 21122) is a standardized codec that may be covered by patents. You are
# solely responsible for determining if your use of jpeg-xs requires any additional licenses. The
# developer of this project is not responsible for obtaining any such licenses, nor liable for any
# licensing fees due, in connection with your use of jpeg-xs. See NOTICE.
set(SPARK_JXS_DIR "" CACHE PATH "MooCUDAJXS source directory (JPEG XS codec; empty = auto-detect)")
set(_spark_jxs_candidates
  "${SPARK_JXS_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/MooCUDAJXS"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../MooCUDAJXS"     # sibling checkout beside this repo
)
set(SPARK_JXS_SOURCE_DIR "")
foreach(_c IN LISTS _spark_jxs_candidates)
  if(_c AND EXISTS "${_c}/include/moocudajxs/moocudajxs.h")
    get_filename_component(SPARK_JXS_SOURCE_DIR "${_c}" ABSOLUTE)
    break()
  endif()
endforeach()
unset(_spark_jxs_candidates)

if(SPARK_JXS_SOURCE_DIR)
  message(STATUS "JPEG XS: MooCUDAJXS at ${SPARK_JXS_SOURCE_DIR}")
  # Build only the library: its tests and interop/live-viewer tools are its own repo's business.
  set(MOO_JXS_BUILD_TESTS OFF)
  set(MOO_JXS_BUILD_INTEROP_DRIVER OFF)
  # Its default is a wide fat binary (sm_75..sm_90); the engine targets GB10 only, and a fat build
  # of the whole codec is minutes of nvcc for architectures this box cannot run.
  set(MOO_JXS_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
  add_subdirectory("${SPARK_JXS_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/moocudajxs")
  # The codec is built with CUDA_SEPARABLE_COMPILATION, so its device symbols must be resolved into
  # the static library itself — the consuming targets here are plain C++ and never device-link.
  set_target_properties(moocudajxs PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
else()
  message(STATUS "JPEG XS: MooCUDAJXS NOT found — building without ST 2110-22 support "
                 "(set -DSPARK_JXS_DIR=/path/to/MooCUDAJXS to enable)")
endif()

# --- ST 2110-20 framing core (pure C++; no DPDK / Holoscan) ---------------------------------------
# Also carries the ST 2110-22 / RFC 9134 JPEG XS framing: same "pure, unit-testable wire format"
# role, and the RX/TX operators need both packetizers side by side to switch codec per direction.
add_library(spark_st2110_core STATIC
  operators/st2110_tx/rtp_st2110.cpp
  operators/st2110_tx/rtp_jxs.cpp
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
  operators/common/dpdk_port.cpp
  operators/st2110_tx/dpdk_tx_backend.cpp
  operators/st2110_rx/dpdk_rx_backend.cpp
)
target_include_directories(spark_st2110_net PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spark_st2110_net PUBLIC PkgConfig::DPDK)

# --- AI super-resolution: fused CUDA executor + embedded pre-trained x2 weights -------------------
# (weights .cpp is GENERATED by tools/sr_weights/convert.py; no Holoscan/NPP deps so tests run bare)
add_library(spark_sr STATIC
  operators/resize/sr_model.cpp
  operators/resize/sr_net.cu
  operators/resize/sr_weights_data.cpp
)
target_include_directories(spark_sr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spark_sr PUBLIC CUDA::cudart)

# --- GPU processing: resize (NPP + AI SR); resize.cpp itself is host-only ---------------------------
add_library(spark_resize STATIC
  operators/resize/gpu_frame.cpp
  operators/resize/resize.cpp
)
target_include_directories(spark_resize PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spark_resize PUBLIC holoscan::core CUDA::cudart CUDA::nppig CUDA::nppc spark_sr)

# --- RFC 4175 pixel codec (CUDA kernels) + unpack/pack operators ----------------------------------
add_library(spark_codec STATIC
  operators/codec/pixel_codec.cu
  operators/codec/ip10_codec.cu
  operators/codec/codec_ops.cpp
)
target_include_directories(spark_codec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spark_codec PUBLIC holoscan::core spark_resize CUDA::cudart)

# --- ST 2110-22 JPEG XS encode/decode operators (MooCUDAJXS) --------------------------------------
# Always built so the pipeline app can wire the operators unconditionally; SPARK_HAVE_JXS decides
# whether they carry a real codec or fail at start() with an actionable message.
add_library(spark_jxs STATIC
  operators/codec/jxs_codec.cpp
)
target_include_directories(spark_jxs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spark_jxs PUBLIC holoscan::core spark_resize spark_st2110_core CUDA::cudart)
if(SPARK_JXS_SOURCE_DIR)
  target_link_libraries(spark_jxs PUBLIC MooCUDAJXS::moocudajxs)
  target_compile_definitions(spark_jxs PUBLIC SPARK_HAVE_JXS)
endif()

# --- video filters: proc amp + unsharp sharpen (CUDA) — modular GpuFrame chain stages -------------
add_library(spark_filters STATIC
  operators/filters/filter_kernels.cu
  operators/filters/filters.cpp
)
target_include_directories(spark_filters PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spark_filters PUBLIC holoscan::core spark_resize CUDA::cudart)

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

# full processing pipeline: rx -> unpack -> [filter chain] -> pack -> tx (shared EAL)
add_executable(st2110_pipeline apps/st2110_pipeline.cpp)
target_link_libraries(st2110_pipeline
  PRIVATE spark_operators spark_codec spark_jxs spark_resize spark_frc spark_filters holoscan::core)

# --- tests (framing round-trip; pure CPU, runs without NIC/root) -----------------------------------
enable_testing()
add_executable(test_rtp_st2110 operators/st2110_tx/test/test_rtp_st2110.cpp)
target_link_libraries(test_rtp_st2110 PRIVATE spark_st2110_core)
add_test(NAME rtp_st2110 COMMAND test_rtp_st2110)

# ST 2110-22 / RFC 9134 JPEG XS framing round-trip (pure CPU, no NIC/root/codec)
add_executable(test_rtp_jxs operators/st2110_tx/test/test_rtp_jxs.cpp)
target_link_libraries(test_rtp_jxs PRIVATE spark_st2110_core)
add_test(NAME rtp_jxs COMMAND test_rtp_jxs)

add_executable(test_pixel_codec operators/codec/test/test_pixel_codec.cpp)
target_link_libraries(test_pixel_codec PRIVATE spark_codec)
add_test(NAME pixel_codec COMMAND test_pixel_codec)

# Blackmagic IP10 (10:8) codec reference — pure CPU (no GPU), so build the CPU source straight in.
add_executable(test_ip10_codec
  operators/codec/test/test_ip10_codec.cpp
  operators/codec/ip10_codec.cpp)
target_include_directories(test_ip10_codec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME ip10_codec COMMAND test_ip10_codec)

# ST 2110 multicast addressing (RFC 1112 MAC + IGMPv3 join); header-only, no DPDK
add_executable(test_net_addr operators/common/test/test_net_addr.cpp)
add_test(NAME net_addr COMMAND test_net_addr)

# Shared-port queue map (plan_queues): the RX+TX-on-one-port layout; header-only, no DPDK
add_executable(test_dpdk_port operators/common/test/test_dpdk_port.cpp)
target_include_directories(test_dpdk_port PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME dpdk_port COMMAND test_dpdk_port)

# RTP timestamp unwrap (absolute capture time — the fixed-latency + lip-sync basis); header-only
add_executable(test_rtp_time operators/common/test/test_rtp_time.cpp)
target_include_directories(test_rtp_time PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME rtp_time COMMAND test_rtp_time)

# FRC uniform-grid tick/phase math (SPARK_FRC=3); header-only, no CUDA/Holoscan
add_executable(test_frc_grid operators/frc/test/test_frc_grid.cpp)
add_test(NAME frc_grid COMMAND test_frc_grid)

# video filter kernels (proc amp / unsharp) vs CPU references — GPU, no NIC/root/Holoscan
add_executable(test_filters
  operators/filters/test/test_filters.cpp
  operators/filters/filter_kernels.cu)
target_include_directories(test_filters PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_filters PRIVATE CUDA::cudart)
add_test(NAME filters COMMAND test_filters)

# ST 2110-30 audio framing round-trip + lip-sync delay (pure CPU, no NIC)
add_executable(test_audio_st2110 operators/audio/test/test_audio_st2110.cpp)
target_link_libraries(test_audio_st2110 PRIVATE spark_audio_core)
add_test(NAME audio_st2110 COMMAND test_audio_st2110)

# AI super-resolution: GPU kernels vs a CPU reference forward of the same embedded weights, plus
# an SR-beats-bilinear PSNR sanity check — GPU, no NIC/root/Holoscan
add_executable(test_sr_net
  operators/resize/test/test_sr_net.cpp
  operators/resize/sr_model.cpp
  operators/resize/sr_net.cu
  operators/resize/sr_weights_data.cpp)
target_include_directories(test_sr_net PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_sr_net PRIVATE CUDA::cudart)
add_test(NAME sr_net COMMAND test_sr_net)
