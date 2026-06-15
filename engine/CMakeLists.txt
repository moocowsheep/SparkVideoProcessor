# Spark Video Processor — engine (C++/CUDA Holoscan pipeline).
# Build against the source-built Holoscan SDK install tree (see docs/SETUP.md step 3):
#   T=~/holoscan-sdk/install-cu13-aarch64-dgpu
#   cmake -G Ninja -S engine -B engine/build -DCMAKE_PREFIX_PATH="$T"
#   cmake --build engine/build
#   ctest --test-dir engine/build                          # framing unit test (no NIC/root)
#   LD_LIBRARY_PATH="$T/lib" ./engine/build/spark_engine    # placeholder ping graph
#   sudo -E LD_LIBRARY_PATH="$T/lib" ./engine/build/st2110_tx_smoke   # gate-4 follow-on (needs CX-7)
cmake_minimum_required(VERSION 3.20)
project(spark_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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

# --- Holoscan operators ---------------------------------------------------------------------------
add_library(spark_operators STATIC
  operators/st2110_tx/st2110_tx.cpp
  operators/st2110_rx/st2110_rx.cpp
  operators/test_pattern/test_pattern.cpp
)
target_include_directories(spark_operators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spark_operators PUBLIC holoscan::core spark_st2110_core spark_st2110_net)

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

# resize benchmark/bring-up (NPP; pure GPU, no NIC/root)
add_executable(resize_smoke apps/resize_smoke.cpp)
target_link_libraries(resize_smoke PRIVATE spark_resize holoscan::core)

# --- tests (framing round-trip; pure CPU, runs without NIC/root) -----------------------------------
enable_testing()
add_executable(test_rtp_st2110 operators/st2110_tx/test/test_rtp_st2110.cpp)
target_link_libraries(test_rtp_st2110 PRIVATE spark_st2110_core)
add_test(NAME rtp_st2110 COMMAND test_rtp_st2110)
