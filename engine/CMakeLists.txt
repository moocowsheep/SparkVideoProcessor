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
find_package(PkgConfig REQUIRED)
pkg_check_modules(DPDK REQUIRED IMPORTED_TARGET libdpdk)  # DPDK 23.11 mlx5 (tx_pp); PMDs dlopen'd by EAL

# --- ST 2110-20 framing core (pure C++; no DPDK / Holoscan) ---------------------------------------
add_library(spark_st2110_core STATIC
  operators/st2110_tx/rtp_st2110.cpp
)
target_include_directories(spark_st2110_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# --- ST 2110 transport backend (all rte_* confined here) ------------------------------------------
add_library(spark_st2110_net STATIC
  operators/st2110_tx/dpdk_tx_backend.cpp
)
target_include_directories(spark_st2110_net PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spark_st2110_net PUBLIC PkgConfig::DPDK)

# --- Holoscan operators ---------------------------------------------------------------------------
add_library(spark_operators STATIC
  operators/st2110_tx/st2110_tx.cpp
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

# --- ST 2110 TX bring-up app (gate-4 follow-on) ---------------------------------------------------
add_executable(st2110_tx_smoke apps/st2110_tx_smoke.cpp)
target_link_libraries(st2110_tx_smoke PRIVATE spark_operators holoscan::core)

# --- tests (framing round-trip; pure CPU, runs without NIC/root) -----------------------------------
enable_testing()
add_executable(test_rtp_st2110 operators/st2110_tx/test/test_rtp_st2110.cpp)
target_link_libraries(test_rtp_st2110 PRIVATE spark_st2110_core)
add_test(NAME rtp_st2110 COMMAND test_rtp_st2110)
