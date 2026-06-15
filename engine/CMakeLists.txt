# Moo Video Processor — engine (C++/CUDA Holoscan pipeline).
# Build against the source-built Holoscan SDK install tree (see docs/SETUP.md step 3):
#   T=~/holoscan-sdk/install-cu13-aarch64-dgpu
#   cmake -G Ninja -S engine -B engine/build -DCMAKE_PREFIX_PATH="$T"
#   cmake --build engine/build
#   LD_LIBRARY_PATH="$T/lib" ./engine/build/moo_engine   # placeholder ping graph (M1 skeleton)
cmake_minimum_required(VERSION 3.20)
project(moo_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(holoscan REQUIRED CONFIG)

add_executable(moo_engine
  src/moo_engine.cpp
)

target_link_libraries(moo_engine
  PRIVATE
    holoscan::core
    holoscan::ops::ping_tx   # placeholder ops (M1 build validation); replaced by real operators
    holoscan::ops::ping_rx
)
