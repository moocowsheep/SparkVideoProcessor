# Spark Video Processor — control daemon (M4). Independent of the engine (no DPDK/Holoscan/CUDA):
# just gRPC + protobuf + libmicrohttpd. It spawns/manages the st2110_pipeline binary.
#   cmake -S control -B control/build && cmake --build control/build
#   ./control/build/spark_controld --pipeline ./engine/build/st2110_pipeline --web web
cmake_minimum_required(VERSION 3.20)
project(spark_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPCPP REQUIRED IMPORTED_TARGET grpc++)
pkg_check_modules(PROTOBUF REQUIRED IMPORTED_TARGET protobuf)
pkg_check_modules(MHD REQUIRED IMPORTED_TARGET libmicrohttpd)
find_program(PROTOC protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

# --- codegen: proto -> C++ (protobuf messages + gRPC service) -------------------------------------
set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../proto)
set(GEN ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${GEN})
set(PB_SRC ${GEN}/spark_control.pb.cc)
set(GRPC_SRC ${GEN}/spark_control.grpc.pb.cc)
add_custom_command(
  OUTPUT ${PB_SRC} ${GEN}/spark_control.pb.h ${GRPC_SRC} ${GEN}/spark_control.grpc.pb.h
  COMMAND ${PROTOC} -I ${PROTO_DIR}
          --cpp_out=${GEN} --grpc_out=${GEN} --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
          ${PROTO_DIR}/spark_control.proto
  DEPENDS ${PROTO_DIR}/spark_control.proto
  COMMENT "protoc spark_control.proto (cpp + grpc)")

add_executable(spark_controld daemon.cpp ${PB_SRC} ${GRPC_SRC})
target_include_directories(spark_controld PRIVATE ${GEN})
target_link_libraries(spark_controld PRIVATE PkgConfig::GRPCPP PkgConfig::PROTOBUF PkgConfig::MHD)

# --- NMOS node (IS-04/IS-05 via nmos-cpp) — optional; build with -DSPARK_WITH_NMOS=ON ------------
# Requires `bash deploy/nmos.sh --packages && bash deploy/nmos.sh --build` (installs nmos-cpp under
# third_party/nmos-cpp/install). Heavy deps (cpprestsdk/Boost/Avahi) stay isolated to this target.
option(SPARK_WITH_NMOS "Build the NMOS node (needs third_party/nmos-cpp/install)" OFF)
if(SPARK_WITH_NMOS)
  set(NMOS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/nmos-cpp)
  # our cpprestsdk shim (Ubuntu libcpprest-dev ships no CMake config) + the installed nmos-cpp package
  set(cpprestsdk_DIR ${NMOS_ROOT}/cmake-shims/cpprestsdk)
  list(APPEND CMAKE_PREFIX_PATH ${NMOS_ROOT}/install)
  find_package(nmos-cpp REQUIRED)
  add_executable(spark_nmos_node nmos/spark_node.cpp)
  target_link_libraries(spark_nmos_node PRIVATE nmos-cpp::nmos-cpp)
endif()
