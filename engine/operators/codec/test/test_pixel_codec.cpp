// Round-trip test for the RFC 4175 pixel codec: packed -> planar -> packed must be byte-identical
// (the 5-byte pgroup <-> 4x10-bit mapping is a bijection). Needs a GPU; no NIC/root.
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "operators/codec/pixel_codec.hpp"

int main() {
  const uint32_t w = 1920, h = 1080;
  const size_t octets = static_cast<size_t>(w / 2) * h * 5;
  std::vector<uint8_t> host(octets), host2(octets, 0);
  for (size_t i = 0; i < octets; ++i) host[i] = static_cast<uint8_t>((i * 131u + 7u) & 0xff);

  uint8_t *dp = nullptr, *dp2 = nullptr;
  uint16_t *y = nullptr, *cb = nullptr, *cr = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&dp), octets);
  cudaMalloc(reinterpret_cast<void**>(&dp2), octets);
  cudaMalloc(reinterpret_cast<void**>(&y), static_cast<size_t>(w) * h * 2);
  cudaMalloc(reinterpret_cast<void**>(&cb), static_cast<size_t>(w / 2) * h * 2);
  cudaMalloc(reinterpret_cast<void**>(&cr), static_cast<size_t>(w / 2) * h * 2);

  cudaMemcpy(dp, host.data(), octets, cudaMemcpyHostToDevice);
  spark::codec::unpack_422_10(dp, y, cb, cr, w, h, 0);
  spark::codec::pack_422_10(dp2, y, cb, cr, w, h, 0);
  cudaDeviceSynchronize();
  const cudaError_t e = cudaGetLastError();
  cudaMemcpy(host2.data(), dp2, octets, cudaMemcpyDeviceToHost);

  size_t diffs = 0;
  for (size_t i = 0; i < octets; ++i)
    if (host[i] != host2[i]) ++diffs;

  std::printf("cuda=%s octets=%zu diffs=%zu -> %s\n", cudaGetErrorString(e), octets, diffs,
              diffs == 0 ? "[PASS] unpack->pack round-trip byte-identical" : "[FAIL]");
  return (e == cudaSuccess && diffs == 0) ? 0 : 1;
}
