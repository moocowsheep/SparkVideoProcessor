// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// SRW1 super-resolution model definitions: a shuffle2-terminated stack of SAME/stride-1 convs on
// the luma plane (the shape both embedded model families — FSRCNN and ESPCN — reduce to). Pure
// C++ (no CUDA/Holoscan) so the loader is testable anywhere; sr_net.cu compiles a NetDef into
// fused GPU kernels.
//
// Blob layout (little-endian u32 header words, then raw f32 payload per layer):
//   magic 'SRW1', version 1, scale, n_layers,
//   per layer: kind(0 conv | 1 shuffle2), k, cin, cout, act(0 none|1 prelu|2 relu|3 tanh), has_bias,
//   conv payload: weights OIHW [cout][cin][k][k], bias[cout] if has_bias, alpha[cout] if prelu.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace spark::sr {

enum class Kind : uint32_t { conv = 0, shuffle2 = 1 };
enum class Act : uint32_t { none = 0, prelu = 1, relu = 2, tanh = 3 };

struct Layer {
  Kind kind = Kind::conv;
  uint32_t k = 0;  // conv kernel size (odd); 0 for shuffle2
  uint32_t cin = 0;
  uint32_t cout = 0;
  Act act = Act::none;           // shuffle2 may carry a post-shuffle activation (espcn: tanh)
  std::vector<float> w;          // conv: OIHW [cout][cin][k][k]
  std::vector<float> bias;       // empty = no bias
  std::vector<float> alpha;      // prelu per-channel slopes
};

struct NetDef {
  std::string name;
  uint32_t scale = 0;  // only 2 is supported by the engine
  std::vector<Layer> layers;
};

// True for interp values that select an AI super-resolution model ("fsrcnn" | "fsrcnn-s" | "espcn").
bool is_sr_interp(const std::string& interp);

// Parse an SRW1 blob. Throws std::runtime_error on malformed/truncated data.
NetDef parse_blob(const uint8_t* data, size_t size, const std::string& name);

// Embedded model by selection key. Throws std::runtime_error if the key is unknown.
NetDef load_embedded(const std::string& name);

// SPARK_SR_WEIGHTS escape hatch: load a (possibly retrained) SRW1 blob from disk.
NetDef load_file(const std::string& path, const std::string& name);

}  // namespace spark::sr
