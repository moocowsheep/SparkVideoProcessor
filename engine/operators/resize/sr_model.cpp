// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

#include "sr_model.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "sr_weights_data.hpp"

namespace spark::sr {
namespace {

class Cursor {
 public:
  Cursor(const uint8_t* data, size_t size, const std::string& name)
      : data_(data), size_(size), name_(name) {}

  uint32_t u32() {
    uint32_t v = 0;
    take(&v, sizeof(v));
    return v;
  }

  void floats(std::vector<float>& out, size_t count) {
    out.resize(count);
    take(out.data(), count * sizeof(float));
  }

  bool exhausted() const { return off_ == size_; }

 private:
  void take(void* dst, size_t n) {
    if (off_ + n > size_)
      throw std::runtime_error("sr model '" + name_ + "': truncated blob");
    std::memcpy(dst, data_ + off_, n);
    off_ += n;
  }

  const uint8_t* data_;
  size_t size_;
  size_t off_ = 0;
  const std::string& name_;
};

}  // namespace

bool is_sr_interp(const std::string& interp) {
  return interp == "fsrcnn" || interp == "fsrcnn-s" || interp == "espcn";
}

NetDef parse_blob(const uint8_t* data, size_t size, const std::string& name) {
  Cursor c(data, size, name);
  NetDef net;
  net.name = name;
  const uint32_t magic = c.u32();
  if (magic != 0x31575253u)  // "SRW1" little-endian
    throw std::runtime_error("sr model '" + name + "': bad magic");
  const uint32_t version = c.u32();
  if (version != 1)
    throw std::runtime_error("sr model '" + name + "': unsupported version " +
                             std::to_string(version));
  net.scale = c.u32();
  const uint32_t n_layers = c.u32();
  if (net.scale != 2)
    throw std::runtime_error("sr model '" + name + "': only scale 2 is supported");
  if (n_layers == 0 || n_layers > 64)
    throw std::runtime_error("sr model '" + name + "': implausible layer count");

  uint32_t prev_out = 1;  // luma in
  for (uint32_t i = 0; i < n_layers; ++i) {
    Layer l;
    l.kind = static_cast<Kind>(c.u32());
    l.k = c.u32();
    l.cin = c.u32();
    l.cout = c.u32();
    l.act = static_cast<Act>(c.u32());
    const bool has_bias = c.u32() != 0;
    if (l.cin != prev_out)
      throw std::runtime_error("sr model '" + name + "': channel mismatch at layer " +
                               std::to_string(i));
    if (l.kind == Kind::conv) {
      if (l.k == 0 || l.k > 9 || (l.k % 2) == 0 || l.cin > 256 || l.cout > 256 || l.cout == 0)
        throw std::runtime_error("sr model '" + name + "': unsupported conv shape at layer " +
                                 std::to_string(i));
      c.floats(l.w, static_cast<size_t>(l.cout) * l.cin * l.k * l.k);
      if (has_bias) c.floats(l.bias, l.cout);
      if (l.act == Act::prelu) c.floats(l.alpha, l.cout);
    } else if (l.kind == Kind::shuffle2) {
      if (i + 1 != n_layers || l.cin != 4 || l.cout != 1)
        throw std::runtime_error("sr model '" + name +
                                 "': shuffle2 must terminate the net with cin=4");
      if (has_bias || l.act == Act::prelu)
        throw std::runtime_error("sr model '" + name + "': shuffle2 carries no weights");
    } else {
      throw std::runtime_error("sr model '" + name + "': unknown layer kind");
    }
    prev_out = l.cout;
    net.layers.push_back(std::move(l));
  }
  if (net.layers.back().kind != Kind::shuffle2)
    throw std::runtime_error("sr model '" + name + "': net is not shuffle2-terminated");
  if (!c.exhausted())
    throw std::runtime_error("sr model '" + name + "': trailing bytes");
  return net;
}

NetDef load_embedded(const std::string& name) {
  size_t count = 0;
  const EmbeddedModel* models = embedded_models(&count);
  for (size_t i = 0; i < count; ++i)
    if (name == models[i].name) return parse_blob(models[i].data, models[i].size, name);
  throw std::runtime_error("sr model '" + name + "': no embedded weights");
}

NetDef load_file(const std::string& path, const std::string& name) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("sr model '" + name + "': cannot open " + path);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
  return parse_blob(bytes.data(), bytes.size(), name);
}

}  // namespace spark::sr
