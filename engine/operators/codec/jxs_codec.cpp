// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

#include "jxs_codec.hpp"

#include <chrono>
#include <cstdlib>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../pipeline_caps.hpp"
#include "../st2110_tx/rtp_jxs.hpp"
#include "host_zerocopy.hpp"

#ifdef SPARK_HAVE_JXS
#include <moocudajxs/moocudajxs.h>
#endif

namespace spark::ops {
namespace {

// One message for both operators, so a misconfigured run says what to do rather than where it crashed.
[[noreturn]] void unavailable(const char* which) {
  throw std::runtime_error(
      std::string("jxs_codec: ") + which +
      " requires MooCUDAJXS, which this engine was not built against. Re-run CMake with "
      "-DSPARK_JXS_DIR=/path/to/MooCUDAJXS (or place it beside this repo) and rebuild.");
}

#ifdef SPARK_HAVE_JXS
constexpr size_t kRing = 10;  // matches unpack/pack: > the inter-op queues + frames held in flight
                              // (the TX holds host codestream buffers for a whole paced frame)

void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("jxs_codec: ") + what + ": " + cudaGetErrorString(e));
}

uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void jxs_check(MooJxsStatus s, const char* what) {
  if (s != MOO_JXS_OK)
    throw std::runtime_error(std::string("jxs_codec: ") + what + ": " + moo_jxs_status_string(s));
}

// The pipeline's intermediate format is planar 10-bit YCbCr 4:2:2 (GpuFrame), which is exactly
// MooCUDAJXS YUV422P at bit_depth 10 — no conversion, the codec reads/writes the GpuFrame planes.
constexpr MooJxsPixelFormat kFormat = MOO_JXS_FORMAT_YUV422P;
constexpr uint32_t kBitDepth = 10;

// Describe a GpuFrame's three planes to the codec. Y is (w x h), Cb/Cr are (w/2 x h), all tightly
// pitched by GpuFrame::alloc, so the pitches come straight from its step helpers.
MooJxsDeviceFrame describe(const spark::gpu::GpuFrame& f) {
  MooJxsDeviceFrame d{};
  d.component_count = 3;
  d.bit_depth = kBitDepth;
  d.format = kFormat;
  d.planes[0] = {f.y, static_cast<size_t>(f.y_step_bytes()), f.width, f.height};
  d.planes[1] = {f.cb, static_cast<size_t>(f.c_step_bytes()), f.chroma_width(), f.height};
  d.planes[2] = {f.cr, static_cast<size_t>(f.c_step_bytes()), f.chroma_width(), f.height};
  return d;
}

// Page-locked host word the codec can write from a kernel and we can read without a device copy.
// Used for the codestream size and the aggregate stage status: both are single words that the CPU
// needs, and pinned host memory is device-addressable on every UVA platform.
template <typename T>
T* alloc_pinned_word() {
  void* p = nullptr;
  cuda_check(cudaHostAlloc(&p, sizeof(T), cudaHostAllocMapped), "cudaHostAlloc status word");
  *static_cast<T*>(p) = T{};
  return static_cast<T*>(p);
}
#endif  // SPARK_HAVE_JXS

}  // namespace

bool jxs_available() {
#ifdef SPARK_HAVE_JXS
  return true;
#else
  return false;
#endif
}

const char* jxs_version() {
#ifdef SPARK_HAVE_JXS
  return moo_jxs_version_string();
#else
  return "unavailable (engine built without MooCUDAJXS)";
#endif
}

// ==================================================================================================
// JxsDecodeOp — VideoFrame (host JPEG XS codestream) -> GpuFrame (device planar 10-bit)
// ==================================================================================================
#ifdef SPARK_HAVE_JXS
struct JxsDecodeOp::Impl {
  cudaStream_t stream = nullptr;
  MooJxsDecoder* dec = nullptr;
  uint32_t w = 0, h = 0;
  bool zerocopy = false;
  uint8_t* dcode = nullptr;  // device staging for the codestream (copy path only)
  size_t dcode_bytes = 0;
  std::vector<spark::gpu::GpuFramePtr> pool;
  size_t idx = 0;

  // One in-flight decode's deferred bookkeeping. The healthy decode path never synchronizes —
  // downstream ops order on GpuFrame::ready — so the codestream size, the status word and (zero-copy)
  // the RX buffer whose bytes the kernels are still reading all have to live until the work lands.
  // The RX recycles any ring buffer whose use_count drops to 1, so holding the shared_ptr here is
  // what stops the next frame overwriting a codestream mid-decode.
  struct Slot {
    cudaEvent_t ev = nullptr;
    size_t* size = nullptr;      // pinned: the codestream length handed to the decoder
    uint32_t* status = nullptr;  // pinned: aggregate device-side decode status
    std::shared_ptr<std::vector<uint8_t>> hold;
  };
  std::deque<Slot> inflight;
  std::vector<Slot> free_slots;
  bool unhealthy = false;  // a decode was rejected: compute() re-syncs per frame until one succeeds

  uint64_t frames = 0, failed = 0;
  size_t logged_skip = SIZE_MAX;  // box-prefix length last reported (log once, not per frame)
  double last_live_s = 0;
};
#else
struct JxsDecodeOp::Impl {};
#endif

void JxsDecodeOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<spark::st2110::VideoFrame>("in");
  spec.output<spark::gpu::GpuFramePtr>("out");
  spec.param(max_precinct_columns_, "max_precinct_columns", "Max precinct columns",
             "worst-case PIH Cw the decoder plans workspace for; a codestream with more is rejected",
             uint32_t(4));
}

void JxsDecodeOp::start() {
#ifndef SPARK_HAVE_JXS
  unavailable("JPEG XS decode");
#else
  impl_ = std::make_shared<Impl>();
  HOLOSCAN_LOG_INFO("jxs_decode: MooCUDAJXS {} (max {} precinct columns)", jxs_version(),
                    max_precinct_columns_.get());
#endif
}

void JxsDecodeOp::drain_inflight(bool wait) {
#ifdef SPARK_HAVE_JXS
  if (!impl_) return;
  auto& s = *impl_;
  while (!s.inflight.empty()) {
    auto& f = s.inflight.front();
    const cudaError_t st = wait ? cudaEventSynchronize(f.ev) : cudaEventQuery(f.ev);
    if (st == cudaErrorNotReady) break;
    // The status word is only meaningful now that the stream reached the copy. A bad codestream
    // (loss the RX could not see, an unsupported feature, too many precinct columns) surfaces here
    // one frame late rather than stalling the decode path on a synchronization — and flips
    // compute() into its per-frame-checked recovery mode so the NEXT bad frame never airs.
    if (*f.status != MOO_JXS_OK) {
      ++s.failed;
      s.unhealthy = true;
      if (s.failed == 1 || (s.failed % 60) == 0)
        HOLOSCAN_LOG_WARN("jxs_decode: decoder rejected a frame ({}) — {} failed so far",
                          moo_jxs_status_string(static_cast<MooJxsStatus>(*f.status)), s.failed);
    }
    *f.status = MOO_JXS_OK;
    f.hold.reset();
    s.free_slots.push_back(f);
    s.inflight.pop_front();
  }
#else
  (void)wait;
#endif
}

void JxsDecodeOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                          holoscan::ExecutionContext&) {
  auto in = op_input.receive<spark::st2110::VideoFrame>("in");
  if (!in || !in.value().data) return;
#ifndef SPARK_HAVE_JXS
  (void)op_output;
  unavailable("JPEG XS decode");
#else
  const auto& vf = in.value();
  const auto& fmt = vf.format;
  const uint64_t bytes = vf.wire_bytes();
  if (bytes == 0) return;  // RX dropped a corrupt frame's payload; nothing to decode

  auto& s = *impl_;
  if (!s.stream) cuda_check(cudaStreamCreate(&s.stream), "decode stream");
  if (s.w != fmt.width || s.h != fmt.height) {
    // Geometry change (first frame, or a re-routed source): rebuild the handle and the pools. The
    // decoder plans its whole worst-case workspace here; frame submission then allocates nothing.
    cudaStreamSynchronize(s.stream);
    drain_inflight(true);
    if (s.dec) {
      moo_jxs_decoder_destroy(s.dec);
      s.dec = nullptr;
    }
    MooJxsDecoderConfig cfg{};
    cfg.abi_version = MOO_JXS_ABI_VERSION;
    cfg.maximum_width = fmt.width;
    cfg.maximum_height = fmt.height;
    cfg.maximum_bit_depth = kBitDepth;
    cfg.output_format = kFormat;
    cfg.maximum_precinct_columns = max_precinct_columns_.get();
    jxs_check(moo_jxs_decoder_create(&cfg, &s.dec), "decoder create");
    s.w = fmt.width;
    s.h = fmt.height;
    s.zerocopy = spark::codec::host_zerocopy();
    s.pool.assign(kRing, nullptr);
    for (auto& f : s.pool) f = std::make_shared<spark::gpu::GpuFrame>(fmt.width, fmt.height);
    s.idx = 0;
    s.unhealthy = false;  // fresh decoder, fresh stream: back to the async path
    HOLOSCAN_LOG_INFO("jxs_decode: {}x{} 4:2:2 10-bit{}", fmt.width, fmt.height,
                      s.zerocopy ? ", zero-copy" : "");
  }
  drain_inflight(false);

  // What the RX reassembled is a picture SEGMENT, not a codestream: RFC 9134 codestream mode carries
  // the ISO/IEC 21122-3 video support boxes ahead of the SOC, and real senders do send them (a RED
  // V-Raptor prefixes 60 bytes of 'jpvs'/'colr'). The codec decodes codestreams, so skip the boxes.
  // The format is fixed per stream, so this is logged once rather than per frame.
  const size_t skip = spark::st2110::jxs_codestream_offset(vf.data->data(), bytes);
  if (skip != s.logged_skip) {
    s.logged_skip = skip;
    if (skip)
      HOLOSCAN_LOG_INFO("jxs_decode: picture segment carries {} bytes of JPEG XS boxes before SOC",
                        skip);
  }
  const size_t code_bytes = bytes - skip;

  // Copy path: stage the codestream on our stream. cudaMemcpyAsync from pageable memory is
  // host-synchronous, so the RX buffer is free on return and needs no lifetime hold. (CUDA also
  // performs a stream sync before a pageable H2D copy, so this path runs one frame deep — frame
  // N+1's submit waits out frame N's decode. Inside budget at 59.94; zero-copy has no such sync.)
  // Zero-copy: the decoder kernels read the RX buffer in place, so the slot below holds it until
  // they finish.
  const uint8_t* src = vf.data->data() + skip;
  size_t capacity = vf.data->size() - skip;
  if (!s.zerocopy) {
    if (s.dcode_bytes < vf.data->size()) {
      if (s.dcode) cudaFree(s.dcode);
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&s.dcode), vf.data->size()),
                 "cudaMalloc codestream");
      s.dcode_bytes = vf.data->size();
    }
    cuda_check(cudaMemcpyAsync(s.dcode, src, code_bytes, cudaMemcpyHostToDevice, s.stream),
               "H2D codestream");
    src = s.dcode;
    capacity = s.dcode_bytes;
  }

  Impl::Slot slot;
  if (!s.free_slots.empty()) {
    slot = s.free_slots.back();
    s.free_slots.pop_back();
  } else {
    cuda_check(cudaEventCreateWithFlags(&slot.ev, cudaEventDisableTiming), "decode event");
    slot.size = alloc_pinned_word<size_t>();
    slot.status = alloc_pinned_word<uint32_t>();
  }
  *slot.size = code_bytes;

  auto dst = s.pool[s.idx];
  s.idx = (s.idx + 1) % s.pool.size();

  MooJxsDeviceCodestream code{};
  code.data = const_cast<uint8_t*>(src);
  code.capacity_bytes = capacity;
  code.size_device = slot.size;
  MooJxsDeviceFrame out = describe(*dst);
  jxs_check(moo_jxs_decode_async(s.dec, &code, &out, s.stream), "decode submit");
  jxs_check(moo_jxs_decoder_status_async(s.dec, slot.status, s.stream), "decode status");
  cuda_check(cudaEventRecord(dst->ready, s.stream), "decode record");  // consumers wait on this
  cuda_check(cudaEventRecord(slot.ev, s.stream), "decode slot record");
  if (s.zerocopy) slot.hold = vf.data;

  bool healthy = true;
  if (!s.unhealthy) {
    s.inflight.push_back(slot);
  } else {
    // Recovery mode. A rejected decode leaves the pool frame partially written (or stale by the
    // pool depth), and the async path only learns of the rejection a frame late — after that frame
    // went on air. So from the first observed failure, wait for THIS frame's status before emitting
    // (one stream sync per frame — the decode's own budget, not on top of it) and emit nothing
    // until the decoder accepts again: a sick stream freezes on the last good frame instead of
    // strobing stale/torn frames. One bad frame can still air per episode — the one that revealed
    // the failure — which is the price of the async healthy path.
    cuda_check(cudaEventSynchronize(slot.ev), "decode recovery sync");
    drain_inflight(false);  // pre-recovery stragglers finished with the sync above; retire them now
    healthy = *slot.status == MOO_JXS_OK;
    if (healthy) {
      s.unhealthy = false;
    } else {
      ++s.failed;
      if (s.failed == 1 || (s.failed % 60) == 0)
        HOLOSCAN_LOG_WARN("jxs_decode: decoder rejected a frame ({}) — {} failed so far",
                          moo_jxs_status_string(static_cast<MooJxsStatus>(*slot.status)), s.failed);
    }
    *slot.status = MOO_JXS_OK;
    slot.hold.reset();
    s.free_slots.push_back(slot);
  }

  dst->t_ingest_ns = now_ns();  // frame enters the GPU graph here; the egress op reads it back
  dst->capture_ts_ns = vf.capture_ts_ns;  // source frame timing -> TX genlock
  ++s.frames;

  const double t =
      std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if (t - s.last_live_s >= 1.0) {
    s.last_live_s = t;
    HOLOSCAN_LOG_INFO("spark_live jxs_dec_frames={} jxs_dec_failed={} jxs_dec_bytes={}", s.frames,
                      s.failed, code_bytes);
  }
  if (healthy) op_output.emit(dst, "out");
#endif
}

void JxsDecodeOp::stop() {
#ifdef SPARK_HAVE_JXS
  if (!impl_) return;  // start() failed or never ran
  auto& s = *impl_;
  if (s.stream) cudaStreamSynchronize(s.stream);
  drain_inflight(true);
  if (s.frames)
    HOLOSCAN_LOG_INFO("jxs_decode stopped: {} frames decoded, {} rejected", s.frames, s.failed);
  for (auto& f : s.free_slots) {
    if (f.ev) cudaEventDestroy(f.ev);
    if (f.size) cudaFreeHost(f.size);
    if (f.status) cudaFreeHost(f.status);
  }
  s.free_slots.clear();
  if (s.dec) {  // only after the last submitted work completed (the sync + drain above)
    moo_jxs_decoder_destroy(s.dec);
    s.dec = nullptr;
  }
  if (s.dcode) {
    cudaFree(s.dcode);
    s.dcode = nullptr;
  }
  if (s.stream) {
    cudaStreamDestroy(s.stream);
    s.stream = nullptr;
  }
#endif
}

// ==================================================================================================
// JxsEncodeOp — GpuFrame (device planar 10-bit) -> VideoFrame (host JPEG XS codestream)
// ==================================================================================================
#ifdef SPARK_HAVE_JXS
struct JxsEncodeOp::Impl {
  cudaStream_t stream = nullptr;
  MooJxsEncoder* enc = nullptr;
  uint32_t w = 0, h = 0;
  bool zerocopy = false;
  size_t target_bytes = 0;   // constant-rate codestream size (encoder workspace plan)
  uint8_t* dcode = nullptr;  // device staging for the codestream (copy path only)
  size_t* size_word = nullptr;
  uint32_t* status_word = nullptr;
  spark::st2110::VideoFormat fmt{};
  std::vector<std::shared_ptr<std::vector<uint8_t>>> host_pool;  // output ring (host codestream)
  size_t idx = 0;

  uint64_t frames = 0, failed = 0;
  double last_live_s = 0;
  size_t last_bytes = 0;  // last successful codestream length (telemetry runs before this frame's)
  uint64_t lat_last = 0, lat_min = UINT64_MAX, lat_max = 0, lat_sum = 0, lat_n = 0;
};
#else
struct JxsEncodeOp::Impl {};
#endif

void JxsEncodeOp::setup(holoscan::OperatorSpec& spec) {
  // Same input-queue contract as PackOp: burst-deep only when this op directly receives FRC's
  // multi-frame burst (SPARK_BURST_SINK), 1-deep otherwise — queue depth here is standing latency.
  const auto cap = static_cast<uint64_t>(spark::pipeline_queue_cap(name()));
  spec.input<spark::gpu::GpuFramePtr>("in")
      .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer, holoscan::Arg("capacity", cap),
                 holoscan::Arg("policy", static_cast<uint64_t>(2)))
      .condition(holoscan::ConditionType::kMessageAvailable,
                 holoscan::Arg("min_size", static_cast<uint64_t>(1)));
  spec.output<spark::st2110::VideoFrame>("out");
  spec.param(out_fps_, "out_fps", "Output fps", "output RTP media rate (TX pacing)", 60000.0 / 1001.0);
  spec.param(bits_per_pixel_, "bits_per_pixel", "Bits per pixel",
             "constant-rate JPEG XS target; the codestream is exactly this size every frame", 4.0);
  spec.param(horizontal_decomposition_, "horizontal_decomposition", "Horizontal decomposition",
             "wavelet decomposition levels (JPEG XS Nl)", uint32_t(5));
  spec.param(vertical_decomposition_, "vertical_decomposition", "Vertical decomposition",
             "vertical wavelet decomposition levels (JPEG XS Ns)", uint32_t(2));
  spec.param(precincts_per_slice_, "precincts_per_slice", "Precincts per slice",
             "PIH Nh: precinct rows per slice", uint32_t(16));
  spec.param(precinct_width_, "precinct_width", "Precinct width",
             "PIH Cw; 0 = one full-width precinct column", uint32_t(0));
  spec.param(profile_, "profile", "Profile", "PIH Ppih written verbatim; 0 = unrestricted",
             uint32_t(0));
  spec.param(level_, "level", "Level", "PIH Plev written verbatim; 0 = unrestricted", uint32_t(0));
}

void JxsEncodeOp::start() {
#ifndef SPARK_HAVE_JXS
  unavailable("JPEG XS encode");
#else
  impl_ = std::make_shared<Impl>();
  HOLOSCAN_LOG_INFO("jxs_encode: MooCUDAJXS {} ({:.2f} bpp, decomposition {}/{}, Nh {}, Cw {})",
                    jxs_version(), bits_per_pixel_.get(), horizontal_decomposition_.get(),
                    vertical_decomposition_.get(), precincts_per_slice_.get(),
                    precinct_width_.get());
#endif
}

void JxsEncodeOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                          holoscan::ExecutionContext&) {
  auto in = op_input.receive<spark::gpu::GpuFramePtr>("in");
  if (!in || !in.value()) return;
#ifndef SPARK_HAVE_JXS
  (void)op_output;
  unavailable("JPEG XS encode");
#else
  const auto& src = *in.value();
  auto& s = *impl_;
  if (!s.stream) cuda_check(cudaStreamCreate(&s.stream), "encode stream");
  if (s.w != src.width || s.h != src.height) {
    cudaStreamSynchronize(s.stream);  // the old handle may still own in-flight work
    if (s.enc) {
      moo_jxs_encoder_destroy(s.enc);
      s.enc = nullptr;
    }
    MooJxsEncoderConfig cfg{};
    cfg.abi_version = MOO_JXS_ABI_VERSION;
    cfg.width = src.width;
    cfg.height = src.height;
    cfg.bit_depth = kBitDepth;
    cfg.format = kFormat;
    cfg.bits_per_pixel = static_cast<float>(bits_per_pixel_.get());
    cfg.horizontal_decomposition = static_cast<uint8_t>(horizontal_decomposition_.get());
    cfg.vertical_decomposition = static_cast<uint8_t>(vertical_decomposition_.get());
    cfg.precincts_per_slice = static_cast<uint16_t>(precincts_per_slice_.get());
    cfg.quantization = MOO_JXS_QUANTIZATION_DEADZONE;
    cfg.precinct_width = static_cast<uint16_t>(precinct_width_.get());
    cfg.profile = static_cast<uint16_t>(profile_.get());
    cfg.level = static_cast<uint16_t>(level_.get());

    // The plan gives the exact constant-rate codestream size, which is both the encoder's required
    // output capacity and the size of every host buffer the TX will packetize.
    MooJxsEncoderWorkspaceRequirements req{};
    jxs_check(moo_jxs_encoder_workspace_requirements(&cfg, &req), "encoder workspace plan");
    jxs_check(moo_jxs_encoder_create(&cfg, &s.enc), "encoder create");
    s.target_bytes = req.target_codestream_bytes;
    s.w = src.width;
    s.h = src.height;
    s.zerocopy = spark::codec::host_zerocopy();
    if (!s.size_word) s.size_word = alloc_pinned_word<size_t>();
    if (!s.status_word) s.status_word = alloc_pinned_word<uint32_t>();
    if (s.dcode) {
      cudaFree(s.dcode);
      s.dcode = nullptr;
    }
    if (!s.zerocopy)
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&s.dcode), s.target_bytes),
                 "cudaMalloc codestream");
    s.fmt = spark::st2110::VideoFormat{src.width, src.height, out_fps_.get(),
                                       spark::st2110::Sampling::YCbCr422_10,
                                       spark::st2110::Codec::JpegXS};
    s.host_pool.assign(kRing, nullptr);
    for (auto& b : s.host_pool) b = std::make_shared<std::vector<uint8_t>>(s.target_bytes);
    s.idx = 0;
    const double mbps = s.target_bytes * 8.0 * out_fps_.get() / 1e6;
    HOLOSCAN_LOG_INFO("jxs_encode: {}x{} 4:2:2 10-bit -> {} B/frame ({:.2f} bpp, ~{:.0f} Mbps){}",
                      src.width, src.height, s.target_bytes, bits_per_pixel_.get(), mbps,
                      s.zerocopy ? ", zero-copy" : "");
  }

  // Pick a host buffer the TX has finished transmitting (use_count()==1 means only the pool holds
  // it). Same rule as PackOp: a blind round-robin can lap an in-flight buffer and corrupt a frame
  // mid-send; grow the pool instead. Chosen before submit — zero-copy encodes straight into it.
  std::shared_ptr<std::vector<uint8_t>> host;
  for (size_t n = 0; n < s.host_pool.size(); ++n) {
    auto& cand = s.host_pool[s.idx];
    s.idx = (s.idx + 1) % s.host_pool.size();
    if (cand.use_count() == 1) {
      host = cand;
      break;
    }
  }
  if (!host) {
    host = std::make_shared<std::vector<uint8_t>>(s.target_bytes);
    s.host_pool.push_back(host);
    HOLOSCAN_LOG_INFO("jxs_encode: grew host pool to {} buffers (TX holding the rest in flight)",
                      s.host_pool.size());
  }

  // Order our stream behind whoever produced this frame, then encode. Unlike decode, this path has
  // to synchronize: the codestream length is a device-side result and the TX needs it on the host
  // to plan its packets. PackOp's D2H sync is the same cost, so the egress hop is unchanged.
  cuda_check(cudaStreamWaitEvent(s.stream, src.ready, 0), "encode wait input");
  MooJxsDeviceFrame frame = describe(src);
  MooJxsDeviceCodestream code{};
  code.data = s.zerocopy ? host->data() : s.dcode;
  code.capacity_bytes = s.target_bytes;
  code.size_device = s.size_word;
  jxs_check(moo_jxs_encode_async(s.enc, &frame, &code, s.stream), "encode submit");
  jxs_check(moo_jxs_encoder_status_async(s.enc, s.status_word, s.stream), "encode status");
  // Copy path: stage the codestream back on OUR stream, ahead of the same sync. The length is a
  // device-side result not known until after the sync, so copy the full constant-rate capacity
  // (produced == target for CBR). A blocking cudaMemcpy after the sync — the old shape — ran on the
  // legacy default stream, which serializes against every other blocking stream in the pipeline.
  if (!s.zerocopy)
    cuda_check(
        cudaMemcpyAsync(host->data(), s.dcode, s.target_bytes, cudaMemcpyDeviceToHost, s.stream),
        "D2H codestream");
  cuda_check(cudaStreamSynchronize(s.stream), "encode sync");

  // Failure paths below all `return` without emitting, so the 1 Hz telemetry has to come FIRST —
  // otherwise a run where every frame fails would report nothing at all, which is exactly the run
  // an operator most needs the counters for. It reports the previous frame's numbers; at 1 Hz on a
  // 60 Hz stream that lag is invisible, and the failure counter is what matters here.
  {
    const double t =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (t - s.last_live_s >= 1.0) {
      s.last_live_s = t;
      HOLOSCAN_LOG_INFO("spark_live jxs_enc_frames={} jxs_enc_failed={} jxs_enc_bytes={}", s.frames,
                        s.failed, s.last_bytes);
      // Same token the daemon already parses for the GPU-chain latency (unpack->pack equivalent).
      if (s.lat_n)
        HOLOSCAN_LOG_INFO("spark_live pipe_latency_us cur={} min={} avg={} max={}",
                          s.lat_last / 1000, s.lat_min / 1000, (s.lat_sum / s.lat_n) / 1000,
                          s.lat_max / 1000);
    }
  }

  if (*s.status_word != MOO_JXS_OK) {
    ++s.failed;
    if (s.failed == 1 || (s.failed % 60) == 0)
      HOLOSCAN_LOG_WARN("jxs_encode: encoder failed ({}) — {} frames dropped so far",
                        moo_jxs_status_string(static_cast<MooJxsStatus>(*s.status_word)), s.failed);
    return;  // drop the frame rather than put an invalid codestream on the wire
  }
  const size_t produced = *s.size_word;
  if (produced == 0 || produced > s.target_bytes) {
    ++s.failed;
    HOLOSCAN_LOG_WARN("jxs_encode: implausible codestream size {} (capacity {}) — frame dropped",
                      produced, s.target_bytes);
    return;
  }
  s.last_bytes = produced;

  if (src.t_ingest_ns) {
    const uint64_t lat = now_ns() - src.t_ingest_ns;
    s.lat_last = lat;
    s.lat_sum += lat;
    ++s.lat_n;
    if (lat < s.lat_min) s.lat_min = lat;
    if (lat > s.lat_max) s.lat_max = lat;
  }

  spark::st2110::VideoFrame out;
  out.data = host;
  out.format = s.fmt;
  out.capture_ts_ns = src.capture_ts_ns;  // source frame timing -> TX genlock
  out.payload_bytes = produced;           // variable-length wire payload; `host` is the capacity
  ++s.frames;
  op_output.emit(out, "out");
#endif
}

void JxsEncodeOp::stop() {
#ifdef SPARK_HAVE_JXS
  if (!impl_) return;  // start() failed or never ran
  auto& s = *impl_;
  if (s.lat_n)
    HOLOSCAN_LOG_INFO("jxs_encode stopped: {} frames ({} failed), pipe_latency_us min/avg/max = {}/{}/{}",
                      s.frames, s.failed, s.lat_min / 1000, (s.lat_sum / s.lat_n) / 1000,
                      s.lat_max / 1000);
  if (s.stream) {
    cudaStreamSynchronize(s.stream);
    cudaStreamDestroy(s.stream);
    s.stream = nullptr;
  }
  if (s.enc) {
    moo_jxs_encoder_destroy(s.enc);
    s.enc = nullptr;
  }
  if (s.dcode) {
    cudaFree(s.dcode);
    s.dcode = nullptr;
  }
  if (s.size_word) {
    cudaFreeHost(s.size_word);
    s.size_word = nullptr;
  }
  if (s.status_word) {
    cudaFreeHost(s.status_word);
    s.status_word = nullptr;
  }
#endif
}

}  // namespace spark::ops
