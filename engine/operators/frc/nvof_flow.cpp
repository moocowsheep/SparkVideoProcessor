// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

#include "nvof_flow.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <cuda.h>

#include "nvOpticalFlowCuda.h"

namespace spark::frc {
namespace {
void cu_check(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    throw std::runtime_error(std::string("nvof: ") + what + ": " + (s ? s : "?"));
  }
}
NV_OF_OUTPUT_VECTOR_GRID_SIZE grid_enum(uint32_t g) {
  switch (g) {
    case 1: return NV_OF_OUTPUT_VECTOR_GRID_SIZE_1;
    case 2: return NV_OF_OUTPUT_VECTOR_GRID_SIZE_2;
    default: return NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
  }
}
bool env_on(const char* name, bool dflt) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) != 0 : dflt;
}
}  // namespace

struct NvofFlow::Impl {
  NV_OF_CUDA_API_FUNCTION_LIST fl{};
  // One NVOF session PER DIRECTION. A single session alternating fwd/bwd executes would poison
  // temporal hints — the previous execute (the OTHER direction) would seed the search — so hints
  // had to stay disabled. Per-direction sessions make consecutive executes temporally consecutive
  // same-direction solves, exactly what the driver's hinting expects on continuous video.
  struct Session {
    NvOFHandle hof = nullptr;
    NvOFGPUBufferHandle bin = nullptr, bref = nullptr, bout = nullptr, bcost = nullptr;
    CUdeviceptr in_dp = 0, ref_dp = 0, out_dp = 0, cost_dp = 0;
    uint32_t in_pitch = 0, ref_pitch = 0, out_pitch = 0, cost_pitch = 0;
  };
  Session fwd, bwd;
  CUcontext ctx = nullptr;  // the (primary) context NVOF was created on
  uint32_t width = 0, height = 0, grid_size = 4, grid_w = 0, grid_h = 0;
  uint32_t cost_elem = 0;      // cost buffer element size: 1 (UINT8) | 4 (UINT); 0 = no cost
  bool temporal_hints = true;  // SPARK_FRC_TEMPORAL_HINTS=0 reverts to hint-free solves

  void of_check(NV_OF_STATUS r, const char* what, NvOFHandle h = nullptr) {
    if (r == NV_OF_SUCCESS) return;
    std::string msg = std::string("nvof: ") + what + " status=" + std::to_string((int)r);
    if (h && fl.nvOFGetLastError) {
      char e[MIN_ERROR_STRING_SIZE] = {0};
      uint32_t sz = sizeof(e);
      fl.nvOFGetLastError(h, e, &sz);
      msg += std::string(" (") + e + ")";
    }
    throw std::runtime_error(msg);
  }

  void destroy_session(Session& s) {
    if (fl.nvOFDestroyGPUBufferCuda) {
      if (s.bin) fl.nvOFDestroyGPUBufferCuda(s.bin);
      if (s.bref) fl.nvOFDestroyGPUBufferCuda(s.bref);
      if (s.bout) fl.nvOFDestroyGPUBufferCuda(s.bout);
      if (s.bcost) fl.nvOFDestroyGPUBufferCuda(s.bcost);
    }
    if (s.hof && fl.nvOFDestroy) fl.nvOFDestroy(s.hof);
    s = Session{};
  }

  // Create + init one session and its buffers. cost_elem 1|4 selects the cost buffer format;
  // returns false (partially-built session, caller destroys) if THAT cost format is rejected,
  // so init() can probe formats. Non-cost failures throw.
  bool build_session(Session& s, uint32_t ce) {
    of_check(fl.nvCreateOpticalFlowCuda(ctx, &s.hof), "nvCreateOpticalFlowCuda");
    NV_OF_INIT_PARAMS ip{};
    ip.width = width;
    ip.height = height;
    ip.outGridSize = grid_enum(grid_size);
    ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    ip.mode = NV_OF_MODE_OPTICALFLOW;
    ip.perfLevel = NV_OF_PERF_LEVEL_SLOW;  // best-quality flow; ~75% GPU headroom to spend on it
    ip.enableOutputCost = ce ? NV_OF_TRUE : NV_OF_FALSE;
    of_check(fl.nvOFInit(s.hof, &ip), "nvOFInit", s.hof);

    auto mk = [&](uint32_t w, uint32_t h, NV_OF_BUFFER_USAGE u, NV_OF_BUFFER_FORMAT f,
                  NvOFGPUBufferHandle* o) {
      NV_OF_BUFFER_DESCRIPTOR d{};
      d.width = w;
      d.height = h;
      d.bufferUsage = u;
      d.bufferFormat = f;
      return fl.nvOFCreateGPUBufferCuda(s.hof, &d, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, o);
    };
    of_check(mk(width, height, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, &s.bin),
             "nvOFCreateGPUBufferCuda in", s.hof);
    of_check(mk(width, height, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, &s.bref),
             "nvOFCreateGPUBufferCuda ref", s.hof);
    of_check(mk(grid_w, grid_h, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2, &s.bout),
             "nvOFCreateGPUBufferCuda out", s.hof);
    if (ce) {
      const NV_OF_BUFFER_FORMAT cf = ce == 1 ? NV_OF_BUFFER_FORMAT_UINT8 : NV_OF_BUFFER_FORMAT_UINT;
      if (mk(grid_w, grid_h, NV_OF_BUFFER_USAGE_COST, cf, &s.bcost) != NV_OF_SUCCESS) return false;
    }

    auto stride = [&](NvOFGPUBufferHandle b) -> uint32_t {
      NV_OF_CUDA_BUFFER_STRIDE_INFO si{};
      of_check(fl.nvOFGPUBufferGetStrideInfo(b, &si), "nvOFGPUBufferGetStrideInfo", s.hof);
      return si.strideInfo[0].strideXInBytes;
    };
    s.in_dp = fl.nvOFGPUBufferGetCUdeviceptr(s.bin);
    s.in_pitch = stride(s.bin);
    s.ref_dp = fl.nvOFGPUBufferGetCUdeviceptr(s.bref);
    s.ref_pitch = stride(s.bref);
    s.out_dp = fl.nvOFGPUBufferGetCUdeviceptr(s.bout);
    s.out_pitch = stride(s.bout);
    if (ce) {
      s.cost_dp = fl.nvOFGPUBufferGetCUdeviceptr(s.bcost);
      s.cost_pitch = stride(s.bcost);
    }
    return true;
  }

  ~Impl() {
    destroy_session(fwd);
    destroy_session(bwd);
  }
};

NvofFlow::NvofFlow() : p_(std::make_unique<Impl>()) {}
NvofFlow::~NvofFlow() = default;

void NvofFlow::init(uint32_t width, uint32_t height, uint32_t grid_size) {
  Impl& im = *p_;
  im.width = width;
  im.height = height;
  // NVOF outputs one vector per 1|2|4 px only; grid_enum() maps anything else to 4, so the grid
  // used to size grid_w/grid_h buffers must match or the output buffer is undersized.
  if (grid_size != 1 && grid_size != 2) grid_size = 4;
  im.grid_size = grid_size;
  im.grid_w = (width + grid_size - 1) / grid_size;
  im.grid_h = (height + grid_size - 1) / grid_size;
  im.temporal_hints = env_on("SPARK_FRC_TEMPORAL_HINTS", true);

  cu_check(cuInit(0), "cuInit");
  CUcontext ctx = nullptr;
  cu_check(cuCtxGetCurrent(&ctx), "cuCtxGetCurrent");
  if (!ctx) {  // share the runtime primary context so NVOF buffers interop with GpuFrame planes
    CUdevice dev;
    cu_check(cuDeviceGet(&dev, 0), "cuDeviceGet");
    cu_check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    cu_check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
  }
  im.ctx = ctx;

  if (NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &im.fl) != NV_OF_SUCCESS)
    throw std::runtime_error("nvof: NvOFAPICreateInstanceCuda failed (OFA not usable)");

  // The per-vector matching cost feeds the warp's confidence weighting (SPARK_FRC_COST=0 skips
  // it). Its buffer format differs across driver generations (UINT8 vs legacy 32-bit UINT), so
  // probe on the fwd session and reuse the winning format for bwd; no-cost is the fallback, never
  // a hard failure.
  im.cost_elem = 0;
  if (env_on("SPARK_FRC_COST", true)) {
    for (uint32_t ce : {1u, 4u}) {
      bool ok = false;
      try {
        ok = im.build_session(im.fwd, ce);
      } catch (const std::exception&) {
        ok = false;
      }
      if (ok) {
        im.cost_elem = ce;
        break;
      }
      im.destroy_session(im.fwd);
    }
  }
  if (!im.fwd.hof && !im.build_session(im.fwd, 0))
    throw std::runtime_error("nvof: fwd session init failed");
  if (!im.build_session(im.bwd, im.cost_elem))
    throw std::runtime_error("nvof: bwd session cost buffer create failed after fwd probe");
  // view() publishes one pitch for both directions; same descriptors must yield same strides.
  if (im.bwd.out_pitch != im.fwd.out_pitch || im.bwd.cost_pitch != im.fwd.cost_pitch)
    throw std::runtime_error("nvof: fwd/bwd buffer pitch mismatch");
}

void NvofFlow::compute(const uint8_t* prevY8, const uint8_t* curY8, void* stream) {
  Impl& im = *p_;
  // FrcOp::compute() may run on any Holoscan worker thread; make our context current so the NVOF
  // driver calls (and the runtime kernels sharing this primary context) target the right device.
  cu_check(cuCtxSetCurrent(im.ctx), "cuCtxSetCurrent");
  const CUstream cs = reinterpret_cast<CUstream>(stream);
  // Bind NVOF input/output processing to the caller's stream so the uploads, the executes, and the
  // downstream interpolate (run on the same stream) order without a device-wide cuCtxSynchronize —
  // that full-context sync was the pipeline's serialization point. nvOFExecute is then asynchronous.
  if (im.fl.nvOFSetIOCudaStreams) {
    im.of_check(im.fl.nvOFSetIOCudaStreams(im.fwd.hof, cs, cs), "nvOFSetIOCudaStreams fwd");
    im.of_check(im.fl.nvOFSetIOCudaStreams(im.bwd.hof, cs, cs), "nvOFSetIOCudaStreams bwd");
  }
  auto upload = [&](CUdeviceptr dst, uint32_t dpitch, const uint8_t* src) {
    CUDA_MEMCPY2D c{};
    c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    c.srcDevice = reinterpret_cast<CUdeviceptr>(src);
    c.srcPitch = im.width;
    c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    c.dstDevice = dst;
    c.dstPitch = dpitch;
    c.WidthInBytes = im.width;
    c.Height = im.height;
    cu_check(cuMemcpy2DAsync(&c, cs), "cuMemcpy2DAsync Y8");
  };
  // Each session owns its input buffers, so both directions upload both frames (D2D, ~8 MB each at
  // 2160p — noise next to the OFA solve). fwd solves prev->cur, bwd solves cur->prev.
  upload(im.fwd.in_dp, im.fwd.in_pitch, prevY8);
  upload(im.fwd.ref_dp, im.fwd.ref_pitch, curY8);
  upload(im.bwd.in_dp, im.bwd.in_pitch, curY8);
  upload(im.bwd.ref_dp, im.bwd.ref_pitch, prevY8);

  auto exec = [&](Impl::Session& s, const char* what) {
    NV_OF_EXECUTE_INPUT_PARAMS ein{};
    ein.inputFrame = s.bin;
    ein.referenceFrame = s.bref;
    // Temporal hints seed the search with this session's previous (same-direction) solve — NVIDIA
    // recommends them on continuous video. On a hard scene cut the hints are transiently wrong but
    // only bias the search start; the field recovers on the next pair.
    ein.disableTemporalHints = im.temporal_hints ? NV_OF_FALSE : NV_OF_TRUE;
    NV_OF_EXECUTE_OUTPUT_PARAMS eout{};
    eout.outputBuffer = s.bout;
    if (im.cost_elem) eout.outputCostBuffer = s.bcost;
    im.of_check(im.fl.nvOFExecute(s.hof, &ein, &eout), what, s.hof);
  };
  // Forward flow: prev -> cur, indexed in prev coords. Backward: cur -> prev, in cur coords. The
  // warp uses fwd<->bwd consistency (photometric + vector) to detect occlusions.
  exec(im.fwd, "nvOFExecute fwd");
  exec(im.bwd, "nvOFExecute bwd");
  // No cuCtxSynchronize: both flow outputs are produced on `stream`; the caller's interpolate kernel
  // on the same stream consumes them in order. Reuse of the per-session buffers across frames is
  // also stream-ordered (FrcOp::compute is serial), so no inter-frame hazard.
}

FlowView NvofFlow::view() const {
  const Impl& im = *p_;
  FlowView v;
  v.fwd = reinterpret_cast<const void*>(im.fwd.out_dp);
  v.bwd = reinterpret_cast<const void*>(im.bwd.out_dp);
  v.pitch_bytes = im.fwd.out_pitch;
  v.grid_w = im.grid_w;
  v.grid_h = im.grid_h;
  v.grid_size = im.grid_size;
  if (im.cost_elem) {
    v.cost_fwd = reinterpret_cast<const void*>(im.fwd.cost_dp);
    v.cost_bwd = reinterpret_cast<const void*>(im.bwd.cost_dp);
    v.cost_pitch_bytes = im.fwd.cost_pitch;
    v.cost_elem_bytes = im.cost_elem;
  }
  return v;
}

const void* NvofFlow::flow_dev() const { return reinterpret_cast<const void*>(p_->fwd.out_dp); }
const void* NvofFlow::flow_dev_bwd() const { return reinterpret_cast<const void*>(p_->bwd.out_dp); }
uint32_t NvofFlow::flow_pitch_bytes() const { return p_->fwd.out_pitch; }
uint32_t NvofFlow::grid_w() const { return p_->grid_w; }
uint32_t NvofFlow::grid_h() const { return p_->grid_h; }
uint32_t NvofFlow::grid_size() const { return p_->grid_size; }
bool NvofFlow::has_cost() const { return p_->cost_elem != 0; }
bool NvofFlow::temporal_hints() const { return p_->temporal_hints; }

}  // namespace spark::frc
