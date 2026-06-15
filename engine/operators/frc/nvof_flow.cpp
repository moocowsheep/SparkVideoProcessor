#include "nvof_flow.hpp"

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
}  // namespace

struct NvofFlow::Impl {
  NV_OF_CUDA_API_FUNCTION_LIST fl{};
  NvOFHandle hof = nullptr;
  NvOFGPUBufferHandle bin = nullptr, bref = nullptr, bout = nullptr;
  CUdeviceptr in_dp = 0, ref_dp = 0, out_dp = 0;
  uint32_t in_pitch = 0, ref_pitch = 0, out_pitch = 0;
  uint32_t width = 0, height = 0, grid_size = 4, grid_w = 0, grid_h = 0;

  void of_check(NV_OF_STATUS r, const char* what) {
    if (r == NV_OF_SUCCESS) return;
    std::string msg = std::string("nvof: ") + what + " status=" + std::to_string((int)r);
    if (hof && fl.nvOFGetLastError) {
      char e[MIN_ERROR_STRING_SIZE] = {0};
      uint32_t sz = sizeof(e);
      fl.nvOFGetLastError(hof, e, &sz);
      msg += std::string(" (") + e + ")";
    }
    throw std::runtime_error(msg);
  }
  ~Impl() {
    if (fl.nvOFDestroyGPUBufferCuda) {
      if (bin) fl.nvOFDestroyGPUBufferCuda(bin);
      if (bref) fl.nvOFDestroyGPUBufferCuda(bref);
      if (bout) fl.nvOFDestroyGPUBufferCuda(bout);
    }
    if (hof && fl.nvOFDestroy) fl.nvOFDestroy(hof);
  }
};

NvofFlow::NvofFlow() : p_(std::make_unique<Impl>()) {}
NvofFlow::~NvofFlow() = default;

void NvofFlow::init(uint32_t width, uint32_t height, uint32_t grid_size) {
  Impl& im = *p_;
  im.width = width;
  im.height = height;
  im.grid_size = grid_size;

  cu_check(cuInit(0), "cuInit");
  CUcontext ctx = nullptr;
  cu_check(cuCtxGetCurrent(&ctx), "cuCtxGetCurrent");
  if (!ctx) {  // share the runtime primary context so NVOF buffers interop with GpuFrame planes
    CUdevice dev;
    cu_check(cuDeviceGet(&dev, 0), "cuDeviceGet");
    cu_check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    cu_check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
  }

  if (NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &im.fl) != NV_OF_SUCCESS)
    throw std::runtime_error("nvof: NvOFAPICreateInstanceCuda failed (OFA not usable)");
  im.of_check(im.fl.nvCreateOpticalFlowCuda(ctx, &im.hof), "nvCreateOpticalFlowCuda");

  NV_OF_INIT_PARAMS ip{};
  ip.width = width;
  ip.height = height;
  ip.outGridSize = grid_enum(grid_size);
  ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
  ip.mode = NV_OF_MODE_OPTICALFLOW;
  ip.perfLevel = NV_OF_PERF_LEVEL_MEDIUM;
  im.of_check(im.fl.nvOFInit(im.hof, &ip), "nvOFInit");

  im.grid_w = (width + grid_size - 1) / grid_size;
  im.grid_h = (height + grid_size - 1) / grid_size;

  auto mkbuf = [&](uint32_t w, uint32_t h, NV_OF_BUFFER_USAGE u, NV_OF_BUFFER_FORMAT f,
                   NvOFGPUBufferHandle* o) {
    NV_OF_BUFFER_DESCRIPTOR d{};
    d.width = w;
    d.height = h;
    d.bufferUsage = u;
    d.bufferFormat = f;
    im.of_check(im.fl.nvOFCreateGPUBufferCuda(im.hof, &d, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, o),
                "nvOFCreateGPUBufferCuda");
  };
  mkbuf(width, height, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, &im.bin);
  mkbuf(width, height, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, &im.bref);
  mkbuf(im.grid_w, im.grid_h, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2, &im.bout);

  auto stride = [&](NvOFGPUBufferHandle b) -> uint32_t {
    NV_OF_CUDA_BUFFER_STRIDE_INFO si{};
    im.of_check(im.fl.nvOFGPUBufferGetStrideInfo(b, &si), "nvOFGPUBufferGetStrideInfo");
    return si.strideInfo[0].strideXInBytes;
  };
  im.in_dp = im.fl.nvOFGPUBufferGetCUdeviceptr(im.bin);
  im.in_pitch = stride(im.bin);
  im.ref_dp = im.fl.nvOFGPUBufferGetCUdeviceptr(im.bref);
  im.ref_pitch = stride(im.bref);
  im.out_dp = im.fl.nvOFGPUBufferGetCUdeviceptr(im.bout);
  im.out_pitch = stride(im.bout);
}

void NvofFlow::compute(const uint8_t* prevY8, const uint8_t* curY8) {
  Impl& im = *p_;
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
    cu_check(cuMemcpy2D(&c), "cuMemcpy2D Y8");
  };
  upload(im.in_dp, im.in_pitch, prevY8);
  upload(im.ref_dp, im.ref_pitch, curY8);

  NV_OF_EXECUTE_INPUT_PARAMS ein{};
  ein.inputFrame = im.bin;
  ein.referenceFrame = im.bref;
  ein.disableTemporalHints = NV_OF_TRUE;
  NV_OF_EXECUTE_OUTPUT_PARAMS eout{};
  eout.outputBuffer = im.bout;
  im.of_check(im.fl.nvOFExecute(im.hof, &ein, &eout), "nvOFExecute");
  cu_check(cuCtxSynchronize(), "cuCtxSynchronize");
}

const void* NvofFlow::flow_dev() const { return reinterpret_cast<const void*>(p_->out_dp); }
uint32_t NvofFlow::flow_pitch_bytes() const { return p_->out_pitch; }
uint32_t NvofFlow::grid_w() const { return p_->grid_w; }
uint32_t NvofFlow::grid_h() const { return p_->grid_h; }
uint32_t NvofFlow::grid_size() const { return p_->grid_size; }

}  // namespace spark::frc
