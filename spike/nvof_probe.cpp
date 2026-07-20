// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// M0 gate 3 probe: confirm the NVIDIA Optical Flow Accelerator (NVOFA) works on GB10 (aarch64).
// Runs HW optical flow (NV_OF_MODE_OPTICALFLOW) between two synthetic frames using the CUDA driver
// API + the on-system libnvidia-opticalflow.so. Success here confirms the v1 motion-comp FRC path.
//
// Build: see spike/build_probes.sh
#include <cuda.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include "nvOpticalFlowCuda.h"

#define CU(x) do { CUresult r=(x); if(r!=CUDA_SUCCESS){ const char* s=nullptr; cuGetErrorString(r,&s); \
    fprintf(stderr,"CUDA error %d (%s) at %s:%d\n", r, s?s:"?", __FILE__, __LINE__); return 2; } } while(0)
#define OF(x) do { NV_OF_STATUS r=(x); if(r!=NV_OF_SUCCESS){ \
    fprintf(stderr,"NVOF error %d at %s:%d\n", (int)r, __FILE__, __LINE__); \
    if(hOF){ char e[MIN_ERROR_STRING_SIZE]={0}; uint32_t sz=sizeof(e); if(fl.nvOFGetLastError) fl.nvOFGetLastError(hOF,e,&sz); \
    fprintf(stderr,"  lastError: %s\n", e);} return 3; } } while(0)

static const uint32_t W = 256, H = 256, GRID = 4, DX = 8;

int main() {
    NvOFHandle hOF = nullptr;
    NV_OF_CUDA_API_FUNCTION_LIST fl{};

    CU(cuInit(0));
    CUdevice dev; CU(cuDeviceGet(&dev, 0));
    char name[128]={0}; CU(cuDeviceGetName(name, sizeof(name), dev));
    CUcontext ctx; CU(cuCtxCreate(&ctx, nullptr, 0, dev));  // CUDA 13: cuCtxCreate_v4(params=NULL)
    printf("[nvof] CUDA device: %s\n", name);

    uint32_t drvApi = 0;
    if (NvOFGetMaxSupportedApiVersion(&drvApi) == NV_OF_SUCCESS)
        printf("[nvof] driver max OF API version: %u.%u (header is %u.%u)\n",
               drvApi >> 4, drvApi & 0xf, NV_OF_API_MAJOR_VERSION, NV_OF_API_MINOR_VERSION);

    if (NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &fl) != NV_OF_SUCCESS) {
        fprintf(stderr, "[nvof] NvOFAPICreateInstanceCuda FAILED — OFA not usable\n"); return 4;
    }
    OF(fl.nvCreateOpticalFlowCuda(ctx, &hOF));

    // Report HW capabilities.
    auto cap = [&](NV_OF_CAPS c)->uint32_t{ uint32_t v=0, n=1; if(fl.nvOFGetCaps(hOF,c,&v,&n)!=NV_OF_SUCCESS) return 0; return v; };
    printf("[nvof] caps: width %u..%u, height %u..%u\n",
           cap(NV_OF_CAPS_WIDTH_MIN), cap(NV_OF_CAPS_WIDTH_MAX),
           cap(NV_OF_CAPS_HEIGHT_MIN), cap(NV_OF_CAPS_HEIGHT_MAX));

    NV_OF_INIT_PARAMS init{};
    init.width = W; init.height = H;
    init.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    init.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    init.mode = NV_OF_MODE_OPTICALFLOW;
    init.perfLevel = NV_OF_PERF_LEVEL_MEDIUM;
    OF(fl.nvOFInit(hOF, &init));

    const uint32_t outW = (W + GRID - 1) / GRID, outH = (H + GRID - 1) / GRID;

    auto mkbuf = [&](uint32_t w, uint32_t ht, NV_OF_BUFFER_USAGE usage, NV_OF_BUFFER_FORMAT fmt,
                     NvOFGPUBufferHandle* out)->NV_OF_STATUS {
        NV_OF_BUFFER_DESCRIPTOR d{}; d.width=w; d.height=ht; d.bufferUsage=usage; d.bufferFormat=fmt;
        return fl.nvOFCreateGPUBufferCuda(hOF, &d, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, out);
    };
    NvOFGPUBufferHandle bIn=nullptr, bRef=nullptr, bOut=nullptr;
    OF(mkbuf(W, H, NV_OF_BUFFER_USAGE_INPUT,  NV_OF_BUFFER_FORMAT_GRAYSCALE8, &bIn));
    OF(mkbuf(W, H, NV_OF_BUFFER_USAGE_INPUT,  NV_OF_BUFFER_FORMAT_GRAYSCALE8, &bRef));
    OF(mkbuf(outW, outH, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2, &bOut));

    // Synthetic frames: textured, non-periodic blocks on gray; reference = same content shifted
    // right by DX px → optical flow input->reference should report flowx ~ +DX, flowy ~ 0.
    std::vector<uint8_t> in(W*H), ref(W*H);
    auto draw = [&](std::vector<uint8_t>& img, int shift){
        std::fill(img.begin(), img.end(), (uint8_t)64);
        uint32_t s = 12345;
        auto rnd = [&](){ s = s*1103515245u + 12345u; return (s>>16) & 0x7fff; };
        for (int i=0;i<60;i++){
            int bw=8+rnd()%24, bh=8+rnd()%24;
            int px=rnd()%((int)W-48), py=rnd()%((int)H-48);
            uint8_t c=(uint8_t)(60+rnd()%195);
            for (int yy=0; yy<bh; yy++) for (int xx=0; xx<bw; xx++){
                int X=px+xx+shift, Y=py+yy;
                if (X>=0 && X<(int)W && Y>=0 && Y<(int)H) img[Y*W+X]=c;
            }
        }
    };
    draw(in, 0); draw(ref, DX);
    auto upload = [&](NvOFGPUBufferHandle b, const uint8_t* src)->NV_OF_STATUS {
        NV_OF_CUDA_BUFFER_STRIDE_INFO si{}; NV_OF_STATUS s=fl.nvOFGPUBufferGetStrideInfo(b,&si); if(s!=NV_OF_SUCCESS) return s;
        CUdeviceptr dp = fl.nvOFGPUBufferGetCUdeviceptr(b);
        uint32_t pitch = si.strideInfo[0].strideXInBytes;
        for(uint32_t y=0;y<H;y++){ if(cuMemcpyHtoD(dp + (size_t)y*pitch, src + (size_t)y*W, W)!=CUDA_SUCCESS) return NV_OF_ERR_GENERIC; }
        return NV_OF_SUCCESS;
    };
    OF(upload(bIn, in.data()));
    OF(upload(bRef, ref.data()));

    NV_OF_EXECUTE_INPUT_PARAMS  ein{};  ein.inputFrame=bIn; ein.referenceFrame=bRef; ein.disableTemporalHints=NV_OF_TRUE;
    NV_OF_EXECUTE_OUTPUT_PARAMS eout{}; eout.outputBuffer=bOut;
    OF(fl.nvOFExecute(hOF, &ein, &eout));
    CU(cuCtxSynchronize());

    // Read back center flow vectors.
    NV_OF_CUDA_BUFFER_STRIDE_INFO so{}; OF(fl.nvOFGPUBufferGetStrideInfo(bOut,&so));
    CUdeviceptr odp = fl.nvOFGPUBufferGetCUdeviceptr(bOut);
    uint32_t opitch = so.strideInfo[0].strideXInBytes;
    std::vector<NV_OF_FLOW_VECTOR> row(outW);
    uint32_t cy = outH/2;
    CU(cuMemcpyDtoH(row.data(), odp + (size_t)cy*opitch, outW*sizeof(NV_OF_FLOW_VECTOR)));
    uint32_t cx = outW/2;
    float fx = row[cx].flowx / 32.0f, fy = row[cx].flowy / 32.0f;
    printf("[nvof] EXECUTE ok. center flow (grid %u,%u) = (%.2f, %.2f) px  [expected ~|x|=%u]\n", cx, cy, fx, fy, DX);

    fl.nvOFDestroyGPUBufferCuda(bIn); fl.nvOFDestroyGPUBufferCuda(bRef); fl.nvOFDestroyGPUBufferCuda(bOut);
    fl.nvOFDestroy(hOF);
    cuCtxDestroy(ctx);
    printf("[nvof] PASS — NVOFA hardware optical flow works on this GB10.\n");
    return 0;
}
