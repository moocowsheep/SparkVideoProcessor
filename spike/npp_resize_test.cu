// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// M0 / M2 prep: validate the NPP resize path on this host and measure latency.
// Resizes a 16-bit single-channel image (stand-in for 10-bit-in-16-bit luma) 1080p -> 2160p and
// times several interpolation modes. Confirms the NPP toolchain works and gives a latency baseline.
//
// Build: see spike/build_probes.sh
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 2; } } while(0)

struct Mode { const char* name; int v; };

int main() {
    const int srcW=1920, srcH=1080, dstW=3840, dstH=2160;
    int srcStep=0, dstStep=0;
    Npp16u* src = nppiMalloc_16u_C1(srcW, srcH, &srcStep);
    Npp16u* dst = nppiMalloc_16u_C1(dstW, dstH, &dstStep);
    if(!src || !dst){ fprintf(stderr,"nppiMalloc failed\n"); return 2; }
    CK(cudaMemset2D(src, srcStep, 0x11, (size_t)srcW*sizeof(Npp16u), srcH));

    NppiSize srcSize{srcW,srcH}; NppiRect srcRoi{0,0,srcW,srcH};
    NppiSize dstSize{dstW,dstH}; NppiRect dstRoi{0,0,dstW,dstH};

    Mode modes[] = {{"LINEAR",NPPI_INTER_LINEAR},{"CUBIC",NPPI_INTER_CUBIC},{"LANCZOS",NPPI_INTER_LANCZOS}};
    // NPP 13 removed nppGetStreamContext — fill the context manually (default stream 0).
    NppStreamContext nctx{};
    nctx.hStream = 0;
    CK(cudaGetDevice(&nctx.nCudaDeviceId));
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, nctx.nCudaDeviceId));
    nctx.nMultiProcessorCount = prop.multiProcessorCount;
    nctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    nctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    nctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    nctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    nctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
    CK(cudaStreamGetFlags(nctx.hStream, &nctx.nStreamFlags));
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    const int iters=200;
    printf("[npp] resize %dx%d -> %dx%d, 16u C1, %d iters\n", srcW,srcH,dstW,dstH,iters);
    int ok=0;
    for (auto& m : modes) {
        NppStatus s = nppiResize_16u_C1R_Ctx(src,srcStep,srcSize,srcRoi, dst,dstStep,dstSize,dstRoi, m.v, nctx);
        if (s != NPP_SUCCESS) { printf("[npp]   %-8s unsupported (status %d)\n", m.name, (int)s); continue; }
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for(int i=0;i<iters;i++) nppiResize_16u_C1R_Ctx(src,srcStep,srcSize,srcRoi, dst,dstStep,dstSize,dstRoi, m.v, nctx);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms=0; CK(cudaEventElapsedTime(&ms,a,b));
        printf("[npp]   %-8s %.3f ms/frame\n", m.name, ms/iters);
        ok++;
    }
    nppiFree(src); nppiFree(dst);
    if(!ok){ fprintf(stderr,"[npp] FAIL — no interpolation mode worked\n"); return 3; }
    printf("[npp] PASS — NPP resize works on this GPU (%d mode(s)).\n", ok);
    return 0;
}
