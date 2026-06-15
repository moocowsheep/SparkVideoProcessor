# M2 — GPU processing operators + full pipeline

Status: **COMPLETE — the full `st2110_rx → unpack → resize → frc → pack → st2110_tx` pipeline runs
end-to-end on the NIC (1080p→2160p NPP upscale + OFA motion-compensated interpolation in flight),
zero loss, `future_err=0` (2026-06-15). Remaining work is perf tuning + production hardening, not
functionality.**

```
st2110_rx → [unpack] → [resize] → [frc] → [pack] → st2110_tx
   ✓          ✓ CUDA     ✓ NPP    ✓ OFA     ✓ CUDA     ✓     ← all wired + running on hardware
```

## Intermediate format: GpuFrame
`engine/operators/resize/gpu_frame.hpp` — planar YCbCr 4:2:2, 10-bit in `uint16`: device planes Y
(w×h), Cb/Cr (w/2×h). RAII over cudaMalloc; passed between operators as `shared_ptr<GpuFrame>`. This
is the GPU working format for the whole processing chain.

## Operators
- **`resize` (NPP)** — `nppiResize_16u_C1R_Ctx` per plane; interp linear|cubic|lanczos. 1080p→2160p
  ~0.23 ms (cubic, steady-state), ~1.5% of the 16.7 ms 60 fps budget (M0 gate 3b). 3-plane, so a bit
  above the M0 1-plane spike. CUDA-13 NPP exposes only `_Ctx` variants and no `nppGetStreamContext`,
  so the app fills `NppStreamContext` from device props.
- **`unpack`/`pack` (CUDA, `codec/pixel_codec.cu`)** — RFC 4175 5-byte pgroup ⟷ planar Y/Cb/Cr. A
  bijection; the round-trip test (`test_pixel_codec`, in ctest) is byte-identical. `UnpackOp`:
  VideoFrame (host packed) → GpuFrame; `PackOp`: the inverse. Default-stream ordered with sync
  H2D/D2H staging.
- **`frc` (OFA, `frc/nvof_flow.cpp` + `frc/frc_kernels.cu`)** — motion-compensated interpolation,
  hand-rolled on the raw NVIDIA Optical Flow API (NvOFFRUC isn't on aarch64). `NvofFlow` wraps the
  NVOF driver API (shares the runtime primary context); a warp/blend kernel synthesizes a frame at
  phase t from prev/cur along the flow. `FrcOp` holds prev and emits an interpolated frame per input
  (1:1; 2× rate-conversion is a follow-on). Validated (`frc_smoke`, pure GPU): synthetic 16px
  translation → flow `(16.00, 0.22)`, interpolation MAE 0.05 vs naive-blend 5.22 (~100× better — the
  flow is actually used, no ghosting). Wired into the live pipeline; `NvofFlow` captures the primary
  context at init and `cuCtxSetCurrent`s it in `compute()` so the NVOF driver calls work on any
  Holoscan worker thread (no context errors in the live graph).

## Build note (GB10)
Enabling the CUDA language sets `CMAKE_CUDA_ARCHITECTURES` to a conservative default (sm_75 here),
whose PTX won't run on Blackwell ("unsupported toolchain"). GB10 is compute capability **12.1**, so
CMake forces `CMAKE_CUDA_ARCHITECTURES=121`.

## End-to-end pipeline (`apps/st2110_pipeline.cpp`)
One process, shared EAL, MultiThreadScheduler:
`st2110_rx → unpack → resize → frc → pack → st2110_tx` (`SPARK_FRC=0` bypasses frc). Validated with a
generator feeding the RX port (1080p → pipeline → 2160p out, 300 frames):

| stage | with FRC | without FRC |
|---|---|---|
| RX (1080p in) | 1,113,300 pkts, **lost=0 bad=0 q_dropped=0 hw_missed=0** | same, lost=0 |
| resize | 1080p→2160p cubic ~0.28 ms | ~0.28 ms |
| frc | interpolated 299/300 (3840×2160) | — |
| TX (2160p out) | 4,448,100 pkts, **future_err=0**, jitter 18 ns | future_err=0, jitter 24 ns |
| ingest latency avg | ~15.6 µs | ~2.8 µs |
| TX past_err | 54,919 (~1.2%, slightly-late, not loss) | ~425 |

Zero loss / zero dropped frames either way — real NPP upscale **and** OFA motion-compensated
interpolation in flight. FRC raises pacing jitter + latency under the heavier load (a tuning item,
below); RX stays clean.

Reproduce (2 processes; pipeline first, then generator after its "st2110_rx started" line):
```bash
sudo -n SPARK_PROFILE=1080p SPARK_OUT_W=3840 SPARK_OUT_H=2160 SPARK_FRC=1 SPARK_RX_PCI=0000:01:00.1 \
        SPARK_TX_PCI=0002:01:00.0 SPARK_DST_MAC=30:c5:99:3e:9d:30 ./engine/build/st2110_pipeline &
sudo -n SPARK_PROFILE=1080p SPARK_WARMUP_MS=1200 SPARK_TX_PCI=0000:01:00.0 \
        SPARK_DST_MAC=30:c5:99:3e:9d:2c ./engine/build/st2110_tx_smoke
```
Resize-only benchmark (pure GPU, no NIC/root): `./engine/build/resize_smoke`.

## Next
1. **Perf tuning** — FRC raised TX `past_err`→1.2% + ingest latency→~15.6 µs under load. Per-operator
   CUDA streams (currently default-stream serialized + per-frame syncs in resize/frc), fewer syncs,
   and GPUDirect to drop the host packed staging copies in unpack/pack.
2. **FRC rate model** — currently 1:1 motion-comp retiming; true 2× rate-conversion (emit prev + mid)
   needs the TX pacer to handle the doubled output rate.
3. **PTP discipline** (`ptp4l`/`phc2sys`, grandmaster) for production timing.
4. **`deploy/` provisioning** — `REAL_TIME_CLOCK_ENABLE=1` + hugepages (M1 open item #5).
