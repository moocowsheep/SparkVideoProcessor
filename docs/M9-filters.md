# M9 — modular filter chain: downscale, sharpen, proc amp

**Status:** implemented + built 2026-07-01, kernel unit tests green (`test_filters`). Hardware
(picture) validation on the BMD rig pending.

The processing stages between unpack and pack are now a **composable chain**. Every filter is a
GpuFrame → GpuFrame operator with the same contract (own CUDA stream, `cudaStreamWaitEvent` on the
producer's ready event, out-of-place into a private pool, `t_ingest_ns`/`capture_ts_ns` carried
through), so any subset in any order wires identically:

```
rx -> unpack -> [ frc | scale | sharpen | procamp ... ] -> pack -> tx
```

The composed chain is logged at startup (`chain: rx -> unpack -> frc -> scale -> ... -> tx`).

## The filters

| stage | what it does | activation |
|---|---|---|
| `frc` | motion-compensated retime / 2× up-convert (NVOF), unchanged | `SPARK_FRC` = 1 or 2 |
| `scale` | NPP resize up **or down**; `interp=auto` (new default) picks supersampling for downscales (anti-aliased — cubic/lanczos only interpolate, they alias on a shrink), cubic for upscales, and the 1:1 passthrough at identity. Explicit `super` falls back to cubic on a non-shrink (NPP restriction). `interp=fsrcnn \| fsrcnn-s \| espcn` selects **AI super-resolution** for exact-2× upscales (below). | always in the default chain |
| `sharpen` | luma unsharp mask (3×3 gaussian): `y + amount·(y − blur)`; chroma passes through (sharpening chroma just rings). Clamps to the 10-bit non-reserved range [4, 1019]. | `SPARK_SHARPEN` > 0 |
| `procamp` | classic video proc amp in native 10-bit YCbCr: brightness (black-level offset, ±1 = ±full swing), contrast (gain about black 64), saturation (chroma gain about 512), hue (chroma phase rotation). Same [4, 1019] clamp. | any parameter non-neutral |

Default order `frc,scale,sharpen,procamp`: flow estimation at native input resolution (NVOFA search
range), sharpening at the delivery resolution, levels trimmed last.

## Configuration

| env (engine) | proto / JSON (daemon + web) | default | notes |
|---|---|---|---|
| `SPARK_INTERP` | `interp` | `auto` | `auto \| linear \| cubic \| lanczos \| super \| fsrcnn \| fsrcnn-s \| espcn` (`ai` = alias for fsrcnn) |
| `SPARK_SR_WEIGHTS` | — | unset | path to a retrained SRW1 blob overriding the embedded AI weights (falls back to cubic with an error log if unreadable) |
| `SPARK_SHARPEN` | `sharpen` | 0 (off) | unsharp gain, useful range ~0.3–1.5 |
| `SPARK_PA_BRIGHT` | `paBrightness` | 0 | ±1 = ±full Y swing |
| `SPARK_PA_CONTRAST` | `paContrast` | 1 | 0/negative reads as **unset → 1.0** (proto3 zero-default safety; use 0.01 for an effective zero) |
| `SPARK_PA_SAT` | `paSaturation` | 1 | same 0-as-unset rule; 0.01 ≈ mono |
| `SPARK_PA_HUE` | `paHueDeg` | 0 | degrees |
| `SPARK_FILTERS` | `filters` | "" (auto) | explicit chain, e.g. `procamp,frc,scale` — order and set override; repeated tokens allowed (ops get unique names); `frc` still requires `SPARK_FRC`≠0 |

The web Configuration card exposes sharpen + the four proc amp fields; the chain-order override
lives under Advanced. The daemon passes everything through as env (`start_locked`).

Downscaling is just `out_width/out_height` smaller than the input (e.g. 3840×2160 in → 1920×1080
out); with `auto` interp it uses `NPPI_INTER_SUPER`. 1080p59.94 raw output easily fits the 10G TX
path (the non-2160p TX pacing profile, validated in M1).

## AI super-resolution (1080p → 2160p)

`interp=fsrcnn | fsrcnn-s | espcn` runs a pre-trained ×2 CNN on the **luma plane** (chroma goes
through NPP cubic — the nets are trained on luma statistics, and 4:2:2 chroma at w/2×h scales 2×
along with the frame). Exact-2× geometries only; anything else falls back to cubic with a
one-shot warn. Selection is per the normal plumbing: `SPARK_INTERP`, proto `interp`, dashboard
Interpolation dropdown.

| model | net (all convs SAME/stride-1, pixel-shuffle output) | ms/frame 1080p→2160p (GB10 @513 MHz / proj. @3 GHz) |
|---|---|---|
| `fsrcnn` | 56-wide FSRCNN d56/s12/m4 (best quality) | ~37 / ~6.5 |
| `fsrcnn-s` | FSRCNN-small d32/s5/m1 (fastest; real-time even at idle clocks) | ~6.7 / ~1.2 |
| `espcn` | ESPCN 64/32 + tanh output (heaviest; different look) | ~68 / ~12 |

**GPU clocks matter**: the GB10 DVFS governor holds the SM clock at ~513 MHz of 3003 even at 95%
utilization, so fsrcnn/espcn only hit 59.94 fps with locked clocks. **The daemon locks the SM
clock to its rated max at every pipeline start** (`lock_gpu_clocks()` in control/daemon.cpp —
root, idempotent, self-heals a manual `-rgc`; opt out with `SPARK_LOCK_GPU_CLOCKS=0` in the
daemon's env). Standalone runs (`resize_smoke`, `test_sr_net`) don't go through the daemon —
lock by hand first: `sudo nvidia-smi -lgc 3003` (undo: `sudo nvidia-smi -rgc`).

Implementation (`engine/operators/resize/sr_net.cu`): hand-rolled fused CUDA — no
TensorRT/cuDNN exists for this aarch64 box, and the nets are tiny enough that fused direct
kernels win anyway. fp16 activations / fp32 accumulation; weights staged in shared memory;
middle/tail convs register-tile 2–4 pixels per thread; 1×1 convs never touch DRAM (fused into
the 5×5 head / the shuffle tail); zero-pad borders match the TF training graphs. 10-bit video
range maps (Y−64)/876 ↔ [0,1] with no input clamp (sub-black/super-white ride through; espcn's
tanh inherently clips super-whites), output clamped [4, 1019] like every other stage.

Weights are **embedded in the binary** (`sr_weights_data.cpp`, generated by
`tools/sr_weights/convert.py` from pinned, sha256-verified Apache-2.0 TF frozen graphs —
Saafke/FSRCNN_Tensorflow, fannymonori/TF-ESPCN). Regenerate with the script (stdlib-only, no TF);
`SPARK_SR_WEIGHTS=<blob>` swaps in retrained weights at runtime — unknown layer shapes run on
generic fallback kernels, so a retrain doesn't require a rebuild unless it wants full speed.

Validation: `ctest -R sr_net` — GPU vs a CPU reference forward of the same weights (±2 codes,
fp16-storage emulation at the fused-plan boundaries), an "SR beats bilinear" PSNR gate on a
detail pattern (a conversion bug fails this hard), and a structural perf tripwire. Picture
checks on the BMD rig: A/B fsrcnn vs cubic on 1080p→2160p59.94 with FRC — pending.

## Costs (latency budget context, see docs/M8-latency.md)

sharpen and procamp are single-pass point/3×3 kernels — well under 1 ms/frame each at 2160p (the
GB10 zero-copy benchmark moved 2×16.6 MB through a trivial kernel in 0.44 ms). Each active stage
adds one pool hop; with the EventBasedScheduler the added pipeline latency per stage is sub-ms and
absorbed by the TX lead + trim servo. FRC remains the only stage with real GPU weight.

## Validation

- `ctest -R filters` (engine/build): GPU kernels vs CPU references (±1 code-value rounding
  tolerance), neutral-parameter identity, extreme-value clamps.
- On the rig: procamp neutral vs non-neutral A/B on the BMD picture; sharpen amount sweep 0→1.5;
  2160p→1080p downscale (auto vs cubic — look for moiré/aliasing on detail); confirm
  `pipe_latency_us` stays in budget with all stages active.
