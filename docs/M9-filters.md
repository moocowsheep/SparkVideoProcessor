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
| `scale` | NPP resize up **or down**; `interp=auto` (new default) picks supersampling for downscales (anti-aliased — cubic/lanczos only interpolate, they alias on a shrink), cubic for upscales, and the 1:1 passthrough at identity. Explicit `super` falls back to cubic on a non-shrink (NPP restriction). | always in the default chain |
| `sharpen` | luma unsharp mask (3×3 gaussian): `y + amount·(y − blur)`; chroma passes through (sharpening chroma just rings). Clamps to the 10-bit non-reserved range [4, 1019]. | `SPARK_SHARPEN` > 0 |
| `procamp` | classic video proc amp in native 10-bit YCbCr: brightness (black-level offset, ±1 = ±full swing), contrast (gain about black 64), saturation (chroma gain about 512), hue (chroma phase rotation). Same [4, 1019] clamp. | any parameter non-neutral |

Default order `frc,scale,sharpen,procamp`: flow estimation at native input resolution (NVOFA search
range), sharpening at the delivery resolution, levels trimmed last.

## Configuration

| env (engine) | proto / JSON (daemon + web) | default | notes |
|---|---|---|---|
| `SPARK_INTERP` | `interp` | `auto` | `auto \| linear \| cubic \| lanczos \| super` |
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
