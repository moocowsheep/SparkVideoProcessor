# M11 — Replacing the Holoscan SDK with `spark::rt`

Status: **✅ CODE COMPLETE, ⚠️ HARDWARE VALIDATION PENDING.** The engine builds, links and passes all
10 unit tests with zero Holoscan dependencies; the GPU operators run under the new runtime with
unchanged numbers. What has *not* been re-measured is the paced ST 2110 path — that needs a cabled
ConnectX (see "What still has to be proven"). Date: 2026-08-25.

## Why
The engine used Holoscan for exactly one thing: an operator graph. An audit of the whole tree found
it used none of the rest — no `holoscan::Tensor`, no allocators, no HoloInfer, no Holoviz, and
despite the M0 decision text, no `advanced_network` (it survived only in four comments as a possible
future backend). Networking is raw DPDK/mlx5, resize is NPP, FRC is NVOF, super-resolution is
hand-written CUDA. The used API surface was:

`Application` · `Operator` (setup/initialize/start/compute/stop) · `OperatorSpec` · `Parameter<T>` ·
`Arg` · `spec.input`/`spec.output` · `receive`/`emit` · `CountCondition` ·
`kMessageAvailable` · `kDownstreamMessageAffordable` · `kDoubleBuffer` · two schedulers · the log macros.

Against that, the dependency cost was an hour-long containerized SDK build, a **CMake >= 3.30.4**
floor that Ubuntu 24.04 cannot satisfy from its archive (3.28.3 — this is what actually blocked the
x86_64 bring-up), a CUDA-13 coupling through GXF, a build container pinned to Ubuntu 22.04 while the
host runs 24.04, and a ~GB install tree.

The deciding structural fact: **every graph in the repo is a strict linear chain.** All twelve
`add_flow` calls across all seven apps form straight pipes (`rx → unpack → [filters] → pack → tx` is
the longest), and no operator declares more than one input or one output port. A general DAG runtime
was being carried for a straight line.

## What `spark::rt` is
`engine/runtime/` — ~600 lines: `runtime.hpp` (API), `runtime.cpp` (executor), `log.hpp` (logging).

**One thread per operator, one bounded blocking queue per edge.** That single choice supplies what
Holoscan spread across schedulers, transmitters/receivers and conditions:

| Holoscan concept | `spark::rt` |
|---|---|
| `EventBasedScheduler` / `MultiThreadScheduler` | nothing — each operator owns a thread and blocks on its input queue |
| `kMessageAvailable` | the blocking pop |
| `kDownstreamMessageAffordable(min_size=n)` | the bounded queue: a full queue blocks the producer |
| `kDoubleBuffer(capacity=n, policy=2)` | queue capacity; `policy` has no analogue — a blocking queue backpressures instead of overflowing, so there is no drop/fault decision left |
| `CountCondition(n)` | operator stops after n computes, then closes its downstream queue |
| `max_duration_ms` | `Application::max_duration_ms()` |
| `HOLOSCAN_LOG_*` (fmt) | `SPARK_LOG_*` — the four format specs the engine actually uses (`{}`, `{:.1f}`, `{:.2f}`, `{:.3f}`) |

Shutdown is ordered: a finished source *closes* (not stops) its downstream queue, the consumer drains
what is in flight, sees end-of-stream, and closes its own — so a bounded run ends cleanly with nothing
dropped. SIGINT/SIGTERM set a stop flag and unblock every queue, which is how a live pipeline ends.

Because the graph is linear and acyclic, blocking-on-full cannot deadlock: the downstream end is
always either draining or shutting down.

### Latency: why this should be *better*, not merely equal
`docs/M8-latency.md` and the comment at `st2110_pipeline.cpp` recorded the largest avoidable latency
term under Holoscan: `MultiThreadScheduler`'s polling thread slept `check_recession_period_ms`
(default **5 ms**) whenever a pass found nothing ready — up to 5 ms *per hop*, across 5 hops. That is
why the default had already moved to `EventBasedScheduler`. `spark::rt` has no polling thread at all:
a blocked operator wakes on its condition variable the instant upstream pushes. `SPARK_SCHED` is now
accepted-and-ignored so existing run scripts keep working.

## The port
Mechanical, by design — the API deliberately mirrors the Holoscan shapes so the operator bodies (the
code carrying the validated video behavior) changed by namespace swap, not rewrite:
`holoscan::` → `spark::rt::`, `HOLOSCAN_LOG_` → `SPARK_LOG_`, `<holoscan/holoscan.hpp>` →
`"runtime/runtime.hpp"`. 29 files. Beyond that only four things needed thought:

1. **Schedulers deleted** from the four apps that configured one (`worker_thread_number`,
   `stop_on_deadlock*` have no analogue); only the run bound survives as `max_duration_ms()`.
2. **`spark_engine`'s ping graph** used `holoscan::ops::ping_tx/ping_rx`; those are now a local
   ten-line pair, so the build/execution self-test needs nothing external.
3. **`audio_rx.cpp` needed `<cstring>`** — it had been getting `memcpy` transitively from a Holoscan
   header.
4. **`FrameSinkOp` is no longer load-bearing.** It existed because Holoscan only ticked an operator
   whose output was connected, so sink-mode `st2110_rx` had to wire its unused output somewhere.
   `spark::rt` ticks every operator and discards emits on an unconnected port. It is kept because it
   also counts frames, which those apps report.

## Evidence so far
* **Build:** clean, no warnings from engine code, with **stock Ubuntu 24.04 cmake 3.28.3** — the
  3.30.4 blocker is gone as a side effect. `ldd engine/build/st2110_pipeline` lists 29 shared
  libraries, **zero** of them Holoscan/GXF/UCX/RMM.
* **Unit tests:** 10/10 pass, including a new `test_runtime` covering delivery + `CountCondition`,
  the multi-emit burst (the FrcOp 2:1 shape), backpressure through a capacity-1 queue, Arg override,
  and the unconnected-output case.
* **GPU operators under the new runtime:** `resize_smoke` (test_gpu_source → resize → sink, 300
  frames) reports 1080p→2160p NPP resize at **0.072/0.120/12.573 ms** min/avg/max — consistent with
  the ~0.1 ms/frame M0 measured under Holoscan. `frc_smoke` passes all three NVOF cases.

## What still has to be proven (needs the NIC)
The paced ST 2110 path is untouched by this change — the DPDK backends were never Holoscan code —
but the *scheduling around it* is entirely new, so the M1/M8 numbers must be reproduced:

1. **`st2110_tx` at rate:** `future_err=0` at 1080p and 2160p (gate-4 Result 3).
2. **`st2110_rx` zero-loss** + ~2–3 µs ingest latency (gate-4 Result 4).
3. **`st2110_pipeline` end-to-end latency vs `docs/M8-latency.md`** — the headline claim of this
   change is that removing the scheduler poll *lowers* latency; that is a prediction until measured.
4. **FrcOp's burst under real backpressure** — unit-tested here, but never yet at 2160p59.94.

On the x86_64 host all four are blocked on the same thing as gate 4 itself: only one of the two
SFP28 ports is cabled (`docs/M1-gate4-pacing.md`). They can be run on the DGX Spark first.

## Reproduce
```bash
cmake -G Ninja -S engine -B engine/build && cmake --build engine/build
ctest --test-dir engine/build          # 10/10, no NIC or root
./engine/build/spark_engine            # ping graph — runtime self-test
./engine/build/resize_smoke            # GPU operators under spark::rt
./engine/build/test_runtime            # dataflow guarantees, verbose
```
