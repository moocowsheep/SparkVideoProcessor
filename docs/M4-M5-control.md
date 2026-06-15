# M4 + M5 — control daemon + web dashboard

Status: **DONE (2026-06-15) — `spark_controld` controls/monitors the pipeline over gRPC and HTTP/JSON;
a vanilla web dashboard drives it. Validated end-to-end (lifecycle, config, stats parse, static
serving). Launching the real DPDK pipeline needs the daemon to run privileged (root).**

```
browser ──HTTP/JSON──┐
                     ├─▶ spark_controld ──spawn/env──▶ st2110_pipeline (engine)
grpc client ──gRPC───┘     (control/daemon.cpp)         (rx→unpack→resize→frc→pack→tx)
```

## Control API (`proto/spark_control.proto`)
`SparkControl` service: `GetStatus`, `SetConfig` (STOPPED only), `Start`, `Stop`. Messages
(`PipelineConfig`, `PipelineStatus`, `PipelineStats`) are defined once and reused by both transports —
the daemon converts them to JSON with protobuf's JSON mapping for the browser.

## Daemon (`control/daemon.cpp` → `spark_controld`)
- Implements the gRPC service **and** an HTTP/JSON + static-file server (libmicrohttpd) on one process,
  so the browser needs no gRPC-web proxy. `GET /api/status`, `POST /api/{config,start,stop}`, and the
  `web/` SPA off `/`.
- Manages the pipeline as a child process: `Start` fork/execs `st2110_pipeline` with the config mapped
  to `SPARK_*` env vars and captures its output; `Stop` SIGTERMs it and parses the final stats
  (rx frames/lost, tx future/past_err, ingest latency, FRC count) from the captured log.
- No DPDK/Holoscan/CUDA deps — builds standalone (gRPC + protobuf + libmicrohttpd).

## Dashboard (`web/`)
Dependency-free SPA (no npm/build): state badge, editable config (disabled while running), Start/Stop,
and a 1 Hz stats poll. Served by the daemon.

## Build & run
```bash
cmake -S control -B control/build && cmake --build control/build
# run privileged so it can launch the DPDK pipeline:
sudo ./control/build/spark_controld --pipeline ./engine/build/st2110_pipeline --web web
#   browser → http://localhost:8080     gRPC → localhost:50051
```

## Validated
Daemon as a normal user with a mock pipeline: HTTP `status`/`config`/`start`/`stop`; full
`start → RUNNING → stop → STOPPED` lifecycle; stats parsed from engine output
(`rxFrames=300 rxLost=0 txPackets=4448100 past_err=68813 ingest=15.6µs frc=299`); dashboard served.
gRPC server starts on `:50051` (same state logic as the HTTP path).

## Live stats (done)
The engine operators emit a periodic (1 Hz) machine-parseable line with unique tokens —
`spark_live rx_frames=N rx_packets=N rx_lost=N rx_latency_us=N`, the `tx_*` equivalents
(`tx_future_err`/`tx_past_err`), and `frc_interpolated=N` — from `compute()` during the run plus a
forced final snapshot at `stop()`. The daemon greps those tokens (latest-wins, log tail only) on
every status poll while RUNNING, so the dashboard's 1 Hz poll shows live values. Validated: counters
advance live while RUNNING; final values retained after stop.

## Follow-ons
- **gRPC exercise** — add `grpc_cli`/a small client (or a unit test) to the suite.
- **Auth / TLS** — the daemon is currently open (InsecureServerCredentials + open HTTP); add for
  anything beyond a trusted LAN.
- **Privilege** — run the daemon as root (or via a setcap/sudo helper) to launch the DPDK pipeline.
