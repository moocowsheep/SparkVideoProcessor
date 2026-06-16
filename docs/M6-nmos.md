# M6 — NMOS discovery (IS-04 / IS-05)

**Goal:** discover/route ST 2110 sources over the network instead of typing raw PCIe BDFs / MACs. The
processor becomes a full **NMOS Node** (Sony **nmos-cpp**): it advertises video (ST 2110-20) + audio
(ST 2110-30) **Receivers** to ingest, plus matching **Senders** for the processed output, all under a
PTP-locked clock — so a broadcast controller (or our own UI) routes sources to us via **IS-05**, and
our output is itself routable. Supports both **registered** (registry / DNS-SD) and **peer-to-peer**
(mDNS) discovery.

```
controller / registry ──IS-04 query / IS-05 connect──▶ spark NMOS Node (control/nmos)
                                                          │  on IS-05 activation (P2)
                                                          ▼
                                          PipelineConfig (mcast/port/format) ─▶ engine
```

## Status
- **P0 — toolchain (done).** `deploy/nmos.sh` builds nmos-cpp from Ubuntu 24.04 **system packages, no
  conan** (cpprestsdk shim + `WEBSOCKETPP_INCLUDE_DIR` + bundled json-schema-validator/jwt-cpp +
  Avahi). Smoke-tested: stock node discovers a registry over mDNS and registers.
- **P1 — Node resource model + PTP clock (done).** `control/nmos/spark_node.cpp`: Node/Device, PTP
  `clk0`, video+audio Senders/Receivers, IS-05 auto-resolver + SDP setter. Registers and emits valid
  ST 2110 SDP (`a=ts-refclk:ptp`). Built behind CMake `-DSPARK_WITH_NMOS=ON`.
- **P2 — IS-05 ↔ engine bridge (done).** The node's `on_connection_activated` handler translates a
  Receiver activation into a `PipelineConfig` and drives the engine **through the existing control
  daemon's HTTP API** (`/api/config` + `/api/start`); releasing the Receiver stops it. The daemon
  passes the new fields to the engine as `SPARK_RX_MCAST/SRC/PORT/IFACE`, `SPARK_IN_W/H/FPS/...`, etc.
  Validated end-to-end against a mock pipeline: a simulated controller PATCH routing a 2110 source
  (multicast `239.50.0.5`, SDP `1920x1080` `YCbCr-4:2:2` 10-bit) reached the engine as the right env.
- **P3 — engine real multicast RX/TX, video (done).** `dpdk_rx_backend` joins the group via an
  **IGMPv3** Membership Report (emitted out a TX queue on the RX port), enables allmulticast, and
  filters received packets by **group + source + dst-port** (was promiscuous + dst-port only).
  `dpdk_tx_backend` derives the egress MAC from the group (**RFC 1112**) for multicast. Addressing
  logic is in the DPDK-free `operators/common/net_addr.hpp` and unit-tested (`test_net_addr`, runs in
  ctest, no NIC). `st2110_pipeline` wires the `SPARK_RX_MCAST/SRC/PORT/IFACE` + `SPARK_TX_MCAST/PORT`
  env into the operators. Legacy promiscuous loopback is preserved (empty group / explicit MAC).
  *Full on-wire validation needs the CX-7 + a real/software sender (P5).*
- **P3b — engine ST 2110-30 audio path (done).** Pure 2110-30 framing core
  (`operators/audio/audio_st2110.{hpp,cpp}`: L16/L24 packetizer/depacketizer + a fixed lip-sync
  `AudioDelayLine`), unit-tested (`test_audio_st2110`, in ctest). Holoscan operators `AudioRxOp` /
  `AudioDelayOp` / `AudioTxOp` and an `audio_passthrough` app (audio_rx → delay → audio_tx, shared
  EAL), reusing the multicast/IGMP RX and tx_pp TX backends. Reads `SPARK_RX_AUDIO_*` /
  `SPARK_TX_AUDIO_*` / `SPARK_AUDIO_{CH,DEPTH,PTIME,DELAY_MS}`. *Runs as its own process today.*
- **P4 — web UI for discovery (done).** The dashboard (`web/`) gained a **Discover sources** panel and
  an NMOS node/PTP-clock status line. Because nmos-cpp APIs are CORS-enabled, the browser talks
  **directly** to the registry (IS-04 Query API) and our node (IS-05 Connection API) — no daemon/proxy
  changes. Picking a sender fetches its SDP and PATCHes our receiver's `/staged` (activate-immediate);
  the node derives transport params from the SDP and the P2 bridge starts the engine. The manual
  PCIe/MAC fields moved into an "Advanced" disclosure. Validated end-to-end via the exact dashboard
  calls (discover → connect → engine RUNNING on the discovered group).
- **P5 — interop / on-wire validation (harness added; on-wire pending hardware).** A dependency-free
  **software ST 2110 sender** (`spike/st2110_software_sender.cpp`) reuses the engine's pure RFC 4175 /
  RFC 3190 framing over kernel multicast sockets (no DPDK/root) to feed the processor without a camera,
  plus a matching `--recv` validator. `deploy/p5_bench.sh` builds it, self-tests over loopback, and
  stands up a registry + sender + NMOS "camera" to drive from the dashboard. Still needs the CX-7 +
  cabling for the real on-wire test, and the shared-port RX demux for one-process video+audio.

## Bench harness (P5)
```bash
bash deploy/p5_bench.sh --build       # compile the software sender (g++, pure C++)
bash deploy/p5_bench.sh --selftest    # send<->recv over the default mcast iface — no NIC needed
bash deploy/p5_bench.sh --run         # registry + software sender + NMOS camera; connect from dashboard
#   env: GROUP=239.100.0.10 PORT=5004 IFACE=<media-NIC-ip> PROFILE=1080p AUDIO=1 REGISTRY=<url>
```
- `st2110_software_sender --send` emits a correct (not 2110-21-paced) RFC 4175 2110-20 stream (+ a
  2110-30 L24 tone with `--audio`); `--recv` joins the group and depacketizes with the real engine
  `Depacketizer`, reporting frames/packets/loss — so the harness self-validates with zero hardware.
- The self-test reconstructs full frames over loopback (a handful of socket-buffer drops on bursts are
  expected — kernel sockets, no HW pacing).
- For the on-wire test, run the processor (`spark_controld` + `spark_nmos_node`) on the CX-7 box,
  point the dashboard's Registry at the bench, and Connect the camera's video source.

## Dashboard (P4)
- **Discover sources** card: editable Registry + Node URLs (saved to localStorage; default
  `http://<host>:3211` / `:3242`), a node/clock status line (label + PTP gmid/traceable), and a list
  of ST 2110 senders (label + format) with **Connect / Disconnect** per sender. Connected sources show
  a "routed" badge (from the receivers' IS-04 `subscription`).
- **Connect** = `GET sender.manifest_href` (SDP) → `PATCH <node>/x-nmos/connection/v1.1/single/
  receivers/<id>/staged` with `{sender_id, master_enable, transport_file, activation:immediate}`.
- Everything else (Start/Stop, processing config, stats) is unchanged and still drives the daemon.

### Audio (P3b) and the shared-port follow-on
`audio_passthrough` currently owns the CX-7 ports as a separate process. To carry **video + audio in
one process on one physical port**, the RX backend needs a **shared-port demux**: one port joins both
groups and tags each `RxPacket` with its UDP dst-port so a splitter routes video (RFC 4175) vs audio
(2110-30). Until then, the daemon would launch `audio_passthrough` alongside `st2110_pipeline` (audio
on the second port), or audio runs standalone. The framing/operators/delay are complete + tested; this
is purely an IO-plane wiring step (tracked for P5/on-wire bring-up).

## Engine multicast (P3, video)
- **`operators/common/net_addr.hpp`** (pure, header-only): `ipv4_is_multicast`, `multicast_mac`
  (RFC 1112), `build_igmpv3_join` (full Eth/IP+Router-Alert/IGMPv3 frame, ASM `EXCLUDE{}` or SSM
  `INCLUDE{src}`), `inet_checksum`. Unit-tested.
- **RX** (`dpdk_rx_backend.cpp`): on init, if a group is set → add a TX queue, `allmulticast_enable`,
  send the IGMPv3 join; `parse_udp` drops anything not matching the group (and `src_ip` if set).
- **TX** (`dpdk_tx_backend.cpp`): when `dst_mac` is unset (all-zero) and `dst_ip` is multicast, the
  dst MAC is derived from the group; an explicit MAC still wins (gate-4 loopback).
- IGMP is sent once at join; answering periodic general queries is a follow-on.

## Bridge (P2)
```
controller ──IS-05 PATCH /staged (master_enable, transport_params, SDP)──▶ spark_nmos_node
   on_connection_activated ─▶ EngineController (worker thread, cpprest http_client)
      Receiver active  ─▶ POST /api/config {rxMcastGroup,rxSrcIp,rxDstPort,inWidth,...} ─▶ POST /api/start
      Receiver released ─▶ POST /api/stop
                                   │
                         spark_controld ──SPARK_* env──▶ st2110_pipeline
```
- The node drives the daemon over **HTTP** (it already links cpprest) — no nmos-cpp deps leak into the
  lean `spark_controld`, which stays the single engine-lifecycle owner.
- HTTP runs on the controller's worker thread, never blocking the nmos-cpp activation thread.
- Video + audio receivers are aggregated; the engine (re)starts when the **video** receiver is active
  (fork/exec engine has no hot-reconfig, so config changes cycle it — a last-applied snapshot avoids
  needless restarts). Sender activation records the output group/port.

## Build
```bash
sudo bash deploy/nmos.sh --packages      # one-time: apt deps (needs password)
bash deploy/nmos.sh --build              # build + install nmos-cpp -> third_party/nmos-cpp/install
/usr/bin/cmake -S control -B control/build -DSPARK_WITH_NMOS=ON   # use cmake 3.x, not /usr/local 4.x
/usr/bin/cmake --build control/build --target spark_nmos_node
```
Heavy nmos-cpp deps (cpprestsdk/Boost/Avahi) stay isolated to the `spark_nmos_node` target; the lean
`spark_controld` build is unchanged.

## Run / verify (local registry + node)
```bash
R=third_party/nmos-cpp/build
$R/nmos-cpp-registry '{"logging_level":-10}' &                                   # registry on 3210/3211/3212
./control/build/spark_nmos_node '{"logging_level":-10,"http_port":3242,"spark_ptp_gmid":"00-00-5e-ff-fe-00-53-00"}' &
curl -s http://127.0.0.1:3211/x-nmos/query/v1.3/senders     # our video/raw + audio/L24 senders
```
- `http_port` multiplexes all the node's APIs onto one port — set it (e.g. 3242) so the node doesn't
  collide with the registry's own 3209/3212/3208 when both run on one host.
- The PTP grandmaster identity comes from `pmc` (e.g. `pmc -u -b 0 'GET PARENT_DATA_SET'`); the daemon
  passes it as `spark_ptp_gmid`. With `spark_ptp_traceable=true` the SDP shows `ts-refclk:...:traceable`;
  set it `false` to embed the explicit gmid.

## Resource model (`control/nmos/spark_node.cpp`)
| Resource | Detail |
|---|---|
| Node | interfaces = host media interfaces; clock `clk0` = `ptp` (gmid, traceable, locked) |
| Device | groups the senders + receivers |
| Sender (video) | flow `video/raw`, 2110-20; SDP `width/height/exactframerate/sampling/depth/...` |
| Sender (audio) | flow `audio/L24`, 2110-30; SDP `L24/48000/2`, `ptime` |
| Receiver (video) | `media_types: [video/raw]`, caps: grain_rate / frame size / sampling / depth |
| Receiver (audio) | `media_types: [audio/L24]`, caps: channel_count / sample_rate / depth / packet_time |

IS-05 callbacks: `on_resolve_auto` (source/destination/interface IP; output multicast groups),
`on_set_transportfile` (builds the sender SDP — PTP `ts-refclk` is derived from the node clock),
`on_connection_activated` (P1: logs; **P2: drives the engine**).

## Notes
- v1 scope: **single-path** (ST 2022-7 dual-path deferred), **video + audio** (ANC/data later).
- nmos-cpp + Avahi logs noisy `DNSServiceCreateConnection ... -65544` warnings — non-fatal; advertise
  and browse both work through the Avahi compat layer.
