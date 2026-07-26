# M11 — JPEG XS (ST 2110-22 / RFC 9134) encode + decode

## Licensing

> JPEG XS (ISO/IEC 21122) is a standardized codec that may be covered by patents, and may be subject
> to licensing terms independent of this project's Apache 2.0 license.
>
> **You are solely responsible for determining if your use of jpeg-xs requires any additional
> licenses. The developer of this project is not responsible for obtaining any such licenses, nor
> liable for any licensing fees due, in connection with your use of jpeg-xs.**

The feature is **off by default** and is only compiled when the separately obtained MooCUDAJXS codec
is present. No JPEG XS codec source is distributed in this repository. The same notice appears in
`NOTICE`, `README.md`, the dashboard's Format panel, and the codec operator's header.

## Why

M7 added Blackmagic IP10 because uncompressed 2160p59.94/60 (~12 Gbps) does not fit a 10G link. IP10
solves that for Blackmagic receivers, but it is a vendor codec at a fixed ~2:1. JPEG XS is the
*interoperable* answer to the same problem — it is what ST 2110-22 was written for, it reaches ~5:1
and beyond at visually lossless quality, and a wide range of third-party 2110 gear both sends and
receives it. Adding it makes the processor a peer on a compressed-2110 plant, in both directions.

Both directions are independent, and independent of IP10:

| | in | out |
|---|---|---|
| GUI | `source is JPEG XS` | `JPEG XS output` |
| env | `SPARK_IN_JXS` | `SPARK_JXS` |
| proto | `inJxs` | `jxs` |

So the engine can decode JPEG XS and emit uncompressed, ingest uncompressed and emit JPEG XS,
transcode JPEG XS→JPEG XS through the whole GPU filter chain, or bridge JPEG XS↔IP10. IP10 and
JPEG XS both describe an entire wire format, so setting both on the *same* direction is
contradictory: the GUI prevents it, and the engine resolves it in favour of JPEG XS with a warning.

## Codec

[MooCUDAJXS](../../MooCUDAJXS) — a CUDA-first JPEG XS implementation whose integration contract is
zero-copy: raw frames are pitched CUDA device planes, codestreams live in CUDA-accessible memory, and
work is enqueued on a caller-owned stream. That matches this engine exactly.

`GpuFrame` (planar 10-bit YCbCr 4:2:2 in `uint16`) **is** MooCUDAJXS `YUV422P` at `bit_depth = 10`,
so the codec reads and writes the pipeline's frames with no conversion and no intermediate copy:

| GpuFrame | MooJxsDevicePlane |
|---|---|
| `y` (w × h), pitch `w*2` | `planes[0]` |
| `cb` (w/2 × h), pitch `(w/2)*2` | `planes[1]` |
| `cr` (w/2 × h), pitch `(w/2)*2` | `planes[2]` |

The encoder is **constant-rate**: every frame is exactly `width × height × bpp / 8` bytes, known at
`moo_jxs_encoder_create` time from the workspace plan. That is what lets the TX size its buffers once
and the SDP declare a real `b=AS`.

Measured on the GB10 (`SPARK_JXS_BPP` default 4.0, synthetic gradient + edge material):

| format | bpp | codestream/frame | rate @59.94 | PSNR (Y) |
|---|---|---|---|---|
| 1920×1080 | 4.0 | 1 036 800 B | 497 Mbps | 78.1 dB |
| 3840×2160 | 4.0 | 4 147 200 B | 1.99 Gbps | 78.7 dB |
| 3840×2160 | 6.0 | 6 220 800 B | 2.98 Gbps | mathematically lossless |

(Uncompressed 2160p59.94 4:2:2 10-bit is ~11.9 Gbps for comparison.)

> MooCUDAJXS v0.8 is an **interoperability checkpoint, not an ISO Part 4 conformance claim** — see
> its `docs/CONFORMANCE.md`. The engine reflects that honestly: the encoder writes PIH `Ppih`/`Plev`
> = 0 (unrestricted) by default rather than stamping a standardized profile it has not been
> certified against, and the generated SDP therefore omits `profile`/`level`. Set
> `SPARK_JXS_PROFILE` / `SPARK_JXS_LEVEL` (and add the matching SDP params) if a specific receiver
> demands them.

## Graph

The codec sits exactly where the RFC 4175 pack/unpack bridges sit, speaking the same message types,
so nothing between them changes:

```
raw in :  st2110_rx -> [unpack]     -> nr,frc,scale,sharpen,... -> [pack]       -> st2110_tx
jxs in :  st2110_rx -> [jxs_decode] -> nr,frc,scale,sharpen,... -> [jxs_encode] -> st2110_tx
```

- **`JxsDecodeOp`** — host codestream → `GpuFrame`. Never synchronizes: downstream operators order on
  `GpuFrame::ready`, and the decode status is checked one frame later off a completion event (a ring
  of pinned status words). In zero-copy mode it also holds the RX ring buffer behind that event, since
  the decoder kernels read the codestream in place and the RX recycles any buffer whose `use_count`
  drops to 1.
- **`JxsEncodeOp`** — `GpuFrame` → host codestream. Must synchronize, because the codestream length is
  a device-side result the TX needs on the host to plan its packets. `PackOp`'s D2H sync costs the
  same, so the egress hop's latency profile is unchanged.

Both own one persistent codec handle with all workspace preallocated; frame submission allocates
nothing. Handles are rebuilt only on a geometry change.

## Wire format (RFC 9134)

Unlike RFC 4175, a JPEG XS packet is **position-implicit**. There is no line/offset descriptor — the
codestream is sliced into MTU-sized fragments and rebuilt by concatenation, in order. Every design
consequence in `engine/operators/st2110_tx/rtp_jxs.hpp` follows from that.

Each packet carries a 4-byte payload header between the RTP header and the fragment:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|T|K|L| I |    F    |        SEP        |         P             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

We send `T=1` (sequential), `K=0` (**codestream** packetization), `I=00` (progressive). `L` mirrors
the RTP marker, as the RFC requires in codestream mode. `F` is the frame counter mod 32. `{SEP,P}` is
one 22-bit packet index within the frame: `SEP` is the high 11 bits, so it advances exactly when the
11-bit `P` overruns.

### The packetization unit is a picture *segment*, not a codestream

RFC 9134 §4.2: *"the packetization unit SHALL be the entire JPEG XS picture segment (i.e., codestream
preceded by boxes)"* — every frame is `VS box + CS box + codestream`. So what the RX reassembles is
**not** something a codestream decoder can be handed directly, and `jxs_codestream_offset()` walks the
ISO/IEC 21122-3 box chain (4-byte big-endian size + 4-byte type) to find the SOC before
`JxsDecodeOp` submits it.

This is not theoretical — it is what a RED V-Raptor sends. Captured 2026-07-26 from its 2160p59.94
`jxsv` flow, the 60 bytes ahead of every frame are:

```
0000: 00 00 00 2a 6a 70 76 73   'jpvs' video support superbox (42 B)
0008:   00 00 00 16 6a 70 76 69   'jpvi' video information    (22 B)
001e:   00 00 00 0c 6a 78 70 6c   'jxpl' profile/level        (12 B)
0026:     4a 40 24 06               Ppih=0x4a40 Plev=0x2406
002a: 00 00 00 12 63 6f 6c 72   'colr' colour specification   (18 B)
003c: ff 10 ...                 SOC — the codestream (4 147 200 B = exactly 4.00 bpp)
```

Note `jxpl` carries the same profile/level as the PIH, and the SDP's `profile=High444.12;
level=4k-2` agrees with `0x4a40`/`0x2406`. The walk is bounded (16 boxes / 4 KiB) and rejects
size-0, 64-bit-extended and overlong boxes by returning 0, so a malformed prefix reaches the decoder
unmodified and fails with the decoder's own error rather than being silently re-cut here.

> **Encoder gap.** `JxsEncodeOp` currently emits a **bare codestream** with no preceding boxes, which
> does not satisfy that SHALL. Receivers that take their format from the SDP (which the NMOS node
> generates in full) are unaffected, but a strict receiver may reject the stream. Emitting them needs
> the ISO/IEC 21122-3 field layout for `jpvi`/`colr` — `jxpl` is simply Ppih+Plev and could be written
> today. Not implemented rather than guessed, because a box that misdescribes the stream is worse
> than no box. Tracked as the main known limitation of the send side.

**Loss handling is the real difference from raw.** A lost RFC 4175 packet leaves a hole in a picture
that is still worth showing; a lost JPEG XS fragment shifts every following byte, so the decoder
either errors out or reconstructs plausible garbage. The RX therefore latches any discontinuity —
a non-dense `{SEP,P}`, a new RTP timestamp before the marker, or a codestream over the buffer budget
— and **drops the whole frame**. On air that is a repeated frame, not a tear. The counters are
`jxs_rx_corrupt` / `jxs_rx_incomplete` (engine log), surfaced as one `jxs lost` tile on the dashboard.

Slice packetization mode (`K=1`) is **not** reassembled: its slices carry their own headers and
concatenating them would produce a malformed stream. A sender using it is reported once, by name,
rather than silently mis-decoded.

### Pacing

A codestream has no raster, so the per-line-burst and gapped shapers the raw path uses have nothing
to key on. `send_jxs_frame` spreads packets **evenly** over the pacing span — which *is* the ST 2110-21
shape without line structure. Narrow still means "finish within the active period"; wide uses the
whole frame interval; `SPARK_TX_FILL` tunes on top. The packet count is recomputed per frame.

Per-packet send-on-timestamp WQE cost, which forced the raw 2160p path onto one-stamp-per-line, is a
non-issue here: at 4 bpp a 2160p frame is ~3k packets against raw's ~15k, so the standard 8192-entry
TX ring already holds a whole frame.

### SDP

The NMOS node rewrites the video sender's SDP (`control/nmos/spark_node.cpp`, `apply_jxs_video`),
alongside the existing IP10 rewrite:

```
a=rtpmap:96 jxsv/90000
a=fmtp:96 sampling=YCbCr-4:2:2; depth=10; width=3840; height=2160; exactframerate=60000/1001; colorimetry=BT709; packetmode=0; transmode=1; SSN=ST2110-22:2022; TP=2110TPN;
b=AS:2068
```

`depth` and `sampling` describe the **decoded** image, as they do for IP10. `b=AS` is computed from
the constant-rate frame size. On the ingest side the video receiver advertises `video/jxsv` and
`parse_video_fmtp` detects a JPEG XS sender from its rtpmap, so routing one in via IS-05
auto-enables decode (`inJxs`), exactly as an IP10 source auto-enables `inIp10`.

## Configuration

| env | proto / JSON | default | meaning |
|---|---|---|---|
| `SPARK_JXS` | `jxs` | 0 | encode the output as JPEG XS |
| `SPARK_IN_JXS` | `inJxs` | 0 | source is JPEG XS (decode on ingest) |
| `SPARK_JXS_BPP` | `jxsBpp` | 4.0 | encoder constant-rate target, bits/pixel |
| `SPARK_JXS_MAX_BPP` | `jxsMaxBpp` | 12.0 | decoder-side worst-case budget, bits/pixel |
| `SPARK_JXS_MAX_COLUMNS` | — | 4 | worst-case PIH `Cw` the decoder plans for |

`jxsBpp` and `jxsMaxBpp` are deliberately separate: the outgoing rate is our encoder's choice, while
the incoming rate belongs to somebody else's encoder and is **not carried in the SDP**. The receive
budget sizes the reassembly buffers; a source above it has frames dropped rather than truncated, and
says so in the log with the value to raise.

## Build

```sh
cmake -G Ninja -S engine -B engine/build \
  -DCMAKE_PREFIX_PATH="$HOLOSCAN_INSTALL" \
  -DSPARK_JXS_DIR=/path/to/MooCUDAJXS      # optional: a sibling checkout is auto-detected
cmake --build engine/build
```

The configure step prints which it chose:

```
-- JPEG XS: MooCUDAJXS at /home/saturn/claude/MooCUDAJXS
-- JPEG XS: MooCUDAJXS NOT found — building without ST 2110-22 support
```

Without the codec the engine builds and runs exactly as before; `JxsDecodeOp`/`JxsEncodeOp` still
exist so the app wires them unconditionally, but they refuse to start with a message naming the CMake
flag. The pipeline app also logs an error at compose time if JPEG XS was requested in such a build.

MooCUDAJXS is built as a subproject with its own tests/tools off and its CUDA architectures narrowed
to the engine's (`sm_121`), and with `CUDA_RESOLVE_DEVICE_SYMBOLS` on — it uses separable
compilation, and the consuming targets here are plain C++ that never device-link.

## Tests

`ctest --test-dir engine/build -R rtp_jxs` — the RFC 9134 framing round trip (pure CPU, no NIC, no
GPU, no codec). It fragments synthetic codestreams, runs every packet through the *real*
`JxsDepacketizer`, and asserts byte-identical reassembly plus the header fields that carry position:
the dense `{SEP,P}` counter across a 2048-packet `P` overrun, `F` mod 32 across a wrap, `L == M`,
`K`/`I`, continuous sequence numbers across frame boundaries, 16-bit sequence extension, and
rejection of malformed / CSRC / header-extension packets.

Codec-level encode→decode fidelity is MooCUDAJXS's own test suite (`ctest` in that repo).

## Live interop — RED V-Raptor (2026-07-26)

Receive path validated against a real ST 2110-22 sender: a V-Raptor at 192.0.2.95 emitting
2160p59.94 4:2:2 10-bit `jxsv` on 232.200.45.110:5004, `packetmode=0`, `sublevel=Sublev4bpp`,
`b=AS:2049820`. Driven through the engine's own `JxsDepacketizer` + `ingest_jxs` reassembly rules +
`JxsDecodeOp`'s decoder configuration over a plain socket (i.e. everything but DPDK ingest):

| | result |
|---|---|
| packets | 3 473 104, **0 malformed** |
| frames reassembled | 1 198, 2 dropped corrupt, 0 slice-mode |
| picture segment | 4 147 260 B every frame — 60 B of boxes + 4 147 200 B = **exactly 4.00 bpp** |
| decode | **400 / 400 frames decoded, 0 failures**, all four stages `success` |
| picture | correct BT.709 image, luma 25–960, chroma off-neutral — verified visually |

Decode throughput on the GB10, measured on a real-content frame from this camera (300 iterations,
one warm-up, persistent handle): **11.19 ms/frame = 89.3 fps**, against the 16.68 ms budget for
2160p59.94 — roughly 1.5× headroom.

The first session decoded flat black, which turned out to be a **faulty network cable** on the
camera, not a codec fault. Worth recording how that was diagnosed, because byte statistics alone
were ambiguous: capturing ten consecutive frames and hashing each codestream showed all ten
**bit-identical**. A live sensor cannot do that — photon and read noise vary every frame — so the
camera had to be emitting synthesized blanking rather than digitising. After the cable was replaced
the same test produced ten distinct codestreams and a correct picture.

Three environment notes from those sessions:

- **A kernel-socket receiver cannot keep up with this flow on a default host.** `rmem_max` is 208 KB
  here, about 0.85 ms of buffering at 2 Gbps, so a test tool that decodes inline on the socket thread
  drops every other frame (measured: 29.9 fps, 50 % packet loss). That is a property of the harness,
  not the pipeline — `St2110RxOp` uses DPDK with a dedicated poll thread and no kernel socket. Do not
  read socket-tool loss figures as engine performance.

- **The sender must be activated.** The V-Raptor advertises the flow in IS-04 while
  `master_enable` is `false`; nothing is on the wire until its IS-05 `/staged` endpoint is PATCHed
  with `{"master_enable": true, "activation": {"mode": "activate_immediate"}}`.
- **The segment's IGMP querier is v2**, which forces v2 compatibility mode and makes source-specific
  (SSM) joins ineffective from the kernel stack — socket-based tools must fall back to an any-source
  join. `St2110RxOp` is unaffected: it builds its own IGMPv3 reports through DPDK
  (`operators/common/net_addr.hpp`).
