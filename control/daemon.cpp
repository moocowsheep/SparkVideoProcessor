// Copyright 2026 Devin Block
// SPDX-License-Identifier: Apache-2.0

// Spark Video Processor — control daemon (M4).
// Exposes the pipeline control as gRPC (SparkControl) AND HTTP/JSON + static web (libmicrohttpd) so
// the vanilla web dashboard can drive it with fetch() — no gRPC-web proxy. Manages the
// st2110_pipeline process (env from config) and parses its final stats. No DPDK/CUDA deps.
#include <arpa/inet.h>
#include <fcntl.h>
#include <microhttpd.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include <grpcpp/grpcpp.h>
#include <google/protobuf/util/json_util.h>

#include "spark_control.grpc.pb.h"

using namespace spark::control;

// ---------------- shared state ----------------
namespace {
std::string g_pipeline = "./engine/build/st2110_pipeline";
std::string g_web = "web";
const char* kLog = "/tmp/spark_pipeline.log";

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct State {
  std::mutex mu;
  PipelineConfig config;
  PipelineState state = STOPPED;
  int pid = -1;
  double start_time = 0;
  PipelineStats stats;
  std::string message = "idle";
  std::string ptp_gmid;  // live PTP grandmaster (from pmc); the NMOS node mirrors it into its clock/SDP

  State() {  // sensible defaults (the gate-4 / pipeline topology)
    config.set_profile("1080p");
    config.set_out_width(3840);
    config.set_out_height(2160);
    config.set_interp("auto");  // supersampling on downscale, cubic on upscale, 1:1 passthrough
    // FRC defaults OFF (frc=false, frc_mode=0 via proto3 defaults) — enable per run from the web
    // UI / NMOS (1=retime, 2=up-convert 30->60, 3=uniform-grid up-convert).
    config.set_pa_contrast(1.0);    // proc amp neutral (0 would read as "unset" -> 1.0 anyway)
    config.set_pa_saturation(1.0);
    config.set_rx_pci("0000:01:00.0");
    config.set_tx_pci("0002:01:00.0");  // up port on this rig (re-cabled 2026-07: .1 is the down link)
    config.set_dst_mac("00:00:5e:00:53:30");
    config.set_frames(0);  // 0 = continuous (live routing); bounded runs set an explicit count
  }
};
State g_state;

// Pull "key=<digits>" (last occurrence) from a blob; returns def if absent.
uint64_t grab(const std::string& s, const std::string& key, uint64_t def = 0) {
  size_t pos = s.rfind(key);
  if (pos == std::string::npos) return def;
  pos += key.size();
  uint64_t v = 0;
  bool any = false;
  while (pos < s.size() && isdigit((unsigned char)s[pos])) {
    v = v * 10 + (s[pos] - '0');
    ++pos;
    any = true;
  }
  return any ? v : def;
}

// Query the live PTP grandmaster from the local ptp4l (root-only pmc UDS — the daemon runs as root)
// and format it as an NMOS gmid: EUI-64, lower-case, dash-separated (e.g. 00-00-5e-ff-fe-00-53-88).
// Returns "" if ptp4l/pmc is unavailable or has no lock, so callers keep the last known value.
std::string read_ptp_gmid() {
  // pmc must be told the PTP domain — it defaults to 0 and ptp4l (ST 2059-2 media domain 127,
  // deploy/ptp4l.conf) silently ignores management messages for other domains, so an unqualified
  // query returns nothing and the gmid poll runs empty forever (the NMOS node then keeps its
  // stale launch seed). SPARK_PTP_DOMAIN overrides.
  const char* dom_env = std::getenv("SPARK_PTP_DOMAIN");
  const int domain = dom_env ? std::atoi(dom_env) : 127;
  // absolute path: a root daemon's popen PATH may not include /usr/sbin where pmc lives
  const std::string cmd = "timeout 2 /usr/sbin/pmc -u -b 0 -d " + std::to_string(domain) +
                          " 'GET PARENT_DATA_SET' 2>/dev/null";
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) return "";
  std::string gmid;
  char buf[512];
  while (fgets(buf, sizeof buf, p)) {
    std::string s(buf);
    const auto pos = s.find("grandmasterIdentity");
    if (pos == std::string::npos) continue;
    std::istringstream iss(s.substr(pos + 19));  // past "grandmasterIdentity"
    std::string tok;  // e.g. "7c2e0d.fffe.1e4b93"
    iss >> tok;
    std::string hex;
    for (char c : tok) if (std::isxdigit((unsigned char)c)) hex += (char)std::tolower((unsigned char)c);
    if (hex.size() == 16)
      for (size_t i = 0; i < 16; i += 2) { if (i) gmid += '-'; gmid += hex.substr(i, 2); }
  }
  pclose(p);
  return gmid;
}

// Lock the GPU SM clock to its rated max before the pipeline starts. The GB10's DVFS governor
// never boosts under CUDA load (SM pinned ~513 MHz of 3003 at 95% util, docs/M9-filters.md), so
// without this the compute-heavy stages (AI super-resolution, FRC) run ~6x slow — and locked
// clocks are what a media box wants anyway: a DVFS ramp mid-stream reads as latency wobble.
// Needs root (the daemon is). Idempotent, so re-locking on every start is free and self-heals a
// manual `nvidia-smi -rgc`. SPARK_LOCK_GPU_CLOCKS=0 (daemon env) opts out. Failure is a warning,
// not fatal: a box without nvidia-smi (or a future non-root daemon) still runs, just slower.
void lock_gpu_clocks() {
  const char* opt = std::getenv("SPARK_LOCK_GPU_CLOCKS");
  if (opt && std::string(opt) == "0") return;
  // absolute path like pmc above: a root daemon's PATH may be minimal
  FILE* p = popen(
      "/usr/bin/nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null",
      "r");
  char buf[64] = {};
  const bool got = p && fgets(buf, sizeof buf, p) != nullptr;
  if (p) pclose(p);
  const int max_sm = got ? std::atoi(buf) : 0;
  if (max_sm <= 0) {
    std::fprintf(stderr, "gpu clocks: nvidia-smi unavailable — leaving DVFS alone\n");
    return;
  }
  const std::string cmd =
      "/usr/bin/nvidia-smi -lgc " + std::to_string(max_sm) + " >/dev/null 2>&1";
  if (std::system(cmd.c_str()) == 0)
    std::fprintf(stderr, "gpu clocks: SM locked to %d MHz for the pipeline (nvidia-smi -rgc undoes)\n",
                 max_sm);
  else
    std::fprintf(stderr,
                 "gpu clocks: WARNING lock failed (nvidia-smi -lgc %d; not root?) — GPU stages "
                 "may run ~6x slow\n",
                 max_sm);
}

void parse_stats(PipelineStats& st) {
  std::ifstream f(kLog, std::ios::binary | std::ios::ate);
  if (!f) return;
  const std::streamoff TAIL = 65536;  // only the tail: latest spark_live lines, O(1) per poll
  const std::streamoff size = f.tellg();
  f.seekg(size > TAIL ? size - TAIL : 0);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string log = ss.str();
  // Engine operators emit periodic "spark_live <op>_<field>=N" lines (unique tokens); latest-wins.
  st.set_rx_frames(grab(log, "rx_frames="));
  st.set_rx_packets(grab(log, "rx_packets="));
  st.set_rx_lost(grab(log, "rx_lost="));
  st.set_tx_frames(grab(log, "tx_frames="));
  st.set_tx_packets(grab(log, "tx_packets="));
  st.set_tx_future_err(grab(log, "tx_future_err="));
  st.set_tx_past_err(grab(log, "tx_past_err="));
  st.set_frc_interpolated(grab(log, "frc_interpolated="));
  st.set_ingest_latency_us(static_cast<double>(grab(log, "rx_latency_us=")));
  // M10: schedule + audio relay health (tokens absent -> 0, e.g. video-only or servo mode)
  st.set_tx_reanchors(grab(log, "tx_reanchors="));
  st.set_tx_skipped(grab(log, "tx_skipped="));
  st.set_fixed_latency_ms(grab(log, "tx_fixed_ms="));
  st.set_tx_margin_us(grab(log, "tx_margin_us="));
  st.set_pipe_latency_us(grab(log, "pipe_latency_us cur="));
  st.set_audio_rx_packets(grab(log, "audio_rx_pkts="));
  st.set_audio_tx_packets(grab(log, "audio_tx_pkts="));
  st.set_audio_late(grab(log, "audio_tx_late=") + grab(log, "audio_tx_drop=") +
                    grab(log, "audio_rx_drop="));
}

// Reap the child if it exited on its own; update state. Caller holds the lock.
void reap_locked(State& s) {
  if (s.state != RUNNING || s.pid <= 0) return;
  int status = 0;
  if (waitpid(s.pid, &status, WNOHANG) > 0) {
    parse_stats(s.stats);
    const bool clean = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    s.state = clean ? STOPPED : ERRORED;
    s.message = clean ? "pipeline exited" : "pipeline exited with error (see " + std::string(kLog) + ")";
    s.pid = -1;
  }
}

bool start_locked(State& s, std::string& msg) {
  reap_locked(s);
  if (s.state == RUNNING) {
    msg = "already running";
    return false;
  }
  lock_gpu_clocks();  // parent, pre-fork: popen/system are not fork-safe in a threaded process
  const pid_t pid = fork();
  if (pid < 0) {
    msg = "fork failed";
    return false;
  }
  if (pid == 0) {  // child: redirect output to the log, set env, exec the pipeline
    int fd = open(kLog, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
      dup2(fd, 1);
      dup2(fd, 2);
      close(fd);
    }
    const PipelineConfig& c = s.config;
    setenv("SPARK_PROFILE", c.profile().c_str(), 1);
    setenv("SPARK_OUT_W", std::to_string(c.out_width()).c_str(), 1);
    setenv("SPARK_OUT_H", std::to_string(c.out_height()).c_str(), 1);
    setenv("SPARK_INTERP", c.interp().c_str(), 1);
    // SPARK_FRC is a mode (0=off, 1=retime, 2=up-convert, 3=uniform-grid up-convert). frc_mode
    // supersedes the legacy frc bool.
    const int frc_mode = c.frc_mode() > 0 ? static_cast<int>(c.frc_mode()) : (c.frc() ? 1 : 0);
    setenv("SPARK_FRC", std::to_string(frc_mode).c_str(), 1);
    setenv("SPARK_IP10", c.ip10() ? "1" : "0", 1);  // Blackmagic IP10 10:8 output (for 2160p60 to BMD)
    setenv("SPARK_IN_IP10", c.in_ip10() ? "1" : "0", 1);  // source is IP10 (decode on RX)
    // Filter chain (M9): sharpen amount, proc amp params (0 contrast/saturation = unset -> the engine
    // treats them as neutral 1.0), and the optional explicit chain-order override.
    setenv("SPARK_SHARPEN", std::to_string(c.sharpen()).c_str(), 1);
    setenv("SPARK_PA_BRIGHT", std::to_string(c.pa_brightness()).c_str(), 1);
    setenv("SPARK_PA_CONTRAST", std::to_string(c.pa_contrast()).c_str(), 1);
    setenv("SPARK_PA_SAT", std::to_string(c.pa_saturation()).c_str(), 1);
    setenv("SPARK_PA_HUE", std::to_string(c.pa_hue_deg()).c_str(), 1);
    // Spatial NR, film grain + A/V delay (0 grain size/mode = unset -> engine defaults 1.5/mono).
    setenv("SPARK_NR_LUMA", std::to_string(c.nr_luma()).c_str(), 1);
    setenv("SPARK_NR_CHROMA", std::to_string(c.nr_chroma()).c_str(), 1);
    setenv("SPARK_GRAIN", std::to_string(c.grain()).c_str(), 1);
    setenv("SPARK_GRAIN_SIZE", std::to_string(c.grain_size()).c_str(), 1);
    setenv("SPARK_GRAIN_MODE", c.grain_mode().c_str(), 1);
    setenv("SPARK_DELAY_VIDEO_MS", std::to_string(c.delay_video_ms()).c_str(), 1);
    setenv("SPARK_DELAY_AUDIO_MS", std::to_string(c.delay_audio_ms()).c_str(), 1);
    setenv("SPARK_FILTERS", c.filters().c_str(), 1);
    setenv("SPARK_RX_PCI", c.rx_pci().c_str(), 1);
    setenv("SPARK_TX_PCI", c.tx_pci().c_str(), 1);
    setenv("SPARK_DST_MAC", c.dst_mac().c_str(), 1);
    setenv("SPARK_FRAMES", std::to_string(c.frames()).c_str(), 1);
    // NMOS / network-layer source + sink (M6) — empty unless an IS-05 connection set them.
    setenv("SPARK_RX_MCAST", c.rx_mcast_group().c_str(), 1);
    setenv("SPARK_RX_SRC", c.rx_src_ip().c_str(), 1);
    setenv("SPARK_RX_PORT", std::to_string(c.rx_dst_port()).c_str(), 1);
    setenv("SPARK_RX_IFACE", c.rx_iface_ip().c_str(), 1);
    setenv("SPARK_RX_AUDIO_MCAST", c.rx_audio_mcast_group().c_str(), 1);
    setenv("SPARK_RX_AUDIO_SRC", c.rx_audio_src_ip().c_str(), 1);
    setenv("SPARK_RX_AUDIO_PORT", std::to_string(c.rx_audio_dst_port()).c_str(), 1);
    setenv("SPARK_IN_W", std::to_string(c.in_width()).c_str(), 1);
    setenv("SPARK_IN_H", std::to_string(c.in_height()).c_str(), 1);
    setenv("SPARK_IN_FPS", c.in_exactframerate().c_str(), 1);
    setenv("SPARK_IN_DEPTH", std::to_string(c.in_depth()).c_str(), 1);
    setenv("SPARK_IN_SAMPLING", c.in_sampling().c_str(), 1);
    setenv("SPARK_TX_MCAST", c.tx_mcast_group().c_str(), 1);
    setenv("SPARK_TX_PORT", std::to_string(c.tx_dst_port()).c_str(), 1);
    // Only override the egress source IP when set (NMOS-resolved); empty keeps the engine default.
    if (!c.tx_src().empty()) setenv("SPARK_TX_SRC", c.tx_src().c_str(), 1);
    // TX pacing fill (narrow-profile): only override when set (>0); 0 keeps the engine default (0.9).
    if (c.tx_fill() > 0.0) setenv("SPARK_TX_FILL", std::to_string(c.tx_fill()).c_str(), 1);
    setenv("SPARK_TX_AUDIO_MCAST", c.tx_audio_mcast_group().c_str(), 1);
    setenv("SPARK_TX_AUDIO_PORT", std::to_string(c.tx_audio_dst_port()).c_str(), 1);
    execl(g_pipeline.c_str(), g_pipeline.c_str(), (char*)nullptr);
    _exit(127);  // exec failed
  }
  s.pid = pid;
  s.state = RUNNING;
  s.start_time = now_s();
  s.message = "started pid " + std::to_string(pid);
  msg = s.message;
  return true;
}

bool stop_locked(State& s, std::string& msg) {
  reap_locked(s);
  if (s.state != RUNNING || s.pid <= 0) {
    msg = "not running";
    return false;
  }
  kill(s.pid, SIGTERM);
  for (int i = 0; i < 30; ++i) {
    if (waitpid(s.pid, nullptr, WNOHANG) > 0) {
      s.pid = -1;
      break;
    }
    usleep(100000);
  }
  if (s.pid > 0) {
    kill(s.pid, SIGKILL);
    waitpid(s.pid, nullptr, 0);
    s.pid = -1;
  }
  parse_stats(s.stats);
  s.state = STOPPED;
  s.message = "stopped";
  msg = s.message;
  return true;
}

PipelineStatus build_status_locked(State& s) {
  reap_locked(s);
  if (s.state == RUNNING) parse_stats(s.stats);  // live refresh from the engine's periodic lines
  PipelineStatus out;
  out.set_state(s.state);
  out.set_pid(s.pid);
  *out.mutable_config() = s.config;
  *out.mutable_stats() = s.stats;
  out.set_message(s.message);
  out.set_uptime_s(s.state == RUNNING ? now_s() - s.start_time : 0.0);
  out.set_ptp_gmid(s.ptp_gmid);
  return out;
}
}  // namespace

// ---------------- gRPC service ----------------
class SparkControlImpl final : public SparkControl::Service {
  grpc::Status GetStatus(grpc::ServerContext*, const Empty*, PipelineStatus* out) override {
    std::lock_guard<std::mutex> lk(g_state.mu);
    *out = build_status_locked(g_state);
    return grpc::Status::OK;
  }
  grpc::Status SetConfig(grpc::ServerContext*, const SetConfigRequest* req, Ack* ack) override {
    std::lock_guard<std::mutex> lk(g_state.mu);
    reap_locked(g_state);
    if (g_state.state == RUNNING) {
      ack->set_ok(false);
      ack->set_message("cannot change config while running");
      return grpc::Status::OK;
    }
    g_state.config = req->config();
    ack->set_ok(true);
    ack->set_message("config updated");
    return grpc::Status::OK;
  }
  grpc::Status Start(grpc::ServerContext*, const Empty*, Ack* ack) override {
    std::lock_guard<std::mutex> lk(g_state.mu);
    std::string m;
    ack->set_ok(start_locked(g_state, m));
    ack->set_message(m);
    return grpc::Status::OK;
  }
  grpc::Status Stop(grpc::ServerContext*, const Empty*, Ack* ack) override {
    std::lock_guard<std::mutex> lk(g_state.mu);
    std::string m;
    ack->set_ok(stop_locked(g_state, m));
    ack->set_message(m);
    return grpc::Status::OK;
  }
};

// ---------------- HTTP / JSON + static web (libmicrohttpd) ----------------
namespace {
std::string json_of(const google::protobuf::Message& m) {
  std::string s;
  google::protobuf::util::JsonPrintOptions o;
  o.add_whitespace = true;
  o.always_print_primitive_fields = true;
  google::protobuf::util::MessageToJsonString(m, &s, o);
  return s;
}

// ---------------- /api/filters — filter-chain descriptors ----------------
// The dashboard's Filter chain panel renders entirely from these (REDStreamer-style): each entry
// is one GPU stage of the M9 chain with its UI params; `cfg` names the PipelineConfig JSON field a
// param maps to. Adding a stage here (+ its engine op + env plumbing in start_locked) puts it in
// the GUI with zero JS changes. Membership + order travel as the config `filters` token list.
const char* kFilterCatalog = R"json({"filters":[
{"name":"nr","label":"Noise reduction",
 "tip":"Edge-preserving spatial denoise (5×5 bilateral) with separate luma / chroma strengths; 0 leaves that plane untouched. Runs best first — before FRC (optical flow matches clean frames, at the input rate) and before Scale, at the native input resolution where the noise lives.",
 "params":[
  {"key":"luma","label":"Luma","type":"slider","cfg":"nrLuma","min":0,"max":1,"step":0.01,"digits":2},
  {"key":"chroma","label":"Chroma","type":"slider","cfg":"nrChroma","min":0,"max":1,"step":0.01,"digits":2}]},
{"name":"frc","label":"FRC — motion interpolation",
 "tip":"Motion-compensated frame-rate conversion (NVOF). Runs at native input resolution, before scale.",
 "params":[
  {"key":"mode","label":"Mode","type":"select","cfg":"frcMode","choices":[
   {"value":1,"label":"retime (1:1)"},
   {"value":2,"label":"up-convert 2×"},
   {"value":3,"label":"up-convert 2× — uniform grid"}]}]},
{"name":"scale","label":"Scale","required":true,
 "tip":"Resize to the delivery resolution (always on: the output raster advertised on the wire/SDP rides Width×Height). auto = anti-aliased supersampling on downscale, cubic on upscale, passthrough at 1:1.",
 "params":[
  {"key":"out_width","label":"Width","type":"number","cfg":"outWidth","int":true,"min":320,"max":7680,"step":2},
  {"key":"out_height","label":"Height","type":"number","cfg":"outHeight","int":true,"min":240,"max":4320,"step":2},
  {"key":"interp","label":"Interpolation","type":"select","cfg":"interp","choices":[
   {"value":"auto","label":"auto"},{"value":"cubic","label":"cubic"},
   {"value":"linear","label":"linear"},{"value":"lanczos","label":"lanczos"},
   {"value":"super","label":"super (AA downscale)"},
   {"value":"fsrcnn","label":"fsrcnn (AI ×2)"},
   {"value":"fsrcnn-s","label":"fsrcnn-s (AI ×2 fast)"},
   {"value":"espcn","label":"espcn (AI ×2)"}]}]},
{"name":"sharpen","label":"Sharpen",
 "tip":"Luma unsharp mask at the delivery resolution. 0 = identity.",
 "params":[
  {"key":"amount","label":"Amount","type":"slider","cfg":"sharpen","min":0,"max":4,"step":0.05,"digits":2}]},
{"name":"procamp","label":"Proc amp",
 "tip":"Classic video corrector on the native 10-bit YCbCr. Neutral = 0 / 1 / 1 / 0.",
 "params":[
  {"key":"brightness","label":"Brightness","type":"slider","cfg":"paBrightness","min":-1,"max":1,"step":0.01,"digits":2},
  {"key":"contrast","label":"Contrast","type":"slider","cfg":"paContrast","min":0,"max":4,"step":0.05,"digits":2,"unset_to":1},
  {"key":"saturation","label":"Saturation","type":"slider","cfg":"paSaturation","min":0,"max":4,"step":0.05,"digits":2,"unset_to":1},
  {"key":"hue","label":"Hue (°)","type":"slider","cfg":"paHueDeg","min":-180,"max":180,"step":1,"digits":0}]},
{"name":"grain","label":"Film grain",
 "tip":"Photochemical-style grain: strongest in the mid-tones, vanishing at pure black/white, re-drawn every frame. mono = one luma grain field (silver-halide look); color adds independent chroma fields (color-negative look). Size is the grain cell in pixels.",
 "params":[
  {"key":"mode","label":"Mode","type":"select","cfg":"grainMode","choices":[
   {"value":"mono","label":"mono"},{"value":"color","label":"color"}]},
  {"key":"amount","label":"Amount","type":"slider","cfg":"grain","min":0,"max":1,"step":0.01,"digits":2},
  {"key":"size","label":"Size (px)","type":"slider","cfg":"grainSize","min":1,"max":4,"step":0.25,"digits":2,"unset_to":1.5}]},
{"name":"delay","label":"A/V delay",
 "tip":"Schedule-level delay, not a GPU stage: each essence's wire time shifts independently on top of the fixed capture→wire latency L (video → L + video ms, audio → L + audio ms — audio-only is a lip-sync trim). Requires fixed-latency mode (default L = 105 ms); video delay grows the RX frame store accordingly.",
 "params":[
  {"key":"video_ms","label":"Video (ms)","type":"slider","cfg":"delayVideoMs","min":0,"max":1000,"step":5,"digits":0},
  {"key":"audio_ms","label":"Audio (ms)","type":"slider","cfg":"delayAudioMs","min":0,"max":1000,"step":5,"digits":0}]}
]})json";

const char* ctype(const std::string& path) {
  if (path.size() > 5 && path.substr(path.size() - 5) == ".html") return "text/html";
  if (path.size() > 3 && path.substr(path.size() - 3) == ".js") return "application/javascript";
  if (path.size() > 4 && path.substr(path.size() - 4) == ".css") return "text/css";
  return "text/plain";
}

bool read_file(const std::string& path, std::string& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::stringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}

MHD_Result reply(struct MHD_Connection* c, int code, const std::string& body, const char* type) {
  struct MHD_Response* r =
      MHD_create_response_from_buffer(body.size(), (void*)body.data(), MHD_RESPMEM_MUST_COPY);
  MHD_add_response_header(r, "Content-Type", type);
  MHD_add_response_header(r, "Access-Control-Allow-Origin", "*");
  MHD_Result ret = MHD_queue_response(c, code, r);
  MHD_destroy_response(r);
  return ret;
}

// ---------------- /api/nmos — same-origin proxy for NMOS control traffic ----------------
// The dashboard must fetch SDPs from, and PATCH IS-05 connection APIs on, OTHER nodes (e.g. the
// Blackmagic BiDirects) to route this processor's inputs AND outputs. Those nodes don't serve CORS
// headers, so the browser can't reach them directly — but the dashboard is served from THIS daemon,
// so a tiny same-origin forwarder removes the whole problem. Locked down to what NMOS control needs:
// plain http, hosts that resolve to private/loopback ranges, and NMOS-shaped paths only.

// Allow only NMOS API paths: IS-04/IS-05 live under /x-nmos/, nmos-cpp serves sender manifests
// (SDPs) under /x-manifest/, and the mDNS proxy bridges non-CORS manifests at /manifest.
bool nmos_path_ok(const std::string& path) {
  return path.rfind("/x-nmos/", 0) == 0 || path.rfind("/x-manifest/", 0) == 0 ||
         path.rfind("/manifest", 0) == 0;
}

bool sockaddr_is_private(const sockaddr* sa) {
  if (sa->sa_family == AF_INET) {
    const uint32_t a = ntohl(reinterpret_cast<const sockaddr_in*>(sa)->sin_addr.s_addr);
    return (a >> 24) == 10 || (a >> 24) == 127 || (a >> 20) == 0xAC1 /*172.16/12*/ ||
           (a >> 16) == 0xC0A8 /*192.168/16*/ || (a >> 16) == 0xA9FE /*169.254/16 link-local*/;
  }
  if (sa->sa_family == AF_INET6) {
    const auto* a6 = reinterpret_cast<const sockaddr_in6*>(sa);
    return IN6_IS_ADDR_LOOPBACK(&a6->sin6_addr) || IN6_IS_ADDR_LINKLOCAL(&a6->sin6_addr) ||
           (a6->sin6_addr.s6_addr[0] & 0xfe) == 0xfc /*ULA fc00::/7*/;
  }
  return false;
}

// http://host[:port]/path -> parts. Returns false for anything but plain http.
bool parse_http_url(const std::string& url, std::string& host, std::string& port, std::string& path) {
  if (url.rfind("http://", 0) != 0) return false;
  const auto hp = url.substr(7);
  const auto slash = hp.find('/');
  if (slash == std::string::npos) return false;
  path = hp.substr(slash);
  auto authority = hp.substr(0, slash);
  const auto colon = authority.rfind(':');
  if (colon != std::string::npos) {
    host = authority.substr(0, colon);
    port = authority.substr(colon + 1);
  } else {
    host = authority;
    port = "80";
  }
  return !host.empty();
}

// Minimal blocking HTTP client (Connection: close), enough for LAN NMOS control-plane calls.
// Returns the upstream status code, or 0 on transport failure (err describes it). Handles chunked
// transfer-encoding; follows one level of GET redirect (some nodes 30x their manifest URLs).
int http_request(const std::string& method, const std::string& url, const std::string& body,
                 std::string& resp_body, std::string& resp_ctype, std::string& err, int redirects = 2) {
  std::string host, port, path;
  if (!parse_http_url(url, host, port, path)) { err = "unsupported url (plain http only)"; return 0; }
  if (!nmos_path_ok(path)) { err = "path not allowed (NMOS APIs only)"; return 0; }

  addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res) { err = "resolve failed"; return 0; }
  std::unique_ptr<addrinfo, void (*)(addrinfo*)> res_guard(res, freeaddrinfo);
  if (!sockaddr_is_private(res->ai_addr)) { err = "host not on a private network"; return 0; }

  const int fd = socket(res->ai_family, SOCK_STREAM, 0);
  if (fd < 0) { err = "socket failed"; return 0; }
  timeval tv{4, 0};  // covers connect (SO_SNDTIMEO) and each read (SO_RCVTIMEO)
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
  if (connect(fd, res->ai_addr, res->ai_addrlen) != 0) { close(fd); err = "connect failed"; return 0; }

  std::string req = method + " " + path + " HTTP/1.1\r\nHost: " + host + ":" + port +
                    "\r\nConnection: close\r\nAccept: */*\r\n";
  if (!body.empty())
    req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
  req += "\r\n" + body;
  for (size_t off = 0; off < req.size();) {
    const ssize_t n = send(fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
    if (n <= 0) { close(fd); err = "send failed"; return 0; }
    off += static_cast<size_t>(n);
  }

  // Read until the response is COMPLETE, not until EOF: some embedded servers (the Blackmagic
  // BiDirects) ignore "Connection: close" and hold the socket open, so an EOF-delimited read eats
  // the whole 4 s timeout and reports failure with the body already in hand. Completion = header
  // seen AND (Content-Length satisfied | terminal chunk seen); EOF stays as the fallback delimiter.
  std::string raw;
  char buf[8192];
  size_t hdr_end = std::string::npos;
  long want_len = -1;  // from Content-Length; -1 = unknown (EOF-delimited)
  bool is_chunked = false;
  const auto complete = [&]() -> bool {
    if (hdr_end == std::string::npos) return false;
    if (is_chunked) return raw.find("\r\n0\r\n", hdr_end + 2) != std::string::npos;
    if (want_len >= 0) return raw.size() - (hdr_end + 4) >= static_cast<size_t>(want_len);
    return false;
  };
  for (;;) {
    const ssize_t n = recv(fd, buf, sizeof buf, 0);
    if (n < 0) { close(fd); err = "read failed/timeout"; return 0; }
    if (n == 0) break;
    raw.append(buf, static_cast<size_t>(n));
    if (hdr_end == std::string::npos) {
      hdr_end = raw.find("\r\n\r\n");
      if (hdr_end != std::string::npos) {
        std::string lower = raw.substr(0, hdr_end);
        for (auto& ch : lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        const auto cl = lower.find("\r\ncontent-length:");
        if (cl != std::string::npos) want_len = std::atol(lower.c_str() + cl + 17);
        const auto te = lower.find("\r\ntransfer-encoding:");
        is_chunked = te != std::string::npos && lower.find("chunked", te) != std::string::npos;
      }
    }
    if (complete()) break;
    if (raw.size() > (16u << 20)) break;  // 16 MB cap — an SDP/JSON reply is KBs
  }
  close(fd);

  if (hdr_end == std::string::npos) { err = "malformed response"; return 0; }
  const std::string head = raw.substr(0, hdr_end);
  resp_body = raw.substr(hdr_end + 4);
  const int status = std::atoi(head.c_str() + head.find(' ') + 1);

  // headers we care about (case-insensitive): Content-Type, Transfer-Encoding, Location
  const auto header = [&](const char* name) -> std::string {
    std::string lower = head;
    for (auto& ch : lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    const auto p = lower.find(std::string("\r\n") + name + ":");
    if (p == std::string::npos) return {};
    const auto v0 = head.find(':', p + 2) + 1;
    const auto v1 = head.find("\r\n", v0);
    auto v = head.substr(v0, v1 - v0);
    v.erase(0, v.find_first_not_of(" \t"));
    return v;
  };
  resp_ctype = header("content-type");
  if (resp_ctype.empty()) resp_ctype = "application/octet-stream";

  if (header("transfer-encoding").find("chunked") != std::string::npos) {  // de-chunk
    std::string out;
    size_t p = 0;
    while (p < resp_body.size()) {
      const auto eol = resp_body.find("\r\n", p);
      if (eol == std::string::npos) break;
      const size_t len = std::strtoul(resp_body.c_str() + p, nullptr, 16);
      if (len == 0) break;
      out.append(resp_body, eol + 2, len);
      p = eol + 2 + len + 2;
    }
    resp_body = std::move(out);
  }

  if (status >= 301 && status <= 308 && method == "GET" && redirects > 0) {
    const auto loc = header("location");
    if (!loc.empty())
      return http_request(method, loc, body, resp_body, resp_ctype, err, redirects - 1);
  }
  return status;
}

// GET /api/nmos?u=<url>  |  POST /api/nmos?u=<url>&m=PATCH (body forwarded) — upstream status/body
// pass straight through so the dashboard treats it like a plain fetch.
MHD_Result nmos_proxy(struct MHD_Connection* conn, const char* method_override,
                      const std::string& body) {
  const char* u = MHD_lookup_connection_value(conn, MHD_GET_ARGUMENT_KIND, "u");
  if (!u || !*u) return reply(conn, 400, "{\"error\":\"missing u=<url>\"}", "application/json");
  const char* m = MHD_lookup_connection_value(conn, MHD_GET_ARGUMENT_KIND, "m");
  const std::string method = method_override ? method_override : (m && *m ? m : "GET");
  if (method != "GET" && method != "PATCH")
    return reply(conn, 400, "{\"error\":\"method must be GET or PATCH\"}", "application/json");
  std::string resp, ctype, err;
  const int status = http_request(method, u, body, resp, ctype, err);
  if (status == 0)
    return reply(conn, 502, "{\"error\":\"" + err + "\"}", "application/json");
  return reply(conn, status, resp, ctype.c_str());
}

MHD_Result http_handler(void*, struct MHD_Connection* conn, const char* url, const char* method,
                        const char*, const char* upload_data, size_t* upload_size, void** con_cls) {
  const std::string u = url, m = method;

  // Accumulate POST body across libmicrohttpd callback invocations.
  if (m == "POST") {
    auto* body = static_cast<std::string*>(*con_cls);
    if (!body) {
      *con_cls = new std::string();
      return MHD_YES;
    }
    if (*upload_size) {
      body->append(upload_data, *upload_size);
      *upload_size = 0;
      return MHD_YES;
    }
    std::unique_ptr<std::string> owned(body);
    *con_cls = nullptr;
    // NMOS proxy first — it must not hold (or wait on) the state lock while talking to other nodes.
    if (u == "/api/nmos") return nmos_proxy(conn, nullptr, *owned);
    Ack ack;
    std::lock_guard<std::mutex> lk(g_state.mu);
    std::string msg;
    if (u == "/api/start") {
      ack.set_ok(start_locked(g_state, msg));
    } else if (u == "/api/stop") {
      ack.set_ok(stop_locked(g_state, msg));
    } else if (u == "/api/config") {
      reap_locked(g_state);
      if (g_state.state == RUNNING) {
        ack.set_ok(false);
        msg = "cannot change config while running";
      } else {
        SetConfigRequest req;
        // Strict parse first, so a typo'd key is never silently dropped; only unknown fields
        // (a newer dashboard posting fields this build lacks) fall back to the lenient parse,
        // and the ack message flags them so version skew / misspellings stay visible.
        auto st = google::protobuf::util::JsonStringToMessage(*owned, req.mutable_config());
        if (!st.ok()) {
          req.Clear();
          google::protobuf::util::JsonParseOptions jopt;
          jopt.ignore_unknown_fields = true;
          st = google::protobuf::util::JsonStringToMessage(*owned, req.mutable_config(), jopt);
          if (st.ok()) msg = "config updated (unknown fields ignored — dashboard/daemon version skew or a typo?)";
        }
        if (st.ok()) {
          g_state.config = req.config();
          ack.set_ok(true);
          if (msg.empty()) msg = "config updated";
        } else {
          ack.set_ok(false);
          msg = "bad JSON config";
        }
      }
    } else {
      return reply(conn, 404, "{\"ok\":false,\"message\":\"unknown\"}", "application/json");
    }
    ack.set_message(msg);
    return reply(conn, 200, json_of(ack), "application/json");
  }

  // GET
  if (u == "/api/nmos") return nmos_proxy(conn, "GET", "");
  if (u == "/api/filters") return reply(conn, 200, kFilterCatalog, "application/json");
  if (u == "/api/status") {
    std::lock_guard<std::mutex> lk(g_state.mu);
    return reply(conn, 200, json_of(build_status_locked(g_state)), "application/json");
  }
  std::string file = (u == "/" || u.empty()) ? "/index.html" : u;
  if (file.find("..") != std::string::npos)
    return reply(conn, 400, "bad path", "text/plain");
  std::string body;
  if (read_file(g_web + file, body)) return reply(conn, 200, body, ctype(file));
  return reply(conn, 404, "not found", "text/plain");
}
}  // namespace

int main(int argc, char** argv) {
  std::string grpc_addr = "0.0.0.0:50051";
  int http_port = 8080;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--pipeline") g_pipeline = next();
    else if (a == "--web") g_web = next();
    else if (a == "--grpc") grpc_addr = next();
    else if (a == "--http") http_port = std::atoi(next());
  }
  signal(SIGPIPE, SIG_IGN);  // SIGCHLD left default: we reap children via waitpid(WNOHANG)

  // Poll the live PTP grandmaster so the NMOS node can mirror it into clk0 + sender SDPs (the GM can
  // change via BMC; a static gmid goes stale and ST 2110 receivers reject the mismatch).
  std::thread([] {
    for (;;) {
      std::string gm = read_ptp_gmid();
      if (!gm.empty()) { std::lock_guard<std::mutex> lk(g_state.mu); g_state.ptp_gmid = gm; }
      std::this_thread::sleep_for(std::chrono::seconds(3));
    }
  }).detach();

  SparkControlImpl svc;
  grpc::ServerBuilder b;
  b.AddListeningPort(grpc_addr, grpc::InsecureServerCredentials());
  b.RegisterService(&svc);
  std::unique_ptr<grpc::Server> server(b.BuildAndStart());

  // Thread-per-connection: /api/nmos blocks for up to ~4 s talking to another node; on the single
  // polling thread that would freeze every status poll (and the UI) for the duration.
  struct MHD_Daemon* http =
      MHD_start_daemon(MHD_USE_INTERNAL_POLLING_THREAD | MHD_USE_THREAD_PER_CONNECTION, http_port,
                       nullptr, nullptr, &http_handler, nullptr, MHD_OPTION_END);
  if (!http) {
    std::fprintf(stderr, "failed to start HTTP on :%d\n", http_port);
    return 1;
  }
  std::printf("spark_controld: gRPC %s | HTTP http://localhost:%d | pipeline=%s | web=%s\n",
              grpc_addr.c_str(), http_port, g_pipeline.c_str(), g_web.c_str());
  server->Wait();
  MHD_stop_daemon(http);
  return 0;
}
