// Spark Video Processor — control daemon (M4).
// Exposes the pipeline control as gRPC (SparkControl) AND HTTP/JSON + static web (libmicrohttpd) so
// the vanilla web dashboard can drive it with fetch() — no gRPC-web proxy. Manages the
// st2110_pipeline process (env from config) and parses its final stats. No DPDK/Holoscan deps.
#include <fcntl.h>
#include <microhttpd.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
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

  State() {  // sensible defaults (the gate-4 / pipeline topology)
    config.set_profile("1080p");
    config.set_out_width(3840);
    config.set_out_height(2160);
    config.set_interp("cubic");
    config.set_frc(true);
    config.set_frc_mode(1);  // 1=retime; web UI / NMOS can pick 2=up-convert (30->60)
    config.set_rx_pci("0000:01:00.1");
    config.set_tx_pci("0002:01:00.1");  // up port on this rig (.0 is the down link)
    config.set_dst_mac("00:00:5e:00:53:30");
    config.set_frames(300);
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
    // SPARK_FRC is a mode (0=off, 1=retime, 2=up-convert). frc_mode supersedes the legacy frc bool.
    const int frc_mode = c.frc_mode() > 0 ? static_cast<int>(c.frc_mode()) : (c.frc() ? 1 : 0);
    setenv("SPARK_FRC", std::to_string(frc_mode).c_str(), 1);
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
        auto st = google::protobuf::util::JsonStringToMessage(*owned, req.mutable_config());
        if (st.ok()) {
          g_state.config = req.config();
          ack.set_ok(true);
          msg = "config updated";
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

  SparkControlImpl svc;
  grpc::ServerBuilder b;
  b.AddListeningPort(grpc_addr, grpc::InsecureServerCredentials());
  b.RegisterService(&svc);
  std::unique_ptr<grpc::Server> server(b.BuildAndStart());

  struct MHD_Daemon* http = MHD_start_daemon(MHD_USE_INTERNAL_POLLING_THREAD, http_port, nullptr,
                                             nullptr, &http_handler, nullptr, MHD_OPTION_END);
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
