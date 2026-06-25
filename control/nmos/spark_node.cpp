// Spark Video Processor — NMOS Node (IS-04 discovery/registration + IS-05 connection management).
//
// This makes the processor a first-class NMOS Node so 2110 sources are discovered/routed over the
// network instead of typing raw PCIe/MAC: it advertises a video (ST 2110-20) and an audio
// (ST 2110-30) Receiver to ingest into the engine, plus matching Senders for the processed output,
// all under a PTP-locked clock (clk0). Built on Sony's nmos-cpp (see deploy/nmos.sh).
//
// P1 scope: the resource model + PTP clock + valid sender SDP, registering in both registered and
// peer-to-peer (mDNS) modes. The IS-05 connection-activation handler is a logging stub here; P2
// turns activation into engine (re)configuration (multicast group/port/format -> SPARK_* / pipeline).
//
// Structure follows nmos-cpp's nmos-cpp-node example (BSD-3): a node_implementation supplies the
// IS-05 callbacks (resolve "auto", build the sender transport file/SDP, on-activated), and a thread
// inserts our fixed resource set into the model once the server is up.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "bst/optional.h"
#include "cpprest/host_utils.h"
#include "cpprest/http_client.h"
#include "nmos/activation_mode.h"
#include "nmos/capabilities.h"
#include "nmos/certificate_handlers.h"
#include "nmos/channels.h"
#include "nmos/clock_name.h"
#include "nmos/colorspace.h"
#include "nmos/connection_api.h"        // nmos::resolve_rtp_auto
#include "nmos/connection_resources.h"  // make_connection_rtp_sender/receiver, resolve_auto
#include "nmos/format.h"
#include "nmos/group_hint.h"
#include "nmos/interlace_mode.h"
#include "nmos/log_gate.h"
#include "nmos/media_type.h"
#include "nmos/model.h"
#include "nmos/node_interfaces.h"
#include "nmos/node_resource.h"     // make_ptp_clock
#include "nmos/node_resources.h"    // make_node/device/source/flow/sender/receiver
#include "nmos/node_server.h"       // make_node_server, node_implementation, insert_node_default_settings
#include "nmos/process_utils.h"     // wait_term_signal
#include "nmos/random.h"
#include "nmos/rational.h"          // rational, make_rational, parse_rational
#include "nmos/resources.h"         // find_resource, modify_resource
#include "nmos/sdp_utils.h"
#include "nmos/server.h"            // server_guard
#include "nmos/settings.h"          // get_host_interfaces
#include "nmos/slog.h"
#include "nmos/transfer_characteristic.h"
#include "nmos/transport.h"
#include "sdp/sdp.h"

namespace spark {
namespace {

using web::json::value;
using web::json::value_of;
using web::json::value_from_elements;

// ----- settings helpers (custom SPARK_* fields the control daemon passes through) -----
utility::string_t str_field(const nmos::settings& s, const utility::string_t& key, const utility::string_t& def) {
  return s.has_field(key) ? s.at(key).as_string() : def;
}
bool bool_field(const nmos::settings& s, const utility::string_t& key, bool def) {
  return s.has_field(key) ? s.at(key).as_bool() : def;
}
unsigned int uint_field(const nmos::settings& s, const utility::string_t& key, unsigned int def) {
  if (!s.has_field(key)) return def;
  const auto& v = s.at(key);
  return v.is_integer() ? (unsigned int)v.as_integer() : (unsigned int)std::stoul(utility::us2s(v.as_string()));
}

// ----- deterministic resource ids derived from the node's seed (stable across restarts) -----
struct Ids {
  nmos::id node, device, source_v, flow_v, sender_v, receiver_v, source_a, flow_a, sender_a, receiver_a;
  explicit Ids(const nmos::settings& settings) {
    const auto seed = nmos::experimental::fields::seed_id(settings);
    const auto id = [&](const utility::string_t& name) { return nmos::make_repeatable_id(seed, name); };
    node = id(U("/spark/node"));
    device = id(U("/spark/device"));
    source_v = id(U("/spark/video/source"));
    flow_v = id(U("/spark/video/flow"));
    sender_v = id(U("/spark/video/sender"));
    receiver_v = id(U("/spark/video/receiver"));
    source_a = id(U("/spark/audio/source"));
    flow_a = id(U("/spark/audio/flow"));
    sender_a = id(U("/spark/audio/sender"));
    receiver_a = id(U("/spark/audio/receiver"));
  }
};

// Output multicast group our senders advertise (resolve "auto" destination_ip to these).
utility::string_t out_multicast(const Ids& ids, const nmos::id& sender_id) {
  if (sender_id == ids.sender_v) return U("239.100.0.10");
  if (sender_id == ids.sender_a) return U("239.100.0.20");
  return U("239.100.0.30");
}

// Find the host interface that owns `address` (for binding sender/receiver legs).
const web::hosts::experimental::host_interface* find_interface(
    const std::vector<web::hosts::experimental::host_interface>& ifaces, const utility::string_t& address) {
  for (const auto& i : ifaces)
    for (const auto& a : i.addresses)
      if (a == address) return &i;
  return nullptr;
}

// ----- IS-05: resolve "auto" transport params at activation -----
nmos::connection_resource_auto_resolver make_spark_auto_resolver(const nmos::settings& settings) {
  const Ids ids(settings);
  return [ids](const nmos::resource& /*resource*/, const nmos::resource& connection_resource, value& transport_params) {
    const auto& id = connection_resource.id;
    const auto& constraints = nmos::fields::endpoint_constraints(connection_resource.data);
    if (id == ids.sender_v || id == ids.sender_a) {
      nmos::details::resolve_auto(transport_params[0], nmos::fields::source_ip, [&] {
        return web::json::front(nmos::fields::constraint_enum(constraints.at(0).at(nmos::fields::source_ip)));
      });
      nmos::details::resolve_auto(transport_params[0], nmos::fields::destination_ip, [&] {
        return value::string(out_multicast(ids, id));
      });
      nmos::resolve_rtp_auto(connection_resource.type, transport_params);
    } else if (id == ids.receiver_v || id == ids.receiver_a) {
      nmos::details::resolve_auto(transport_params[0], nmos::fields::interface_ip, [&] {
        return web::json::front(nmos::fields::constraint_enum(constraints.at(0).at(nmos::fields::interface_ip)));
      });
      nmos::resolve_rtp_auto(connection_resource.type, transport_params);
    }
  };
}

// PTP domain to embed in the SDP's ts-refclk (a=ts-refclk:ptp=IEEE1588-2008:<gmid>:<domain>).
// Omitted (RFC 7273 allows that) when spark_ptp_domain is unset. ST 2110 receivers like the Blackmagic
// BiDirect validate the refclk and reject the bare ":traceable" form, so we advertise gmid[:domain].
bst::optional<int> ptp_domain_setting(const nmos::settings& s) {
  return s.has_field(U("spark_ptp_domain")) ? bst::optional<int>((int)uint_field(s, U("spark_ptp_domain"), 0))
                                            : bst::nullopt;
}

// ST 2110-21 sender compliance profile advertised in the SDP (TP=). Default Narrow (2110TPN) matches
// our active-period gapped pacing; SPARK_TX_TP=wide advertises Wide (2110TPW) so the receiver sizes its
// (larger) wide buffer — must match the engine's SPARK_TX_TP, which switches the actual wire pacing.
inline bool sdp_tp_wide() {
  const char* v = std::getenv("SPARK_TX_TP");
  if (!v) return false;
  const std::string s(v);
  return s == "wide" || s == "W" || s == "2110TPW";
}

// Rewrite a raw-video SDP into Blackmagic IP10 (10:8) form. We reuse nmos-cpp's raw-video SDP for the
// ts-refclk/mediaclk/framerate/components, then swap the codec signalling per the published IP10 spec:
// encoding name "vnd.blackmagicdesign.ip10", SSN=ST2110-22:2022, TP=2110TPN, and the Blackmagic
// scheme=10:8 attribute; PM is omitted (defaults to 2110GPM). The wire payload is an 8-bit 4:2:2 RFC
// 4175 stream of IP10 codewords (see engine ip10_codec), and depth stays 10 (the SOURCE sample depth).
void apply_ip10_video(nmos::sdp_parameters& sdp) {
  // Encoding name: match the spec's *device* example SDP (captured from a real Blackmagic unit), which
  // uses the hyphenated vendor form "vnd.blackmagic-design.ip10" — Blackmagic receivers parse that. (The
  // spec prose's unhyphenated "vnd.blackmagicdesign.ip10" is rejected with "SDP parsing" on the wire.)
  sdp.rtpmap.encoding_name = U("vnd.blackmagic-design.ip10");
  nmos::sdp_parameters::fmtp_t out;
  const auto carry = [&](const utility::string_t& key) {
    const auto it = std::find_if(sdp.fmtp.begin(), sdp.fmtp.end(),
                                 [&](const std::pair<utility::string_t, utility::string_t>& p) { return p.first == key; });
    if (it != sdp.fmtp.end()) out.push_back(*it);
  };
  // Keep (in the spec's example order) the format params the IP10 SDP carries; drop raw-only ones
  // (PM, TCS, TSMODE, TSDELAY, …) by simply not copying them.
  carry(U("sampling"));
  carry(U("depth"));
  carry(U("width"));
  carry(U("height"));
  carry(U("exactframerate"));
  carry(U("colorimetry"));
  out.emplace_back(U("SSN"), U("ST2110-22:2022"));
  out.emplace_back(U("TP"), sdp_tp_wide() ? U("2110TPW") : U("2110TPN"));
  out.emplace_back(U("scheme"), U("10:8"));
  sdp.fmtp = std::move(out);

  // Declare b=AS (application-specific bandwidth). Blackmagic's own IP10 SDP carries it (e.g. b=AS:8262000
  // for 2160p59.94) and a receiver may size its buffer from it; without it BiDirect can under-allocate and
  // drop the frame tail (bottom-of-frame). Compute from the 8-bit IP10 geometry: (w/2)*h*4 octets/frame
  // × 8 × fps, + ~4% header overhead, in kbps.
  const auto fval = [&](const utility::string_t& k) -> utility::string_t {
    const auto it = std::find_if(sdp.fmtp.begin(), sdp.fmtp.end(),
                                 [&](const std::pair<utility::string_t, utility::string_t>& p) { return p.first == k; });
    return it != sdp.fmtp.end() ? it->second : utility::string_t{};
  };
  const auto num = [](const utility::string_t& s) -> double {
    const std::string v = utility::us2s(s);
    const auto sl = v.find('/');
    try {
      if (sl == std::string::npos) return std::stod(v);
      const double n = std::stod(v.substr(0, sl)), d = std::stod(v.substr(sl + 1));
      return d != 0.0 ? n / d : 0.0;
    } catch (...) { return 0.0; }
  };
  const double w = num(fval(U("width"))), h = num(fval(U("height"))), fps = num(fval(U("exactframerate")));
  if (w > 0 && h > 0 && fps > 0) {
    const uint64_t kbps = static_cast<uint64_t>((w / 2.0) * h * 4.0 * 8.0 * fps / 1000.0 * 1.04);
    sdp.bandwidth = nmos::sdp_parameters::bandwidth_t{ sdp::bandwidth_types::application_specific, kbps };
  }
}

// Build an ST 2110 SDP transport file for a sender from its CURRENT node/source/flow + resolved
// IS-05 transport params. The flow is read LIVE, so updating its frame_width/height/grain_rate and
// re-running this regenerates the SDP. `ip10` rewrites the VIDEO sender's SDP to Blackmagic IP10 (10:8)
// — required for 2160p59.94/60 into Blackmagic receivers. Returns null for an unknown sender. Caller
// holds the model lock.
value build_sender_transportfile(const nmos::resources& node_resources, const Ids& ids,
                                 const nmos::resource& sender, const nmos::resource& connection_sender,
                                 bst::optional<int> ptp_domain, bool ip10) {
  nmos::id source_id, flow_id;
  if (connection_sender.id == ids.sender_v) { source_id = ids.source_v; flow_id = ids.flow_v; }
  else if (connection_sender.id == ids.sender_a) { source_id = ids.source_a; flow_id = ids.flow_a; }
  else return value::null();

  auto node = nmos::find_resource(node_resources, { ids.node, nmos::types::node });
  auto source = nmos::find_resource(node_resources, { source_id, nmos::types::source });
  auto flow = nmos::find_resource(node_resources, { flow_id, nmos::types::flow });
  if (node_resources.end() == node || node_resources.end() == source || node_resources.end() == flow)
    throw std::logic_error("spark nmos: node/source/flow not found for sender transportfile");

  const std::vector<utility::string_t> mids{ U("PRIMARY") };  // single-path (no ST 2022-7 in v1)
  const nmos::format format{ nmos::fields::format(flow->data) };
  const auto tp = sdp_tp_wide() ? sdp::type_parameters::type_W : sdp::type_parameters::type_N;
  auto sdp_params = (nmos::formats::video == format)
      ? nmos::make_video_sdp_parameters(node->data, source->data, flow->data, sender.data,
                                        nmos::details::payload_type_video_default, mids, ptp_domain, tp)
      : nmos::make_audio_sdp_parameters(node->data, source->data, flow->data, sender.data,
                                        nmos::details::payload_type_audio_default, mids, ptp_domain, 1.0 /*ptime ms*/);
  if (ip10 && nmos::formats::video == format) apply_ip10_video(sdp_params);  // raw 10-bit -> IP10 10:8

  auto& transport_params = nmos::fields::transport_params(nmos::fields::endpoint_active(connection_sender.data));
  auto session_description = nmos::make_session_description(sdp_params, transport_params);
  auto sdp = utility::s2us(sdp::make_session_description(session_description));
  return nmos::make_connection_rtp_sender_transportfile(sdp);
}

// ----- IS-05: build each sender's /transportfile (the ST 2110 SDP) at activation -----
// Because the node clock is PTP, make_*_sdp_parameters emit a=ts-refclk:ptp=IEEE1588-2008:<gmid>.
nmos::connection_sender_transportfile_setter make_spark_transportfile_setter(
    const nmos::resources& node_resources, const nmos::settings& settings) {
  const Ids ids(settings);
  const auto ptp_domain = ptp_domain_setting(settings);
  return [&node_resources, ids, ptp_domain](const nmos::resource& sender, const nmos::resource& connection_sender,
                                value& endpoint_transportfile) {
    // ip10=false here: at activation the daemon's IP10 choice isn't known yet. NodeStateSync polls
    // /api/status and rebuilds the video SDP with the real IP10 state within a few seconds (same
    // eventual-consistency path it uses for output resolution and the PTP grandmaster).
    auto tf = build_sender_transportfile(node_resources, ids, sender, connection_sender, ptp_domain, false);
    if (!tf.is_null()) endpoint_transportfile = tf;  // model mutex already held by the calling thread
  };
}

// ----- helpers to read IS-05 transport params + the connected sender's SDP -----
utility::string_t tp_str(const value& tp, const utility::string_t& key) {
  if (!tp.has_field(key)) return {};
  const auto& v = tp.at(key);
  return v.is_string() ? v.as_string() : utility::string_t{};
}
uint32_t tp_uint(const value& tp, const utility::string_t& key) {
  if (!tp.has_field(key)) return 0;
  const auto& v = tp.at(key);
  if (v.is_integer()) return (uint32_t)v.as_integer();
  if (v.is_string()) { try { return (uint32_t)std::stoul(utility::us2s(v.as_string())); } catch (...) {} }
  return 0;
}
utility::string_t active_sdp(const value& endpoint_active) {
  if (!endpoint_active.has_field(U("transport_file"))) return {};
  const auto& tf = endpoint_active.at(U("transport_file"));
  return (tf.has_field(U("data")) && tf.at(U("data")).is_string()) ? tf.at(U("data")).as_string() : utility::string_t{};
}

// ----- bridge: IS-05 activation -> control daemon (HTTP) -> engine -----
// Aggregates the video + audio receiver/sender connection state. Whenever the VIDEO receiver is
// active, it pushes a PipelineConfig (camelCase protobuf-JSON) to the control daemon and starts the
// engine; releasing the video receiver stops it. All HTTP runs on a worker thread so it never blocks
// the nmos-cpp activation thread / model lock. A last-applied snapshot avoids needless restarts.
class EngineController {
 public:
  struct Rtp { utility::string_t group, src, iface; uint32_t port = 0; bool active = false; };
  struct VideoFmt { uint32_t width = 0, height = 0, depth = 0; utility::string_t fps, sampling; bool ip10 = false; };

  EngineController(const utility::string_t& control_url, slog::base_gate& gate)
      : client_(control_url), gate_(gate), worker_([this] { run(); }) {}
  ~EngineController() {
    { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  void set_video_rx(const Rtp& s, const VideoFmt& f) { std::lock_guard<std::mutex> lk(mu_); vrx_ = s; vfmt_ = f; dirty(); }
  void set_audio_rx(const Rtp& s) { std::lock_guard<std::mutex> lk(mu_); arx_ = s; dirty(); }
  void set_video_tx(const Rtp& s) { std::lock_guard<std::mutex> lk(mu_); vtx_ = s; dirty(); }
  void set_audio_tx(const Rtp& s) { std::lock_guard<std::mutex> lk(mu_); atx_ = s; dirty(); }

 private:
  void dirty() { dirty_ = true; cv_.notify_all(); }

  void run() {
    std::unique_lock<std::mutex> lk(mu_);
    while (!stop_) {
      cv_.wait(lk, [this] { return dirty_ || stop_; });
      if (stop_) break;
      dirty_ = false;
      const auto vrx = vrx_, arx = arx_, vtx = vtx_, atx = atx_;
      const auto vfmt = vfmt_;
      lk.unlock();
      reconcile(vrx, arx, vtx, atx, vfmt);
      lk.lock();
    }
  }

  void reconcile(const Rtp& vrx, const Rtp& arx, const Rtp& vtx, const Rtp& atx, const VideoFmt& vfmt) {
    try {
      if (!vrx.active) {
        if (!last_applied_.empty()) {
          post(U("/api/stop"), U(""));
          last_applied_.clear();
          slog::log<slog::severities::info>(gate_, SLOG_FLF) << "spark nmos: video receiver released -> engine stop";
        }
        return;
      }
      // Start from the daemon's CURRENT config so the operator's transport (rx/tx PCI, dst MAC) and
      // processing choices (output size, interp, FRC mode) persist — NMOS routing only swaps the SOURCE.
      value c = current_config();
      // Hard fallbacks so a real launch never gets empty PCI even if /api/status was unreachable.
      if (!c.has_field(U("rxPci")) || c.at(U("rxPci")).as_string().empty())
        c[U("rxPci")] = value::string(U("0000:01:00.1"));
      if (!c.has_field(U("txPci")) || c.at(U("txPci")).as_string().empty())
        c[U("txPci")] = value::string(U("0002:01:00.1"));
      // RX video source (from the activated sender's transport params + SDP):
      c[U("rxMcastGroup")] = value::string(vrx.group);
      c[U("rxSrcIp")] = value::string(vrx.src);
      c[U("rxDstPort")] = vrx.port;
      c[U("rxIfaceIp")] = value::string(vrx.iface);
      // RX audio: set when active, else clear any stale route.
      if (arx.active) {
        c[U("rxAudioMcastGroup")] = value::string(arx.group);
        c[U("rxAudioSrcIp")] = value::string(arx.src);
        c[U("rxAudioDstPort")] = arx.port;
      } else {
        c[U("rxAudioMcastGroup")] = value::string(U(""));
        c[U("rxAudioSrcIp")] = value::string(U(""));
        c[U("rxAudioDstPort")] = 0;
      }
      // Input format from the SDP (overrides `profile` when width != 0). inIp10 is derived from the
      // source SDP codec, so it's set (not preserved) — connecting an IP10 source auto-enables decode.
      if (vfmt.width) {
        c[U("inWidth")] = vfmt.width; c[U("inHeight")] = vfmt.height;
        c[U("inExactframerate")] = value::string(vfmt.fps);
        c[U("inDepth")] = vfmt.depth; c[U("inSampling")] = value::string(vfmt.sampling);
        c[U("inIp10")] = value::boolean(vfmt.ip10);
      }
      // TX egress groups: set when our matching sender is activated, else clear. txSrc MUST match the
      // sender SDP's source-filter (vtx.src = the sender's resolved source_ip) or SSM receivers drop us.
      if (!vtx.group.empty()) {
        c[U("txMcastGroup")] = value::string(vtx.group); c[U("txDstPort")] = vtx.port;
        c[U("txSrc")] = value::string(vtx.src);
      } else {
        c[U("txMcastGroup")] = value::string(U("")); c[U("txDstPort")] = 0; c[U("txSrc")] = value::string(U(""));
      }
      if (!atx.group.empty()) { c[U("txAudioMcastGroup")] = value::string(atx.group); c[U("txAudioDstPort")] = atx.port; }
      else { c[U("txAudioMcastGroup")] = value::string(U("")); c[U("txAudioDstPort")] = 0; }

      const auto body = c.serialize();
      if (body == last_applied_) return;  // already running with this exact config
      post(U("/api/stop"), U(""));  // fork/exec engine has no hot-reconfig; cycle it
      post(U("/api/config"), body);
      post(U("/api/start"), U(""));
      last_applied_ = body;
      slog::log<slog::severities::info>(gate_, SLOG_FLF)
          << "spark nmos: pushed config + start (video rx " << utility::us2s(vrx.group) << ":" << vrx.port
          << (arx.active ? ", audio rx " + utility::us2s(arx.group) : "") << ")";
    } catch (const std::exception& e) {
      slog::log<slog::severities::error>(gate_, SLOG_FLF) << "spark nmos: engine control failed: " << e.what();
    }
  }

  // Read the daemon's live PipelineConfig (the `config` object from /api/status) so reconcile() can
  // preserve operator-set fields it doesn't own (PCI BDFs, dst MAC, output size, interp, frc_mode).
  // Returns an empty object on any failure — reconcile() then falls back to PCI defaults.
  value current_config() {
    try {
      auto resp = client_.request(web::http::methods::GET, U("/api/status")).get();
      if (resp.status_code() == web::http::status_codes::OK) {
        const auto body = resp.extract_json().get();
        if (body.has_field(U("config"))) return body.at(U("config"));
      }
    } catch (const std::exception& e) {
      slog::log<slog::severities::warning>(gate_, SLOG_FLF)
          << "spark nmos: could not read current daemon config (" << e.what() << ") — using PCI fallbacks";
    }
    return value::object();
  }

  void post(const utility::string_t& path, const utility::string_t& body) {
    web::http::http_request req(web::http::methods::POST);
    req.set_request_uri(path);
    req.set_body(body, U("application/json"));
    client_.request(req).get();  // blocking, but on this worker thread only
  }

  web::http::client::http_client client_;
  slog::base_gate& gate_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool dirty_ = false, stop_ = false;
  Rtp vrx_, arx_, vtx_, atx_;
  VideoFmt vfmt_;
  utility::string_t last_applied_;
  std::thread worker_;
};

// Parse width/height/exactframerate/depth/sampling out of an ST 2110-20 SDP's fmtp line.
void parse_video_fmtp(const utility::string_t& sdp_u, EngineController::VideoFmt& f) {
  const std::string sdp = utility::us2s(sdp_u);
  const auto grab = [&](const std::string& key) -> std::string {
    auto p = sdp.find(key);
    if (p == std::string::npos) return {};
    p += key.size();
    return sdp.substr(p, sdp.find_first_of("; \r\n", p) - p);
  };
  const auto w = grab("width="), h = grab("height="), d = grab("depth=");
  try { if (!w.empty()) f.width = std::stoul(w); } catch (...) {}
  try { if (!h.empty()) f.height = std::stoul(h); } catch (...) {}
  try { if (!d.empty()) f.depth = std::stoul(d); } catch (...) {}
  f.fps = utility::s2us(grab("exactframerate="));
  f.sampling = utility::s2us(grab("sampling="));
  // Blackmagic IP10 source: the rtpmap encoding name (or the 10:8 scheme attribute) marks it. The wire
  // is 8-bit codeword pgroups; the engine RX must IP10-decode them. Drives inIp10 in the pushed config.
  // Match both vendor spellings ("vnd.blackmagic-design.ip10" / "...blackmagicdesign...") + the scheme.
  f.ip10 = sdp.find("scheme=10:8") != std::string::npos ||
           (sdp.find("blackmagic") != std::string::npos && sdp.find("ip10") != std::string::npos);
}

// ----- IS-05 on-activation: translate connection state into engine control via the daemon -----
nmos::connection_activation_handler make_spark_activation_handler(
    std::shared_ptr<EngineController> ctrl, const nmos::settings& settings, slog::base_gate& gate) {
  const Ids ids(settings);
  return [ctrl, ids, &gate](const nmos::resource& resource, const nmos::resource& connection_resource) {
    const auto& active = nmos::fields::endpoint_active(connection_resource.data);
    const bool enabled = nmos::fields::master_enable(active);
    const auto& tps = nmos::fields::transport_params(active);
    const value tp0 = (tps.is_array() && tps.as_array().size()) ? tps.at(0) : value::object();

    EngineController::Rtp s;
    s.active = enabled;
    s.src = tp_str(tp0, U("source_ip"));
    s.iface = tp_str(tp0, U("interface_ip"));
    s.port = tp_uint(tp0, U("destination_port"));

    const auto& id = resource.id;
    if (id == ids.receiver_v) {
      s.group = tp_str(tp0, U("multicast_ip"));
      EngineController::VideoFmt f;
      parse_video_fmtp(active_sdp(active), f);
      ctrl->set_video_rx(s, f);
    } else if (id == ids.receiver_a) {
      s.group = tp_str(tp0, U("multicast_ip"));
      ctrl->set_audio_rx(s);
    } else if (id == ids.sender_v) {
      s.group = tp_str(tp0, U("destination_ip"));  // a sender transmits TO destination_ip
      ctrl->set_video_tx(s);
    } else if (id == ids.sender_a) {
      s.group = tp_str(tp0, U("destination_ip"));
      ctrl->set_audio_tx(s);
    }
    slog::log<slog::severities::info>(gate, SLOG_FLF)
        << "spark nmos: activated " << resource.type.name << " " << resource.id
        << " (master_enable=" << std::boolalpha << enabled << ")";
  };
}

// ----- create our fixed resource set and insert it into the model -----
void insert_spark_resources(nmos::node_model& model, slog::base_gate& gate) {
  auto lock = model.write_lock();
  const auto& settings = model.settings;
  const Ids ids(settings);

  const auto insert = [&](nmos::resources& resources, nmos::resource&& r) {
    if (!nmos::insert_resource(resources, std::move(r)).second)
      throw std::logic_error("spark nmos: resource insert failed");
  };

  // PTP clock clk0 — gmid/traceable/locked supplied by the control daemon (from `pmc`).
  const auto gmid = str_field(settings, U("spark_ptp_gmid"), U("00-00-00-00-00-00-00-00"));
  const auto traceable = bool_field(settings, U("spark_ptp_traceable"), true);
  const auto locked = bool_field(settings, U("spark_ptp_locked"), true);
  const auto clocks = value_of({ nmos::make_ptp_clock(nmos::clock_names::clk0, traceable, gmid, locked) });

  const auto host_interfaces = nmos::get_host_interfaces(settings);
  const auto interfaces = nmos::experimental::node_interfaces(host_interfaces);

  // Bind senders/receivers to the media interface that owns the host_address (single leg in v1).
  const auto& host_address = nmos::fields::host_address(settings);
  const auto* media_if = find_interface(host_interfaces, host_address);
  if (!media_if) throw std::logic_error("spark nmos: no interface for host_address " + utility::us2s(host_address));
  const std::vector<utility::string_t> interface_names{ media_if->name };

  // Human-readable name for controllers / the BMD source list (resources were unlabeled -> blank).
  const auto node_label = str_field(settings, U("spark_label"), U("Spark Video Processor"));

  // Node + Device
  {
    auto node = nmos::make_node(ids.node, clocks, nmos::make_node_interfaces(interfaces), settings);
    node.data[nmos::fields::label] = value::string(node_label);
    node.data[nmos::fields::description] = value::string(node_label + U(" - ST 2110 processor"));
    insert(model.node_resources, std::move(node));
  }
  {
    auto device = nmos::make_device(ids.device, ids.node, { ids.sender_v, ids.sender_a },
                                    { ids.receiver_v, ids.receiver_a }, settings);
    device.data[nmos::fields::label] = value::string(node_label);
    insert(model.node_resources, std::move(device));
  }

  // Initial output format (4:2:2 10-bit video; stereo 48k/24-bit audio). These are seed values only:
  // OutputFormatSync polls the control daemon and keeps the video flow + sender SDP in step with the
  // operator's selected output resolution / FRC mode at runtime (no node restart needed).
  const nmos::rational frame_rate{ 60000, 1001 };
  const unsigned int frame_width = uint_field(settings, U("spark_out_w"), 1920);
  const unsigned int frame_height = uint_field(settings, U("spark_out_h"), 1080);

  // ---- video sender chain (source -> flow -> sender + IS-05 connection sender) ----
  {
    auto source = nmos::make_video_source(ids.source_v, ids.device, nmos::clock_names::clk0, frame_rate, settings);
    auto flow = nmos::make_raw_video_flow(ids.flow_v, ids.source_v, ids.device, frame_rate, frame_width, frame_height,
                                          nmos::interlace_modes::progressive, nmos::colorspaces::BT709,
                                          nmos::transfer_characteristics::SDR, sdp::samplings::YCbCr_4_2_2, 10, settings);
    insert(model.node_resources, std::move(source));  // setter looks these up, so insert before sender
    insert(model.node_resources, std::move(flow));

    const auto manifest = nmos::experimental::make_manifest_api_manifest(ids.sender_v, settings);
    // rtp.mcast (not generic rtp): ST 2110 senders are multicast, and controllers (e.g. the Blackmagic
    // source dropdown) filter to senders whose transport matches the receiver's rtp.mcast — a generic
    // rtp sender is hidden even though a forced IS-05 PATCH still connects.
    auto sender = nmos::make_sender(ids.sender_v, ids.flow_v, nmos::transports::rtp_mcast, ids.device,
                                    manifest.to_string(), interface_names, settings);
    sender.data[nmos::fields::label] = value::string(node_label + U(" - ST 2110-20 video"));
    auto connection_sender = nmos::make_connection_rtp_sender(ids.sender_v, false /*smpte2022_7*/);
    connection_sender.data[nmos::fields::endpoint_constraints][0][nmos::fields::source_ip] =
        value_of({ { nmos::fields::constraint_enum, value_from_elements(media_if->addresses) } });
    if (bool_field(settings, U("spark_activate_senders"), true)) {
      auto& staged = connection_sender.data[nmos::fields::endpoint_staged];
      staged[nmos::fields::master_enable] = value::boolean(true);
      staged[nmos::fields::activation] = value_of({
          { nmos::fields::mode, nmos::activation_modes::activate_scheduled_relative.name },
          { nmos::fields::requested_time, U("0:0") },
          { nmos::fields::activation_time, nmos::make_version() } });
    }
    insert(model.node_resources, std::move(sender));
    insert(model.connection_resources, std::move(connection_sender));
  }

  // ---- audio sender chain (ST 2110-30, stereo L24 @ 48k) ----
  {
    const std::vector<nmos::channel> channels{
        { U("Left"), nmos::channel_symbols::L }, { U("Right"), nmos::channel_symbols::R } };
    auto source = nmos::make_audio_source(ids.source_a, ids.device, nmos::clock_names::clk0, frame_rate, channels, settings);
    auto flow = nmos::make_raw_audio_flow(ids.flow_a, ids.source_a, ids.device, 48000, 24, settings);
    flow.data[nmos::fields::grain_rate] = nmos::make_rational(frame_rate);
    insert(model.node_resources, std::move(source));
    insert(model.node_resources, std::move(flow));

    const auto manifest = nmos::experimental::make_manifest_api_manifest(ids.sender_a, settings);
    auto sender = nmos::make_sender(ids.sender_a, ids.flow_a, nmos::transports::rtp_mcast, ids.device,
                                    manifest.to_string(), interface_names, settings);
    sender.data[nmos::fields::label] = value::string(node_label + U(" - ST 2110-30 audio"));
    auto connection_sender = nmos::make_connection_rtp_sender(ids.sender_a, false);
    connection_sender.data[nmos::fields::endpoint_constraints][0][nmos::fields::source_ip] =
        value_of({ { nmos::fields::constraint_enum, value_from_elements(media_if->addresses) } });
    if (bool_field(settings, U("spark_activate_senders"), true)) {
      auto& staged = connection_sender.data[nmos::fields::endpoint_staged];
      staged[nmos::fields::master_enable] = value::boolean(true);
      staged[nmos::fields::activation] = value_of({
          { nmos::fields::mode, nmos::activation_modes::activate_scheduled_relative.name },
          { nmos::fields::requested_time, U("0:0") },
          { nmos::fields::activation_time, nmos::make_version() } });
    }
    insert(model.node_resources, std::move(sender));
    insert(model.connection_resources, std::move(connection_sender));
  }

  // ---- video receiver (ingest: ST 2110-20 raw 4:2:2 10-bit) ----
  {
    // Accept raw AND Blackmagic IP10 (so an IP10 2160p59.94/60 sender's SDP can stage/activate on this
    // receiver). The format params (sampling/depth/width/height/rate) match the raw constraint_sets below.
    auto receiver = nmos::make_receiver(ids.receiver_v, ids.device, nmos::transports::rtp, interface_names,
                                        nmos::formats::video,
                                        { nmos::media_types::video_raw,
                                          nmos::media_type{ U("video/vnd.blackmagicdesign.ip10") } },
                                        settings);
    receiver.data[nmos::fields::label] = value::string(node_label + U(" - video in"));
    receiver.data[nmos::fields::caps][nmos::fields::constraint_sets] = value_of({ value_of({
        // Ingest receiver: accept the common broadcast rates (the FRC operator handles rate
        // conversion downstream). Constraining to one rate makes nmos-cpp reject a real sender's
        // SDP as "unsupported format-specific parameters" — e.g. a 30000/1001 Blackmagic source.
        { nmos::caps::format::grain_rate, nmos::make_caps_rational_constraint({
            nmos::rational{ 24, 1 }, nmos::rational{ 24000, 1001 }, nmos::rational{ 25, 1 },
            nmos::rational{ 30, 1 }, nmos::rational{ 30000, 1001 }, nmos::rational{ 50, 1 },
            nmos::rational{ 60, 1 }, nmos::rational{ 60000, 1001 } }) },
        { nmos::caps::format::frame_width, nmos::make_caps_integer_constraint({ 1920, 3840 }) },
        { nmos::caps::format::frame_height, nmos::make_caps_integer_constraint({ 1080, 2160 }) },
        { nmos::caps::format::interlace_mode, nmos::make_caps_string_constraint({ nmos::interlace_modes::progressive.name }) },
        { nmos::caps::format::color_sampling, nmos::make_caps_string_constraint({ sdp::samplings::YCbCr_4_2_2.name }) },
        { nmos::caps::format::component_depth, nmos::make_caps_integer_constraint({ 10 }) }
    }) });
    receiver.data[nmos::fields::version] = receiver.data[nmos::fields::caps][nmos::fields::version] = value(nmos::make_version());
    auto connection_receiver = nmos::make_connection_rtp_receiver(ids.receiver_v, false);
    connection_receiver.data[nmos::fields::endpoint_constraints][0][nmos::fields::interface_ip] =
        value_of({ { nmos::fields::constraint_enum, value_from_elements(media_if->addresses) } });
    insert(model.node_resources, std::move(receiver));
    insert(model.connection_resources, std::move(connection_receiver));
  }

  // ---- audio receiver (ingest: ST 2110-30 PCM L16/L24 @ 48k) ----
  {
    auto receiver = nmos::make_audio_receiver(ids.receiver_a, ids.device, nmos::transports::rtp, interface_names, 24, settings);
    receiver.data[nmos::fields::label] = value::string(node_label + U(" - audio in"));
    receiver.data[nmos::fields::caps][nmos::fields::constraint_sets] = value_of({ value_of({
        { nmos::caps::format::channel_count, nmos::make_caps_integer_constraint({}, 1, 8) },
        { nmos::caps::format::sample_rate, nmos::make_caps_rational_constraint({ { 48000, 1 } }) },
        { nmos::caps::format::sample_depth, nmos::make_caps_integer_constraint({ 16, 24 }) },
        { nmos::caps::transport::packet_time, nmos::make_caps_number_constraint({ 1, 0.125 }) }
    }) });
    receiver.data[nmos::fields::version] = receiver.data[nmos::fields::caps][nmos::fields::version] = value(nmos::make_version());
    auto connection_receiver = nmos::make_connection_rtp_receiver(ids.receiver_a, false);
    connection_receiver.data[nmos::fields::endpoint_constraints][0][nmos::fields::interface_ip] =
        value_of({ { nmos::fields::constraint_enum, value_from_elements(media_if->addresses) } });
    insert(model.node_resources, std::move(receiver));
    insert(model.connection_resources, std::move(connection_receiver));
  }

  model.notify();
  slog::log<slog::severities::info>(gate, SLOG_FLF)
      << "spark nmos: inserted node/device + video & audio senders/receivers (clk0=ptp gmid=" << utility::us2s(gmid) << ")";
}

// ----- keep the node's advertised state (output format + PTP grandmaster) in step with reality -------
// Two things drift from a static node config: (1) the operator's output resolution / FRC mode, set on
// the control daemon via the web UI (outWidth/outHeight/frcMode); (2) the live PTP grandmaster, which
// the daemon reads from pmc (ptpGmid) and which can change via BMC. Both feed our sender SDPs (the GM
// into ts-refclk on BOTH senders), and ST 2110 receivers reject a mismatch — so we poll /api/status and
// mirror changes into the video flow + node clk0 + the sender transport files, live, no restart.
class NodeStateSync {
 public:
  NodeStateSync(nmos::node_model& model, slog::base_gate& gate)
      : model_(model), gate_(gate), ids_(model.settings),
        ptp_domain_(ptp_domain_setting(model.settings)),
        client_(str_field(model.settings, U("spark_control_url"), U("http://127.0.0.1:8080"))),
        thread_([this] { run(); }) {}
  ~NodeStateSync() {
    { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
  }

 private:
  void run() {
    std::unique_lock<std::mutex> lk(mu_);
    while (!stop_) {
      // wait first: lets the auto-activated senders resolve their transport params before the first rebuild
      cv_.wait_for(lk, std::chrono::seconds(3), [this] { return stop_; });
      if (stop_) break;
      lk.unlock();
      try { sync_once(); }
      catch (const std::exception& e) {
        slog::log<slog::severities::more_info>(gate_, SLOG_FLF) << "spark nmos: node-state sync skipped: " << e.what();
      }
      lk.lock();
    }
  }

  void sync_once() {
    auto resp = client_.request(web::http::methods::GET, U("/api/status")).get();
    if (resp.status_code() != web::http::status_codes::OK) return;
    const auto body = resp.extract_json().get();
    if (!body.has_field(U("config"))) return;
    const auto& c = body.at(U("config"));
    const uint32_t w = cfg_uint(c, U("outWidth")), h = cfg_uint(c, U("outHeight"));
    if (0 == w || 0 == h) return;  // daemon output not configured yet
    const auto gmid = (body.has_field(U("ptpGmid")) && body.at(U("ptpGmid")).is_string())
                          ? body.at(U("ptpGmid")).as_string() : utility::string_t{};
    apply(w, h, output_rate(cfg_str(c, U("inExactframerate")), cfg_uint(c, U("frcMode"))), gmid,
          cfg_bool(c, U("ip10")));
  }

  void apply(uint32_t w, uint32_t h, const nmos::rational& rate, const utility::string_t& gmid, bool ip10) {
    auto lock = model_.write_lock();
    auto flow = nmos::find_resource(model_.node_resources, { ids_.flow_v, nmos::types::flow });
    if (model_.node_resources.end() == flow) return;  // resources not inserted yet
    const auto ver = value(nmos::make_version());
    bool updated = false;

    // (1) output format -> video flow (resolution / rate); affects the video SDP only.
    if ((uint32_t)nmos::fields::frame_width(flow->data) != w ||
        (uint32_t)nmos::fields::frame_height(flow->data) != h ||
        nmos::parse_rational(nmos::fields::grain_rate(flow->data)) != rate) {
      nmos::modify_resource(model_.node_resources, ids_.flow_v, [&](nmos::resource& r) {
        r.data[nmos::fields::frame_width] = value((int)w);
        r.data[nmos::fields::frame_height] = value((int)h);
        r.data[nmos::fields::grain_rate] = nmos::make_rational(rate);
        r.data[nmos::fields::version] = ver;
      });
      nmos::modify_resource(model_.node_resources, ids_.source_v, [&](nmos::resource& r) {
        r.data[nmos::fields::grain_rate] = nmos::make_rational(rate);
        r.data[nmos::fields::version] = ver;
      });
      v_sdp_dirty_ = true;
      updated = true;
    }

    // (2) PTP grandmaster -> node clk0; affects ts-refclk in BOTH sender SDPs.
    if (!gmid.empty() && gmid != current_gmid()) {
      nmos::modify_resource(model_.node_resources, ids_.node, [&](nmos::resource& r) {
        for (auto& clk : r.data[nmos::fields::clocks].as_array())
          if (nmos::fields::name(clk) == nmos::clock_names::clk0.name)
            clk[nmos::fields::gmid] = value::string(gmid);
        r.data[nmos::fields::version] = ver;
      });
      v_sdp_dirty_ = a_sdp_dirty_ = true;
      updated = true;
    }

    // (2b) IP10 codec on/off -> video SDP signalling (rtpmap/fmtp). Video sender SDP only.
    if (ip10 != ip10_) {
      ip10_ = ip10;
      v_sdp_dirty_ = true;
      updated = true;
    }

    // (3) regenerate any dirty sender SDP, once that (auto-activated) sender's transport params resolve.
    if (v_sdp_dirty_ && rebuild_sender(ids_.sender_v)) { v_sdp_dirty_ = false; updated = true; }
    if (a_sdp_dirty_ && rebuild_sender(ids_.sender_a)) { a_sdp_dirty_ = false; updated = true; }

    if (updated) {
      model_.notify();
      slog::log<slog::severities::info>(gate_, SLOG_FLF)
          << "spark nmos: advert -> " << w << "x" << h << " @ " << rate.numerator() << "/" << rate.denominator()
          << (ip10_ ? " IP10" : "") << " gmid=" << utility::us2s(current_gmid());
    }
  }

  // Regenerate one sender's transport file (SDP) from the current node/source/flow. No-op (returns
  // false) until the sender's IS-05 transport params have resolved, so the dirty flag retries later.
  bool rebuild_sender(const nmos::id& sender_id) {  // caller holds the write lock
    auto csender = nmos::find_resource(model_.connection_resources, { sender_id, nmos::types::sender });
    auto sender = nmos::find_resource(model_.node_resources, { sender_id, nmos::types::sender });
    if (model_.connection_resources.end() == csender || model_.node_resources.end() == sender) return false;
    if (!sender_resolved(*csender)) return false;
    nmos::modify_resource(model_.connection_resources, sender_id, [&](nmos::resource& cr) {
      // ip10_ only affects the video sender (build_sender_transportfile ignores it for audio).
      auto tf = build_sender_transportfile(model_.node_resources, ids_, *sender, cr, ptp_domain_, ip10_);
      if (!tf.is_null()) cr.data[nmos::fields::endpoint_transportfile] = tf;
    });
    return true;
  }

  utility::string_t current_gmid() {  // clk0's advertised grandmaster; caller holds the lock
    auto node = nmos::find_resource(model_.node_resources, { ids_.node, nmos::types::node });
    if (model_.node_resources.end() == node) return {};
    for (const auto& clk : nmos::fields::clocks(node->data))
      if (nmos::fields::name(clk) == nmos::clock_names::clk0.name) return nmos::fields::gmid(clk);
    return {};
  }

  // out rate mirrors the engine (st2110_pipeline): up-convert (frc_mode 2) doubles the source rate,
  // else 1:1; fall back to 59.94 when no source is connected (the engine's base_fps default).
  static nmos::rational output_rate(const utility::string_t& in_fps, uint32_t frc_mode) {
    nmos::rational base = nmos::rates::rate59_94;
    const auto s = utility::us2s(in_fps);
    if (!s.empty()) {
      try {
        const auto slash = s.find('/');
        const int64_t num = std::stoll(s.substr(0, slash));
        const int64_t den = (std::string::npos == slash) ? 1 : std::stoll(s.substr(slash + 1));
        if (num > 0 && den > 0) base = nmos::rational(num, den);
      } catch (...) {}
    }
    return 2 == frc_mode ? nmos::rational(base.numerator() * 2, base.denominator()) : base;
  }

  static bool sender_resolved(const nmos::resource& connection_sender) {
    const auto& active = nmos::fields::endpoint_active(connection_sender.data);
    if (!active.is_object() || !active.has_field(U("transport_params"))) return false;
    const auto& tps = active.at(U("transport_params"));
    if (!tps.is_array() || 0 == tps.as_array().size()) return false;
    const auto& tp0 = tps.at(0);
    if (!tp0.has_field(U("destination_ip"))) return false;
    const auto& d = tp0.at(U("destination_ip"));
    return d.is_string() && !d.as_string().empty() && U("auto") != d.as_string();
  }

  static uint32_t cfg_uint(const value& c, const utility::string_t& k) {
    if (!c.has_field(k)) return 0;
    const auto& v = c.at(k);
    if (v.is_integer()) return (uint32_t)v.as_integer();
    if (v.is_string()) { try { return (uint32_t)std::stoul(utility::us2s(v.as_string())); } catch (...) {} }
    return 0;
  }
  static utility::string_t cfg_str(const value& c, const utility::string_t& k) {
    return (c.has_field(k) && c.at(k).is_string()) ? c.at(k).as_string() : utility::string_t{};
  }
  static bool cfg_bool(const value& c, const utility::string_t& k) {
    if (!c.has_field(k)) return false;
    const auto& v = c.at(k);
    if (v.is_boolean()) return v.as_bool();
    if (v.is_integer()) return 0 != v.as_integer();
    if (v.is_string()) { const auto s = utility::us2s(v.as_string()); return s == "1" || s == "true"; }
    return false;
  }

  nmos::node_model& model_;
  slog::base_gate& gate_;
  Ids ids_;
  bst::optional<int> ptp_domain_;
  web::http::client::http_client client_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  bool v_sdp_dirty_ = false;
  bool a_sdp_dirty_ = false;
  bool ip10_ = false;  // last-applied IP10 state (from the daemon config); flips -> rebuild video SDP
  std::thread thread_;
};

nmos::experimental::node_implementation make_spark_node_implementation(nmos::node_model& model, slog::base_gate& gate) {
  // The engine controller (HTTP -> control daemon) lives as long as the activation handler that holds it.
  auto ctrl = std::make_shared<EngineController>(
      str_field(model.settings, U("spark_control_url"), U("http://127.0.0.1:8080")), gate);
  return nmos::experimental::node_implementation()
      .on_load_server_certificates(nmos::make_load_server_certificates_handler(model.settings, gate))
      .on_load_dh_param(nmos::make_load_dh_param_handler(model.settings, gate))
      .on_load_ca_certificates(nmos::make_load_ca_certificates_handler(model.settings, gate))
      .on_resolve_auto(make_spark_auto_resolver(model.settings))
      .on_set_transportfile(make_spark_transportfile_setter(model.node_resources, model.settings))
      .on_connection_activated(make_spark_activation_handler(ctrl, model.settings, gate));
}

}  // namespace
}  // namespace spark

int main(int argc, char* argv[]) {
  nmos::node_model node_model;
  nmos::experimental::log_model log_model;
  std::filebuf error_log_buf;
  std::ostream error_log(std::cerr.rdbuf());
  std::filebuf access_log_buf;
  std::ostream access_log(&access_log_buf);
  nmos::experimental::log_gate gate(error_log, access_log, log_model);

  try {
    slog::log<slog::severities::info>(gate, SLOG_FLF) << "Starting Spark NMOS node";

    if (argc > 1) {
      std::error_code error;
      node_model.settings = web::json::value::parse(utility::s2us(argv[1]), error);
      if (error) {
        std::ifstream file(argv[1]);
        file.exceptions(std::ios_base::failbit);
        node_model.settings = web::json::value::parse(file);
        node_model.settings.as_object();
      }
    }

    nmos::insert_node_default_settings(node_model.settings);
    log_model.settings = node_model.settings;
    log_model.level = nmos::fields::logging_level(log_model.settings);

    slog::log<slog::severities::info>(gate, SLOG_FLF) << "Process ID: " << nmos::details::get_process_id();
    slog::log<slog::severities::info>(gate, SLOG_FLF) << "Build settings: " << nmos::get_build_settings_info();
    slog::log<slog::severities::info>(gate, SLOG_FLF) << "Initial settings: " << node_model.settings.serialize();

    auto node_implementation = spark::make_spark_node_implementation(node_model, gate);
    auto node_server = nmos::experimental::make_node_server(node_model, node_implementation, log_model, gate);

    // insert our resources once the server is up (model lock + notify inside)
    node_server.thread_functions.push_back([&] { spark::insert_spark_resources(node_model, gate); });

    slog::log<slog::severities::info>(gate, SLOG_FLF) << "Preparing for connections";
    nmos::server_guard node_server_guard(node_server);
    slog::log<slog::severities::info>(gate, SLOG_FLF) << "Ready for connections";

    // Mirror the daemon's output format + live PTP grandmaster into our flow/clock/SDPs (no restart).
    spark::NodeStateSync node_state_sync(node_model, gate);

    nmos::details::wait_term_signal();
    slog::log<slog::severities::info>(gate, SLOG_FLF) << "Closing connections";
  } catch (const std::exception& e) {
    slog::log<slog::severities::error>(gate, SLOG_FLF) << "Fatal: " << e.what();
    return 1;
  }
  slog::log<slog::severities::info>(gate, SLOG_FLF) << "Stopping Spark NMOS node";
  return 0;
}
