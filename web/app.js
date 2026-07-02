// Spark Video Processor dashboard.
//   * Engine control: talks to the control daemon's HTTP/JSON API (/api/*).
//   * Discover: talks DIRECTLY to the NMOS registry (IS-04 Query API) and our NMOS node (IS-05
//     Connection API) — both CORS-enabled — to list 2110 senders and route one to this processor.
// protobuf JSON: enums are strings, fields are camelCase, 64-bit ints are strings.
'use strict';

const $ = (id) => document.getElementById(id);
const CFG = ['profile', 'interp', 'out_width', 'out_height', 'frc_mode', 'ip10', 'frames',
  'sharpen', 'pa_brightness', 'pa_contrast', 'pa_saturation', 'pa_hue_deg',
  'rx_pci', 'tx_pci', 'dst_mac', 'in_ip10', 'filters'];
const CAMEL = { out_width: 'outWidth', out_height: 'outHeight', frc_mode: 'frcMode', rx_pci: 'rxPci',
  tx_pci: 'txPci', dst_mac: 'dstMac', in_ip10: 'inIp10', pa_brightness: 'paBrightness',
  pa_contrast: 'paContrast', pa_saturation: 'paSaturation', pa_hue_deg: 'paHueDeg' };
// float-valued config (proc amp / sharpen) — parseInt would truncate 0.85 to 0
const FLOAT = new Set(['sharpen', 'pa_brightness', 'pa_contrast', 'pa_saturation', 'pa_hue_deg']);
let formLoaded = false;
// Last full config from /api/status. The form covers only the processing knobs, but /api/config
// REPLACES the daemon's whole config — posting the bare form would wipe the NMOS/SDP-derived
// routing fields (rxMcastGroup, rxSrcIp, in* format, tx*...), so Save overlays the form onto this.
let lastConfig = null;

async function api(path, opts) {
  const r = await fetch(path, opts);
  return r.json();
}

// ---------------- engine config (control daemon) ----------------
function fillForm(c) {
  for (const id of CFG) {
    const el = $(id);
    const v = c[CAMEL[id] || id];
    if (v === undefined) continue;
    if (el.type === 'checkbox') el.checked = !!v;
    else el.value = v;
  }
}
function readForm() {
  const c = {};
  for (const id of CFG) {
    const el = $(id);
    const key = CAMEL[id] || id;
    if (el.type === 'checkbox') c[key] = el.checked;
    else if (FLOAT.has(id)) c[key] = parseFloat(el.value) || 0;
    else if (el.type === 'number' || id === 'frc_mode') c[key] = parseInt(el.value, 10) || 0;
    else c[key] = el.value;
  }
  return c;
}
function renderStats(s) {
  const rows = [
    ['RX frames', s.rxFrames], ['RX packets', s.rxPackets], ['RX lost', s.rxLost],
    ['TX frames', s.txFrames], ['TX packets', s.txPackets],
    ['TX future_err', s.txFutureErr], ['TX past_err', s.txPastErr],
    ['FRC interpolated', s.frcInterpolated], ['Ingest latency (µs)', (s.ingestLatencyUs || 0).toFixed(1)],
  ];
  $('stats').innerHTML = rows.map(([k, v]) => {
    const bad = (k === 'RX lost' && +v > 0);
    return `<div class="stat"><span>${k}</span><b class="${bad ? 'bad' : ''}">${v ?? '–'}</b></div>`;
  }).join('');
}
async function poll() {
  let st;
  try { st = await api('/api/status'); } catch { $('state').textContent = 'daemon offline'; return; }
  const state = st.state || 'STOPPED';
  const running = state === 'RUNNING';
  const badge = $('state');
  badge.textContent = state;
  badge.className = 'badge ' + (running ? 'run' : state === 'ERRORED' ? 'err' : 'idle');
  $('pid').textContent = st.pid > 0 ? st.pid : '–';
  $('uptime').textContent = Math.round(st.uptimeS || 0);
  $('msg').textContent = st.message || '';
  if (st.stats) renderStats(st.stats);
  if (st.config) lastConfig = st.config;
  if (st.config && (!formLoaded || running)) { fillForm(st.config); formLoaded = true; }
  for (const id of CFG) $(id).disabled = running;
  $('save').disabled = running;
  $('start').disabled = running;
  $('stop').disabled = !running;
}

$('save').onclick = async () => {
  const a = await api('/api/config', { method: 'POST', body: JSON.stringify({ ...(lastConfig || {}), ...readForm() }) });
  $('msg').textContent = a.message || '';
  poll();
};
$('start').onclick = async () => { const a = await api('/api/start', { method: 'POST', body: '' }); $('msg').textContent = a.message || ''; poll(); };
$('stop').onclick = async () => { const a = await api('/api/stop', { method: 'POST', body: '' }); $('msg').textContent = a.message || ''; poll(); };

// ---------------- NMOS discovery (registry OR mDNS proxy + node, direct) ----------------
// `registry` is the IS-04 Query API base the dashboard reads. In registry-free (P2P) mode it
// points at the mDNS proxy (deploy/nmos_mdns_proxy.py), which browses _nmos-node._tcp and
// re-serves the Query API shape — so no registry is needed. The Node base (IS-05) is unchanged.
const NMOS = { registry: '', registryUrl: '', node: '', proxy: '', p2p: false, receivers: null, senders: null };

// Facility NMOS registry (IS-04 Query API). External registry on registry-host.
// Per-browser override: edit the Registry field in the Discover card (saved to localStorage).
const DEFAULT_REGISTRY = 'http://192.0.2.41:8010';

function nmosDefaults() {
  const h = location.hostname || 'localhost';
  return { registry: DEFAULT_REGISTRY, node: `http://${h}:3242`, proxy: `http://${h}:3290` };
}
function applyDiscoverySource() {
  NMOS.registry = NMOS.p2p ? NMOS.proxy : NMOS.registryUrl;
  $('nmos_registry').value = NMOS.registry;
  $('nmos_registry').disabled = NMOS.p2p;  // proxy URL is fixed when P2P is on
}
function loadNmosCfg() {
  const d = nmosDefaults();
  NMOS.registryUrl = localStorage.getItem('nmos_registry') || d.registry;
  NMOS.node = localStorage.getItem('nmos_node') || d.node;
  NMOS.proxy = localStorage.getItem('nmos_proxy') || d.proxy;
  NMOS.p2p = localStorage.getItem('nmos_p2p') === '1';
  $('nmos_node').value = NMOS.node;
  $('nmos_p2p').checked = NMOS.p2p;
  applyDiscoverySource();
}
function saveNmosCfg() {
  NMOS.p2p = $('nmos_p2p').checked;
  if (!NMOS.p2p) NMOS.registryUrl = $('nmos_registry').value.replace(/\/$/, '');
  NMOS.node = $('nmos_node').value.replace(/\/$/, '');
  localStorage.setItem('nmos_p2p', NMOS.p2p ? '1' : '0');
  localStorage.setItem('nmos_registry', NMOS.registryUrl);
  localStorage.setItem('nmos_node', NMOS.node);
  applyDiscoverySource();
  NMOS.receivers = null;  // re-resolve against the new node
  NMOS.senders = null;
}
// All NMOS traffic rides the daemon's same-origin /api/nmos forwarder. The dashboard has to read
// SDPs from — and PATCH the IS-05 connection APIs of — OTHER nodes (Blackmagic units serve no CORS
// headers, so the browser can't fetch them directly; a cross-origin PATCH preflight always fails).
// Proxying uniformly removes every CORS trap in both registry and mDNS (P2P) modes.
const prox = (url) => `/api/nmos?u=${encodeURIComponent(url)}`;
async function pget(url, asText) {
  const r = await fetch(prox(url));
  if (!r.ok) throw new Error(`GET ${url} -> ${r.status}`);
  return asText ? r.text() : r.json();
}
async function ppatch(url, body) {
  const r = await fetch(`${prox(url)}&m=PATCH`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`PATCH -> ${r.status} ${await r.text()}`);
  return r.json();
}
async function jget(base, path) { return pget(base + path); }

// our node's receiver ids, keyed by media format ('video' | 'audio')
async function ourReceivers() {
  if (NMOS.receivers) return NMOS.receivers;
  const rxs = await jget(NMOS.node, '/x-nmos/node/v1.3/receivers');
  const map = {};
  for (const r of rxs) {
    const fmt = (r.format || '').split(':').pop();
    if (fmt === 'video' || fmt === 'audio') map[fmt] = r.id;
  }
  NMOS.receivers = map;
  return map;
}

// our node's senders ({id, manifest}), keyed by media format — the output side of routing.
// manifest_href serves our sender's SDP (regenerated live by the node: format/IP10/gmid).
async function ourSenders() {
  if (NMOS.senders) return NMOS.senders;
  const [snds, flows] = await Promise.all([
    jget(NMOS.node, '/x-nmos/node/v1.3/senders'),
    jget(NMOS.node, '/x-nmos/node/v1.3/flows'),
  ]);
  const fmtOf = Object.fromEntries(flows.map((f) => [f.id, (f.format || '').split(':').pop()]));
  const map = {};
  for (const s of snds) {
    const fmt = fmtOf[s.flow_id];
    if (fmt === 'video' || fmt === 'audio') map[fmt] = { id: s.id, manifest: s.manifest_href };
  }
  NMOS.senders = map;
  return map;
}

// A device's IS-05 Connection API base (its controls list) — where receiver PATCHes must go.
function connectionHref(device) {
  const ctrls = ((device && device.controls) || [])
    .filter((c) => (c.type || '').startsWith('urn:x-nmos:control:sr-ctrl/'))
    .sort((a, b) => (b.type || '').localeCompare(a.type || ''));  // highest version first
  return ctrls.length ? ctrls[0].href.replace(/\/$/, '') : '';
}

async function loadNodeStatus() {
  const el = $('nmos_status');
  try {
    const self = await jget(NMOS.node, '/x-nmos/node/v1.3/self');
    const clk = (self.clocks || []).find((c) => c.ref_type === 'ptp');
    const clkTxt = clk
      ? `PTP <b class="${clk.traceable ? 'ok' : ''}">${clk.gmid || '?'}</b>${clk.traceable ? ' · traceable' : ''}`
      : 'clock: internal';
    el.innerHTML = `<span class="ok">●</span> node <b>${self.label || self.id.slice(0, 8)}</b> · ${clkTxt}`;
  } catch (e) {
    el.innerHTML = `<span class="err">●</span> node unreachable at ${NMOS.node}`;
  }
}

function fmtLabel(media) {
  if (media === 'video/raw') return 'ST 2110-20 video';
  if (media && media.startsWith('audio/L')) return `ST 2110-30 audio (${media.split('/').pop()})`;
  return media || '?';
}

// External receivers we can send our output to, kept between renders so button handlers can
// look up the receiver + its node's Connection API href by id.
const DESTS = {};

async function loadSources() {
  const box = $('sources');
  const dbox = $('dests');
  let senders, flows, receivers, devices;
  try {
    [senders, flows, receivers, devices] = await Promise.all([
      jget(NMOS.registry, '/x-nmos/query/v1.3/senders'),
      jget(NMOS.registry, '/x-nmos/query/v1.3/flows'),
      jget(NMOS.registry, '/x-nmos/query/v1.3/receivers'),
      jget(NMOS.registry, '/x-nmos/query/v1.3/devices'),
    ]);
  } catch (e) {
    box.innerHTML = `<p class="err">registry unreachable at ${NMOS.registry}</p>`;
    dbox.innerHTML = '';
    return;
  }
  const flowById = Object.fromEntries(flows.map((f) => [f.id, f]));
  const deviceById = Object.fromEntries(devices.map((d) => [d.id, d]));
  // which sender is each of OUR receivers currently subscribed to (active route)?
  const ourRx = await ourReceivers().catch(() => ({}));
  const ourSnd = await ourSenders().catch(() => ({}));
  const ourRxIds = new Set(Object.values(ourRx));
  const ourSndIds = new Set(Object.values(ourSnd).map((s) => s.id));
  const connectedSender = {};
  for (const r of receivers) {
    if (ourRxIds.has(r.id) && r.subscription && r.subscription.active && r.subscription.sender_id)
      connectedSender[r.subscription.sender_id] = r.format.split(':').pop();
  }

  // ---- inputs: external RTP senders we can route INTO the processor ----
  // match the whole RTP family: urn:x-nmos:transport:rtp, rtp.mcast, rtp.ucast
  const rtp = senders.filter((s) => (s.transport || '').split(':').pop().startsWith('rtp') && !ourSndIds.has(s.id));
  box.innerHTML = rtp.length ? rtp.map((s) => {
    const fl = flowById[s.flow_id] || {};
    const media = fl.media_type || '';
    const kind = (fl.format || '').split(':').pop();     // 'video' | 'audio' | ...
    const routable = kind === 'video' || kind === 'audio';
    const connected = !!connectedSender[s.id];
    const label = s.label || s.id.slice(0, 8);
    const btn = !routable
      ? `<span class="hint">unsupported</span>`
      : connected
        ? `<button class="btn stop" data-disc="${ourRx[kind]}">Disconnect</button>`
        : `<button class="btn go" data-sender="${s.id}" data-kind="${kind}">Connect</button>`;
    return `<div class="src ${connected ? 'on' : ''}">
        <div class="src-main"><b>${label}</b><span class="src-fmt">${fmtLabel(media)}</span></div>
        <div class="src-act">${connected ? '<span class="badge run">routed</span>' : ''}${btn}</div>
      </div>`;
  }).join('') : '<p class="hint">no ST 2110 (RTP) senders discovered.</p>';

  box.querySelectorAll('button[data-sender]').forEach((b) => {
    b.onclick = () => connectSource(b.dataset.sender, b.dataset.kind, b);
  });
  box.querySelectorAll('button[data-disc]').forEach((b) => {
    b.onclick = () => disconnectReceiver(b.dataset.disc, b);
  });

  // ---- outputs: external RTP receivers we can send the processed stream TO ----
  // Routing = PATCH the TARGET node's IS-05 with our sender's SDP (via the daemon proxy — the
  // Blackmagic connection APIs have no CORS). Its Connection API base comes from its device controls.
  for (const k of Object.keys(DESTS)) delete DESTS[k];
  const drtp = receivers.filter((r) => (r.transport || '').split(':').pop().startsWith('rtp') && !ourRxIds.has(r.id));
  dbox.innerHTML = drtp.length ? drtp.map((r) => {
    const kind = (r.format || '').split(':').pop();
    const href = connectionHref(deviceById[r.device_id]);
    const routable = (kind === 'video' || kind === 'audio') && !!href && !!ourSnd[kind];
    const sub = r.subscription || {};
    const fromUs = !!(sub.active && sub.sender_id && ourSndIds.has(sub.sender_id));
    const busy = !!(sub.active && sub.sender_id && !fromUs);  // routed to some other sender
    const label = r.label || r.id.slice(0, 8);
    DESTS[r.id] = { id: r.id, kind, href };
    const btn = !routable
      ? `<span class="hint">${href ? 'unsupported' : 'no connection API'}</span>`
      : fromUs
        ? `<button class="btn stop" data-drop="${r.id}">Release</button>`
        : `<button class="btn go" data-dest="${r.id}">Send</button>`;
    return `<div class="src ${fromUs ? 'on' : ''}">
        <div class="src-main"><b>${label}</b><span class="src-fmt">${kind}${busy ? ' · routed elsewhere' : ''}</span></div>
        <div class="src-act">${fromUs ? '<span class="badge run">receiving us</span>' : ''}${btn}</div>
      </div>`;
  }).join('') : '<p class="hint">no ST 2110 (RTP) receivers discovered.</p>';

  dbox.querySelectorAll('button[data-dest]').forEach((b) => {
    b.onclick = () => connectDest(DESTS[b.dataset.dest], b);
  });
  dbox.querySelectorAll('button[data-drop]').forEach((b) => {
    b.onclick = () => disconnectDest(DESTS[b.dataset.drop], b);
  });
}

async function patchStaged(rxId, body) {
  return ppatch(`${NMOS.node}/x-nmos/connection/v1.1/single/receivers/${rxId}/staged`, body);
}

// Fetch a sender's SDP (transport file) and insist it IS one — previously a 404 body or an empty
// string was silently PATCHed as the "SDP", which the receiver then rejected (or worse, half-took).
async function senderSdp(manifestHref, what) {
  if (!manifestHref) throw new Error(`${what} exposes no SDP (manifest_href empty)`);
  const sdp = await pget(manifestHref, true);
  if (!sdp || !sdp.trimStart().startsWith('v=')) throw new Error(`${what} returned no usable SDP`);
  return sdp;
}

async function connectSource(senderId, kind, btn) {
  btn.disabled = true; btn.textContent = 'Connecting…';
  try {
    const rx = await ourReceivers();
    const rxId = rx[kind];
    if (!rxId) throw new Error(`no ${kind} receiver on this node`);
    const senders = await jget(NMOS.registry, '/x-nmos/query/v1.3/senders');
    const sender = senders.find((s) => s.id === senderId);
    const sdp = await senderSdp(sender && sender.manifest_href, 'sender');
    await patchStaged(rxId, {
      sender_id: senderId,
      master_enable: true,
      activation: { mode: 'activate_immediate' },
      transport_file: { type: 'application/sdp', data: sdp },
    });
    $('msg').textContent = `routed ${kind} source → engine`;
    setTimeout(() => { refreshNmos(); poll(); }, 600);
  } catch (e) {
    $('msg').textContent = 'connect failed: ' + e.message;
    btn.disabled = false; btn.textContent = 'Connect';
  }
}

async function disconnectReceiver(rxId, btn) {
  btn.disabled = true; btn.textContent = 'Disconnecting…';
  try {
    await patchStaged(rxId, { master_enable: false, activation: { mode: 'activate_immediate' } });
    $('msg').textContent = 'source released';
    setTimeout(() => { refreshNmos(); poll(); }, 600);
  } catch (e) {
    $('msg').textContent = 'disconnect failed: ' + e.message;
    btn.disabled = false; btn.textContent = 'Disconnect';
  }
}

// ---- output routing: hand OUR sender's SDP to an external receiver's IS-05 ----
async function connectDest(dest, btn) {
  btn.disabled = true; btn.textContent = 'Sending…';
  try {
    const snd = (await ourSenders())[dest.kind];
    if (!snd) throw new Error(`no ${dest.kind} sender on this node`);
    const sdp = await senderSdp(snd.manifest, 'our sender');
    await ppatch(`${dest.href}/single/receivers/${dest.id}/staged`, {
      sender_id: snd.id,
      master_enable: true,
      activation: { mode: 'activate_immediate' },
      transport_file: { type: 'application/sdp', data: sdp },
    });
    $('msg').textContent = `routed engine ${dest.kind} → destination`;
    setTimeout(refreshNmos, 600);
  } catch (e) {
    $('msg').textContent = 'send failed: ' + e.message;
    btn.disabled = false; btn.textContent = 'Send';
  }
}

async function disconnectDest(dest, btn) {
  btn.disabled = true; btn.textContent = 'Releasing…';
  try {
    await ppatch(`${dest.href}/single/receivers/${dest.id}/staged`,
      { master_enable: false, activation: { mode: 'activate_immediate' } });
    $('msg').textContent = 'destination released';
    setTimeout(refreshNmos, 600);
  } catch (e) {
    $('msg').textContent = 'release failed: ' + e.message;
    btn.disabled = false; btn.textContent = 'Release';
  }
}

function refreshNmos() { loadNodeStatus(); loadSources(); }

$('nmos_refresh').onclick = () => { saveNmosCfg(); refreshNmos(); };
$('nmos_p2p').onchange = () => { saveNmosCfg(); refreshNmos(); };

// ---------------- boot ----------------
loadNmosCfg();
refreshNmos();
poll();
setInterval(poll, 1000);
setInterval(refreshNmos, 5000);
