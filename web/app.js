// Spark Video Processor dashboard (REDStreamer-style).
//   * Engine control: HTTP/JSON to the control daemon (/api/*), 1 s status poll.
//   * Filter chain: the panel renders entirely from /api/filters descriptors (same idea as the
//     REDStreamer filters panel — a new stage appears here with zero JS changes). The chain array
//     (membership + order) maps onto config.filters; per-stage params map onto their config fields.
//     The daemon locks config while the engine runs, so edits apply at the next pipeline start.
//   * Routing: NMOS registry (IS-04) + nodes (IS-05) via the daemon's same-origin /api/nmos proxy.
// protobuf JSON: enums are strings, fields are camelCase, 64-bit ints are strings.
'use strict';

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const r = await fetch(path, opts);
  return r.json();
}

// ---------------- engine config model ----------------
// Simple (non-filter) form fields; the filter panel owns the rest via descriptors.
const SIMPLE = ['profile', 'frames', 'ip10', 'in_ip10', 'rx_pci', 'tx_pci', 'dst_mac'];
const CAMEL = { in_ip10: 'inIp10', rx_pci: 'rxPci', tx_pci: 'txPci', dst_mac: 'dstMac' };

let CATALOG = null;     // filter descriptors from /api/filters (or the fallback below)
let chain = [];         // enabled filters in run order -> config.filters
const sliders = {};     // cfg key -> {update(v), setDisabled(b)}
let formLoaded = false;
let dirty = false;      // unsaved edits: the poll must not clobber them
let running = false;
// Last full config from /api/status. /api/config REPLACES the daemon's whole config — posting the
// bare form would wipe the NMOS/SDP-derived routing fields (rxMcastGroup, in* format, tx*...), so
// Save overlays the form onto this.
let lastConfig = null;

// Mirrors the catalog the daemon serves at /api/filters; used if the endpoint is missing (older
// daemon binary). cfg = the PipelineConfig JSON key each param maps to.
const FALLBACK_CATALOG = [
  { name: 'frc', label: 'FRC — motion interpolation',
    tip: 'Motion-compensated frame-rate conversion (NVOF). Runs at native input resolution, before scale.',
    params: [
      { key: 'mode', label: 'Mode', type: 'select', cfg: 'frcMode', choices: [
        { value: 1, label: 'retime (1:1)' },
        { value: 2, label: 'up-convert 2×' },
        { value: 3, label: 'up-convert 2× — uniform grid' }] }] },
  { name: 'scale', label: 'Scale',
    tip: 'Resize to the delivery resolution. auto = anti-aliased supersampling on downscale, cubic on upscale, passthrough at 1:1.',
    params: [
      { key: 'out_width', label: 'Width', type: 'number', cfg: 'outWidth', int: true, min: 320, max: 7680, step: 2 },
      { key: 'out_height', label: 'Height', type: 'number', cfg: 'outHeight', int: true, min: 240, max: 4320, step: 2 },
      { key: 'interp', label: 'Interpolation', type: 'select', cfg: 'interp', choices: [
        { value: 'auto', label: 'auto' }, { value: 'cubic', label: 'cubic' },
        { value: 'linear', label: 'linear' }, { value: 'lanczos', label: 'lanczos' },
        { value: 'super', label: 'super (AA downscale)' },
        { value: 'fsrcnn', label: 'fsrcnn (AI ×2)' },
        { value: 'fsrcnn-s', label: 'fsrcnn-s (AI ×2 fast)' },
        { value: 'espcn', label: 'espcn (AI ×2)' }] }] },
  { name: 'sharpen', label: 'Sharpen',
    tip: 'Luma unsharp mask at the delivery resolution. 0 = identity.',
    params: [
      { key: 'amount', label: 'Amount', type: 'slider', cfg: 'sharpen', min: 0, max: 4, step: 0.05, digits: 2 }] },
  { name: 'procamp', label: 'Proc amp',
    tip: 'Classic video corrector on the native 10-bit YCbCr. Neutral = 0 / 1 / 1 / 0.',
    params: [
      { key: 'brightness', label: 'Brightness', type: 'slider', cfg: 'paBrightness', min: -1, max: 1, step: 0.01, digits: 2 },
      { key: 'contrast', label: 'Contrast', type: 'slider', cfg: 'paContrast', min: 0, max: 4, step: 0.05, digits: 2, unset_to: 1 },
      { key: 'saturation', label: 'Saturation', type: 'slider', cfg: 'paSaturation', min: 0, max: 4, step: 0.05, digits: 2, unset_to: 1 },
      { key: 'hue', label: 'Hue (°)', type: 'slider', cfg: 'paHueDeg', min: -180, max: 180, step: 1, digits: 0 }] },
];

function markDirty() {
  if (running || !formLoaded) return;
  dirty = true;
  $('msg').textContent = 'unsaved changes — Save applies at next start';
}

// ---------------- widgets ----------------
function setChip(el, text, cls) {
  el.textContent = text;
  el.classList.remove('on', 'off', 'warn');
  if (cls) el.classList.add(cls);
}

function mkSlider(el, { label, cfg, min, max, step, digits = 2 }) {
  el.innerHTML = `
    <div class="top"><label>${label}</label><input type="number" id="fp-${cfg}"></div>
    <input type="range">`;
  const num = el.querySelector('input[type=number]');
  const rng = el.querySelector('input[type=range]');
  for (const i of [num, rng]) { i.min = min; i.max = max; i.step = step; }
  rng.addEventListener('input', () => {
    num.value = Number(rng.value).toFixed(digits);
    markDirty();
  });
  num.addEventListener('change', () => {
    const v = Math.min(max, Math.max(min, Number(num.value) || 0));
    num.value = Number(v).toFixed(digits);
    rng.value = v;
    markDirty();
  });
  sliders[cfg] = {
    update(v) { rng.value = v; num.value = Number(v).toFixed(digits); },
    setDisabled(b) { el.classList.toggle('disabled', b); num.disabled = rng.disabled = b; },
  };
}

// ---------------- filter chain panel ----------------
// Rows display in chain order (enabled stages first, run top to bottom), then the disabled rest in
// catalog order. Enabling appends at the end of the chain; disabled filters keep their params.

async function loadCatalog() {
  try {
    const j = await api('/api/filters');
    CATALOG = Array.isArray(j.filters) && j.filters.length ? j.filters : FALLBACK_CATALOG;
  } catch {
    CATALOG = FALLBACK_CATALOG;
  }
  buildFilterPanel();
}

function buildFilterPanel() {
  const list = $('filter-list');
  list.innerHTML = '';
  for (const d of CATALOG) {
    const row = document.createElement('div');
    row.className = 'frow off';
    row.id = `frow-${d.name}`;

    const head = document.createElement('div');
    head.className = 'fhead';
    head.title = d.tip || '';
    head.innerHTML = `
      <label class="toggle"><input type="checkbox" id="fen-${d.name}"><span>${d.label}</span></label>
      <button class="fmove" id="fup-${d.name}" title="run earlier">&#9650;</button>
      <button class="fmove" id="fdn-${d.name}" title="run later">&#9660;</button>`;
    row.appendChild(head);

    const body = document.createElement('div');
    body.className = 'fbody';
    for (const p of d.params || []) {
      if (p.type === 'slider') {
        const div = document.createElement('div');
        div.className = 'slider';
        body.appendChild(div);
        mkSlider(div, { label: p.label, cfg: p.cfg, min: p.min, max: p.max, step: p.step, digits: p.digits ?? 2 });
      } else {
        const r = document.createElement('div');
        r.className = 'row';
        const id = `fp-${p.cfg}`;
        r.innerHTML = p.type === 'select'
          ? `<label for="${id}">${p.label}</label><select id="${id}"></select>`
          : `<label for="${id}">${p.label}</label><input type="number" id="${id}" min="${p.min ?? 0}" max="${p.max ?? ''}" step="${p.step ?? 1}">`;
        body.appendChild(r);
        if (p.type === 'select') {
          const sel = r.querySelector('select');
          for (const c of p.choices || []) {
            const o = document.createElement('option');
            o.value = String(c.value);
            o.textContent = c.label;
            sel.appendChild(o);
          }
        }
        r.querySelector('select, input').addEventListener('change', markDirty);
      }
    }
    row.appendChild(body);
    list.appendChild(row);

    $(`fen-${d.name}`).addEventListener('change', () => {
      const on = $(`fen-${d.name}`).checked;
      chain = chain.filter((n) => n !== d.name);
      if (on) chain.push(d.name);
      markDirty();
      applyChainState();
    });
    $(`fup-${d.name}`).addEventListener('click', () => moveFilter(d.name, -1));
    $(`fdn-${d.name}`).addEventListener('click', () => moveFilter(d.name, +1));
  }
}

function moveFilter(name, dir) {
  const i = chain.indexOf(name);
  const j = i + dir;
  if (i < 0 || j < 0 || j >= chain.length) return;
  [chain[i], chain[j]] = [chain[j], chain[i]];
  markDirty();
  applyChainState();
}

function applyChainState() {
  if (!CATALOG) return;
  const list = $('filter-list');
  const rest = CATALOG.map((d) => d.name).filter((n) => !chain.includes(n));
  for (const n of [...chain, ...rest]) {
    const row = $(`frow-${n}`);
    if (row) list.appendChild(row);  // appendChild moves existing nodes
  }
  for (const d of CATALOG) {
    const i = chain.indexOf(d.name);
    $(`frow-${d.name}`).classList.toggle('off', i < 0);
    $(`fen-${d.name}`).checked = i >= 0;
    $(`fup-${d.name}`).disabled = running || i <= 0;
    $(`fdn-${d.name}`).disabled = running || i < 0 || i === chain.length - 1;
  }
}

// config.filters -> chain array. Empty/absent = the engine's automatic composition (mirror
// st2110_pipeline: frc when mode>0, scale always, sharpen when amount>0, procamp when non-neutral).
// The GUI dedupes; hand-set duplicates ("procamp,procamp") stay an API-only feature.
function deriveChain(c) {
  const known = new Set(CATALOG.map((d) => d.name));
  const f = (c.filters || '').trim();
  if (f) {
    const seen = new Set();
    return f.split(',').map((t) => t.trim())
      .filter((t) => known.has(t) && !seen.has(t) && seen.add(t));
  }
  const ch = [];
  if (+c.frcMode > 0 || c.frc) ch.push('frc');
  ch.push('scale');
  if (+c.sharpen > 0) ch.push('sharpen');
  const contrast = +c.paContrast > 0 ? +c.paContrast : 1;
  const sat = +c.paSaturation > 0 ? +c.paSaturation : 1;
  if (+c.paBrightness !== 0 || contrast !== 1 || sat !== 1 || +c.paHueDeg !== 0) ch.push('procamp');
  return ch;
}

// ---------------- config <-> form ----------------
function fillForm(c) {
  if (!CATALOG) return;  // catalog not loaded yet; retry on the next poll
  for (const id of SIMPLE) {
    const el = $(id);
    const v = c[CAMEL[id] || id];
    if (v === undefined) continue;
    if (el.type === 'checkbox') el.checked = !!v;
    else el.value = v;
  }
  for (const d of CATALOG) {
    for (const p of d.params || []) {
      let v = c[p.cfg];
      if (v === undefined) continue;
      // proto3 zero-default: 0 for contrast/saturation (and frcMode) means "unset", not zero
      if (p.unset_to !== undefined && !(+v > 0)) v = p.unset_to;
      if (p.type === 'slider') sliders[p.cfg].update(+v);
      else if (p.type === 'select') { if (+v > 0 || typeof v === 'string') $(`fp-${p.cfg}`).value = String(v); }
      else $(`fp-${p.cfg}`).value = v;
    }
  }
  chain = deriveChain(c);
  applyChainState();
  formLoaded = true;
}

function readForm() {
  const c = {};
  for (const id of SIMPLE) {
    const el = $(id);
    const key = CAMEL[id] || id;
    if (el.type === 'checkbox') c[key] = el.checked;
    else if (el.type === 'number') c[key] = parseInt(el.value, 10) || 0;
    else c[key] = el.value;
  }
  for (const d of CATALOG) {
    for (const p of d.params || []) {
      const el = $(`fp-${p.cfg}`);
      if (p.type === 'select') {
        const numeric = (p.choices || []).some((ch) => typeof ch.value === 'number');
        c[p.cfg] = numeric ? Number(el.value) : el.value;
      } else {
        c[p.cfg] = p.int ? Math.round(Number(el.value) || 0) : Number(el.value) || 0;
      }
    }
  }
  // Chain membership + order. "," = explicitly-empty chain (the engine parses zero tokens; a truly
  // empty string would re-trigger its automatic composition). FRC enable rides frcMode: the daemon
  // defaults frc_mode=1, and a stale frc:true would re-enable it, so both are forced together.
  c.filters = chain.length ? chain.join(',') : ',';
  c.frc = chain.includes('frc');
  c.frcMode = c.frc ? (c.frcMode || 1) : 0;
  return c;
}

// ---------------- status poll ----------------
function renderStats(s) {
  const fixed = +s.fixedLatencyMs > 0;
  const put = (id, v, warn) => {
    const el = $(id);
    el.textContent = v;
    el.classList.toggle('warn', !!warn);
  };
  put('st-rxf', s.rxFrames ?? 0);
  put('st-rxlost', s.rxLost ?? 0, +s.rxLost > 0);
  put('st-txf', s.txFrames ?? 0);
  put('st-skip', s.txSkipped ?? 0, +s.txSkipped > 0);
  put('st-errs', `${s.txPastErr ?? 0} / ${s.txFutureErr ?? 0}`, +s.txPastErr > 0 || +s.txFutureErr > 0);
  put('st-frc', s.frcInterpolated ?? 0);
  put('st-lat', fixed ? `${s.fixedLatencyMs} ms` : 'servo');
  put('st-margin', fixed ? s.txMarginUs : '—', fixed && +s.txMarginUs === 0);
  put('st-pipe', (+s.pipeLatencyUs / 1000 || 0).toFixed(1));
  put('st-ingest', (+s.ingestLatencyUs || 0).toFixed(0));
  put('st-audio', `${s.audioRxPackets ?? 0} / ${s.audioTxPackets ?? 0}`);
  put('st-avlate', s.audioLate ?? 0, +s.audioLate > 0);
}

function setRunning(r) {
  running = r;
  $('panel-format').classList.toggle('inactive', r);
  $('panel-filters').classList.toggle('inactive', r);
  for (const id of SIMPLE) $(id).disabled = r;
  for (const d of CATALOG || []) {
    $(`fen-${d.name}`).disabled = r;
    for (const p of d.params || []) {
      if (p.type === 'slider') sliders[p.cfg].setDisabled(r);
      else $(`fp-${p.cfg}`).disabled = r;
    }
  }
  if (CATALOG) applyChainState();  // refresh the move buttons' disabled state
  $('save').disabled = r;
  const power = $('power');
  power.textContent = r ? '■ STOP' : '▶ START';
  power.classList.toggle('running', r);
}

async function poll() {
  let st;
  try {
    st = await api('/api/status');
  } catch {
    setChip($('conn'), 'offline', 'off');
    return;
  }
  setChip($('conn'), 'online', 'on');
  const state = st.state || 'STOPPED';
  setChip($('state'), state.toLowerCase(),
          state === 'RUNNING' ? 'on' : state === 'ERRORED' ? 'off' : '');
  const ptp = $('ptp');
  ptp.hidden = !st.ptpGmid;
  if (st.ptpGmid) ptp.textContent = `PTP ${st.ptpGmid}`;
  $('meta').textContent = st.pid > 0 ? `pid ${st.pid} · ${Math.round(st.uptimeS || 0)}s` : '';
  if (!dirty) $('msg').textContent = st.message || '';
  if (st.stats) renderStats(st.stats);
  if (st.config) lastConfig = st.config;
  const run = state === 'RUNNING';
  if (st.config && (!formLoaded || run) && !dirty) fillForm(st.config);
  setRunning(run);
}

async function saveConfig() {
  const a = await api('/api/config', {
    method: 'POST',
    body: JSON.stringify({ ...(lastConfig || {}), ...readForm() }),
  });
  if (a.ok) dirty = false;
  $('msg').textContent = a.message || '';
  return !!a.ok;
}

$('save').onclick = async () => { await saveConfig(); poll(); };
$('power').onclick = async () => {
  const power = $('power');
  power.disabled = true;
  try {
    if (running) {
      const a = await api('/api/stop', { method: 'POST', body: '' });
      $('msg').textContent = a.message || '';
    } else {
      if (dirty && !(await saveConfig())) return;  // unsaved edits ride along on START
      const a = await api('/api/start', { method: 'POST', body: '' });
      $('msg').textContent = a.message || '';
    }
  } finally {
    power.disabled = false;
    poll();
  }
};
for (const id of SIMPLE) $(id).addEventListener('change', markDirty);

// ---------------- NMOS discovery (registry OR mDNS proxy + node, via the daemon proxy) ----------
// `registry` is the IS-04 Query API base the dashboard reads. In registry-free (P2P) mode it
// points at the mDNS proxy (deploy/nmos_mdns_proxy.py), which browses _nmos-node._tcp and
// re-serves the Query API shape — so no registry is needed. The Node base (IS-05) is unchanged.
const NMOS = { registry: '', registryUrl: '', node: '', proxy: '', p2p: false, receivers: null, senders: null };

// Facility NMOS registry (IS-04 Query API). External registry on registry-host.
// Per-browser override: edit the Registry field in the Routing panel (saved to localStorage).
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
        <div class="src-act">${connected ? '<span class="badge">routed</span>' : ''}${btn}</div>
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
        <div class="src-act">${fromUs ? '<span class="badge">receiving us</span>' : ''}${btn}</div>
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
loadCatalog().then(poll);
setInterval(poll, 1000);
setInterval(refreshNmos, 5000);
