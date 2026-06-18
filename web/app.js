// Spark Video Processor dashboard.
//   * Engine control: talks to the control daemon's HTTP/JSON API (/api/*).
//   * Discover: talks DIRECTLY to the NMOS registry (IS-04 Query API) and our NMOS node (IS-05
//     Connection API) — both CORS-enabled — to list 2110 senders and route one to this processor.
// protobuf JSON: enums are strings, fields are camelCase, 64-bit ints are strings.
'use strict';

const $ = (id) => document.getElementById(id);
const CFG = ['profile', 'interp', 'out_width', 'out_height', 'frc_mode', 'frames', 'rx_pci', 'tx_pci', 'dst_mac'];
const CAMEL = { out_width: 'outWidth', out_height: 'outHeight', frc_mode: 'frcMode', rx_pci: 'rxPci', tx_pci: 'txPci', dst_mac: 'dstMac' };
let formLoaded = false;

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
  if (st.config && (!formLoaded || running)) { fillForm(st.config); formLoaded = true; }
  for (const id of CFG) $(id).disabled = running;
  $('save').disabled = running;
  $('start').disabled = running;
  $('stop').disabled = !running;
}

$('save').onclick = async () => {
  const a = await api('/api/config', { method: 'POST', body: JSON.stringify(readForm()) });
  $('msg').textContent = a.message || '';
  poll();
};
$('start').onclick = async () => { const a = await api('/api/start', { method: 'POST', body: '' }); $('msg').textContent = a.message || ''; poll(); };
$('stop').onclick = async () => { const a = await api('/api/stop', { method: 'POST', body: '' }); $('msg').textContent = a.message || ''; poll(); };

// ---------------- NMOS discovery (registry OR mDNS proxy + node, direct) ----------------
// `registry` is the IS-04 Query API base the dashboard reads. In registry-free (P2P) mode it
// points at the mDNS proxy (deploy/nmos_mdns_proxy.py), which browses _nmos-node._tcp and
// re-serves the Query API shape — so no registry is needed. The Node base (IS-05) is unchanged.
const NMOS = { registry: '', registryUrl: '', node: '', proxy: '', p2p: false, receivers: null };

function nmosDefaults() {
  const h = location.hostname || 'localhost';
  return { registry: `http://${h}:3211`, node: `http://${h}:3242`, proxy: `http://${h}:3290` };
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
}
async function jget(base, path) {
  const r = await fetch(base + path);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

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

async function loadSources() {
  const box = $('sources');
  let senders, flows, receivers;
  try {
    [senders, flows, receivers] = await Promise.all([
      jget(NMOS.registry, '/x-nmos/query/v1.3/senders'),
      jget(NMOS.registry, '/x-nmos/query/v1.3/flows'),
      jget(NMOS.registry, '/x-nmos/query/v1.3/receivers'),
    ]);
  } catch (e) {
    box.innerHTML = `<p class="err">registry unreachable at ${NMOS.registry}</p>`;
    return;
  }
  const flowById = Object.fromEntries(flows.map((f) => [f.id, f]));
  // which sender is each of OUR receivers currently subscribed to (active route)?
  const ourRx = await ourReceivers().catch(() => ({}));
  const ourRxIds = new Set(Object.values(ourRx));
  const connectedSender = {};
  for (const r of receivers) {
    if (ourRxIds.has(r.id) && r.subscription && r.subscription.active && r.subscription.sender_id)
      connectedSender[r.subscription.sender_id] = r.format.split(':').pop();
  }

  const rtp = senders.filter((s) => (s.transport || '').endsWith('rtp'));
  if (!rtp.length) { box.innerHTML = '<p class="hint">no ST 2110 (RTP) senders registered.</p>'; return; }

  box.innerHTML = rtp.map((s) => {
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
  }).join('');

  box.querySelectorAll('button[data-sender]').forEach((b) => {
    b.onclick = () => connectSource(b.dataset.sender, b.dataset.kind, b);
  });
  box.querySelectorAll('button[data-disc]').forEach((b) => {
    b.onclick = () => disconnectReceiver(b.dataset.disc, b);
  });
}

async function patchStaged(rxId, body) {
  const r = await fetch(`${NMOS.node}/x-nmos/connection/v1.1/single/receivers/${rxId}/staged`,
    { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`connect -> ${r.status} ${await r.text()}`);
  return r.json();
}

async function connectSource(senderId, kind, btn) {
  btn.disabled = true; btn.textContent = 'Connecting…';
  try {
    const rx = await ourReceivers();
    const rxId = rx[kind];
    if (!rxId) throw new Error(`no ${kind} receiver on this node`);
    const senders = await jget(NMOS.registry, '/x-nmos/query/v1.3/senders');
    const sender = senders.find((s) => s.id === senderId);
    const sdp = sender && sender.manifest_href ? await (await fetch(sender.manifest_href)).text() : '';
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

function refreshNmos() { loadNodeStatus(); loadSources(); }

$('nmos_refresh').onclick = () => { saveNmosCfg(); refreshNmos(); };
$('nmos_p2p').onchange = () => { saveNmosCfg(); refreshNmos(); };

// ---------------- boot ----------------
loadNmosCfg();
refreshNmos();
poll();
setInterval(poll, 1000);
setInterval(refreshNmos, 5000);
