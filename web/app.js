// Spark Video Processor dashboard — talks to the control daemon's HTTP/JSON API.
// protobuf JSON: enums are strings, fields are camelCase, 64-bit ints are strings.
'use strict';

const $ = (id) => document.getElementById(id);
const CFG = ['profile', 'interp', 'out_width', 'out_height', 'frc', 'frames', 'rx_pci', 'tx_pci', 'dst_mac'];
// snake_case input id -> protobuf JSON camelCase key
const CAMEL = { out_width: 'outWidth', out_height: 'outHeight', rx_pci: 'rxPci', tx_pci: 'txPci', dst_mac: 'dstMac' };
let formLoaded = false;

async function api(path, opts) {
  const r = await fetch(path, opts);
  return r.json();
}

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
    else if (el.type === 'number') c[key] = parseInt(el.value, 10) || 0;
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

poll();
setInterval(poll, 1000);
