// Dashboard controller: polls the REST API, listens to /ws/events, and drives the 3D scene and panels.
import { AgentScene } from './scene.js';
import { VoiceListener, canSpeak, isMonitorInput, isSpeaking, listen, recordUtterance, speak, speechText, stopSpeaking, transcribe } from './voice.js';

const $ = (id) => document.getElementById(id);
const scene = new AgentScene($('scene'), $('labels'));
if (new URLSearchParams(location.search).has('debug')) window.__scene = scene; // dev hook: drive the 3D view from the console

const TERMINAL = new Set(['completed', 'failed', 'cancelled']);
const state = {
  agents: new Map(),       // name -> AgentInfo
  tasks: new Map(),        // task_id -> TaskResult (latest seen)
  plans: new Map(),        // task_id -> Map(step_id -> step)
  running: new Map(),      // `${task}|${step}` -> {task, step, agent, started, label}
  lastResult: new Map(),   // agent -> {state, text, at}
  events: new Map(),       // task_id -> [event]
  selected: null,
  history: { cpu: [], ram: [], gpu: [], vram: [] },
};

// ------------------------------------------------------------------ helpers
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.message || body.detail?.[0]?.msg || body.detail || `HTTP ${res.status}`);
  return body;
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}
const gb = (s) => (s ? parseFloat(s) : null);
const fmtSec = (s) => (s == null ? '–' : s < 1 ? `${Math.round(s * 1000)}ms` : s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m${Math.round(s % 60)}s`);
const fmtClock = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
function fmtUptime(s) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${Math.floor(s % 60)}s` : `${Math.floor(s)}s`;
}
function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

// ------------------------------------------------------------------ agents in the scene
function idleText(name) {
  if (name === 'user') return 'you';
  if (name === 'aggregator') return 'final synthesis';
  if (name === 'speech') return speechIdleText();
  const info = state.agents.get(name);
  return info?.model_key ? info.model_key : info?.role === 'control' && !info.model_key ? 'rule-based' : '';
}

function sourcesFor(taskId, stepId, agent) {
  if (agent === 'manager') return ['user'];
  if (agent === 'planner') return ['manager'];
  const plan = state.plans.get(taskId);
  if (agent === 'aggregator') {
    if (!plan) return ['planner'];
    const deps = new Set([...plan.values()].flatMap((s) => s.depends_on));
    const leaves = [...plan.values()].filter((s) => !deps.has(s.id)).map((s) => s.agent);
    return leaves.length ? [...new Set(leaves)] : ['planner'];
  }
  const step = plan?.get(stepId);
  if (!step || !step.depends_on.length) return ['planner'];
  return [...new Set(step.depends_on.map((d) => plan.get(d)?.agent).filter(Boolean))];
}

function refreshAgentVisual(agent) {
  const mine = [...state.running.values()].filter((r) => r.agent === agent);
  if (mine.length) {
    const label = mine.length > 1 ? `running ×${mine.length}` : `running · ${mine[0].step}`;
    scene.setState(agent, 'running', label);
    scene.setEmitter(agent, [...new Set(mine.flatMap((r) => r.sources))]);
    return;
  }
  scene.setEmitter(agent, null);
  const last = state.lastResult.get(agent);
  const age = last ? Date.now() - last.at : Infinity;
  if (last && last.state === 'done' && age < 4000) scene.setState(agent, 'done', last.text);
  else if (last && last.state === 'failed' && age < 8000) scene.setState(agent, 'failed', last.text);
  else scene.setState(agent, 'idle', idleText(agent));
}

function markFinished(agent, kind, text) {
  state.lastResult.set(agent, { state: kind, text, at: Date.now() });
  refreshAgentVisual(agent);
  setTimeout(() => refreshAgentVisual(agent), kind === 'failed' ? 8100 : 4100);
}

// ------------------------------------------------------------------ live events
function handleEvent(e, { live = true } = {}) {
  const tid = e.task_id;
  if (!state.events.has(tid)) state.events.set(tid, []);
  const list = state.events.get(tid);
  if (!list.some((x) => x.timestamp === e.timestamp && x.event === e.event && x.step_id === e.step_id)) list.push(e);

  const key = (step) => `${tid}|${step}`;
  switch (e.event) {
    case 'task_started':
      if (live) scene.flow('user', 'manager', undefined, 4);
      break;
    case 'plan_created':
      state.plans.set(tid, new Map(e.steps.map((s) => [s.id, s])));
      break;
    case 'agent_started': {
      const step = e.step_id || e.agent;
      state.running.set(key(step), {
        task: tid, step, agent: e.agent, label: e.task, started: e.timestamp,
        sources: sourcesFor(tid, step, e.agent), attempt: e.attempt, device: e.device,
      });
      refreshAgentVisual(e.agent);
      break;
    }
    case 'aggregation_started':
      state.running.set(key('aggregate'), {
        task: tid, step: 'aggregate', agent: 'aggregator', label: 'Synthesize the final answer',
        started: e.timestamp, sources: sourcesFor(tid, 'aggregate', 'aggregator'),
      });
      refreshAgentVisual('aggregator');
      break;
    case 'agent_completed':
    case 'agent_skipped':
    case 'agent_failed': {
      const step = e.step_id || e.agent;
      state.running.delete(key(step));
      const failed = e.event === 'agent_failed';
      markFinished(e.agent, failed ? 'failed' : 'done',
        failed ? 'failed' : e.event === 'agent_skipped' ? 'skipped' : `done · ${fmtSec(e.execution_time)}`);
      break;
    }
    case 'agent_retry':
      scene.setState(e.agent, 'running', e.action === 'fallback_cpu' ? 'retry on CPU' : 'retrying');
      break;
    case 'agent_progress': {
      const r = state.running.get(key(e.step_id || e.agent));
      if (r && e.message) r.label = e.message;
      break;
    }
    case 'task_completed':
    case 'task_failed':
    case 'task_cancelled': {
      const touched = new Set();
      for (const [k, r] of state.running) if (r.task === tid) { touched.add(r.agent); state.running.delete(k); }
      for (const a of touched) markFinished(a, e.event === 'task_completed' ? 'done' : 'failed', e.event.replace('task_', ''));
      if (e.event === 'task_completed') {
        markFinished('aggregator', 'done', `done · ${fmtSec(e.duration)}`);
        if (live) scene.flow('aggregator', 'user');
      }
      break;
    }
  }
  if (live) {
    renderRunning();
    if (['task_queued', 'task_started', 'task_completed', 'task_failed', 'task_cancelled', 'plan_created'].includes(e.event)) loadTasks();
    renderBotSoon(tid);
    if (tid === state.selected) loadDetailSoon();
  }
}

let ws, wsRetry = 0;
function connectEvents() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/events`);
  ws.onopen = () => {
    wsRetry = 0;
    setConn(true);
    resyncRunning();
  };
  ws.onmessage = (m) => handleEvent(JSON.parse(m.data));
  ws.onclose = () => {
    setConn(false);
    setTimeout(connectEvents, Math.min(8000, 500 * 2 ** wsRetry++));
  };
}
function setConn(ok) {
  $('conn').classList.toggle('ok', ok);
  $('conn').querySelector('.label').textContent = ok ? 'live' : 'reconnecting…';
}

// Tasks that were already running when the page (re)connected: replay their history to rebuild the scene.
async function resyncRunning() {
  try {
    const running = await api('/api/tasks?status=running&limit=20');
    for (const t of running) {
      const history = await api(`/api/tasks/${t.task_id}/events`);
      for (const e of history) handleEvent(e, { live: false });
    }
    renderRunning();
  } catch { /* server not ready yet; the next reconnect retries */ }
}

// ------------------------------------------------------------------ running list
function renderRunning() {
  const ul = $('runningList');
  ul.replaceChildren();
  const items = [...state.running.values()].sort((a, b) => a.started - b.started);
  if (!items.length) { ul.append(el('li', 'muted', 'Nothing running')); return; }
  const now = Date.now() / 1000;
  for (const r of items) {
    const li = el('li');
    li.append(el('b', null, agentLabel(r.agent)));
    li.append(el('span', null, r.label || r.step));
    li.append(el('span', 'elapsed', fmtSec(Math.max(0, now - r.started))));
    li.title = `${r.task} · step ${r.step}${r.device ? ' · ' + r.device : ''}`;
    ul.append(li);
  }
}
setInterval(renderRunning, 500);

// ------------------------------------------------------------------ resources
function pushHist(k, v) {
  const h = state.history[k];
  h.push(v);
  if (h.length > 90) h.shift();
}
function drawSpark(canvas, data, color, max = 100) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const pts = data.filter((v) => v != null);
  if (pts.length < 2) return;
  const step = w / 89;
  const off = (90 - data.length) * step;
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = off + i * step, y = h - 2 - ((v ?? 0) / max) * (h - 4);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.lineTo(off + (data.length - 1) * step, h);
  ctx.lineTo(off, h);
  ctx.closePath();
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, color + '55');
  g.addColorStop(1, color + '00');
  ctx.fillStyle = g;
  ctx.fill();
}
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function setGauge(k, pct, text) {
  $(`${k}Val`).textContent = text;
  $(`${k}Bar`).style.width = `${pct == null ? 0 : Math.min(100, pct)}%`;
  pushHist(k, pct);
  drawSpark($(`${k}Spark`), state.history[k], css(`--${k}`));
}

async function loadHealth() {
  try {
    const h = await api('/api/health');
    $('device').textContent = h.device.toUpperCase();
    $('gpuName').textContent = h.gpu || 'none';
    $('uptime').textContent = fmtUptime(h.uptime_seconds);
    $('offline').querySelector('span:last-child').textContent = h.offline_mode ? 'on' : 'off';
    setGauge('cpu', h.cpu_percent, h.cpu_percent == null ? 'n/a' : `${h.cpu_percent.toFixed(0)}%`);
    const ru = gb(h.ram_used), rt = gb(h.ram_total);
    setGauge('ram', rt ? (ru / rt) * 100 : null, rt ? `${ru.toFixed(1)} / ${rt.toFixed(0)} GB` : 'n/a');
    setGauge('gpu', h.gpu_utilization, h.gpu_utilization == null ? 'n/a' : `${h.gpu_utilization.toFixed(0)}%`);
    const vu = gb(h.vram_used), vt = gb(h.vram_total);
    setGauge('vram', vt ? (vu / vt) * 100 : null, vt ? `${vu.toFixed(1)} / ${vt.toFixed(0)} GB` : 'n/a');
    $('cRunning').textContent = h.tasks_running;
    $('cQueued').textContent = h.tasks_queued;
    $('cAgents').textContent = h.agents_active;
    $('resources').title = `System resources · updated ${new Date().toLocaleTimeString()}`;
    $('resources').classList.remove('stale');
  } catch {
    $('resources').title = 'Server unreachable';
    $('resources').classList.add('stale');
  }
}

// ------------------------------------------------------------------ models & performance
async function loadModels() {
  try {
    const models = await api('/api/models');
    const tbody = $('modelRows');
    tbody.replaceChildren();
    for (const m of models) {
      const tr = el('tr');
      const name = el('td', null, m.name);
      name.title = [m.hf_id || m.path, m.architecture, m.engine && `engine: ${m.engine}`, m.format && `format: ${m.format}`,
        m.quantization && `quant: ${m.quantization}`, m.est_vram_gb && `est. VRAM ~${m.est_vram_gb} GB`].filter(Boolean).join('\n');
      name.append(el('span', 'muted small', ` ${m.type}`));
      tr.append(name);
      const st = el('td');
      st.append(el('span', `state ${m.installed ? m.state : 'FAILED'}`, m.installed ? m.state : 'MISSING'));
      if (m.device) st.append(el('span', 'muted small', ` ${m.device}`));
      tr.append(st, el('td', 'num', m.memory_gb != null ? `${m.memory_gb.toFixed(2)} GB` : m.disk_size_gb != null ? `${m.disk_size_gb.toFixed(1)} GB disk` : '–'));
      const act = el('td');
      if (m.installed && !m.name.includes('@')) {
        const loaded = ['LOADED', 'ACTIVE', 'IDLE'].includes(m.state);
        const b = el('button', 'ghost', loaded ? 'Unload' : 'Load');
        b.onclick = async () => {
          b.disabled = true;
          b.textContent = loaded ? 'Unloading…' : 'Loading…';
          try { await api(`/api/models/${encodeURIComponent(m.name)}/${loaded ? 'unload' : 'load'}`, { method: 'POST' }); }
          catch (err) { alertMsg(err.message); }
          loadModels();
        };
        act.append(b);
      }
      tr.append(act);
      tbody.append(tr);
    }
  } catch { /* retried on next poll */ }
}

async function loadStatus() {
  try {
    const st = await api('/api/status');
    const chip = $('localChip');
    chip.classList.toggle('ok', st.cuda_available);
    chip.classList.toggle('warn', !st.cuda_available);
    $('laDevice').textContent = st.cuda_available ? 'CUDA' : 'no CUDA GPU';
    $('laGpu').textContent = st.gpu ? st.gpu.replace(/^NVIDIA\s+(GeForce\s+)?/, '') : '–';
    $('laVram').textContent = st.vram_total_gb ? `${st.vram_used_gb.toFixed(1)}/${st.vram_total_gb.toFixed(0)} GB` : '–';
    const rm = st.research_model || {};
    $('laModel').textContent = rm.resolved ? `${rm.resolved}${rm.loaded ? '' : ' (idle)'}` : 'no model';
    $('laWeb').textContent = st.web_enabled ? 'Web on' : 'Offline';
    chip.title = [st.notice, st.mode, st.gpu_message, rm.path && `Research model: ${rm.path}`,
      st.cloud_keys_ignored?.length && `Ignored cloud keys: ${st.cloud_keys_ignored.join(', ')}`].filter(Boolean).join('\n');
    $('webToggle').disabled = !st.research_enabled;
  } catch { /* retried on next poll */ }
}

async function loadDownloads() {
  try {
    const jobs = await api('/api/models/downloads');
    const ul = $('dlJobs');
    ul.replaceChildren();
    for (const j of jobs.slice(0, 4)) {
      const pct = j.progress != null ? ` ${j.progress.toFixed(0)}%` : '';
      const li = el('li', `dl ${j.status}`, `${j.hf_id} · ${j.status}${j.status === 'running' ? pct : ''}`);
      li.title = j.error || j.log.join('\n');
      ul.append(li);
    }
  } catch { /* ignore */ }
}
$('dlForm').onsubmit = async (e) => {
  e.preventDefault();
  const hf_id = $('dlRepo').value.trim();
  if (!hf_id) return;
  try {
    await api('/api/models/download', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ hf_id, type: $('dlType').value }),
    });
    $('dlRepo').value = '';
    alertMsg(`Downloading ${hf_id} to models/ — it will appear in the list when complete.`, false);
  } catch (err) { alertMsg(err.message); }
  loadDownloads();
};

async function loadMetrics() {
  try {
    const m = await api('/api/metrics?limit=1');
    const gpu = m.scheduler?.gpu;
    $('cGpuPool').textContent = gpu ? `${gpu.in_use}/${gpu.capacity}${gpu.waiting ? ` (+${gpu.waiting})` : ''}` : '–';
    const tbody = $('perfRows');
    tbody.replaceChildren();
    const rows = Object.entries(m.summary.agents || {}).sort((a, b) => b[1].calls - a[1].calls);
    if (!rows.length) {
      const tr = el('tr');
      const td = el('td', 'muted', 'No agent runs yet');
      td.colSpan = 6;
      tr.append(td);
      tbody.append(tr);
    }
    for (const [name, s] of rows) {
      const model = state.agents.get(name)?.model_key || (name === 'aggregator' ? 'base' : '');
      const tps = s.avg_tokens_per_second ?? m.summary.models?.[model]?.avg_tokens_per_second;
      const tr = el('tr');
      tr.append(el('td', null, name), el('td', 'muted', model || '–'), el('td', 'num', s.calls),
        el('td', `num${s.errors ? ' state FAILED' : ''}`, s.errors), el('td', 'num', fmtSec(s.avg_latency)),
        el('td', 'num', tps != null ? tps.toFixed(1) : '–'));
      tbody.append(tr);
    }
  } catch { /* retried on next poll */ }
}

async function loadAgents() {
  try {
    for (const a of await api('/api/agents')) {
      state.agents.set(a.name, a);
      scene.ensureNode(a.name);
      if (!state.running.size) refreshAgentVisual(a.name);
    }
  } catch { setTimeout(loadAgents, 2000); }
}

// ------------------------------------------------------------------ chat
const chat = { turns: new Map(), localImages: new Map(), sigs: new Map(), research: new Set(), answered: new Set(),
  voice: new Set(), spoken: new Set() };  // voice: tasks asked by voice, whose replies are read aloud

function inlineMd(s) {
  return s
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
}
// Minimal Markdown (paragraphs, lists, headings, code). Text is HTML-escaped before any tags are added.
function renderMarkdown(text) {
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  let html = '';
  text.split('```').forEach((block, i) => {
    if (i % 2) { html += `<pre><code>${esc(block.replace(/^[\w-]*\n/, ''))}</code></pre>`; return; }
    let list = null, para = [];
    const flushP = () => { if (para.length) { html += `<p>${para.join('<br>')}</p>`; para = []; } };
    const flushL = () => { if (list) { html += `</${list}>`; list = null; } };
    const openL = (kind) => { flushP(); if (list !== kind) { flushL(); html += `<${kind}>`; list = kind; } };
    for (const line of esc(block).split('\n')) {
      let m;
      if ((m = line.match(/^\s*[-*•]\s+(.*)/))) { openL('ul'); html += `<li>${inlineMd(m[1])}</li>`; }
      else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) { openL('ol'); html += `<li>${inlineMd(m[1])}</li>`; }
      else if ((m = line.match(/^#{1,6}\s+(.*)/))) { flushP(); flushL(); html += `<h4>${inlineMd(m[1])}</h4>`; }
      else if (!line.trim()) { flushP(); flushL(); }
      else { flushL(); para.push(inlineMd(line)); }
    }
    flushP(); flushL();
  });
  return html;
}

// Agent steps of a task, in order, from live events (or stored agent results for older tasks).
function taskSteps(t) {
  const steps = new Map();
  const evs = state.events.get(t.task_id) || [];
  if (evs.some((e) => e.event === 'agent_started')) {
    for (const e of evs) {
      const step = e.step_id || e.agent;
      if (e.event === 'agent_started' || e.event === 'agent_retry') steps.set(step, { agent: e.agent, state: 'running' });
      else if (e.event === 'aggregation_started') steps.set('aggregate', { agent: 'aggregator', state: 'running' });
      else if (e.event === 'agent_completed') steps.set(step, { agent: e.agent, state: 'done' });
      else if (e.event === 'agent_failed') steps.set(step, { agent: e.agent, state: 'failed' });
      else if (e.event === 'agent_skipped') steps.set(step, { agent: e.agent, state: 'skipped' });
    }
  } else {
    for (const [step, r] of Object.entries(t.agent_results || {})) {
      steps.set(step, { agent: r.agent, state: r.status === 'completed' ? 'done' : r.status === 'skipped' ? 'skipped' : 'failed' });
    }
    if (t.status === 'completed') steps.set('aggregate', { agent: 'aggregator', state: 'done' });
  }
  if (TERMINAL.has(t.status)) {
    for (const s of steps.values()) if (s.state === 'running') s.state = t.status === 'completed' ? 'done' : 'failed';
  }
  return [...steps.values()];
}

function taskStart(t) {
  const started = (state.events.get(t.task_id) || []).find((e) => e.event === 'task_started');
  return started?.timestamp || t.created_at;
}

function nearBottom() {
  const m = $('messages');
  return m.scrollHeight - m.scrollTop - m.clientHeight < 140;
}
function scrollToBottom() {
  const m = $('messages');
  m.scrollTop = m.scrollHeight;
}

function buildTurn(t) {
  const turn = el('div', 'turn');
  const user = el('div', 'msg user');
  const ub = el('div', 'bubble');
  const local = chat.localImages.get(t.task_id);
  if (local) {
    const img = el('img');
    img.src = local;
    img.alt = 'attached image';
    ub.append(img);
  } else if (t.image_path) {
    ub.append(el('span', 'img-chip', `🖼 ${t.image_path.split('/').pop()}`), document.createElement('br'));
  }
  ub.append(document.createTextNode(t.prompt || ''));
  user.append(ub);

  const bot = el('div', 'msg bot');
  bot.append(el('div', 'avatar'));
  bot.append(el('div', 'bubble'));
  turn.append(user, bot);
  return turn;
}

function renderBot(tid) {
  const t = state.tasks.get(tid);
  const turn = chat.turns.get(tid);
  if (!t || !turn) return;
  const steps = taskSteps(t);
  const research = isResearch(t);
  const rEvents = research ? researchEvents(tid) : [];
  const control = isControl(t);
  const cEvents = control ? controlEvents(tid) : [];
  const mNotes = modelNotes(tid);
  const sig = JSON.stringify([t.status, t.answer?.length, t.error, steps, rEvents.length, cEvents.length, mNotes.length,
    chat.answered.size]);
  if (chat.sigs.get(tid) === sig) return;
  chat.sigs.set(tid, sig);

  const bot = turn.querySelector('.msg.bot');
  bot.className = `msg bot ${t.status}`;
  const b = bot.querySelector('.bubble');
  b.replaceChildren();

  if (steps.length) {
    const row = el('div', 'steps');
    for (const s of steps) row.append(el('span', `step-chip ${s.state}`, agentLabel(s.agent)));
    b.append(row);
  }

  if (mNotes.length && !TERMINAL.has(t.status)) {
    b.append(el('div', 'model-notes', mNotes.map((e) => e.message).join(' · ')));
  }
  if (control) renderControl(b, t, cEvents);
  if (research) renderResearch(b, t, rEvents);
  const controlAfter = control && TERMINAL.has(t.status);
  if (research && t.answer) {
    renderResearchAnswer(b, t);
  } else if (t.answer) {
    const a = el('div', 'answer-text');
    a.innerHTML = renderMarkdown(t.answer);
    b.append(a);
  } else if (t.error || t.status === 'failed' || t.status === 'cancelled') {
    b.append(el('div', 'err-text', t.error || `Task ${t.status}.`));
  } else {
    const typing = el('div', 'typing');
    typing.append(el('i'), el('i'), el('i'));
    b.append(typing);
  }

  if (controlAfter) renderControlHistory(b, t, cEvents);

  const foot = el('div', 'msg-foot');
  foot.append(el('span', `badge ${t.status}`, t.status));
  if (research) foot.append(el('span', 'badge web', '🌐 web research'));
  if (control) foot.append(el('span', 'badge control', isJobs(t) ? '💼 Job Agent' : '🖥 AI DYOT'));
  if (t.plan?.intent) foot.append(el('span', null, t.plan.intent));
  const time = el('span', 'elapsed');
  time.dataset.task = tid;
  time.textContent = TERMINAL.has(t.status) ? fmtSec(t.duration) : '';
  foot.append(time);
  if (t.created_at) foot.append(el('span', null, fmtClock(t.created_at)));
  if (!TERMINAL.has(t.status)) {
    const stop = el('button', 'link-btn', 'Stop');
    stop.style.color = 'var(--failed)';
    stop.onclick = () => cancelTask(tid);
    foot.append(stop);
  }
  const det = el('button', 'link-btn', 'Details →');
  if (!TERMINAL.has(t.status)) det.style.marginLeft = '0';
  det.onclick = () => openDrawer(tid);
  foot.append(det);
  const said = TERMINAL.has(t.status) && (t.answer ? speechText(t.answer) : t.error || `Task ${t.status}.`);
  if (said && canSpeak()) {
    const listen = el('button', 'link-btn', '🔊 Listen');
    listen.style.marginLeft = '0';
    listen.title = 'Read this reply aloud (click again to stop)';
    listen.onclick = () => (isSpeaking() ? stopVoice() : speakReply(said));
    foot.append(listen);
  }
  b.append(foot);
  if (said && chat.voice.has(tid) && !chat.spoken.has(tid)) {
    chat.spoken.add(tid);
    speakReply(said);
  }
}

// ------------------------------------------------------------------ control panel turns
const AGENT_LABELS = { control_panel: 'AI DYOT', web_tester: 'Web Tester', job_agent: 'Job Agent', speech: 'Speech', airllm: 'AirLLM 120B' };
// Agents whose progress events can carry an approval prompt (`control.request`), rendered by renderControl.
const ASKING_AGENTS = new Set(['control_panel', 'job_agent']);
const agentLabel = (name) => AGENT_LABELS[name] || name;
const controlPayload = (t) => Object.values(t.agent_results || {}).find((r) => r.agent === 'control_panel')?.result || null;
const isControl = (t) => t.plan?.steps?.some((s) => ASKING_AGENTS.has(s.agent)) || !!controlPayload(t) ||
  (state.events.get(t.task_id) || []).some((e) => ASKING_AGENTS.has(e.agent));
const isJobs = (t) => t.plan?.steps?.some((s) => s.agent === 'job_agent');
const controlEvents = (tid) =>
  (state.events.get(tid) || []).filter((e) => e.event === 'agent_progress' && ASKING_AGENTS.has(e.agent));
const modelNotes = (tid) =>
  (state.events.get(tid) || []).filter((e) => e.event === 'agent_progress' && e.agent === 'manager' && e.model_action);

function renderControl(b, t, evs) {
  const done = TERMINAL.has(t.status);
  const lines = evs.filter((e) => e.message && !e.control?.request);
  const url = evs.find((e) => e.panel_url)?.panel_url || controlPayload(t)?.panel_url;
  // the newest request that has not been answered yet
  const answered = new Set(evs.filter((e) => e.control?.event === 'user_response').map((e) => e.control.request_id));
  const pending = !done && [...evs].reverse().find((e) => e.control?.request && !answered.has(e.control.request.id) &&
    !chat.answered.has(e.control.request.id));
  if (!done && lines.length) {
    const box = el('ul', 'r-status cp-lines');
    for (const e of lines.slice(-6)) box.append(el('li', e === lines[lines.length - 1] ? 'now' : 'past', e.message));
    b.append(box);
  }
  if (pending) b.append(controlAsk(pending.control.request));
  if (!done && url) {
    const a = el('a', 'cp-link', 'Watch the live screen ↗');
    a.href = url; a.target = '_blank'; a.rel = 'noopener';
    b.append(a);
  }
}

// After the answer: every desktop action (from the agent's stored result, so it survives a page reload).
function renderControlHistory(b, t, evs) {
  const payload = controlPayload(t);
  const acts = payload?.actions || [];
  const lines = acts.length
    ? acts.map((a) => `${a.ok ? '✓' : '✗'} ${String(a.label || '').replace(/\n/g, '⏎')} — ${a.message || ''}`)
    : evs.filter((e) => e.message && !e.control?.request).map((e) => e.message);
  if (lines.length) {
    const d = el('details', 'cp-history');
    d.append(el('summary', null, `Desktop actions (${payload?.action_count ?? lines.length})`));
    const ol = el('ol');
    for (const line of lines) ol.append(el('li', null, line));
    d.append(ol);
    b.append(d);
  }
  const url = payload?.panel_url || evs.find((e) => e.panel_url)?.panel_url;
  if (url) {
    const a = el('a', 'cp-link', 'Open AI DYOT ↗');
    a.href = url; a.target = '_blank'; a.rel = 'noopener';
    b.append(a);
  }
}

function controlAsk(req) {
  const box = el('div', `cp-ask ${req.kind === 'confirm' ? 'confirm' : ''}`);
  box.append(el('div', 'q', req.message));
  if (req.screenshot) {
    const img = el('img');
    img.src = `data:image/jpeg;base64,${req.screenshot}`;
    img.alt = 'screen at the time of the question';
    box.append(img);
  }
  const opts = el('div', 'opts');
  const send = async (answer) => {
    for (const btn of opts.querySelectorAll('button, input')) btn.disabled = true;
    try {
      await api(req.endpoint || `/api/control-panel/requests/${req.id}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ answer }),
      });
      chat.answered.add(req.id);
      box.classList.add('resolved');
    } catch (err) {
      alertMsg(err.message);
      for (const btn of opts.querySelectorAll('button, input')) btn.disabled = false;
    }
  };
  if (req.kind === 'confirm') {
    const ok = el('button', 'cp-btn primary', 'Approve'); ok.type = 'button'; ok.onclick = () => send('approve');
    const no = el('button', 'cp-btn danger', 'Deny'); no.type = 'button'; no.onclick = () => send('deny');
    opts.append(ok, no);
  } else {
    for (const c of req.choices || []) {
      const btn = el('button', 'cp-btn', c); btn.type = 'button'; btn.onclick = () => send(c); opts.append(btn);
    }
    const input = el('input');
    input.placeholder = req.choices?.length ? 'or type an answer…' : 'Type your answer…';
    input.onkeydown = (e) => { if (e.key === 'Enter' && input.value.trim()) send(input.value.trim()); };
    opts.append(input);
  }
  box.append(opts);
  return box;
}

// ------------------------------------------------------------------ web research turns
const researchPayload = (t) => Object.values(t.agent_results || {}).find((r) => r.agent === 'research')?.result || null;
const isResearch = (t) =>
  t.plan?.steps?.some((s) => s.agent === 'research') || chat.research.has(t.task_id) || !!researchPayload(t);
const researchEvents = (tid) =>
  (state.events.get(tid) || []).filter((e) => e.event === 'agent_progress' || e.event === 'research_source');

function renderResearch(b, t, evs) {
  const done = TERMINAL.has(t.status);
  const statuses = evs.filter((e) => e.event === 'agent_progress');
  const sources = evs.filter((e) => e.event === 'research_source');
  if (!done && statuses.length) {
    const box = el('ul', 'r-status');
    for (const e of statuses.slice(-5)) {
      const li = el('li', e === statuses[statuses.length - 1] ? 'now' : 'past', e.message);
      box.append(li);
    }
    b.append(box);
  }
  if (!done && sources.length) {
    const row = el('div', 'r-sources');
    for (const s of sources) {
      const ok = s.status === 'ok';
      const chip = el(ok ? 'a' : 'span', `r-src ${ok ? 'ok' : 'bad'}`, ok ? `[${s.source_id}] ${s.domain}` : s.domain || s.url);
      if (ok) { chip.href = s.url; chip.target = '_blank'; chip.rel = 'noopener noreferrer'; }
      chip.title = ok ? s.title : `${s.status}: ${s.error || ''}`;
      row.append(chip);
    }
    b.append(row);
  }
}

function renderResearchAnswer(b, t) {
  const payload = researchPayload(t);
  const sources = payload?.sources || [];
  const byId = new Map(sources.map((s) => [s.id, s]));
  const body = t.answer.split(/\n#{1,6}\s*Sources\b/)[0];
  const a = el('div', 'answer-text research');
  a.innerHTML = renderMarkdown(body).replace(/\[(\d{1,3})\]/g, (m, n) => {
    const s = byId.get(Number(n));
    if (!s) return m;
    if (!/^https?:\/\//.test(s.url)) return m;
    const attr = (x) => x.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
    return `<a class="cite" href="${attr(s.url)}" target="_blank" rel="noopener noreferrer" title="${attr(s.title)}">${n}</a>`;
  });
  b.append(a);
  if (payload?.confidence) {
    const c = el('div', `confidence ${payload.confidence.toLowerCase()}`, `Confidence: ${payload.confidence}`);
    if (payload.confidence_reason) c.title = payload.confidence_reason;
    b.append(c);
  }
  const cited = new Set(payload?.cited || []);
  const listed = sources.filter((s) => cited.has(s.id));
  const others = sources.filter((s) => !cited.has(s.id));
  if (listed.length || others.length) {
    const box = el('div', 'src-list');
    box.append(el('div', 'src-head', 'Sources'));
    const ol = el('ol');
    for (const s of listed.length ? listed : others) {
      const li = el('li');
      li.value = s.id;
      const link = el('a', null, s.title || s.url);
      link.href = s.url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      li.append(link, el('span', 'muted small', ` ${s.domain}${s.published_date ? ` · ${s.published_date.slice(0, 10)}` : ''}`));
      ol.append(li);
    }
    box.append(ol);
    if (listed.length && others.length) box.append(el('div', 'muted small', `Also read: ${others.map((s) => s.domain).join(', ')}`));
    const failed = payload?.failed_sources || [];
    if (failed.length) {
      const f = el('div', 'muted small', `${failed.length} page(s) could not be used (blocked, unavailable or unsupported).`);
      f.title = failed.map((x) => `${x.url}: ${x.error || x.status}`).join('\n');
      box.append(f);
    }
    b.append(box);
  }
}

const pendingBots = new Set();
const flushBots = debounce(() => {
  const stick = nearBottom();
  for (const tid of pendingBots) renderBot(tid);
  pendingBots.clear();
  if (stick) scrollToBottom();
}, 80);
function renderBotSoon(tid) {
  pendingBots.add(tid);
  flushBots();
}

// Live "working · 3.2s" counters on unfinished replies.
setInterval(() => {
  const now = Date.now() / 1000;
  for (const span of document.querySelectorAll('.msg-foot .elapsed')) {
    const t = state.tasks.get(span.dataset.task);
    if (t && !TERMINAL.has(t.status)) {
      const start = taskStart(t);
      span.textContent = t.status === 'queued' ? 'waiting in queue' : `working · ${fmtSec(Math.max(0, now - start))}`;
    }
  }
}, 500);

const loadTasks = debounce(async (forceScroll = false) => {
  try {
    const tasks = (await api('/api/tasks?limit=50')).reverse();
    const stick = forceScroll || nearBottom();
    $('taskCount').textContent = tasks.length ? `${tasks.length} message${tasks.length > 1 ? 's' : ''}` : '';
    $('chatEmpty').hidden = tasks.length > 0;
    const box = $('messages');
    for (const t of tasks) {
      state.tasks.set(t.task_id, t);
      if (!chat.turns.has(t.task_id)) {
        const turn = buildTurn(t);
        chat.turns.set(t.task_id, turn);
        box.append(turn);
      }
      renderBot(t.task_id);
    }
    if (stick) scrollToBottom();
  } catch { /* retried on next poll */ }
}, 120);

async function cancelTask(id) {
  try { await api(`/api/tasks/${id}`, { method: 'DELETE' }); } catch (err) { alertMsg(err.message); }
  loadTasks();
  if (id === state.selected) loadDetail(false);
}

// ------------------------------------------------------------------ task detail drawer (hidden until opened)
function openDrawer(id) {
  state.selected = id;
  $('drawer').classList.add('open');
  $('drawer').setAttribute('aria-hidden', 'false');
  $('scrim').hidden = false;
  document.body.classList.add('drawer-open');
  $('detailEmpty').hidden = false;
  $('detailEmpty').textContent = 'Loading…';
  $('detailBody').hidden = true;
  loadDetail(true);
}
function closeDrawer() {
  state.selected = null;
  $('drawer').classList.remove('open');
  $('drawer').setAttribute('aria-hidden', 'true');
  $('scrim').hidden = true;
  document.body.classList.remove('drawer-open');
}
$('drawerClose').onclick = closeDrawer;
$('scrim').onclick = closeDrawer;
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (isSpeaking()) stopVoice();
  else if (!$('help').hidden) closeHelp();
  else if (state.selected) closeDrawer();
});
$('cancelBtn').onclick = () => state.selected && cancelTask(state.selected);

const loadDetailSoon = debounce(() => loadDetail(false), 250);

async function loadDetail(withHistory) {
  const id = state.selected;
  if (!id) return;
  try {
    const [t, history] = await Promise.all([
      api(`/api/tasks/${id}`),
      withHistory ? api(`/api/tasks/${id}/events`) : Promise.resolve(null),
    ]);
    if (history) state.events.set(id, history);
    if (id === state.selected) renderDetail(t);
  } catch (err) {
    $('detailEmpty').hidden = false;
    $('detailBody').hidden = true;
    $('detailEmpty').textContent = `Could not load task: ${err.message}`;
  }
}

function stepCard(step, res, running) {
  const status = running ? 'running' : res?.status || 'pending';
  const card = el('div', `step ${status}`);
  const h = el('div', 'h');
  h.append(el('span', null, `${step.agent}`));
  h.append(el('span', `badge ${status}`, running ? 'running' : res ? (res.execution_time != null ? `${res.status} · ${fmtSec(res.execution_time)}` : res.status) : 'pending'));
  card.append(h, el('div', 't', `${step.id}: ${step.task}`));
  if (step.optional) card.querySelector('.t').textContent += ' (optional)';
  if (res) {
    const det = el('details');
    det.append(el('summary', null, res.error ? 'error' : 'output'));
    const body = res.error || (typeof res.result?.text === 'string' && res.result.text) || JSON.stringify(res.result, null, 2);
    det.append(el('pre', null, body));
    if (res.models?.length) det.append(el('div', 'muted small', `model: ${res.models.join(', ')} · attempts: ${res.attempts}`));
    card.append(det);
  }
  return card;
}

function renderDetail(t) {
  $('detailEmpty').hidden = true;
  $('detailBody').hidden = false;
  $('detailId').textContent = t.task_id;
  $('cancelBtn').hidden = TERMINAL.has(t.status);
  $('dPrompt').textContent = t.prompt || '';

  const meta = $('dMeta');
  meta.replaceChildren();
  const add = (s) => meta.append(el('span', null, s));
  meta.append(el('span', `badge ${t.status}`, t.status));
  if (t.plan?.intent) add(`intent: ${t.plan.intent}`);
  if (t.priority) add(`priority: ${t.priority}`);
  if (t.duration != null) add(`duration: ${fmtSec(t.duration)}`);
  if (t.image_path) add(`image: ${t.image_path.split('/').pop()}`);

  // Plan: control steps first, then the planner's parallel groups.
  const plan = $('dPlan');
  plan.replaceChildren();
  const isRunning = (step) => state.running.has(`${t.task_id}|${step}`);
  const control = el('div', 'grp');
  for (const a of ['manager', 'planner']) {
    const res = t.agent_results?.[a];
    if (res || isRunning(a)) control.append(stepCard({ id: a, agent: a, task: a === 'manager' ? 'route the request' : 'build the execution plan' }, res, isRunning(a)));
  }
  if (control.children.length) plan.append(control);
  if (t.plan) {
    const steps = new Map(t.plan.steps.map((s) => [s.id, s]));
    const groups = t.plan.parallel_groups?.length ? t.plan.parallel_groups : t.plan.steps.map((s) => [s.id]);
    for (const g of groups) {
      const row = el('div', 'grp');
      for (const sid of g) {
        const s = steps.get(sid);
        if (s) row.append(stepCard(s, t.agent_results?.[sid], isRunning(sid)));
      }
      plan.append(row);
    }
    if (isRunning('aggregate') || t.status === 'completed') {
      const row = el('div', 'grp');
      row.append(stepCard({ id: 'aggregate', agent: 'aggregator', task: 'synthesize the final answer' },
        t.status === 'completed' ? { status: 'completed', execution_time: null, result: { text: 'merged into the answer below' } } : null,
        isRunning('aggregate')));
      plan.append(row);
    }
  } else if (!control.children.length) {
    plan.append(el('div', 'muted', t.status === 'queued' ? 'Waiting in queue…' : 'No plan yet.'));
  }

  const ans = $('dAnswer');
  if (t.answer) { ans.className = 'answer answer-text'; ans.innerHTML = renderMarkdown(t.answer); }
  else if (t.error) { ans.className = 'answer error'; ans.textContent = t.error; }
  else { ans.className = 'answer pending'; ans.textContent = TERMINAL.has(t.status) ? 'No answer.' : 'Agents are working…'; }

  const tl = $('dTimeline');
  tl.replaceChildren();
  const evs = state.events.get(t.task_id) || [];
  const t0 = evs[0]?.timestamp;
  for (const e of evs) {
    const kind = e.event.endsWith('started') || e.event === 'agent_retry' ? 'started'
      : e.event.endsWith('failed') || e.event.endsWith('cancelled') ? 'failed'
      : e.event.endsWith('completed') || e.event === 'plan_created' ? 'completed' : '';
    const li = el('li', `ev-${kind}`);
    li.append(el('span', 'ts', `+${(e.timestamp - t0).toFixed(2)}s`));
    li.append(el('span', 'ic', kind === 'started' ? '▸' : kind === 'completed' ? '✓' : kind === 'failed' ? '✕' : '•'));
    let text = e.event.replace(/_/g, ' ');
    if (e.agent) text = `${agentLabel(e.agent)}${e.step_id && e.step_id !== e.agent ? ` [${e.step_id}]` : ''} — ${text}`;
    if (e.execution_time != null) text += ` (${fmtSec(e.execution_time)})`;
    if (e.event === 'plan_created') text += ` · ${e.intent} · ${e.steps.length} steps`;
    if (e.action) text += ` · ${e.action}`;
    if (e.error) text += ` · ${e.error}`;
    li.append(el('span', 'ev', text));
    tl.append(li);
  }
  tl.scrollTop = tl.scrollHeight;
}

// ------------------------------------------------------------------ composer
function alertMsg(text, isErr = true) {
  const m = $('formMsg');
  m.textContent = text;
  m.classList.toggle('err', isErr);
}

let previewUrl = null;
function setFile(file) {
  const input = $('image');
  if (file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
  }
  const f = input.files[0];
  if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
  $('attachPreview').hidden = !f;
  if (f) {
    previewUrl = URL.createObjectURL(f);
    $('attachImg').src = previewUrl;
    $('attachName').textContent = `${f.name} · ${(f.size / 1024).toFixed(0)} KB`;
  }
}
$('image').onchange = () => setFile();
$('attachClear').onclick = () => { $('image').value = ''; setFile(); };
const dz = $('dropZone');
dz.ondragover = (e) => { e.preventDefault(); dz.classList.add('drag'); };
dz.ondragleave = () => dz.classList.remove('drag');
dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove('drag'); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); };

const promptEl = $('prompt');
const autosize = () => { promptEl.style.height = 'auto'; promptEl.style.height = `${Math.min(promptEl.scrollHeight, 180)}px`; };
promptEl.oninput = autosize;
promptEl.onkeydown = (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $('taskForm').requestSubmit();
  }
};
promptEl.onpaste = (e) => {
  const img = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith('image/'));
  if (img) { e.preventDefault(); setFile(img); }
};
for (const pill of document.querySelectorAll('.suggest .pill')) {
  pill.onclick = () => {
    promptEl.value = pill.textContent.replace(/^🌐\s*/, '');
    if (pill.dataset.web) setWebMode(true);
    autosize();
    promptEl.focus();
  };
}

let webMode = false;
function setWebMode(on) {
  webMode = on;
  $('webToggle').setAttribute('aria-pressed', String(on));
  $('depth').hidden = !on;
  promptEl.placeholder = on
    ? 'Research question: the agent searches the web, reads sources and cites them…'
    : 'Message the agents…  (Shift+Enter for a new line)';
}
$('webToggle').onclick = () => setWebMode(!webMode);

// fromComposer: the prompt (and attached image) came from the message box, which is cleared afterwards.
// Hands-free voice requests leave whatever you were typing alone.
async function submitPrompt(prompt, { byVoice = false, fromComposer = true } = {}) {
  if (!prompt) return;
  const priority = $('priority').value;
  const file = fromComposer ? $('image').files[0] : null;
  const btn = $('submitBtn');
  btn.disabled = true;
  alertMsg('', false);
  try {
    let res;
    if (file) {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('prompt', prompt);
      fd.append('priority', priority);
      if (forcedAgent) fd.append('agents', forcedAgent);
      res = await api('/api/image', { method: 'POST', body: fd });
      chat.localImages.set(res.task_id, URL.createObjectURL(file));
    } else if (webMode) {
      res = await api('/api/research', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: prompt, depth: $('depth').value, priority }),
      });
      chat.research.add(res.research_id);
    } else {
      res = await api('/api/tasks', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt, priority, ...(forcedAgent && { options: { agents: [forcedAgent] } }) }),
      });
    }
    if (byVoice) chat.voice.add(res.task_id || res.research_id);
    if (fromComposer) {
      promptEl.value = '';
      autosize();
      $('image').value = '';
      setFile();
    }
    loadTasks(true);
  } catch (err) {
    alertMsg(err.message);
    if (byVoice) speakReply(`Sorry, that did not go through. ${err.message}`);
  } finally {
    btn.disabled = false;
    if (fromComposer) promptEl.focus();
  }
}
$('taskForm').onsubmit = (e) => {
  e.preventDefault();
  submitPrompt(promptEl.value.trim());
};

// ------------------------------------------------------------------ voice: the Speech agent
// 🎤 = push-to-talk (one message). "Hey CC" = hands-free: always listening, but only utterances that start with the
// wake word are acted on; everything else is dropped after transcription. Replies to voice requests are spoken.
const micBtn = $('micBtn'), earBtn = $('handsFreeBtn'), voiceLine = $('voiceLine');
const speech = { wake: 'Hey CC', needWake: false, endMs: 4000, followUpMs: 8000, listener: null, followUntil: 0, queue: Promise.resolve(), busy: 0 };
let recording = null;
const PREF_KEY = 'maa.handsFree';
const pref = { get: () => { try { return localStorage.getItem(PREF_KEY) === '1'; } catch { return false; } },
  set: (on) => { try { localStorage.setItem(PREF_KEY, on ? '1' : '0'); } catch { /* private window */ } } };

function speechIdleText() {
  if (!speech.needWake) return speech.listener ? 'listening · just talk' : 'off';
  if (!speech.listener) return `say “${speech.wake}” · off`;
  return Date.now() < speech.followUntil ? 'listening for your request…' : `listening for “${speech.wake}”`;
}
function speechState(st, text) {
  if (!state.agents.has('speech')) return;
  if (st === 'idle') refreshAgentVisual('speech');
  else scene.setState('speech', st, text);
}
function say(text) { voiceLine.hidden = !text; voiceLine.textContent = text || ''; }

function micState(name) {
  micBtn.setAttribute('aria-pressed', String(name === 'recording'));
  micBtn.classList.toggle('busy', name === 'busy');
  micBtn.classList.toggle('speaking', name === 'speaking');
  micBtn.style.setProperty('--level', 0);
  micBtn.title = {
    recording: 'Listening… stop talking to send (click to send now)',
    busy: 'Transcribing on this computer…',
    speaking: 'Reading the reply aloud (click or Esc to stop)',
  }[name] || 'Speak one message: it is transcribed on this computer and sent when you stop talking. The reply is read aloud.';
}

async function speakReply(text) {
  if (recording) return;
  micState('speaking');
  speechState('running', 'speaking');
  speech.listener?.mute(true); // don't let the agent hear itself
  const ok = await speak(text);
  speech.listener?.mute(false);
  if (micBtn.classList.contains('speaking')) micState('idle');
  speechState('idle');
  if (!ok) alertMsg('Could not play the reply: the Piper voice is unavailable and the browser has no on-device voice.');
}
function stopVoice() { stopSpeaking(); micState('idle'); speechState('idle'); }

micBtn.onclick = async () => {
  if (recording) { recording.abort(); return; }           // send what was said so far
  if (micBtn.classList.contains('busy')) return;
  if (isSpeaking()) { stopVoice(); return; }
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
    alertMsg('This browser cannot record audio here.');
    return;
  }
  alertMsg('', false);
  recording = new AbortController();
  micState('recording');
  speech.listener?.mute(true);
  try {
    const blob = await recordUtterance({ signal: recording.signal, onLevel: (v) => micBtn.style.setProperty('--level', v.toFixed(2)) });
    recording = null;
    speech.listener?.mute(false);
    if (!blob) { micState('idle'); alertMsg('Did not hear anything. Click the microphone and speak.', false); return; }
    micState('busy');
    const text = await transcribe(blob);
    micState('idle');
    if (!text) { alertMsg('Could not make out any words. Try again a little closer to the mic.', false); return; }
    promptEl.value = promptEl.value.trim() ? `${promptEl.value.trim()} ${text}` : text;
    autosize();
    submitPrompt(promptEl.value.trim(), { byVoice: true });
  } catch (err) {
    recording = null;
    speech.listener?.mute(false);
    micState('idle');
    alertMsg(micError(err));
  }
};
const micError = (err) => ({
  NotAllowedError: 'Microphone access was blocked. Allow it in the address bar and try again.',
  NotFoundError: 'No microphone found. Is your headset connected? A Bluetooth headset must be in its Headset ' +
    '(HSP/HFP) profile, not "High Fidelity Playback", for its microphone to exist.',
  NotReadableError: 'The microphone is busy or stopped responding (another app may be using it, or the device ' +
    'was disconnected). Reconnect it and try again.',
}[err.name] || err.message);

// One utterance from the hands-free listener (handled one at a time, in order).
async function onUtterance(blob) {
  const followUp = Date.now() < speech.followUntil;
  speech.busy++;
  speechState('running', 'transcribing…');
  let heard;
  try {
    heard = await listen(blob, followUp);
  } catch (err) {
    say(`Speech agent: ${err.message}`);
    return;
  } finally {
    speech.busy--;
    if (!speech.busy) speechState('idle');
  }
  if (!heard.woke) {                           // noise / filler (or, with a wake word, not addressed to CC)
    if (heard.text && speech.needWake) {         // shown briefly (only here) so a misheard wake word is visible
      say(`Heard “${heard.text.slice(0, 80)}” — no “${speech.wake}”, ignored.`);
      clearTimeout(speech.heardTimer);
      speech.heardTimer = setTimeout(() => { if (voiceLine.textContent.startsWith('Heard “')) say(''); }, 5000);
    }
    return;
  }
  speech.followUntil = 0;
  const command = heard.command.trim();
  if (!command) {                              // "Hey CC" on its own: ask, then take the next utterance as the request
    say(`“${speech.wake}” — listening…`);
    await speakReply('Yes?');
    speech.followUntil = Date.now() + speech.followUpMs;
    speechState('idle');
    setTimeout(() => { if (Date.now() >= speech.followUntil) { speechState('idle'); if (voiceLine.textContent.endsWith('listening…')) say(''); } }, speech.followUpMs + 50);
    return;
  }
  if (/^(stop|cancel|never ?mind|be quiet|shut up)\W*$/i.test(command)) { stopVoice(); say(''); return; }
  say(`🎙 “${command}”`);
  if (state.agents.has('manager')) scene.flow('speech', 'manager');
  submitPrompt(command, { byVoice: true, fromComposer: false });
}

async function setHandsFree(on, { quiet = false } = {}) {
  if (on && !speech.listener) {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
      alertMsg('This browser cannot record audio here.');
      return;
    }
    const l = new VoiceListener({
      silenceMs: speech.endMs,  // wait this long after you stop: pausing to think does not send half a request
      onUtterance: (blob) => { speech.queue = speech.queue.then(() => onUtterance(blob)); },
      onLevel: (v) => earBtn.style.setProperty('--level', v.toFixed(2)),
      onSpeech: (talking) => earBtn.classList.toggle('hearing', talking),
    });
    try {
      speech.listener = l;
      const live = await l.start();
      if (isMonitorInput(l.deviceLabel)) {
        alertMsg(`The microphone in use is “${l.deviceLabel}”: that records the speakers, not you. A Bluetooth ` +
          'headset in its music (A2DP) profile has no microphone: switch it to the Headset (HSP/HFP) profile in ' +
          'Settings → Sound, then turn Hey CC off and on.');
      }
      if (!live) {
        say(speech.needWake ? `Click anywhere on the page to start listening for “${speech.wake}”.`
          : 'Click anywhere on the page to start listening.');
        document.addEventListener('pointerdown', async () => { await l.resume(); say(''); }, { once: true });
      } else if (!quiet) {
        say(speech.needWake ? `Listening for “${speech.wake}”. Say it, then your request.`
          : 'Listening — just talk. Everything you say goes to the agents, and they answer out loud.');
      }
    } catch (err) {
      speech.listener = null;
      l.stop();
      alertMsg(micError(err));
      on = false;
    }
  } else if (!on && speech.listener) {
    speech.listener.stop();
    speech.listener = null;
    speech.followUntil = 0;
    say('');
  }
  earBtn.setAttribute('aria-pressed', String(on));
  earBtn.classList.remove('hearing');
  pref.set(on);
  speechState('idle');
}
earBtn.onclick = () => setHandsFree(!speech.listener);

api('/api/speech').then((cfg) => {
  speech.wake = cfg.wake_phrase || speech.wake;
  speech.followUpMs = (cfg.follow_up_seconds || 8) * 1000;
  speech.needWake = !!cfg.require_wake_word;
  if (cfg.end_of_speech_seconds) {
    speech.endMs = cfg.end_of_speech_seconds * 1000;
    if (speech.listener) speech.listener.silenceMs = speech.endMs;  // already listening (setting remembered)
  }
  earBtn.querySelector('span').textContent = speech.needWake ? speech.wake : 'Talk';
  earBtn.title = speech.needWake
    ? `Speech agent: always listening; acts only when you say “${speech.wake}” (then your request). ` +
      'Speech is transcribed on this computer and anything else is ignored.'
    : 'Conversation mode: while on, everything you say is sent to the agents and they answer out loud. ' +
      'Turn it off when you are talking to someone else. Speech is transcribed on this computer.';
  if (!cfg.installed) earBtn.disabled = micBtn.disabled = true;
  speechState('idle');
}).catch(() => { /* older server: keep the defaults */ });
if (pref.get()) setHandsFree(true, { quiet: true });

// ------------------------------------------------------------------ help: how to use agents and models
// Example prompts for the built-in agents; custom agents fall back to their description.
const AGENT_EXAMPLES = {
  text: { prompt: 'Summarize: Running models locally keeps data private, avoids per-token costs and works offline.' },
  reasoning: { prompt: "Explain step by step why two 7B models don't fit on one 8 GB GPU." },
  validation: { prompt: 'Check this claim and say if it is accurate: the RTX 3090 has 24 GB of VRAM.' },
  vision: { prompt: 'Describe this image and list the objects you can see.', image: true },
  multimodal: { prompt: 'What is wrong with this screenshot?', image: true },
};
const help = { tab: 'agents', routing: null };
let forcedAgent = null;

function setForcedAgent(name) {
  forcedAgent = name;
  $('forcedChip').hidden = !name;
  $('forcedName').textContent = name || '';
}
$('forcedClear').onclick = () => setForcedAgent(null);

function useInChat(prompt, agent) {
  closeHelp();
  promptEl.value = prompt;
  autosize();
  setForcedAgent(agent);
  promptEl.focus();
}

function codeBlock(text) {
  const pre = el('pre', 'code');
  pre.textContent = text;
  return pre;
}

function renderHelpAgents(agents, routing) {
  const pane = $('helpAgents');
  pane.replaceChildren();
  pane.append(el('p', 'lead',
    'You don\'t have to pick agents. The manager reads each message and chooses them for you. ' +
    'To run one specific agent, click "Use only this agent" on its card: your next chat messages skip routing and go straight to that agent until you remove it with ✕ above the message box.'));
  const cards = el('div', 'cards');
  const order = (a) => (a.role === 'control' ? 0 : 1);
  for (const a of [...agents].sort((x, y) => order(x) - order(y) || x.name.localeCompare(y.name))) {
    const card = el('div', 'card');
    const h = el('div', 'card-h');
    h.append(el('b', null, a.name), el('span', `tag ${a.role}`, a.role === 'control' ? 'always runs' : 'specialist'));
    card.append(h, el('p', null, a.description));
    const model = el('div', 'kv');
    model.append('Model: ');
    model.append(el('b', null, a.model_key || 'none (rule-based)'));
    card.append(model);
    const caps = el('div', 'caps');
    for (const c of a.capabilities) caps.append(el('span', 'cap', c));
    card.append(caps);

    if (a.role === 'control') {
      card.append(el('div', 'kv', a.name === 'manager'
        ? 'Runs first on every message: finds the intent and picks the specialists.'
        : 'Runs second on every message: turns the manager\'s choice into steps and runs independent steps in parallel.'));
    } else {
      const intents = (routing || []).filter((r) => r.agents.includes(a.name));
      if (intents.length) {
        const trig = el('div', 'kv');
        trig.append('Picked automatically for: ');
        const caps2 = el('span', 'caps');
        caps2.style.display = 'inline-flex';
        for (const r of intents) {
          const phr = r.phrases.slice(0, 3).map((p) => `"${p}"`).join(', ');
          caps2.append(el('span', 'phr', `${r.intent}${r.input === 'image' ? ' (image)' : ''}${phr ? ': ' + phr : ' (fallback)'}`));
        }
        trig.append(caps2);
        card.append(trig);
      }
      const ex = AGENT_EXAMPLES[a.name] || { prompt: `Ask the ${a.name} agent: ${a.description}` };
      card.append(el('div', 'example', `Example: ${ex.prompt}${ex.image ? '  (attach an image)' : ''}`));
      const acts = el('div', 'acts');
      const tryBtn = el('button', 'ghost', 'Try example');
      tryBtn.onclick = () => useInChat(ex.prompt, null);
      const onlyBtn = el('button', 'ghost', 'Use only this agent');
      onlyBtn.onclick = () => useInChat(ex.prompt, a.name);
      acts.append(tryBtn, onlyBtn);
      card.append(acts);
    }
    cards.append(card);
  }
  pane.append(cards);

  const how = el('div', 'howto');
  how.append(el('h4', null, 'From the API'));
  how.append(el('p', null, 'Leave out "options" for automatic routing, or list the worker agents to run:'));
  how.append(codeBlock(`curl -X POST http://${location.host}/api/tasks \\
  -H 'Content-Type: application/json' \\
  -d '{"prompt": "Explain recursion", "options": {"agents": ["reasoning"]}}'

# image + a specific agent
curl -X POST http://${location.host}/api/image \\
  -F file=@photo.png -F prompt="Describe this image" -F agents=vision`));
  pane.append(how);
}

function renderHelpModels(models, agents) {
  const pane = $('helpModels');
  pane.replaceChildren();
  pane.append(el('p', 'lead',
    'Models run locally from the models/ folder and load on first use. ' +
    'Each agent uses the model set in config.yaml. If a model is missing, its fallback (base ↔ fast_text, vision ↔ multimodal) is used instead.'));
  const cards = el('div', 'cards');
  for (const m of models) {
    const users = agents.filter((a) => a.model_key === m.name).map((a) => a.name);
    if (m.name === 'base') users.push('aggregator');
    const card = el('div', 'card');
    const h = el('div', 'card-h');
    h.append(el('b', null, m.name), el('span', 'cap', m.type));
    const st = el('span', `state ${m.installed ? m.state : 'FAILED'} tag`, m.installed ? m.state : 'NOT DOWNLOADED');
    h.append(st);
    card.append(h);
    if (m.hf_id) {
      const k = el('div', 'kv');
      k.append('Hugging Face: ', el('b', null, m.hf_id));
      card.append(k);
    }
    const mem = el('div', 'kv');
    mem.append('Size: ', el('b', null,
      m.memory_gb != null ? `${m.memory_gb.toFixed(2)} GB in ${m.device}` : m.disk_size_gb != null ? `${m.disk_size_gb.toFixed(1)} GB on disk` : 'unknown'));
    card.append(mem);
    const used = el('div', 'kv');
    used.append('Used by: ');
    const caps = el('span', 'caps');
    caps.style.display = 'inline-flex';
    if (users.length) for (const u of users) caps.append(el('span', 'phr', u));
    else caps.append(el('span', 'muted', 'no agent (available as a fallback)'));
    used.append(caps);
    card.append(used);
    const acts = el('div', 'acts');
    if (!m.installed) {
      card.append(codeBlock(`python scripts/download_models.py --model ${m.hf_id || '<hf-id>'} --type ${m.type}`));
    } else {
      const loaded = ['LOADED', 'ACTIVE', 'IDLE'].includes(m.state);
      const b = el('button', 'ghost', loaded ? 'Unload' : 'Load now');
      b.onclick = async () => {
        b.disabled = true;
        b.textContent = loaded ? 'Unloading…' : 'Loading…';
        try { await api(`/api/models/${encodeURIComponent(m.name)}/${loaded ? 'unload' : 'load'}`, { method: 'POST' }); } catch { /* shown in state */ }
        loadModels();
        openHelp('models');
      };
      acts.append(b);
      card.append(acts);
    }
    cards.append(card);
  }
  pane.append(cards);

  const how = el('div', 'howto');
  how.append(el('h4', null, 'Add a model or switch an agent to another model'));
  how.append(el('p', null, '1. Download it into models/ (run in the project folder):'));
  how.append(codeBlock('python scripts/download_models.py --model Qwen/Qwen2.5-3B-Instruct --type text --name qwen3b'));
  how.append(el('p', null, '2. Point an agent at it in config.yaml, then restart the server:'));
  how.append(codeBlock('agents:\n  reasoning: { model: qwen3b }'));
  how.append(el('p', null, 'Bigger models give better answers but need more VRAM. Text models are loaded with AutoModelForCausalLM; vision/multimodal models must be image-text-to-text (SmolVLM, Idefics3, Qwen2-VL, LLaVA).'));
  pane.append(how);
}

function renderHelpRouting(routing) {
  const pane = $('helpRouting');
  pane.replaceChildren();
  const flow = el('div', 'flow');
  ['your message', 'manager', 'planner', 'specialists (in parallel)', 'validation', 'aggregator', 'answer'].forEach((s, i) => {
    if (i) flow.append(el('i', null, '→'));
    flow.append(el('span', null, s));
  });
  pane.append(flow);
  pane.append(el('p', 'lead',
    'The manager checks your message against the phrases below, top to bottom; the first match sets the intent, and the intent decides which agents run. ' +
    'Messages with an image use the image rules. If nothing matches, the intent is "generate".'));
  const table = el('table', 'rt');
  const head = el('tr');
  for (const t of ['Intent', 'Input', 'Trigger phrases', 'Agents']) head.append(el('th', null, t));
  table.append(el('thead'));
  table.tHead.append(head);
  const body = el('tbody');
  for (const r of routing) {
    const tr = el('tr');
    tr.append(el('td', null, r.intent), el('td', 'muted', r.input));
    const ph = el('td');
    const pc = el('div', 'caps');
    if (r.phrases.length) for (const p of r.phrases) pc.append(el('span', 'phr', p));
    else pc.append(el('span', 'muted small', 'anything else'));
    ph.append(pc);
    const ag = el('td');
    const ac = el('div', 'caps');
    for (const a of r.agents) ac.append(el('span', 'cap', a));
    ag.append(ac);
    tr.append(ph, ag);
    body.append(tr);
  }
  table.append(body);
  pane.append(table);
}

function showTab(tab) {
  help.tab = tab;
  for (const b of document.querySelectorAll('#help .tab')) b.classList.toggle('active', b.dataset.tab === tab);
  for (const p of document.querySelectorAll('#help .pane')) p.hidden = p.dataset.pane !== tab;
}

async function openHelp(tab) {
  showTab(tab || help.tab);
  $('help').hidden = false;
  $('helpScrim').hidden = false;
  document.body.classList.add('drawer-open');
  try {
    const [agents, models, routing] = await Promise.all([api('/api/agents'), api('/api/models'), api('/api/routing')]);
    help.routing = routing;
    renderHelpAgents(agents, routing);
    renderHelpModels(models, agents);
    renderHelpRouting(routing);
  } catch (err) {
    $('helpAgents').replaceChildren(el('p', 'err-text', `Could not load help: ${err.message}`));
  }
}
function closeHelp() {
  $('help').hidden = true;
  $('helpScrim').hidden = true;
  if (!state.selected) document.body.classList.remove('drawer-open');
}
$('helpBtn').onclick = () => openHelp();

// ---------- light / dark theme ----------
// No saved choice: follow the OS. Clicking pins the opposite of what is shown and remembers it.
const darkQuery = window.matchMedia('(prefers-color-scheme: dark)');
function syncTheme() {
  const pinned = document.documentElement.dataset.theme;
  const dark = pinned ? pinned === 'dark' : darkQuery.matches;
  document.documentElement.classList.toggle('is-dark', dark);
  $('themeBtn').title = dark ? 'Switch to light theme' : 'Switch to dark theme';
  window.dispatchEvent(new Event('themechange'));
}
$('themeBtn').onclick = () => {
  const next = document.documentElement.classList.contains('is-dark') ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('theme', next); } catch {}
  syncTheme();
};
darkQuery.addEventListener('change', syncTheme);
syncTheme();
$('helpClose').onclick = closeHelp;

// ------------------------------------------------------------------ desktop screen view
// Show/hide only changes what you see: the Control Panel agent captures the screen itself, so its tasks keep
// executing while this is hidden, and nothing is fetched for the view then.
const screenView = { on: false, timer: null };
function setScreen(on) {
  screenView.on = on;
  $('screenView').hidden = !on;
  document.querySelector('main.grid').classList.toggle('with-screen', on);
  $('screenBtn').setAttribute('aria-pressed', String(on));
  $('screenBtn').querySelector('span').textContent = on ? 'Hide screen' : 'Show screen';
  try { localStorage.setItem('screenView', on ? 'on' : 'off'); } catch {}
  clearTimeout(screenView.timer);
  if (on) nextFrame();
  else $('screenImg').removeAttribute('src');
}
function nextFrame() {
  if (!screenView.on) return;
  const img = $('screenImg');
  const done = (ok) => {
    $('screenStatus').textContent = ok ? `live · ${new Date().toLocaleTimeString()}` : 'AI DYOT is off: it starts with your next desktop task and closes when it is done';
    if (screenView.on) screenView.timer = setTimeout(nextFrame, ok ? 700 : 3000);
  };
  const frame = new Image(); // load off-screen first so a failed frame never blanks the last good one
  frame.onload = () => { img.src = frame.src; done(true); };
  frame.onerror = () => done(false);
  frame.src = `/api/control-panel/screen.jpg?t=${Date.now()}`;
}
$('screenBtn').onclick = () => setScreen(!screenView.on);
try { if (localStorage.getItem('screenView') === 'on') setScreen(true); } catch {}
$('helpScrim').onclick = closeHelp;
for (const b of document.querySelectorAll('#help .tab')) b.onclick = () => showTab(b.dataset.tab);

// ------------------------------------------------------------------ boot
// Panels size themselves to the viewport below the (wrapping) top bar.
new ResizeObserver(() => document.documentElement.style.setProperty('--hdr', `${$('topbar').offsetHeight}px`)).observe($('topbar'));
loadAgents();
loadHealth();
loadModels();
loadMetrics();
loadStatus();
loadDownloads();
loadTasks(true);
connectEvents();
setInterval(loadHealth, 1000);
setInterval(() => { loadModels(); loadMetrics(); loadStatus(); loadDownloads(); }, 3000);
setInterval(() => loadTasks(), 5000);
