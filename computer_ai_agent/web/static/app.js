// Computer Agent control panel: live screen + action overlay, per-agent conversation, human-in-the-loop controls.
'use strict';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
};

const state = {
  ws: null,
  tasks: new Map(),          // task_id -> {turn, stream, history, actions, status, command}
  pendingLines: new Map(),   // `${task}:${step}` -> line element (action started, waiting for completion)
  agents: new Map(),         // agent name -> line count
  tab: '*',
  current: null,             // running task id
  screen: { w: 0, h: 0 },
  elements: [],
  target: null,              // {point, bbox, label, ts, done}
};

// ---------- theme ----------
const darkQuery = window.matchMedia('(prefers-color-scheme: dark)');
function syncTheme() {
  const pinned = document.documentElement.dataset.theme;
  const dark = pinned ? pinned === 'dark' : darkQuery.matches;
  document.documentElement.classList.toggle('is-dark', dark);
  $('themeBtn').title = dark ? 'Switch to light theme' : 'Switch to dark theme';
}
$('themeBtn').onclick = () => {
  const next = document.documentElement.classList.contains('is-dark') ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('theme', next); } catch {}
  syncTheme();
};
darkQuery.addEventListener('change', syncTheme);
syncTheme();

// ---------- status bar ----------
async function refreshStatus() {
  try {
    const s = await (await fetch('/api/status')).json();
    const m = s.models.planner || {};
    $('modelName').textContent = `${m.model || '?'}${m.loaded ? '' : ' (not loaded)'}`;
    $('modelChip').classList.toggle('ok', !!s.models.ok);
    $('modelChip').title = m.error || `backend ${m.backend}, ${m.vram_gb || 0} GB VRAM, avg ${s.model_stats.avg_ms} ms/call`;
    $('a11yChip').textContent = s.desktop.accessibility ? 'a11y on' : 'a11y off (OCR)';
    $('subDesktop').textContent = `${s.desktop.backend.toUpperCase()} ${s.desktop.display} · ${s.desktop.screen.join('×')}` +
      ` · ${s.plugins.length} plugins · ${s.settings.require_confirmation ? 'confirmations on' : 'confirmations OFF'}`;
    for (const a of s.agents) ensureTab(a.name);
    if (!state.current && s.active_window) $('stApp').textContent = s.active_window.title || '–';
  } catch { /* server restarting */ }
}

function setState(name, detail) {
  const st = $('stState');
  st.textContent = name;
  st.className = 'state ' + ({ COMPLETED: 'done', FAILED: 'bad', WAITING_FOR_USER: 'wait', IDLE: '' }[name] ?? 'run');
  if (detail) st.title = detail;
}

function setControls() {
  const t = state.current && state.tasks.get(state.current);
  const running = !!t && ['running', 'paused'].includes(t.status);
  $('pauseBtn').disabled = !running;
  $('stopBtn').disabled = !running;
  $('pauseBtn').hidden = running && t.status === 'paused';
  $('resumeBtn').hidden = !(running && t.status === 'paused');
  $('sendBtn').disabled = false;
}

// ---------- tabs ----------
function ensureTab(name) {
  if (!name || name === 'MasterAgent' || state.agents.has(name)) return;
  state.agents.set(name, 0);
  const b = el('button', 'tab', name.replace(/SubAgent$/, ' Agent'));
  b.type = 'button';
  b.dataset.agent = name;
  b.onclick = () => selectTab(name);
  $('tabs').append(b);
}
function selectTab(name) {
  state.tab = name;
  for (const b of $('tabs').children) b.classList.toggle('active', b.dataset.agent === name);
  applyFilter();
}
$('tabs').querySelector('[data-agent="*"]').onclick = () => selectTab('*');
function applyFilter() {
  for (const line of document.querySelectorAll('.line[data-agent], .ask[data-agent]')) {
    line.hidden = state.tab !== '*' && line.dataset.agent !== state.tab;
  }
}

// ---------- conversation ----------
function taskView(taskId, command) {
  let t = state.tasks.get(taskId);
  if (t) return t;
  $('empty').hidden = true;
  const turn = el('div', 'turn');
  const user = el('div', 'user-msg', command || '');
  const stream = el('div', 'agent-stream');
  turn.append(user, stream);
  $('messages').append(turn);
  t = { turn, user, stream, actions: [], status: 'running', command };
  state.tasks.set(taskId, t);
  return t;
}
function nearBottom() { const m = $('messages'); return m.scrollHeight - m.scrollTop - m.clientHeight < 160; }
function scroll(force) { const m = $('messages'); if (force || nearBottom()) m.scrollTop = m.scrollHeight; }

function addLine(t, agent, cls, html, opts = {}) {
  const stick = nearBottom();
  const line = el('div', `line ${cls || ''}`);
  line.dataset.agent = agent || 'MasterAgent';
  const who = el('span', 'who', (agent || 'Master').replace(/SubAgent$/, '').replace(/Agent$/, ''));
  const what = el('span', 'what');
  if (opts.text) what.textContent = html; else what.innerHTML = html;
  line.append(who, what);
  t.stream.append(line);
  if (state.tab !== '*' && line.dataset.agent !== state.tab) line.hidden = true;
  if (stick) scroll(true);
  return line;
}
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmtLabel = (s) => esc(s).replace(/\n/g, '⏎');

function renderAsk(t, req, resolved) {
  const box = el('div', `ask ${req.kind === 'confirm' ? '' : 'info'}`);
  box.dataset.agent = req.agent || 'MasterAgent';
  box.id = `ask_${req.id}`;
  box.append(el('div', 'q', req.message));
  if (req.screenshot) {
    const img = el('img');
    img.src = `data:image/jpeg;base64,${req.screenshot}`;
    img.alt = 'screen at the time of the question';
    box.append(img);
  }
  const opts = el('div', 'opts');
  const send = (answer) => {
    state.ws?.send(JSON.stringify({ type: 'respond', id: req.id, answer }));
    box.classList.add('resolved');
  };
  if (req.kind === 'confirm') {
    const ok = el('button', 'btn primary', 'Approve'); ok.type = 'button'; ok.onclick = () => send('approve');
    const no = el('button', 'btn danger', 'Deny'); no.type = 'button'; no.onclick = () => send('deny');
    opts.append(ok, no);
  } else {
    for (const c of req.choices || []) {
      const b = el('button', 'btn', c); b.type = 'button'; b.onclick = () => send(c); opts.append(b);
    }
    const input = el('input'); input.placeholder = req.choices?.length ? 'or type an answer…' : 'Type your answer…';
    input.onkeydown = (e) => { if (e.key === 'Enter' && input.value.trim()) send(input.value.trim()); };
    opts.append(input);
  }
  box.append(opts);
  if (resolved) box.classList.add('resolved');
  t.stream.append(box);
  scroll(true);
  setState('WAITING_FOR_USER');
}

function finishTask(t, e, ok) {
  t.status = e.event === 'task_stopped' ? 'stopped' : ok ? 'completed' : 'failed';
  const res = el('div', `result ${ok ? 'ok' : 'bad'}`, e.result || (ok ? 'Done' : 'Failed'));
  t.stream.append(res);
  t.stream.append(el('div', 'meta', `${e.duration}s · ${e.actions} action${e.actions === 1 ? '' : 's'}`));
  if (t.actions.length) {
    const d = el('details', 'history');
    d.append(el('summary', null, `Action history (${t.actions.length})`));
    const ol = el('ol');
    for (const a of t.actions) {
      const li = el('li');
      li.innerHTML = `${a.ok ? '✓' : '✗'} <b>${fmtLabel(a.label)}</b> <span class="meta">${esc(a.agent.replace(/SubAgent$/, ''))}</span>` +
        `<span class="ms">${a.ms} ms</span>${a.message ? ` — <span class="meta">${esc(a.message)}</span>` : ''}`;
      ol.append(li);
    }
    d.append(ol);
    t.stream.append(d);
  }
  if (state.current === e.task_id) state.current = null;
  setState(ok ? 'COMPLETED' : 'FAILED');
  $('stAction').textContent = '–';
  $('stNext').textContent = '–';
  setControls();
  scroll(true);
}

// ---------- event handling ----------
function handle(e, replay = false) {
  const tid = e.task_id;
  const t = tid ? (state.tasks.get(tid) || (e.event === 'task_started' ? taskView(tid, e.command) : null)) : null;
  switch (e.event) {
    case 'task_accepted':
      taskView(e.task_id, e.command);
      break;
    case 'task_started':
      state.current = tid;
      t.user.textContent = e.command;
      t.status = 'running';
      setControls();
      break;
    case 'plan_created': {
      if (!t) break;
      const ol = el('ol', 'plan');
      for (const s of e.subtasks) ol.append(el('li', null, `[${s.app}] ${s.instruction || 'open it'}`));
      const line = addLine(t, 'MasterAgent', 'dim', `Plan (${esc(e.source)}):`);
      line.querySelector('.what').append(ol);
      break;
    }
    case 'application_detected':
      if (t) addLine(t, 'MasterAgent', 'dim', `${esc(e.application)} is ${e.running ? 'running' : 'not running'}`);
      $('stApp').textContent = e.application;
      break;
    case 'application_opened':
      if (t) addLine(t, 'MasterAgent', '', `${e.launched ? 'Opened' : 'Switched to'} <b>${esc(e.application)}</b>`);
      break;
    case 'agent_created':
      ensureTab(e.agent);
      if (t) addLine(t, 'MasterAgent', 'dim', `Created ${esc(e.agent)}${e.plugin ? ` (plugin: ${esc(e.plugin)})` : ' (generic)'} · sees ` +
        Object.entries(e.ui_summary.element_counts || {}).map(([k, v]) => `${v} ${k}`).join(', ') +
        ` via ${(e.ui_summary.sources || []).join('+') || 'pixels'}`);
      break;
    case 'agent_message': {
      const m = e.message;
      ensureTab(m.to); ensureTab(m.from);
      $('stAgent').textContent = m.to === 'MasterAgent' ? m.from : m.to;
      if (t && m.task) addLine(t, m.from, 'dim', `→ ${m.to}: ${m.task}`, { text: true });
      break;
    }
    case 'state_changed':
      if (!replay && (e.agent !== 'MasterAgent' || ['COMPLETED', 'FAILED', 'UNDERSTANDING'].includes(e.state))) setState(e.state, e.detail);
      if (e.agent && e.agent !== 'MasterAgent') $('stAgent').textContent = e.agent;
      break;
    case 'action_started':
      if (!t) break;
      $('stAction').textContent = e.label.replace(/\n/g, '⏎');
      $('stNext').textContent = e.expect || '–';
      state.pendingLines.set(`${tid}:${e.step}`, addLine(t, e.agent, 'act',
        `▶ <b>${fmtLabel(e.label)}</b>${e.reason ? ` <span class="meta">— ${esc(e.reason)}</span>` : ''}`));
      if (!replay) state.target = { point: e.point, bbox: e.bbox, label: e.label, ts: performance.now(), done: false };
      break;
    case 'action_completed': {
      if (!t) break;
      t.actions.push({ label: e.label || e.action?.action, ok: e.ok, ms: e.duration_ms ?? 0, message: e.message, agent: e.agent || '' });
      const key = `${tid}:${e.step}`;
      const line = state.pendingLines.get(key);
      if (line) {
        line.classList.add(e.ok ? 'ok' : 'bad');
        line.querySelector('.what').insertAdjacentHTML('afterbegin', `<span class="mark">${e.ok ? '✓' : '✗'}</span> `);
        line.querySelector('.what').insertAdjacentHTML('beforeend', `<span class="ms">${e.duration_ms ?? ''} ms</span>` +
          (e.ok ? '' : `<br><span class="meta">${esc(e.message)}</span>`));
        state.pendingLines.delete(key);
      } else {
        addLine(t, e.agent, e.ok ? 'ok' : 'bad', `${e.ok ? '✓' : '✗'} ${esc(e.message)}`);
      }
      if (state.target) state.target.done = true;
      break;
    }
    case 'recovery':
      if (t) addLine(t, e.agent, 'warn', `<span class="mark">↻</span> ${esc(e.reason)}`);
      break;
    case 'verification_completed':
      if (t && e.what === 'goal') addLine(t, e.agent, e.ok ? 'ok' : 'warn', `<span class="mark">${e.ok ? '✓' : '…'}</span> goal check: ${esc(e.detail)}`);
      break;
    case 'confirmation_required':
    case 'user_input_required':
      if (t) renderAsk(t, e.request, replay);
      break;
    case 'user_response': {
      const box = document.getElementById(`ask_${e.request_id}`);
      if (box) box.classList.add('resolved');
      break;
    }
    case 'task_paused':
      if (t) { t.status = 'paused'; addLine(t, 'MasterAgent', 'dim', 'Paused'); setState('PAUSED'); setControls(); }
      break;
    case 'task_resumed':
      if (t) { t.status = 'running'; addLine(t, 'MasterAgent', 'dim', 'Resumed'); setControls(); }
      break;
    case 'task_completed': if (t) finishTask(t, e, true); break;
    case 'task_failed':
    case 'task_stopped': if (t) finishTask(t, e, false); break;
    case 'observation':
      state.elements = e.elements || [];
      break;
    case 'screen_updated':
      showFrame(e);
      break;
    default: break;
  }
}

// ---------- live screen + overlay ----------
function showFrame(e) {
  const img = $('screenImg');
  img.src = `data:image/jpeg;base64,${e.image}`;
  state.screen = { w: e.width, h: e.height };
  $('screenEmpty').hidden = true;
  if (!state.current && e.active_window) $('stApp').textContent = e.active_window.title || '–';
}
$('screenImg').onload = drawOverlay;
window.addEventListener('resize', drawOverlay);
for (const id of ['showEls', 'showOcr']) $(id).onchange = drawOverlay;

function drawOverlay() {
  const img = $('screenImg'), cv = $('overlay'), wrap = $('screenWrap');
  if (!img.naturalWidth || !state.screen.w) return;
  const r = img.getBoundingClientRect(), wr = wrap.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  cv.style.left = `${r.left - wr.left}px`; cv.style.top = `${r.top - wr.top}px`;
  cv.style.width = `${r.width}px`; cv.style.height = `${r.height}px`;
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  const s = r.width / state.screen.w;
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  if ($('showEls').checked || $('showOcr').checked) {
    ctx.lineWidth = 1;
    ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
    for (const e of state.elements) {
      const [x, y, w, h] = e.bbox;
      const isOcr = e.source === 'ocr';
      if (isOcr ? !$('showOcr').checked : !$('showEls').checked) continue;
      ctx.strokeStyle = isOcr ? css('--overlay-ocr') : css('--overlay-el');
      ctx.strokeRect(x * s, y * s, w * s, h * s);
      if (isOcr && $('showOcr').checked && e.text) {
        ctx.fillStyle = css('--overlay-ocr');
        ctx.fillText(e.text.slice(0, 40), x * s, Math.max(9, y * s - 2));
      }
    }
  }
  const t = state.target;
  if (t) {
    const age = performance.now() - t.ts;
    const alpha = t.done ? Math.max(0, 1 - (age - 400) / 2600) : 1;
    if (alpha > 0) {
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 3;
      if (t.bbox) ctx.strokeRect(t.bbox[0] * s - 2, t.bbox[1] * s - 2, t.bbox[2] * s + 4, t.bbox[3] * s + 4);
      if (t.point) {
        const [px, py] = [t.point[0] * s, t.point[1] * s];
        ctx.beginPath(); ctx.arc(px, py, 11, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(px - 18, py); ctx.lineTo(px + 18, py); ctx.moveTo(px, py - 18); ctx.lineTo(px, py + 18);
        ctx.lineWidth = 1.5; ctx.stroke();
        const label = t.label.replace(/\n/g, '⏎').slice(0, 48);
        ctx.font = '600 12px ui-sans-serif, system-ui, sans-serif';
        const tw = ctx.measureText(label).width;
        const lx = Math.min(Math.max(4, px + 14), r.width - tw - 14), ly = Math.max(20, py - 16);
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath(); ctx.roundRect(lx - 6, ly - 15, tw + 12, 21, 10); ctx.fill();
        ctx.fillStyle = '#1f2328'; ctx.fillText(label, lx, ly);
      }
      ctx.globalAlpha = 1;
      if (t.done && alpha > 0) requestAnimationFrame(drawOverlay);
    }
  }
}

// ---------- socket ----------
function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/agent`);
  state.ws = ws;
  ws.onopen = () => { $('connChip').classList.add('ok'); $('connChip').lastElementChild.textContent = 'live'; refreshStatus(); };
  ws.onclose = () => {
    $('connChip').classList.remove('ok'); $('connChip').lastElementChild.textContent = 'offline';
    setTimeout(connect, 1500);
  };
  ws.onmessage = (m) => {
    const e = JSON.parse(m.data);
    if (e.event === 'hello') {
      if (!state.tasks.size) {
        for (const h of e.history) handle(h, true);
        const pendingIds = new Set(e.pending.map((p) => p.id));
        for (const box of document.querySelectorAll('.ask')) {
          if (pendingIds.has(box.id.slice(4))) box.classList.remove('resolved');
        }
        scroll(true);
      }
      return;
    }
    handle(e);
  };
}

// ---------- composer + controls ----------
const cmd = $('cmd');
function autosize() { cmd.style.height = 'auto'; cmd.style.height = `${Math.min(cmd.scrollHeight, 180)}px`; }
cmd.addEventListener('input', autosize);
cmd.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('composer').requestSubmit(); }
});
$('composer').onsubmit = (e) => {
  e.preventDefault();
  const text = cmd.value.trim();
  if (!text || !state.ws || state.ws.readyState !== 1) return;
  state.ws.send(JSON.stringify({ type: 'command', command: text }));
  cmd.value = ''; autosize();
  scroll(true);
};
for (const p of document.querySelectorAll('.suggest .pill')) p.onclick = () => { cmd.value = p.textContent; autosize(); cmd.focus(); };
const control = (type) => () => state.current && state.ws?.send(JSON.stringify({ type, task_id: state.current }));
$('pauseBtn').onclick = control('pause');
$('resumeBtn').onclick = control('resume');
$('stopBtn').onclick = control('stop');

connect();
setInterval(refreshStatus, 10000);
