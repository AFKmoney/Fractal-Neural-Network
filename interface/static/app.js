/* ═══════════════════════════════════════════════════════════════
   NFN Interface — app.js
   Neural Fractal Network chat / code / agent / training UI
   ═══════════════════════════════════════════════════════════════ */

'use strict';

// ─── State ──────────────────────────────────────────────────────────────────
const state = {
  messages: [],          // chat history [{role,content}]
  ws: null,              // streaming WebSocket
  trainWs: null,         // training metrics WebSocket
  lossHistory: [],       // [{step,loss}]
  isStreaming: false,
  isTraining: false,
  modelInfo: null,
  currentMotif: 'binary_tree',
  phaseAnim: null,       // requestAnimationFrame id
};

// ─── API helpers ─────────────────────────────────────────────────────────────
async function api(path, method = 'GET', body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  return res.json();
}

function getParams() {
  return {
    temperature: parseFloat($('temperature').value),
    top_p: parseFloat($('top_p').value),
    top_k: 50,
    max_tokens: parseInt($('max_tokens').value),
    strategy: $('strategy').value,
  };
}

function $(id) { return document.getElementById(id); }

// ─── Tab switching ────────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const panel = $('tab-' + btn.dataset.tab);
      if (panel) panel.classList.add('active');
      if (btn.dataset.tab === 'info') refreshInfo();
    });
  });
}

// ─── Sidebar sliders ──────────────────────────────────────────────────────────
function initSliders() {
  [['temperature', 'temp-val'], ['top_p', 'topp-val'], ['max_tokens', 'maxtok-val']].forEach(([id, disp]) => {
    const el = $(id);
    el.addEventListener('input', () => { $(disp).textContent = el.value; });
  });
}

// ─── Status ───────────────────────────────────────────────────────────────────
async function refreshStatus() {
  try {
    const data = await api('/api/status');
    const dot = $('status-dot');
    const txt = $('status-text');
    if (data.status === 'ready') {
      dot.className = 'ready';
      state.modelInfo = data.model;
      txt.textContent = `${data.model.params} · ${data.model.device}`;
    } else {
      dot.className = 'error';
      txt.textContent = 'Modèle non chargé';
    }
  } catch {
    $('status-dot').className = 'error';
    $('status-text').textContent = 'Serveur injoignable';
  }
}

// ─── Chat tab ─────────────────────────────────────────────────────────────────
function initChat() {
  const input = $('chat-input');
  const send = $('chat-send');
  const clear = $('chat-clear');
  const msgs = $('chat-messages');

  // Auto-resize textarea
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  });

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
  });
  send.addEventListener('click', sendChat);
  clear.addEventListener('click', () => {
    state.messages = [];
    msgs.innerHTML = '';
  });

  function appendMsg(role, content, streaming = false) {
    const div = document.createElement('div');
    div.className = `chat-msg ${role}`;
    div.innerHTML = `
      <div class="msg-role">${role === 'user' ? 'Vous' : 'NFN'}</div>
      <div class="msg-bubble${streaming ? ' streaming' : ''}">${escapeHtml(content)}</div>
    `;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return div.querySelector('.msg-bubble');
  }

  function sendChat() {
    const text = input.value.trim();
    if (!text || state.isStreaming) return;
    input.value = '';
    input.style.height = 'auto';

    state.messages.push({ role: 'user', content: text });
    appendMsg('user', text);

    const bubble = appendMsg('assistant', '', true);
    state.isStreaming = true;
    send.disabled = true;

    let accum = '';
    const ws = openStreamWS({
      mode: 'chat',
      messages: [...state.messages],
      ...getParams(),
    });

    ws.onmessage = e => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'token') {
        accum += msg.text;
        bubble.textContent = accum;
        bubble.classList.add('streaming');
        msgs.scrollTop = msgs.scrollHeight;
      } else if (msg.type === 'end') {
        bubble.classList.remove('streaming');
        state.messages.push({ role: 'assistant', content: accum });
        state.isStreaming = false;
        send.disabled = false;
      } else if (msg.type === 'error') {
        bubble.textContent = '⚠ ' + msg.text;
        bubble.classList.remove('streaming');
        state.isStreaming = false;
        send.disabled = false;
      }
    };
    ws.onerror = () => {
      bubble.textContent = '⚠ Erreur de connexion';
      bubble.classList.remove('streaming');
      state.isStreaming = false;
      send.disabled = false;
    };
  }
}

// ─── WebSocket streaming ──────────────────────────────────────────────────────
function openStreamWS(payload) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
  ws.onopen = () => ws.send(JSON.stringify(payload));
  return ws;
}

// ─── Code tab ─────────────────────────────────────────────────────────────────
function initCode() {
  async function runCode(task) {
    const code = $('code-input').value;
    const instruction = $('code-instruction').value;
    const language = $('code-language').value;
    const output = $('code-output');
    const params = getParams();

    output.textContent = '⏳ NFN génère…';
    try {
      const res = await api('/api/code', 'POST', {
        code,
        instruction,
        language,
        max_tokens: params.max_tokens,
        temperature: task === 'generate' ? params.temperature : 0.4,
      });
      output.textContent = res.result || res.explanation || res.refactored || JSON.stringify(res);
    } catch (e) {
      output.textContent = '⚠ Erreur: ' + e.message;
    }
  }

  $('code-complete').addEventListener('click', () => runCode('complete'));
  $('code-explain').addEventListener('click', () => runCode('explain'));
  $('code-refactor').addEventListener('click', () => runCode('refactor'));
  $('code-generate').addEventListener('click', () => runCode('generate'));
}

// ─── Agent tab ────────────────────────────────────────────────────────────────
function initAgent() {
  const trace = $('agent-trace');
  const goalInput = $('agent-goal');
  const runBtn = $('agent-run');
  const clearBtn = $('agent-clear');

  clearBtn.addEventListener('click', () => { trace.innerHTML = ''; });

  goalInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') runAgent();
  });
  runBtn.addEventListener('click', runAgent);

  async function runAgent() {
    const goal = goalInput.value.trim();
    if (!goal) return;
    const maxSteps = parseInt($('agent-max-steps').value) || 5;

    trace.innerHTML = '';
    runBtn.disabled = true;
    runBtn.textContent = '⏳ Raisonnement…';

    // Show thinking indicator
    const thinkDiv = document.createElement('div');
    thinkDiv.className = 'agent-step';
    thinkDiv.innerHTML = `<div class="agent-step-num">Objectif</div>
      <div class="agent-thought">${escapeHtml(goal)}</div>`;
    trace.appendChild(thinkDiv);

    try {
      const res = await api('/api/agent/run', 'POST', { goal, history: [], max_steps: maxSteps });

      for (const step of res.steps) {
        const div = document.createElement('div');
        div.className = `agent-step${step.final ? ' agent-final' : ''}`;
        let html = `<div class="agent-step-num">Étape ${step.step}</div>
          <div class="agent-thought">${escapeHtml(step.thought || '')}</div>`;
        if (step.action) {
          html += `<div class="agent-action">${escapeHtml(step.action)}</div>`;
        }
        if (step.observation) {
          html += `<div class="agent-obs">→ ${escapeHtml(step.observation)}</div>`;
        }
        div.innerHTML = html;
        trace.appendChild(div);
      }

      if (res.final_answer) {
        const finalDiv = document.createElement('div');
        finalDiv.className = 'agent-step agent-final';
        finalDiv.innerHTML = `<div class="agent-step-num">✅ Réponse finale</div>
          <div class="agent-thought" style="color:var(--green)">${escapeHtml(res.final_answer)}</div>`;
        trace.appendChild(finalDiv);
      }
    } catch (e) {
      trace.innerHTML += `<div class="agent-step"><div class="agent-thought" style="color:var(--red)">⚠ ${e.message}</div></div>`;
    }

    runBtn.disabled = false;
    runBtn.textContent = 'Exécuter';
    trace.scrollTop = trace.scrollHeight;
  }
}

// ─── Training tab ─────────────────────────────────────────────────────────────
const SAMPLE_TEXT = `Le Neural Fractal Network (NFN) est une architecture révolutionnaire.
Il combine la géométrie fractale, les oscillations sinusoïdales paramétriques et l'apprentissage profond.
Chaque nœud possède une phase θ et une fréquence naturelle Ω.
Les connexions sont des fonctions sinusoïdales apprises : Γ(t) = A·sin(ω·t + φ).
La topologie est auto-similaire : chaque sous-réseau est une copie contractée de l'ensemble.
Le réseau superpose plusieurs motifs fractals : arbre binaire, Cantor, Sierpinski.
L'entraînement utilise la rétropropagation à travers les phases (BPTP).
La perte multi-objectif inclut L_tâche + λ_phase·L_phase + λ_freq·L_freq.
Le NFN peut modéliser des dépendances à très longue portée sans explosion paramétrique.
La synchronisation spontanée des oscillateurs implémente le liage temporel.
`;

function initTraining() {
  const startBtn = $('train-start');
  const stopBtn = $('train-stop');
  const sampleBtn = $('train-load-sample');
  const badge = $('train-status-badge');
  const log = $('train-log');

  sampleBtn.addEventListener('click', () => {
    $('train-text').value = SAMPLE_TEXT;
  });

  startBtn.addEventListener('click', async () => {
    const text = $('train-text').value.trim();
    if (!text) { addLog('⚠ Texte vide', 'bad'); return; }

    const body = {
      text,
      n_epochs: parseInt($('train-epochs').value),
      seq_len: parseInt($('train-seq-len').value),
      batch_size: parseInt($('train-batch').value),
      lr: parseFloat($('train-lr').value),
      config_name: $('train-config').value,
    };

    const res = await api('/api/train/start', 'POST', body);
    if (res.error) { addLog('⚠ ' + res.error, 'bad'); return; }

    addLog('▶ Entraînement démarré', 'good');
    startBtn.disabled = true;
    stopBtn.disabled = false;
    state.isTraining = true;
    badge.className = 'running';
    badge.textContent = '● Entraînement';
    state.lossHistory = [];
    connectTrainWS();
  });

  stopBtn.addEventListener('click', async () => {
    await api('/api/train/stop', 'POST');
    addLog('⏹ Arrêt demandé…');
    stopBtn.disabled = true;
  });

  function addLog(msg, cls = '') {
    const div = document.createElement('div');
    div.className = 'log-line' + (cls ? ' ' + cls : '');
    div.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  function connectTrainWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/train`);
    state.trainWs = ws;

    ws.onmessage = e => {
      const { type, data } = JSON.parse(e.data);
      if (type !== 'metrics') return;

      state.lossHistory.push({ step: data.step, loss: data.loss, phase: data.loss_phase });
      drawLossChart();

      $('m-loss').textContent = data.loss?.toFixed(4) ?? '—';
      $('m-phase').textContent = data.loss_phase?.toFixed(5) ?? '—';
      $('m-step').textContent = data.step;
      $('m-lr').textContent = data.lr?.toExponential(2) ?? '—';

      if (data.step % 10 === 0) {
        addLog(`step ${data.step} | loss ${data.loss?.toFixed(4)} | phase ${data.loss_phase?.toFixed(5)}`);
      }
    };

    ws.onclose = () => {
      state.isTraining = false;
      startBtn.disabled = false;
      stopBtn.disabled = true;
      badge.className = '';
      badge.textContent = 'Terminé';
      addLog('✅ Entraînement terminé', 'good');
    };
  }
}

// ─── Loss chart ───────────────────────────────────────────────────────────────
function drawLossChart() {
  const canvas = $('loss-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const hist = state.lossHistory;
  if (!hist.length) return;

  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#12141c';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = '#252840';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (i / 4) * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  const maxLoss = Math.max(...hist.map(h => h.loss || 0), 0.1);
  const minLoss = Math.min(...hist.map(h => h.loss || 0));
  const range = maxLoss - minLoss || 1;

  function toXY(i, val) {
    const x = (i / Math.max(hist.length - 1, 1)) * W;
    const y = H - ((val - minLoss) / range) * (H - 20) - 10;
    return [x, y];
  }

  // Main loss line (gradient)
  const grad = ctx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, '#6ee7f7');
  grad.addColorStop(1, '#a78bfa');
  ctx.strokeStyle = grad;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  hist.forEach((h, i) => {
    const [x, y] = toXY(i, h.loss || 0);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Phase loss (dimmer)
  if (hist.some(h => h.phase)) {
    const maxPhase = Math.max(...hist.map(h => h.phase || 0), 0.001);
    ctx.strokeStyle = '#f472b660';
    ctx.lineWidth = 1;
    ctx.beginPath();
    hist.forEach((h, i) => {
      const x = (i / Math.max(hist.length - 1, 1)) * W;
      const y = H - ((h.phase || 0) / maxPhase) * (H - 20) - 10;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Current loss label
  const last = hist[hist.length - 1];
  ctx.fillStyle = '#6ee7f7';
  ctx.font = '11px JetBrains Mono, monospace';
  ctx.fillText(`loss: ${(last.loss || 0).toFixed(4)}  step: ${last.step}`, 8, 16);
}

// ─── Info tab ─────────────────────────────────────────────────────────────────
function initInfo() {
  $('info-save').addEventListener('click', async () => {
    const res = await api('/api/save_model', 'POST');
    alert(res.path ? `Sauvegardé: ${res.path}` : 'Erreur');
  });

  $('info-reload').addEventListener('click', refreshInfo);

  // Viz buttons
  document.querySelectorAll('.viz-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.viz-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.currentMotif = btn.dataset.motif;
      drawFractal(state.currentMotif);
    });
  });

  // Load checkpoint
  $('ckpt-load-btn').addEventListener('click', async () => {
    const path = $('ckpt-path').value.trim();
    if (!path) return;
    const res = await api('/api/load_model', 'POST', { path });
    if (res.error) alert('Erreur: ' + res.error);
    else { alert(`Modèle chargé: ${res.params}`); refreshStatus(); refreshInfo(); }
  });

  drawFractal('binary_tree');
  startPhaseAnim();
}

function refreshInfo() {
  if (!state.modelInfo) return;
  const grid = $('model-params-grid');
  const info = state.modelInfo;
  const fields = [
    ['Paramètres', info.params],
    ['Vocabulaire', info.vocab_size],
    ['Dimension', info.d_model],
    ['Niveaux (K)', info.n_levels],
    ['Blocs NFN', info.n_blocks],
    ['Motifs', info.motifs?.join(', ')],
    ['Contexte max', info.max_seq_len],
    ['Dispositif', info.device],
  ];
  grid.innerHTML = fields.map(([k, v]) =>
    `<div class="info-param"><div class="info-param-key">${k}</div><div class="info-param-val">${v ?? '—'}</div></div>`
  ).join('');
}

// ─── Fractal visualiser ───────────────────────────────────────────────────────
function drawFractal(motif) {
  const canvas = $('fractal-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#12141c';
  ctx.fillRect(0, 0, W, H);

  if (motif === 'binary_tree') drawBinaryTree(ctx, W, H);
  else if (motif === 'cantor') drawCantor(ctx, W, H);
  else if (motif === 'sierpinski') drawSierpinski(ctx, W, H);
}

function drawBinaryTree(ctx, W, H) {
  const K = 5;
  const nodeRadius = 5;
  const levels = [];
  for (let k = 0; k <= K; k++) {
    const n = Math.pow(2, K - k);
    const y = 20 + k * ((H - 40) / K);
    const nodes = [];
    for (let i = 0; i < n; i++) {
      const x = (W / (n + 1)) * (i + 1);
      nodes.push({ x, y, k, i });
    }
    levels.push(nodes);
  }

  // Draw edges (bottom-up)
  for (let k = 0; k < K; k++) {
    const children = levels[k];
    const parents = levels[k + 1];
    ctx.lineWidth = 1;
    children.forEach((child, i) => {
      const parent = parents[Math.floor(i / 2)];
      const t = k / K;
      ctx.strokeStyle = `rgba(${110 + t * 57}, ${231 - t * 60}, ${247 - t * 90}, ${0.4 + t * 0.3})`;
      ctx.beginPath();
      ctx.moveTo(child.x, child.y);
      ctx.lineTo(parent.x, parent.y);
      ctx.stroke();
    });
  }

  // Draw nodes
  levels.forEach((nodes, k) => {
    nodes.forEach(n => {
      const t = k / K;
      const r = nodeRadius - t * 1.5;
      const grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r);
      grad.addColorStop(0, `rgba(110,231,247,${0.9 - t * 0.3})`);
      grad.addColorStop(1, `rgba(167,139,250,${0.5 - t * 0.2})`);
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
    });
  });

  // Labels
  ctx.fillStyle = '#8b8fa8';
  ctx.font = '10px Inter';
  levels.forEach((nodes, k) => {
    ctx.fillText(`L${k}`, 4, levels[k][0].y + 4);
  });
}

function drawCantor(ctx, W, H) {
  function cantor(x, y, length, depth) {
    if (depth === 0 || length < 2) return;
    const t = (5 - depth) / 5;
    ctx.fillStyle = `rgba(${110 + t * 57}, ${231 - t * 50}, 247, ${0.8 - t * 0.3})`;
    ctx.fillRect(x, y, length, 8);
    const third = length / 3;
    cantor(x, y + 28, third, depth - 1);
    cantor(x + third * 2, y + 28, third, depth - 1);
  }
  cantor(20, 20, W - 40, 5);
}

function drawSierpinski(ctx, W, H) {
  const cx = W / 2, cy = H - 20;
  const size = Math.min(W, H) - 40;
  function triangle(x1, y1, x2, y2, x3, y3, depth) {
    if (depth === 0) {
      const t = 0;
      ctx.beginPath();
      ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.lineTo(x3, y3);
      ctx.closePath();
      ctx.fillStyle = `rgba(167, 139, 250, 0.4)`;
      ctx.fill();
      ctx.strokeStyle = '#a78bfa80';
      ctx.lineWidth = 0.5;
      ctx.stroke();
      return;
    }
    const mx12 = (x1 + x2) / 2, my12 = (y1 + y2) / 2;
    const mx23 = (x2 + x3) / 2, my23 = (y2 + y3) / 2;
    const mx13 = (x1 + x3) / 2, my13 = (y1 + y3) / 2;
    triangle(x1, y1, mx12, my12, mx13, my13, depth - 1);
    triangle(mx12, my12, x2, y2, mx23, my23, depth - 1);
    triangle(mx13, my13, mx23, my23, x3, y3, depth - 1);
  }
  const h = size * Math.sqrt(3) / 2;
  triangle(cx, cy - h, cx - size / 2, cy, cx + size / 2, cy, 5);
}

// ─── Phase oscillation animation ──────────────────────────────────────────────
function startPhaseAnim() {
  const canvas = $('phase-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  let t = 0;

  const oscillators = [
    { freq: 1.0, amp: 0.8, phase: 0,    color: '#6ee7f7' },
    { freq: 0.5, amp: 0.6, phase: 1.2,  color: '#a78bfa' },
    { freq: 0.25, amp: 0.4, phase: 2.4, color: '#f472b6' },
    { freq: 2.0, amp: 0.2, phase: 0.6,  color: '#4ade80' },
  ];

  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#12141c';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#252840';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();

    oscillators.forEach((osc, idx) => {
      ctx.beginPath();
      ctx.strokeStyle = osc.color + (idx === 0 ? 'ff' : '99');
      ctx.lineWidth = idx === 0 ? 2 : 1;
      for (let px = 0; px < W; px++) {
        const tx = (px / W) * 4 * Math.PI + t * osc.freq;
        const val = osc.amp * Math.sin(tx + osc.phase);
        const y = H / 2 - val * (H / 2 - 12);
        px === 0 ? ctx.moveTo(px, y) : ctx.lineTo(px, y);
      }
      ctx.stroke();
    });

    // Labels
    oscillators.forEach((osc, i) => {
      ctx.fillStyle = osc.color;
      ctx.font = '10px JetBrains Mono';
      ctx.fillText(`Ω${i} = ω₀/${Math.pow(2, i)}`, 6, 14 + i * 14);
    });

    t += 0.015;
    state.phaseAnim = requestAnimationFrame(draw);
  }
  draw();
}

// ─── Background fractal canvas ────────────────────────────────────────────────
function initBackground() {
  const canvas = $('bg-canvas');
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  const ctx = canvas.getContext('2d');

  function drawSierpBg(x1, y1, x2, y2, x3, y3, depth) {
    if (depth === 0 || Math.abs(x2 - x1) < 4) {
      ctx.beginPath();
      ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.lineTo(x3, y3);
      ctx.closePath();
      ctx.strokeStyle = '#6ee7f7';
      ctx.lineWidth = 0.3;
      ctx.stroke();
      return;
    }
    const mx12 = (x1 + x2) / 2, my12 = (y1 + y2) / 2;
    const mx23 = (x2 + x3) / 2, my23 = (y2 + y3) / 2;
    const mx13 = (x1 + x3) / 2, my13 = (y1 + y3) / 2;
    drawSierpBg(x1, y1, mx12, my12, mx13, my13, depth - 1);
    drawSierpBg(mx12, my12, x2, y2, mx23, my23, depth - 1);
    drawSierpBg(mx13, my13, mx23, my23, x3, y3, depth - 1);
  }

  const W = canvas.width, H = canvas.height;
  const size = Math.max(W, H) * 1.1;
  const cx = W / 2, cy = H * 1.05;
  const height = size * Math.sqrt(3) / 2;
  drawSierpBg(cx, cy - height, cx - size / 2, cy, cx + size / 2, cy, 6);

  window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const W2 = canvas.width, H2 = canvas.height;
    const sz2 = Math.max(W2, H2) * 1.1;
    const h2 = sz2 * Math.sqrt(3) / 2;
    drawSierpBg(W2 / 2, H2 * 1.05 - h2, W2 / 2 - sz2 / 2, H2 * 1.05, W2 / 2 + sz2 / 2, H2 * 1.05, 6);
  });
}

// ─── Utilities ────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initBackground();
  initTabs();
  initSliders();
  initChat();
  initCode();
  initAgent();
  initTraining();
  initInfo();
  initExplore();
  initAdapt();

  refreshStatus();
  setInterval(refreshStatus, 10000);
});

// ══════════════════════════════════════════════════════════════════════════════
// ── Explore tab ───────────────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════

const exploreState = {
  ws: null,
  running: false,
  normHistory: [],   // [{t, norm}] for the adapt tab chart
};

function initExplore() {
  $('explore-fetch-btn').addEventListener('click', () => {
    const url = $('explore-url-input').value.trim();
    if (!url) return;
    const adapt = $('explore-adapt-check').checked;
    fetchUrl(url, adapt);
  });

  $('explore-url-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') $('explore-fetch-btn').click();
  });

  $('explore-auto-btn').addEventListener('click', () => {
    const seed = $('explore-seed-input').value.trim();
    if (!seed) return;
    const nPages = parseInt($('explore-npages').value) || 5;
    const kwRaw = $('explore-keywords-input').value.trim();
    const keywords = kwRaw ? kwRaw.split(',').map(k => k.trim()).filter(Boolean) : [];
    startAutoExplore(seed, nPages, keywords);
  });

  $('explore-stop-btn').addEventListener('click', () => {
    if (exploreState.ws) {
      exploreState.ws.close();
      exploreState.ws = null;
    }
    exploreState.running = false;
    setExploreRunning(false);
  });

  $('explore-adapt-text-btn').addEventListener('click', () => {
    const text = $('explore-text-area').value.trim();
    if (!text) return;
    adaptText(text);
  });
}

/**
 * Fetch a single URL via POST /api/explore/url
 * Renders a result card in the explore feed.
 */
async function fetchUrl(url, adapt = true) {
  const feed = $('explore-feed');
  const placeholder = feed.querySelector('div[style]');
  if (placeholder) placeholder.remove();

  // Pending card
  const card = document.createElement('div');
  card.className = 'explore-card';
  card.innerHTML = `<div class="explore-card-url">${escapeHtml(url)}</div>
    <div class="explore-card-meta" style="color:var(--yellow)">Chargement…</div>`;
  feed.prepend(card);

  try {
    const res = await api('/api/explore/url', 'POST', { url, adapt });
    renderExploreCard(card, res);
  } catch (e) {
    card.innerHTML = `<div class="explore-card-url">${escapeHtml(url)}</div>
      <div class="explore-card-meta" style="color:var(--red)">Erreur: ${escapeHtml(e.message)}</div>`;
    card.classList.add('error-card');
  }
}

/**
 * Render a fetched-page result card.
 */
function renderExploreCard(card, data) {
  if (data.error && !data.title) {
    card.classList.add('error-card');
    card.innerHTML = `
      <div class="explore-card-url">${escapeHtml(data.url || '')}</div>
      <div class="explore-card-meta" style="color:var(--red)">Erreur: ${escapeHtml(data.error)}</div>`;
    return;
  }

  const chars = data.n_chars ? `${data.n_chars.toLocaleString()} chars` : '';
  const pplBefore = data.ppl_before != null ? data.ppl_before.toFixed(1) : null;
  const pplAfter  = data.ppl_after  != null ? data.ppl_after.toFixed(4)  : null;
  const skipped   = data.skipped;

  let pplHtml = '';
  if (pplBefore != null) {
    const maxPpl = 100;
    const bPct = Math.min(100, (parseFloat(pplBefore) / maxPpl) * 100).toFixed(1);
    pplHtml = `<div class="ppl-bars">
      <span class="ppl-label">Perplexité</span>
      <div class="ppl-bar-wrap"><div class="ppl-bar-fill" style="width:${bPct}%"></div></div>
      <span class="ppl-val">${pplBefore}</span>
    </div>`;
  }
  if (pplAfter != null && !skipped) {
    const maxPpl = 100;
    const aPct = Math.min(100, (parseFloat(pplAfter) / maxPpl) * 100).toFixed(1);
    pplHtml += `<div class="ppl-bars">
      <span class="ppl-label">Loss après</span>
      <div class="ppl-bar-wrap"><div class="ppl-bar-fill after" style="width:${aPct}%"></div></div>
      <span class="ppl-val">${pplAfter}</span>
    </div>`;
  }

  const adaptBadge = skipped
    ? `<span style="color:var(--text3);font-size:11px"> · déjà connu</span>`
    : (pplBefore != null ? `<span style="color:var(--green);font-size:11px"> · adapté</span>` : '');

  card.innerHTML = `
    <div class="explore-card-title">${escapeHtml(data.title || '(sans titre)')}</div>
    <div class="explore-card-url">${escapeHtml(data.url || '')}</div>
    <div class="explore-card-meta">${chars}${adaptBadge}</div>
    ${pplHtml}
    ${data.text_preview ? `<div class="explore-card-preview">${escapeHtml(data.text_preview)}</div>` : ''}
  `;
}

/**
 * Start autonomous exploration via WS /ws/explore.
 */
function startAutoExplore(seedUrl, nPages, keywords) {
  if (exploreState.running) return;

  const feed = $('explore-feed');
  feed.innerHTML = '';
  setExploreRunning(true);

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws/explore`);
  exploreState.ws = ws;
  exploreState.running = true;

  ws.onopen = () => {
    ws.send(JSON.stringify({ seed_url: seedUrl, n_pages: nPages, keywords }));
  };

  ws.onmessage = e => {
    const msg = JSON.parse(e.data);

    if (msg.type === 'page') {
      const card = document.createElement('div');
      card.className = 'explore-card';
      feed.appendChild(card);
      renderExploreCard(card, {
        url:          msg.url,
        title:        msg.title,
        n_chars:      msg.n_chars,
        ppl_before:   msg.ppl_before,
        ppl_after:    msg.ppl_after,
        skipped:      msg.skipped,
        text_preview: null,
      });
      feed.scrollTop = feed.scrollHeight;

    } else if (msg.type === 'error') {
      const card = document.createElement('div');
      card.className = 'explore-card error-card';
      card.innerHTML = `<div class="explore-card-url">${escapeHtml(msg.url || '')}</div>
        <div class="explore-card-meta" style="color:var(--red)">Erreur: ${escapeHtml(msg.error)}</div>`;
      feed.appendChild(card);

    } else if (msg.type === 'done') {
      const banner = document.createElement('div');
      banner.className = 'explore-done-banner';
      banner.textContent = `Exploration terminée — ${msg.n_pages} pages, ${(msg.total_chars || 0).toLocaleString()} caractères`;
      feed.appendChild(banner);
      feed.scrollTop = feed.scrollHeight;
      setExploreRunning(false);
      exploreState.running = false;
    }
  };

  ws.onerror = () => {
    setExploreRunning(false);
    exploreState.running = false;
  };
  ws.onclose = () => {
    setExploreRunning(false);
    exploreState.running = false;
  };
}

/**
 * Adapt from raw text via POST /api/explore/text.
 */
async function adaptText(text) {
  const resultEl = $('explore-text-result');
  resultEl.textContent = 'Adaptation en cours…';
  resultEl.style.color = 'var(--yellow)';
  try {
    const res = await api('/api/explore/text', 'POST', { text });
    if (res.error) {
      resultEl.textContent = 'Erreur: ' + res.error;
      resultEl.style.color = 'var(--red)';
    } else if (res.skipped) {
      resultEl.textContent = `Ignoré — perplexité ${res.ppl?.toFixed(1)} < seuil`;
      resultEl.style.color = 'var(--text3)';
    } else {
      resultEl.textContent = `Adapté — ppl ${res.ppl?.toFixed(1)}, loss ${res.loss?.toFixed(4)}, norme ${res.adapter_norm?.toFixed(4)}`;
      resultEl.style.color = 'var(--green)';
      // Refresh adapt stats if visible
      loadTTLStats();
    }
  } catch (err) {
    resultEl.textContent = 'Erreur: ' + err.message;
    resultEl.style.color = 'var(--red)';
  }
}

function setExploreRunning(running) {
  $('explore-auto-btn').disabled = running;
  $('explore-stop-btn').disabled = !running;
  const badge = $('explore-status-badge');
  if (running) {
    badge.textContent = 'Exploration en cours…';
    badge.style.color = 'var(--green)';
    badge.style.borderColor = 'var(--green)';
  } else {
    badge.textContent = 'Prêt';
    badge.style.color = 'var(--text2)';
    badge.style.borderColor = 'var(--border)';
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// ── Adapt (TTL) tab ───────────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════

const adaptState = {
  normHistory: [],     // [float] last 50 mean_adapter_norm readings
  statsInterval: null,
};

function initAdapt() {
  const toggle = $('ttl-toggle');
  toggle.addEventListener('change', async () => {
    if (toggle.checked) {
      await enableTTL(
        parseInt($('adapt-rank').value),
        parseInt($('adapt-lr').value) * 1e-5,
        parseInt($('adapt-steps').value),
        parseFloat($('adapt-gate').value),
      );
    } else {
      await disableTTL();
    }
  });

  $('adapt-refresh-btn').addEventListener('click', loadTTLStats);
  $('adapt-reset-btn').addEventListener('click', resetAdapters);

  // Poll stats when tab is active
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (btn.dataset.tab === 'adapt') {
        loadTTLStats();
        if (!adaptState.statsInterval) {
          adaptState.statsInterval = setInterval(loadTTLStats, 3000);
        }
      } else {
        clearInterval(adaptState.statsInterval);
        adaptState.statsInterval = null;
      }
    });
  });
}

/**
 * GET /api/ttl/stats → update the stats grid and norm gauge.
 */
async function loadTTLStats() {
  try {
    const data = await api('/api/ttl/stats');

    // Sync toggle state
    $('ttl-toggle').checked = !!data.enabled;

    const msg = $('adapt-status-msg');
    if (!data.enabled) {
      msg.textContent = 'TTL désactivé — le modèle n\'apprend pas en temps réel.';
      msg.style.color = 'var(--text2)';
      return;
    }

    msg.textContent = 'TTL actif — le modèle s\'adapte à chaque nouvelle page.';
    msg.style.color = 'var(--green)';

    $('ttl-n-adapters').textContent = data.n_adapters ?? '—';
    $('ttl-params').textContent     = data.adapter_params != null
      ? (data.adapter_params / 1000).toFixed(1) + 'k' : '—';
    $('ttl-calls').textContent   = data.adapt_calls   ?? '—';
    $('ttl-skipped').textContent = data.skipped        ?? '—';
    $('ttl-avg-loss').textContent = data.avg_loss      != null
      ? data.avg_loss.toFixed(4) : '—';
    $('ttl-ratio').textContent   = data.adapter_ratio  ?? '—';

    const norm = data.mean_adapter_norm ?? 0;
    const normPct = Math.min(100, norm * 1000).toFixed(1);
    $('norm-gauge-fill').style.width = normPct + '%';
    $('norm-gauge-val').textContent  = norm.toFixed(4);

    adaptState.normHistory.push(norm);
    if (adaptState.normHistory.length > 60) adaptState.normHistory.shift();
    drawNormChart();

  } catch {
    // Silently ignore — model may not be ready
  }
}

/**
 * POST /api/ttl/enable
 */
async function enableTTL(rank, lr, steps, gate) {
  try {
    const res = await api('/api/ttl/enable', 'POST', {
      adapter_rank: rank,
      online_lr:    lr,
      n_steps:      steps,
      ppl_gate:     gate,
    });
    if (res.error) {
      $('adapt-status-msg').textContent = 'Erreur: ' + res.error;
      $('adapt-status-msg').style.color = 'var(--red)';
      $('ttl-toggle').checked = false;
    } else {
      await loadTTLStats();
    }
  } catch (e) {
    $('adapt-status-msg').textContent = 'Erreur: ' + e.message;
    $('ttl-toggle').checked = false;
  }
}

/**
 * POST /api/ttl/disable
 */
async function disableTTL() {
  try {
    await api('/api/ttl/disable', 'POST');
    await loadTTLStats();
    adaptState.normHistory = [];
    drawNormChart();
  } catch { /* ignore */ }
}

/**
 * POST /api/ttl/reset
 */
async function resetAdapters() {
  try {
    const res = await api('/api/ttl/reset', 'POST');
    if (res.error) {
      alert('Erreur: ' + res.error);
    } else {
      adaptState.normHistory = [];
      await loadTTLStats();
    }
  } catch (e) {
    alert('Erreur: ' + e.message);
  }
}

/**
 * Draw the adapter-norm-over-time mini-chart on the norm-chart canvas.
 */
function drawNormChart() {
  const canvas = $('norm-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || canvas.width;
  const H = canvas.height;
  canvas.width = W;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#12141c';
  ctx.fillRect(0, 0, W, H);

  const hist = adaptState.normHistory;
  if (hist.length < 2) return;

  const maxV = Math.max(...hist, 0.001);

  // Grid line
  ctx.strokeStyle = '#252840';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();

  // Line
  const grad = ctx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, '#6ee7f7');
  grad.addColorStop(1, '#a78bfa');
  ctx.strokeStyle = grad;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  hist.forEach((v, i) => {
    const x = (i / Math.max(hist.length - 1, 1)) * W;
    const y = H - (v / maxV) * (H - 8) - 4;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Label
  ctx.fillStyle = '#6ee7f7';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillText(`norm: ${hist[hist.length - 1].toFixed(4)}`, 4, 12);
}
