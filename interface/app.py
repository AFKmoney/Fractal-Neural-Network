"""
NFN AGI Interface v4.0 — FastAPI server

Endpoints:
  GET  /                      → web UI (chat + tools + memory inspector)
  GET  /api/status            → model info, active features
  POST /api/chat              → multi-turn chat (JSON, blocking)
  POST /api/think             → think + answer (shows reasoning rounds)
  POST /api/agent/run         → tool-calling agent task
  POST /api/learn             → learn new text into episodic memory
  POST /api/retrieve          → search episodic memory
  POST /api/rag               → generate with RAG context
  POST /api/generate          → raw text generation
  POST /api/train/start       → start background training (AGITrainer)
  POST /api/train/stop        → stop training
  GET  /api/train/status      → live training metrics
  POST /api/memory/reset      → reset working memory slots
  GET  /api/memory/state      → inspect memory state
  POST /api/memory/save       → persist knowledge store
  WS   /ws/stream             → streaming generation (WebSocket)
  WS   /ws/chat               → streaming chat (WebSocket)
  WS   /ws/train              → live training metrics (WebSocket)
"""

import asyncio
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nfn.config import NFNConfig
from nfn.tokenizer import NFNTokenizer
from nfn.agi_model import build_agi_model
from inference.engine import AGIInferenceEngine
from training.agi_trainer import AGITrainer
from interface.agents import ChatAgent, ThinkAgent, ToolAgent, LearnAgent, CodeAgent, ReasoningAgent


# ─────────────────────────────────────────────────────────────────────────────
# App & CORS
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="NFN AGI Interface", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Global model state
# ─────────────────────────────────────────────────────────────────────────────

_engine:   Optional[AGIInferenceEngine] = None
_trainer:  Optional[AGITrainer]         = None
_train_thread: Optional[threading.Thread] = None
_train_metrics: List[Dict]               = []
_ws_clients: List[WebSocket]             = []

KNOWLEDGE_STORE_PATH = str(ROOT / "checkpoints" / "knowledge_store.json.gz")


def get_engine() -> AGIInferenceEngine:
    global _engine
    if _engine is None:
        raise RuntimeError("Model not loaded — call /api/load_model first or restart with a checkpoint")
    return _engine


def _init_default_model():
    """Initialise a tiny model for immediate use (no checkpoint needed)."""
    global _engine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = NFNTokenizer()

    model = build_agi_model(
        vocab_size           = tokenizer.vocab_size,
        d_model              = 256,
        n_blocks             = 4,
        use_reasoning        = True,
        use_predictive_coding = True,
        use_free_energy      = True,
        use_self_consistency = True,
        use_plan_executor    = True,
        use_mod              = True,
        use_mtp              = True,
        use_hyper            = True,
    ).to(device)

    _engine = AGIInferenceEngine(
        model,
        tokenizer,
        knowledge_store_path=KNOWLEDGE_STORE_PATH,
    )
    print(f"[NFN AGI] Model loaded: {sum(p.numel() for p in model.parameters()):,} params on {device}")


# ─────────────────────────────────────────────────────────────────────────────
# Request/Response models
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages:      List[Dict[str, str]]
    system:        Optional[str]  = None
    max_tokens:    int            = 512
    temperature:   float          = 0.8
    top_k:         int            = 50
    top_p:         float          = 0.95
    think_rounds:  int            = 0
    use_tools:     bool           = False
    use_rag:       bool           = True

class GenerateRequest(BaseModel):
    prompt:        str
    max_tokens:    int   = 256
    temperature:   float = 0.8
    top_k:         int   = 50
    top_p:         float = 0.95
    greedy:        bool  = False
    think_rounds:  int   = 0
    speculative:   bool  = False

class ThinkRequest(BaseModel):
    question:      str
    n_rounds:      int   = 3
    max_answer_tokens: int = 512
    temperature:   float = 0.7

class AgentRequest(BaseModel):
    task:          str
    system:        Optional[str] = None
    max_tokens:    int   = 1024
    temperature:   float = 0.7

class LearnRequest(BaseModel):
    text:   str
    source: str = "user"

class RetrieveRequest(BaseModel):
    query:  str
    top_k:  int = 4

class RagRequest(BaseModel):
    prompt:     str
    top_k:      int   = 3
    max_tokens: int   = 256
    temperature: float = 0.8

class TrainRequest(BaseModel):
    text:             str
    n_epochs:         int   = 1
    seq_len:          int   = 256
    batch_size:       int   = 4
    lr:               float = 3e-4
    agi_loss_start:   int   = 100
    agi_loss_ramp:    int   = 50

class LoadModelRequest(BaseModel):
    checkpoint_path: str
    device:          str = "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    _init_default_model()


# ─────────────────────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    try:
        return JSONResponse(get_engine().status())
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=503)


# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    engine = get_engine()
    agent = ChatAgent(
        engine,
        use_rag      = req.use_rag,
        think_rounds = req.think_rounds,
        use_tools    = req.use_tools,
    )
    try:
        result = agent.reply(
            req.messages,
            system      = req.system,
            max_tokens  = req.max_tokens,
            temperature = req.temperature,
            top_k       = req.top_k,
            top_p       = req.top_p,
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


# ─────────────────────────────────────────────────────────────────────────────
# Think
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/think")
async def think(req: ThinkRequest):
    engine = get_engine()
    agent  = ThinkAgent(engine, default_rounds=req.n_rounds)
    try:
        result = agent.think_and_answer(
            req.question,
            n_rounds          = req.n_rounds,
            max_answer_tokens = req.max_answer_tokens,
            temperature       = req.temperature,
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Agent
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/agent/run")
async def agent_run(req: AgentRequest):
    engine = get_engine()
    agent  = ToolAgent(engine)
    try:
        result = agent.run(
            req.task,
            system     = req.system,
            max_new_tokens = req.max_tokens,
            temperature = req.temperature,
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


# ─────────────────────────────────────────────────────────────────────────────
# Generate (raw)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    engine = get_engine()
    try:
        text = engine.generate(
            req.prompt,
            max_new_tokens   = req.max_tokens,
            temperature      = req.temperature,
            top_k            = req.top_k,
            top_p            = req.top_p,
            greedy           = req.greedy,
            think_rounds     = req.think_rounds,
            use_speculative  = req.speculative,
        )
        return JSONResponse({"text": text, "prompt": req.prompt})
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


# ─────────────────────────────────────────────────────────────────────────────
# Continual Learning
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/learn")
async def learn(req: LearnRequest):
    engine = get_engine()
    try:
        result = engine.learn(req.text, source=req.source)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/retrieve")
async def retrieve(req: RetrieveRequest):
    engine = get_engine()
    try:
        results = engine.retrieve(req.query, top_k=req.top_k)
        return JSONResponse({"query": req.query, "results": results})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/rag")
async def rag(req: RagRequest):
    engine = get_engine()
    try:
        text = engine.generate_with_rag(
            req.prompt,
            top_k       = req.top_k,
            max_new_tokens = req.max_tokens,
            temperature = req.temperature,
        )
        return JSONResponse({"text": text, "prompt": req.prompt})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────────────────────────────────────────
# Memory management
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/memory/reset")
async def memory_reset():
    engine = get_engine()
    for block in engine.model.blocks:
        if hasattr(block, "reset_working_memory"):
            block.reset_working_memory()
    return JSONResponse({"reset": True})


@app.get("/api/memory/state")
async def memory_state():
    engine = get_engine()
    return JSONResponse({
        "knowledge_entries": len(engine.learner.store),
        "stats":             engine.learner.stats(),
    })


@app.post("/api/memory/save")
async def memory_save():
    engine = get_engine()
    path = KNOWLEDGE_STORE_PATH
    engine.learner.save_store(path)
    return JSONResponse({"saved": path, "entries": len(engine.learner.store)})


# ─────────────────────────────────────────────────────────────────────────────
# Load model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/load_model")
async def load_model(req: LoadModelRequest):
    global _engine
    device = torch.device(req.device)
    try:
        ckpt = torch.load(req.checkpoint_path, map_location=device, weights_only=False)
        cfg  = NFNConfig.from_dict(ckpt["cfg"])
        model = build_agi_model(
            vocab_size = cfg.vocab_size,
            d_model    = cfg.d_model,
            n_blocks   = cfg.n_blocks,
            use_mod    = cfg.use_mixture_of_depths,
            use_mtp    = cfg.use_multi_token_pred,
            use_hyper  = cfg.use_hyper_net,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        tokenizer = NFNTokenizer()
        _engine = AGIInferenceEngine(model, tokenizer, knowledge_store_path=KNOWLEDGE_STORE_PATH)
        return JSONResponse({"loaded": req.checkpoint_path, **_engine.status()})
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/train/start")
async def train_start(req: TrainRequest):
    global _trainer, _train_thread, _train_metrics
    engine = get_engine()

    if _train_thread and _train_thread.is_alive():
        return JSONResponse({"error": "Training already running"}, status_code=400)

    _train_metrics = []
    _trainer = AGITrainer(
        engine.model,
        engine.tokenizer,
        lr                  = req.lr,
        agi_loss_start_step = req.agi_loss_start,
        agi_loss_ramp_steps = req.agi_loss_ramp,
        step_callback       = lambda m: _train_metrics.append(m),
    )

    def _run():
        _trainer.train(
            req.text,
            n_epochs   = req.n_epochs,
            seq_len    = req.seq_len,
            batch_size = req.batch_size,
        )

    _train_thread = threading.Thread(target=_run, daemon=True)
    _train_thread.start()
    return JSONResponse({"started": True})


@app.post("/api/train/stop")
async def train_stop():
    global _trainer
    if _trainer:
        _trainer.stop()
    return JSONResponse({"stopped": True})


@app.get("/api/train/status")
async def train_status():
    running = _train_thread is not None and _train_thread.is_alive()
    recent  = _train_metrics[-20:] if _train_metrics else []
    return JSONResponse({"running": running, "recent_metrics": recent, "total_steps": len(_train_metrics)})


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — streaming generation
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    try:
        data = await ws.receive_json()
        engine = get_engine()
        prompt = data.get("prompt", "")
        kwargs = {
            "max_new_tokens": data.get("max_tokens", 256),
            "temperature":    data.get("temperature", 0.8),
            "top_k":          data.get("top_k", 50),
            "top_p":          data.get("top_p", 0.95),
            "think_rounds":   data.get("think_rounds", 0),
        }
        n_tokens = 0
        async for token in engine.astream(prompt, **kwargs):
            await ws.send_json({"token": token, "n": n_tokens})
            n_tokens += 1
        await ws.send_json({"done": True, "n_tokens": n_tokens})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    """Streaming chat WebSocket — maintains conversation history on server."""
    await ws.accept()
    history: List[Dict[str, str]] = []
    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action", "chat")

            if action == "reset":
                history = []
                await ws.send_json({"action": "reset", "ok": True})
                continue

            if action == "learn":
                engine = get_engine()
                result = engine.learn(data.get("text", ""), source="ws_chat")
                await ws.send_json({"action": "learn", **result})
                continue

            if action == "retrieve":
                engine = get_engine()
                results = engine.retrieve(data.get("query", ""), top_k=data.get("top_k", 3))
                await ws.send_json({"action": "retrieve", "results": results})
                continue

            # Chat
            user_msg = data.get("content", "")
            history.append({"role": "user", "content": user_msg})

            engine = get_engine()
            agent  = ChatAgent(
                engine,
                use_rag      = data.get("use_rag", True),
                think_rounds = data.get("think_rounds", 0),
                use_tools    = data.get("use_tools", False),
            )
            result = agent.reply(
                history,
                max_tokens  = data.get("max_tokens", 512),
                temperature = data.get("temperature", 0.8),
            )
            reply = result["reply"]
            history.append({"role": "assistant", "content": reply})

            await ws.send_json({
                "action":    "reply",
                "content":   reply,
                "retrieved": result.get("retrieved", []),
                "thoughts":  result.get("think_thoughts", []),
                "elapsed_s": result.get("elapsed_s", 0),
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


@app.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    """Push training metrics to WebSocket clients in real time."""
    await ws.accept()
    _ws_clients.append(ws)
    last_sent = 0
    try:
        while True:
            if len(_train_metrics) > last_sent:
                metrics = _train_metrics[last_sent:]
                await ws.send_json({"metrics": metrics})
                last_sent = len(_train_metrics)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws) if hasattr(_ws_clients, "discard") else None


# ─────────────────────────────────────────────────────────────────────────────
# Web UI
# ─────────────────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NFN AGI v4.0</title>
<style>
  :root {
    --bg: #0a0a0f; --surface: #12121a; --border: #1e1e2e;
    --accent: #7c3aed; --accent2: #06b6d4; --text: #e2e8f0;
    --muted: #64748b; --green: #10b981; --red: #ef4444;
    --yellow: #f59e0b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'JetBrains Mono', monospace; height: 100vh; display: flex; flex-direction: column; }

  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 20px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.1rem; color: var(--accent); letter-spacing: 0.05em; }
  .badge { background: var(--border); border-radius: 4px; padding: 2px 8px; font-size: 0.7rem; color: var(--accent2); }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); margin-left: auto; }

  .layout { display: flex; flex: 1; overflow: hidden; }

  /* Sidebar */
  .sidebar { width: 280px; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 16px; gap: 16px; overflow-y: auto; }
  .sidebar h2 { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
  .control-group { display: flex; flex-direction: column; gap: 8px; }
  label { font-size: 0.75rem; color: var(--muted); }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  input[type=number], select, textarea { background: var(--bg); border: 1px solid var(--border); color: var(--text); border-radius: 6px; padding: 6px 10px; font-size: 0.8rem; width: 100%; font-family: inherit; }
  .val { font-size: 0.75rem; color: var(--accent2); float: right; }
  .sep { border: none; border-top: 1px solid var(--border); }
  .feature-list { display: flex; flex-direction: column; gap: 4px; }
  .feature { display: flex; align-items: center; gap: 6px; font-size: 0.72rem; }
  .dot { width: 6px; height: 6px; border-radius: 50%; }
  .dot.on { background: var(--green); } .dot.off { background: var(--muted); }

  /* Chat area */
  .chat-area { flex: 1; display: flex; flex-direction: column; }
  .tabs { display: flex; border-bottom: 1px solid var(--border); background: var(--surface); }
  .tab { padding: 10px 20px; font-size: 0.8rem; cursor: pointer; color: var(--muted); border-bottom: 2px solid transparent; transition: all 0.15s; }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab-content { display: none; flex: 1; flex-direction: column; overflow: hidden; }
  .tab-content.active { display: flex; }

  /* Messages */
  .messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
  .msg { display: flex; gap: 12px; max-width: 85%; }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg .avatar { width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; flex-shrink: 0; }
  .msg.user .avatar { background: var(--accent); }
  .msg.assistant .avatar { background: var(--accent2); color: #000; }
  .bubble { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px 16px; font-size: 0.85rem; line-height: 1.6; max-width: 100%; }
  .msg.user .bubble { background: #1e1040; border-color: var(--accent); }
  .meta { font-size: 0.65rem; color: var(--muted); margin-top: 4px; }
  .retrieved-ctx { font-size: 0.7rem; color: var(--accent2); margin-top: 6px; border-top: 1px solid var(--border); padding-top: 6px; }
  .thoughts-ctx { font-size: 0.7rem; color: var(--yellow); margin-top: 6px; border-top: 1px solid var(--border); padding-top: 6px; }
  .thinking-indicator { color: var(--yellow); font-style: italic; }

  /* Input area */
  .input-area { border-top: 1px solid var(--border); padding: 16px; display: flex; gap: 10px; background: var(--surface); }
  .input-area textarea { flex: 1; background: var(--bg); border: 1px solid var(--border); color: var(--text); border-radius: 8px; padding: 10px 14px; font-size: 0.85rem; resize: none; font-family: inherit; height: 60px; }
  .input-area textarea:focus { outline: none; border-color: var(--accent); }
  .btn { background: var(--accent); color: white; border: none; border-radius: 8px; padding: 0 20px; cursor: pointer; font-size: 0.85rem; font-family: inherit; transition: opacity 0.15s; white-space: nowrap; }
  .btn:hover { opacity: 0.85; } .btn:disabled { opacity: 0.4; cursor: default; }
  .btn.secondary { background: var(--surface); border: 1px solid var(--border); color: var(--text); }
  .btn.danger { background: var(--red); }
  .btn.cyan { background: var(--accent2); color: #000; }

  /* Memory panel */
  .memory-panel { flex: 1; overflow-y: auto; padding: 20px; }
  .memory-entry { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-bottom: 12px; }
  .memory-entry .score { color: var(--accent2); font-size: 0.75rem; }
  .memory-entry .text { font-size: 0.8rem; margin-top: 4px; line-height: 1.5; }
  .search-bar { display: flex; gap: 8px; margin-bottom: 16px; }
  .search-bar input { flex: 1; }

  /* Think panel */
  .think-panel { flex: 1; overflow-y: auto; padding: 20px; }
  .thought-block { background: #1a1025; border: 1px solid #3b1d5c; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
  .thought-block .round { font-size: 0.7rem; color: var(--yellow); margin-bottom: 6px; }
  .thought-block .content { font-size: 0.8rem; line-height: 1.5; }
  .answer-block { background: #0d1f1a; border: 1px solid var(--green); border-radius: 8px; padding: 16px; margin-top: 16px; }
  .answer-block .label { font-size: 0.7rem; color: var(--green); margin-bottom: 8px; }

  /* Learn panel */
  .learn-panel { flex: 1; padding: 20px; display: flex; flex-direction: column; gap: 12px; overflow-y: auto; }
  .learn-panel textarea { height: 120px; }
  .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
  .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
  .stat-card .val { font-size: 1.2rem; color: var(--accent2); display: block; margin-top: 4px; }

  pre { white-space: pre-wrap; word-break: break-word; }
  code { background: #1a1a2e; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; }
</style>
</head>
<body>
<header>
  <h1>⬡ NFN AGI v4.0</h1>
  <span class="badge" id="param-badge">loading...</span>
  <span class="badge" id="memory-badge">0 memories</span>
  <div class="status-dot" id="status-dot" title="Model status"></div>
</header>

<div class="layout">
  <!-- Sidebar -->
  <div class="sidebar">
    <h2>Génération</h2>
    <div class="control-group">
      <label>Température <span class="val" id="temp-val">0.8</span></label>
      <input type="range" id="temperature" min="0.1" max="2.0" step="0.05" value="0.8"
             oninput="document.getElementById('temp-val').textContent=this.value">
      <label>Top-P <span class="val" id="topp-val">0.95</span></label>
      <input type="range" id="top_p" min="0.5" max="1.0" step="0.01" value="0.95"
             oninput="document.getElementById('topp-val').textContent=this.value">
      <label>Max tokens</label>
      <input type="number" id="max_tokens" value="512" min="16" max="2048">
    </div>
    <hr class="sep">
    <h2>Capacités AGI</h2>
    <div class="control-group">
      <label>Rounds de réflexion <span class="val" id="think-val">0</span></label>
      <input type="range" id="think_rounds" min="0" max="5" step="1" value="0"
             oninput="document.getElementById('think-val').textContent=this.value">
      <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
        <input type="checkbox" id="use_tools"> Outils natifs
      </label>
      <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
        <input type="checkbox" id="use_rag" checked> Mémoire RAG
      </label>
    </div>
    <hr class="sep">
    <h2>Features actives</h2>
    <div class="feature-list" id="feature-list"></div>
    <hr class="sep">
    <button class="btn secondary" onclick="resetChat()">↺ Réinitialiser chat</button>
    <button class="btn secondary" onclick="saveMemory()">💾 Sauvegarder mémoire</button>
  </div>

  <!-- Main content -->
  <div class="chat-area">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('chat')">💬 Chat</div>
      <div class="tab" onclick="switchTab('think')">🧠 Raisonner</div>
      <div class="tab" onclick="switchTab('memory')">🗂 Mémoire</div>
      <div class="tab" onclick="switchTab('learn')">📥 Apprendre</div>
    </div>

    <!-- Chat tab -->
    <div class="tab-content active" id="tab-chat">
      <div class="messages" id="messages"></div>
      <div class="input-area">
        <textarea id="chat-input" placeholder="Message…" onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat()}"></textarea>
        <button class="btn" id="send-btn" onclick="sendChat()">Envoyer</button>
        <button class="btn secondary" onclick="clearMessages()">✕</button>
      </div>
    </div>

    <!-- Think tab -->
    <div class="tab-content" id="tab-think">
      <div class="think-panel" id="think-panel">
        <div style="display:flex;gap:8px;margin-bottom:16px">
          <input type="text" id="think-input" placeholder="Question à raisonner…" style="flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:8px;padding:10px 14px;font-size:0.85rem;font-family:inherit">
          <select id="think-rounds-select" style="width:120px">
            <option value="1">1 round</option>
            <option value="2">2 rounds</option>
            <option value="3" selected>3 rounds</option>
            <option value="5">5 rounds</option>
          </select>
          <button class="btn cyan" onclick="runThink()">Raisonner</button>
        </div>
        <div id="think-results"></div>
      </div>
    </div>

    <!-- Memory tab -->
    <div class="tab-content" id="tab-memory">
      <div class="memory-panel">
        <div class="search-bar">
          <input type="text" id="memory-search" placeholder="Rechercher dans la mémoire…">
          <button class="btn" onclick="searchMemory()">Chercher</button>
          <input type="number" id="memory-topk" value="5" min="1" max="20" style="width:60px">
        </div>
        <div id="memory-results"></div>
      </div>
    </div>

    <!-- Learn tab -->
    <div class="tab-content" id="tab-learn">
      <div class="learn-panel">
        <h2 style="color:var(--text);text-transform:none;letter-spacing:0">Apprendre du nouveau contenu</h2>
        <textarea id="learn-text" placeholder="Colle ici le texte à apprendre (article, code, documentation, conversation…)"></textarea>
        <input type="text" id="learn-source" placeholder="Source (optionnel, ex: wikipedia, docs)">
        <div style="display:flex;gap:8px">
          <button class="btn" onclick="learnText()">📥 Apprendre</button>
          <button class="btn secondary" onclick="loadStatus()">↺ Actualiser stats</button>
        </div>
        <div id="learn-result" style="font-size:0.8rem;color:var(--green)"></div>
        <h2 style="color:var(--text);text-transform:none;letter-spacing:0;margin-top:8px">Statistiques mémoire</h2>
        <div class="stats-grid" id="stats-grid"></div>
      </div>
    </div>
  </div>
</div>

<script>
let history = [];
let ws = null;

// ── WebSocket chat ───────────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws/chat`);
  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.action === 'reply') {
      addMessage('assistant', data.content, data.retrieved, data.thoughts, data.elapsed_s);
      document.getElementById('send-btn').disabled = false;
    } else if (data.action === 'learn') {
      document.getElementById('learn-result').textContent = `✓ Appris: ${data.chars} chars, ${data.store_size} entrées en mémoire`;
    } else if (data.action === 'retrieve') {
      renderMemoryResults(data.results);
    } else if (data.error) {
      addMessage('assistant', `[Erreur] ${data.error}`, [], [], 0);
      document.getElementById('send-btn').disabled = false;
    }
  };
  ws.onclose = () => setTimeout(connectWS, 2000);
}

// ── Tab switching ────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => { t.classList.remove('active'); t.style.display='none'; });
  const tabs = ['chat','think','memory','learn'];
  const idx = tabs.indexOf(name);
  document.querySelectorAll('.tab')[idx].classList.add('active');
  const el = document.getElementById('tab-'+name);
  el.classList.add('active'); el.style.display='flex';
  if (name === 'memory') loadStatus();
}

// ── Chat ─────────────────────────────────────────────────────────────────────
function sendChat() {
  const input = document.getElementById('chat-input');
  const text  = input.value.trim();
  if (!text || !ws || ws.readyState !== 1) return;
  addMessage('user', text);
  history.push({role:'user', content:text});
  input.value = '';
  document.getElementById('send-btn').disabled = true;
  ws.send(JSON.stringify({
    action:       'chat',
    content:      text,
    temperature:  parseFloat(document.getElementById('temperature').value),
    top_p:        parseFloat(document.getElementById('top_p').value),
    max_tokens:   parseInt(document.getElementById('max_tokens').value),
    think_rounds: parseInt(document.getElementById('think_rounds').value),
    use_tools:    document.getElementById('use_tools').checked,
    use_rag:      document.getElementById('use_rag').checked,
  }));
}

function addMessage(role, content, retrieved=[], thoughts=[], elapsed=0) {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const avatar = role === 'user' ? '👤' : '⬡';
  let extra = '';
  if (thoughts && thoughts.length > 0) {
    extra += `<div class="thoughts-ctx">💭 ${thoughts.length} round(s): ${thoughts[0].slice(0,120)}…</div>`;
  }
  if (retrieved && retrieved.length > 0) {
    extra += `<div class="retrieved-ctx">🗂 ${retrieved.length} mémoire(s) retrouvée(s): ${retrieved[0].text.slice(0,80)}…</div>`;
  }
  div.innerHTML = `
    <div class="avatar">${avatar}</div>
    <div>
      <div class="bubble"><pre>${escapeHtml(content)}</pre>${extra}</div>
      <div class="meta">${role === 'assistant' ? `${elapsed}s` : 'maintenant'}</div>
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  if (role === 'assistant') history.push({role:'assistant', content});
}

function clearMessages() { document.getElementById('messages').innerHTML = ''; history = []; }
function resetChat() {
  clearMessages();
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({action:'reset'}));
}

// ── Think ─────────────────────────────────────────────────────────────────────
async function runThink() {
  const q = document.getElementById('think-input').value.trim();
  const rounds = parseInt(document.getElementById('think-rounds-select').value);
  if (!q) return;
  const panel = document.getElementById('think-results');
  panel.innerHTML = '<div class="thinking-indicator">⚙ Raisonnement en cours…</div>';
  try {
    const r = await fetch('/api/think', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question:q, n_rounds:rounds, temperature:0.7}),
    });
    const data = await r.json();
    let html = data.thoughts.map((t,i) =>
      `<div class="thought-block"><div class="round">Round ${i+1}</div><div class="content">${escapeHtml(t)}</div></div>`
    ).join('');
    html += `<div class="answer-block"><div class="label">✓ Réponse finale</div><pre>${escapeHtml(data.answer)}</pre></div>`;
    html += `<div class="meta" style="margin-top:8px;color:var(--muted);">${data.elapsed_s}s</div>`;
    panel.innerHTML = html;
  } catch(e) { panel.innerHTML = `<div style="color:var(--red)">${e}</div>`; }
}

// ── Memory ────────────────────────────────────────────────────────────────────
async function searchMemory() {
  const q = document.getElementById('memory-search').value.trim();
  const k = parseInt(document.getElementById('memory-topk').value);
  if (!q) return;
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({action:'retrieve', query:q, top_k:k}));
  } else {
    const r = await fetch('/api/retrieve', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,top_k:k})});
    const data = await r.json();
    renderMemoryResults(data.results);
  }
}

function renderMemoryResults(results) {
  const el = document.getElementById('memory-results');
  if (!results || results.length === 0) { el.innerHTML = '<div style="color:var(--muted)">Aucun résultat.</div>'; return; }
  el.innerHTML = results.map(r =>
    `<div class="memory-entry"><div class="score">Score: ${r.score}</div><div class="text">${escapeHtml(r.text)}</div></div>`
  ).join('');
}

// ── Learn ─────────────────────────────────────────────────────────────────────
async function learnText() {
  const text   = document.getElementById('learn-text').value.trim();
  const source = document.getElementById('learn-source').value.trim() || 'user';
  if (!text) return;
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({action:'learn', text, source}));
  } else {
    const r = await fetch('/api/learn',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,source})});
    const data = await r.json();
    document.getElementById('learn-result').textContent = `✓ Appris: ${data.chars} chars, ${data.store_size} entrées en mémoire`;
  }
  await loadStatus();
}

async function loadStatus() {
  try {
    const r = await fetch('/api/status');
    const data = await r.json();
    document.getElementById('param-badge').textContent = `${data.params_M}M params`;
    document.getElementById('memory-badge').textContent = `${data.knowledge_entries} mémoires`;

    // Feature list
    const fl = document.getElementById('feature-list');
    fl.innerHTML = Object.entries(data.features || {}).map(([k,v]) =>
      `<div class="feature"><div class="dot ${v?'on':'off'}"></div>${k.replace(/_/g,' ')}</div>`
    ).join('');

    // Stats grid
    const sg = document.getElementById('stats-grid');
    sg.innerHTML = `
      <div class="stat-card"><label>Paramètres</label><span class="val">${data.params_M}M</span></div>
      <div class="stat-card"><label>d_model</label><span class="val">${data.d_model}</span></div>
      <div class="stat-card"><label>Blocs</label><span class="val">${data.n_blocks}</span></div>
      <div class="stat-card"><label>Vocab</label><span class="val">${data.vocab_size}</span></div>
      <div class="stat-card"><label>Mémoires</label><span class="val">${data.knowledge_entries}</span></div>
      <div class="stat-card"><label>Outils</label><span class="val">${(data.tools||[]).length}</span></div>
    `;
    document.getElementById('status-dot').style.background = 'var(--green)';
  } catch(e) {
    document.getElementById('status-dot').style.background = 'var(--red)';
  }
}

async function saveMemory() {
  const r = await fetch('/api/memory/save',{method:'POST'});
  const data = await r.json();
  alert(`Mémoire sauvegardée: ${data.entries} entrées → ${data.saved}`);
}

function escapeHtml(t) {
  return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Init ──────────────────────────────────────────────────────────────────────
connectWS();
loadStatus();
// Init all tab-contents as hidden except first
document.querySelectorAll('.tab-content').forEach((el,i) => { if(i>0) el.style.display='none'; });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(_HTML)


# ─────────────────────────────────────────────────────────────────────────────
# Web Explorer & Test-Time Learning (TTL) routes
# ─────────────────────────────────────────────────────────────────────────────

try:
    from nfn.web_explorer import WebExplorer as _WebExplorerClass
    _explorer = _WebExplorerClass()
except Exception:
    _explorer = None  # type: ignore[assignment]

_online_learner: Optional["OnlineLearner"] = None  # noqa: F821
_explore_clients: List[WebSocket] = []


# ── Request models ────────────────────────────────────────────────────────────

class ExploreUrlRequest(BaseModel):
    url:   str
    adapt: bool = True

class ExploreTextRequest(BaseModel):
    text: str

class TTLEnableRequest(BaseModel):
    adapter_rank: int   = 8
    online_lr:    float = 2e-4
    n_steps:      int   = 4
    ppl_gate:     float = 30.0


# ── /api/explore/url ─────────────────────────────────────────────────────────

@app.post("/api/explore/url")
async def explore_url(req: ExploreUrlRequest):
    if _explorer is None:
        return JSONResponse({"error": "WebExplorer not available"}, status_code=503)

    try:
        page = await asyncio.to_thread(_explorer.fetch, req.url)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    result = {
        "url":          page.get("url", req.url),
        "title":        page.get("title", ""),
        "n_chars":      page.get("n_chars", 0),
        "ppl_before":   None,
        "ppl_after":    None,
        "skipped":      True,
        "text_preview": page.get("text", "")[:200],
        "error":        page.get("error"),
    }

    if req.adapt and _online_learner is not None and not page.get("error"):
        text = page.get("text", "")
        if text:
            try:
                stats = await asyncio.to_thread(_online_learner.adapt_from_text, text)
                result["ppl_before"] = stats.get("ppl")
                result["skipped"]    = stats.get("skipped", True)
                result["ppl_after"]  = stats.get("loss")   # proxy
            except Exception as e:
                result["adapt_error"] = str(e)

    return JSONResponse(result)


# ── /api/explore/text ────────────────────────────────────────────────────────

@app.post("/api/explore/text")
async def explore_text(req: ExploreTextRequest):
    if _online_learner is None:
        return JSONResponse({"error": "TTL not enabled — call /api/ttl/enable first"}, status_code=400)
    try:
        stats = await asyncio.to_thread(_online_learner.adapt_from_text, req.text)
        return JSONResponse(stats)
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


# ── /ws/explore ──────────────────────────────────────────────────────────────

@app.websocket("/ws/explore")
async def ws_explore(ws: WebSocket):
    """
    Client sends:
      {"seed_url": str, "n_pages": int, "keywords": [str]}

    Server streams:
      {"type": "page",  "url", "title", "n_chars", "ppl_before", "ppl_after", "skipped"}
      {"type": "done",  "n_pages", "total_chars"}
      {"type": "error", "url", "error"}
    """
    await ws.accept()
    _explore_clients.append(ws)
    try:
        data = await ws.receive_json()
        seed_url = data.get("seed_url", "")
        n_pages  = int(data.get("n_pages", 5))
        keywords = data.get("keywords", [])

        if not seed_url:
            await ws.send_json({"type": "error", "url": "", "error": "seed_url is required"})
            return

        if _explorer is None:
            await ws.send_json({"type": "error", "url": seed_url, "error": "WebExplorer not available"})
            return

        total_chars = 0
        pages_done  = 0
        visited: set = set()

        # Generator runs in a thread; we iterate asynchronously
        def _run_explore():
            return list(_explorer.explore(seed_url, n_pages=n_pages,
                                          keywords=keywords, visited=visited))

        pages = await asyncio.to_thread(_run_explore)

        for page in pages:
            if page.get("error"):
                await ws.send_json({
                    "type":  "error",
                    "url":   page.get("url", ""),
                    "error": page["error"],
                })
                continue

            ppl_before = None
            ppl_after  = None
            skipped    = True

            if _online_learner is not None:
                text = page.get("text", "")
                if text:
                    try:
                        stats = await asyncio.to_thread(
                            _online_learner.adapt_from_text, text
                        )
                        ppl_before = stats.get("ppl")
                        skipped    = stats.get("skipped", True)
                        ppl_after  = stats.get("loss")
                    except Exception:
                        pass

            total_chars += page.get("n_chars", 0)
            pages_done  += 1

            await ws.send_json({
                "type":      "page",
                "url":       page.get("url", ""),
                "title":     page.get("title", ""),
                "n_chars":   page.get("n_chars", 0),
                "ppl_before": ppl_before,
                "ppl_after":  ppl_after,
                "skipped":   skipped,
            })

        await ws.send_json({
            "type":        "done",
            "n_pages":     pages_done,
            "total_chars": total_chars,
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "url": "", "error": str(e)})
        except Exception:
            pass
    finally:
        try:
            _explore_clients.remove(ws)
        except ValueError:
            pass


# ── /api/ttl/stats ───────────────────────────────────────────────────────────

@app.get("/api/ttl/stats")
async def ttl_stats():
    if _online_learner is None:
        return JSONResponse({"enabled": False})
    s = _online_learner.stats()
    s["enabled"] = True
    return JSONResponse(s)


# ── /api/ttl/enable ──────────────────────────────────────────────────────────

@app.post("/api/ttl/enable")
async def ttl_enable(req: TTLEnableRequest):
    global _online_learner
    try:
        from nfn.online_learner import OnlineLearner
        engine = get_engine()
        _online_learner = OnlineLearner(
            engine.model,
            engine.tokenizer,
            adapter_rank = req.adapter_rank,
            online_lr    = req.online_lr,
            n_steps      = req.n_steps,
            ppl_gate     = req.ppl_gate,
        )
        return JSONResponse({"enabled": True, **_online_learner.stats()})
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


# ── /api/ttl/disable ─────────────────────────────────────────────────────────

@app.post("/api/ttl/disable")
async def ttl_disable():
    global _online_learner
    _online_learner = None
    return JSONResponse({"enabled": False})


# ── /api/ttl/reset ───────────────────────────────────────────────────────────

@app.post("/api/ttl/reset")
async def ttl_reset():
    if _online_learner is None:
        return JSONResponse({"error": "TTL not enabled"}, status_code=400)
    try:
        _online_learner.reset()
        return JSONResponse({"reset": True, **_online_learner.stats()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
