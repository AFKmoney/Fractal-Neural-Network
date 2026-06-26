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

# Single-threaded BLAS — prevents ~100ms thread-spawn overhead per numpy matmul.
# Must come before any numpy/torch import.
import os as _os
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "interface" / "static"
sys.path.insert(0, str(ROOT))

from nfn.config import FNNConfig
from nfn.tokenizer import NFNTokenizer
from nfn.model import build_fnn_model
from inference.engine import AGIInferenceEngine
from training.agi_trainer import AGITrainer
from interface.agents import ChatAgent, ThinkAgent, ToolAgent, LearnAgent, CodeAgent, ReasoningAgent
from interface.remote_trainer import RemoteTrainer

_remote_trainer = RemoteTrainer()


# ─────────────────────────────────────────────────────────────────────────────
# App & CORS
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="NFN AGI Interface", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve CSS / JS from interface/static/
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
    """Initialise a fast CPU-optimised model using numpy inference."""
    global _engine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = NFNTokenizer()

    # Build the smallest viable model. All optional AGI features are disabled
    # so the AGIBlock reduces to a plain FNNBlock call. The numpy
    # inference patch then bypasses PyTorch entirely for ~70× speedup on CPU.
    gpu = device.type == "cuda"
    model = build_fnn_model(
        vocab_size            = tokenizer.vocab_size,
        d_model               = 128,
        n_blocks              = 2,
        use_reasoning         = False,
        use_predictive_coding = False,
        use_free_energy       = False,
        use_self_consistency  = False,
        use_plan_executor     = False,
        use_mod               = False,
        use_mtp               = False,
        use_hyper             = False,
        use_memory            = False,
        use_working_memory    = False,
        use_causal            = False,
        use_goal              = False,
        use_bayesian          = False,
    ).to(device)

    # On CPU, apply numpy inference patch to bypass PyTorch's ~70ms/op overhead.
    # On CUDA the standard PyTorch path is already fast.
    if device.type == "cpu":
        from inference.fast_numpy import apply_numpy_patch
        apply_numpy_patch(model)
        print("[NFN AGI] NumPy inference patch applied (CPU fast path).")

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
    task:          Optional[str] = None
    goal:          Optional[str] = None   # JS alias for task
    system:        Optional[str] = None
    max_tokens:    int   = 1024
    max_steps:     int   = 5
    temperature:   float = 0.7

class CodeRequest(BaseModel):
    code:        str   = ""
    instruction: str   = ""
    language:    str   = "python"
    max_tokens:  int   = 512
    temperature: float = 0.4
    task:        str   = "complete"   # complete | explain | refactor | generate

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
    config_name:      str   = ""    # ignored — always trains current model

class LoadModelRequest(BaseModel):
    checkpoint_path: Optional[str] = None
    path:            Optional[str] = None   # JS alias
    device:          str = "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    # Run blocking model init in a thread so the event loop stays responsive
    await asyncio.to_thread(_init_default_model)


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
    agent = ChatAgent(engine, use_rag=req.use_rag,
                      think_rounds=req.think_rounds, use_tools=req.use_tools)
    try:
        import functools
        result = await asyncio.to_thread(functools.partial(
            agent.reply, req.messages,
            system=req.system, max_tokens=req.max_tokens,
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
        ))
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
        import functools
        result = await asyncio.to_thread(functools.partial(
            agent.think_and_answer, req.question,
            n_rounds=req.n_rounds, max_answer_tokens=req.max_answer_tokens,
            temperature=req.temperature,
        ))
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
    task   = req.task or req.goal or ""
    try:
        import functools
        result = await asyncio.to_thread(functools.partial(
            agent.run, task,
            system=req.system, max_new_tokens=req.max_tokens, temperature=req.temperature,
        ))
        # Normalise to what the JS expects: {steps, final_answer}
        if "result" in result and "steps" not in result:
            result["steps"] = [{"step": 1, "thought": result["result"], "final": True}]
            result["final_answer"] = result["result"]
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


@app.post("/api/code")
async def code_endpoint(req: CodeRequest):
    engine = get_engine()
    agent  = CodeAgent(engine)
    try:
        import functools
        task = req.task or ("generate" if not req.code else "complete")
        if task == "generate":
            desc = req.instruction or req.code
            fn = functools.partial(agent.generate, desc,
                                   language=req.language, max_tokens=req.max_tokens)
        elif task == "explain":
            fn = functools.partial(agent.explain, req.code,
                                   language=req.language, max_tokens=req.max_tokens)
        elif task == "refactor":
            # Use generate with a refactor instruction
            desc = f"Refactor this {req.language} code{': ' + req.instruction if req.instruction else ''}.\n\n```{req.language}\n{req.code}\n```"
            fn = functools.partial(agent.generate, desc,
                                   language=req.language, max_tokens=req.max_tokens)
        else:  # complete
            fn = functools.partial(agent.complete, req.code,
                                   language=req.language, max_tokens=req.max_tokens)
        result = await asyncio.to_thread(fn)
        return JSONResponse({"result": result})
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


# ─────────────────────────────────────────────────────────────────────────────
# Generate (raw)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    engine = get_engine()
    try:
        import functools
        text = await asyncio.to_thread(functools.partial(
            engine.generate, req.prompt,
            max_new_tokens=req.max_tokens, temperature=req.temperature,
            top_k=req.top_k, top_p=req.top_p, greedy=req.greedy,
            think_rounds=req.think_rounds, use_speculative=req.speculative,
        ))
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

@app.post("/api/save_model")
async def save_model_endpoint():
    engine = get_engine()
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / "nfn_saved.pt")
    ks_path   = str(ckpt_dir / "knowledge_store.json.gz")
    try:
        # Save model weights
        ckpt = {
            "model_state": engine.model.state_dict(),
            "cfg":         engine.cfg.__dict__,
        }
        await asyncio.to_thread(torch.save, ckpt, ckpt_path)
        # Save knowledge store separately
        engine.learner.save_store(ks_path)
        return JSONResponse({
            "path":    ckpt_path,
            "ks_path": ks_path,
            "entries": len(engine.learner.store),
        })
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


@app.post("/api/load_model")
async def load_model(req: LoadModelRequest):
    global _engine
    device = torch.device(req.device)
    ckpt_path = req.checkpoint_path or req.path or ""
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg  = FNNConfig.from_dict(ckpt["cfg"])
        model = build_fnn_model(
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
        return JSONResponse({"loaded": ckpt_path, **_engine.status()})
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
        mode = data.get("mode", "generate")

        if mode == "chat":
            import functools
            messages = data.get("messages", [])
            agent = ChatAgent(engine, use_rag=data.get("use_rag", True),
                              think_rounds=data.get("think_rounds", 0))
            result = await asyncio.to_thread(functools.partial(
                agent.reply, messages,
                max_tokens=data.get("max_tokens", 512),
                temperature=data.get("temperature", 0.8),
                top_k=data.get("top_k", 50),
                top_p=data.get("top_p", 0.95),
            ))
            reply = result.get("reply", "")
            # Stream token by token
            words = reply.split(" ")
            for i, w in enumerate(words):
                text = ("" if i == 0 else " ") + w
                await ws.send_json({"type": "token", "text": text})
            await ws.send_json({"type": "end", "n_tokens": len(words)})
        else:
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
                await ws.send_json({"type": "token", "text": token})
                n_tokens += 1
            await ws.send_json({"type": "end", "n_tokens": n_tokens})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "text": str(e)})
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
            import functools
            result = await asyncio.to_thread(functools.partial(
                agent.reply, history,
                max_tokens=data.get("max_tokens", 512),
                temperature=data.get("temperature", 0.8),
            ))
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


def _normalize_metrics(m: Dict) -> Dict:
    """Normalize trainer metric keys to the format the JS chart expects."""
    loss = m.get("loss", m.get("lm", m.get("total", 0.0)))
    loss_phase = m.get("loss_phase", m.get("phase", m.get("soliton", None)))
    out = dict(m)
    out["loss"] = loss
    if loss_phase is not None:
        out["loss_phase"] = loss_phase
    return out


@app.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    """Push training metrics to WebSocket clients in real time."""
    await ws.accept()
    _ws_clients.append(ws)
    last_sent = 0
    try:
        while True:
            if len(_train_metrics) > last_sent:
                for m in _train_metrics[last_sent:]:
                    await ws.send_json({"type": "metrics", "data": _normalize_metrics(m)})
                last_sent = len(_train_metrics)
            # If training finished, close the WS so the JS shows "Done"
            running = _train_thread is not None and _train_thread.is_alive()
            if not running and last_sent > 0 and last_sent == len(_train_metrics):
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


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
  .tabs { display: flex; border-bottom: 1px solid var(--border); background: var(--surface); overflow-x: auto; }
  .tab { padding: 10px 20px; font-size: 0.8rem; cursor: pointer; color: var(--muted); border-bottom: 2px solid transparent; transition: all 0.15s; white-space: nowrap; }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab-content { display: none; flex: 1; flex-direction: column; overflow: hidden; }
  .tab-content.active { display: flex; }

  /* Remote training tab */
  .remote-panel { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
  .remote-section { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
  .remote-section h2 { font-size: 0.75rem; color: var(--accent2); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
  .form-row { display: flex; gap: 8px; flex-wrap: wrap; }
  .form-row input, .form-row select { flex: 1; min-width: 120px; }
  .form-col { display: flex; flex-direction: column; gap: 4px; }
  .form-col label { font-size: 0.72rem; color: var(--muted); }
  .dtabs { display: flex; gap: 4px; margin-bottom: 8px; }
  .dtab { padding: 4px 12px; font-size: 0.75rem; border-radius: 4px; cursor: pointer; background: var(--border); color: var(--muted); border: none; font-family: inherit; }
  .dtab.active { background: var(--accent); color: #fff; }
  .log-output { flex: 1; background: #0d0d14; border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 0.72rem; line-height: 1.5; overflow-y: auto; max-height: 320px; white-space: pre-wrap; word-break: break-all; color: #a0aec0; }
  .metrics-bar { display: flex; gap: 16px; padding: 8px 12px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; font-size: 0.75rem; flex-wrap: wrap; }
  .metrics-bar span { color: var(--muted); }
  .metrics-bar span b { color: var(--accent2); }
  .ckpt-list { display: flex; flex-direction: column; gap: 6px; }
  .ckpt-row { display: flex; align-items: center; gap: 10px; padding: 6px 10px; background: var(--bg); border-radius: 6px; font-size: 0.78rem; }
  .ckpt-row button { margin-left: auto; padding: 3px 10px; font-size: 0.72rem; }
  .hf-result { padding: 6px 10px; background: var(--bg); border-radius: 6px; cursor: pointer; font-size: 0.78rem; display: flex; flex-direction: column; gap: 2px; border: 1px solid transparent; }
  .hf-result:hover, .hf-result.selected { border-color: var(--accent); }
  .hf-result .hf-id { color: var(--accent2); font-weight: bold; }
  .hf-result .hf-desc { color: var(--muted); font-size: 0.7rem; }
  .conn-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; }
  .conn-badge.connected { background: #10b98122; color: var(--green); }
  .conn-badge.disconnected { background: #ef444422; color: var(--red); }
  .conn-badge.training { background: #7c3aed22; color: var(--accent); }
  .conn-badge.setting_up { background: #f59e0b22; color: var(--yellow); }

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
      <div class="tab" onclick="switchTab('remote')">🖥 Remote Train</div>
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

    <!-- Remote Train tab -->
    <div class="tab-content" id="tab-remote">
      <div class="remote-panel">

        <!-- 1. Connect -->
        <div class="remote-section">
          <h2>1. Connect to GPU &nbsp; <span id="conn-badge" class="conn-badge disconnected">disconnected</span></h2>
          <div class="form-row">
            <div class="form-col" style="flex:2">
              <label>Host / IP</label>
              <input type="text" id="ssh-host" placeholder="123.45.67.89 or hostname">
            </div>
            <div class="form-col" style="flex:0 0 70px">
              <label>Port</label>
              <input type="number" id="ssh-port" value="22">
            </div>
            <div class="form-col" style="flex:1">
              <label>Username</label>
              <input type="text" id="ssh-user" placeholder="root">
            </div>
          </div>
          <div class="form-col">
            <label>Password (leave empty if using SSH key below)</label>
            <input type="password" id="ssh-password" placeholder="Password">
          </div>
          <div class="form-col">
            <label>SSH Private Key (paste full key — -----BEGIN ... PRIVATE KEY-----)</label>
            <textarea id="ssh-key" rows="4" placeholder="-----BEGIN OPENSSH PRIVATE KEY-----&#10;...&#10;-----END OPENSSH PRIVATE KEY-----&#10;&#10;Leave empty if using password above" style="font-size:0.7rem;line-height:1.4;resize:vertical"></textarea>
          </div>
          <div style="display:flex;gap:8px;align-items:center">
            <button class="btn" id="connect-btn" onclick="remoteConnect()">🔌 Connect</button>
            <button class="btn secondary" id="setup-btn" onclick="remoteSetup()" disabled>⚙ Install / Update Repo</button>
            <span id="connect-msg" style="font-size:0.78rem;color:var(--muted)"></span>
          </div>
        </div>

        <!-- 2. Dataset -->
        <div class="remote-section">
          <h2>2. Dataset</h2>
          <div class="dtabs">
            <button class="dtab active" id="dtab-builtin-btn" onclick="switchDTab('builtin')">Built-in</button>
            <button class="dtab" id="dtab-hf-btn" onclick="switchDTab('hf')">HuggingFace Search</button>
            <button class="dtab" id="dtab-hfid-btn" onclick="switchDTab('hfid')">HuggingFace ID</button>
          </div>

          <div id="dtab-builtin" class="dtab-pane">
            <select id="builtin-dataset" style="width:100%">
              <option value="tiny-shakespeare">tiny-shakespeare — 1 MB, Shakespeare (quick test)</option>
              <option value="gutenberg-top100">gutenberg-top100 — 20 MB, classic books</option>
              <option value="wikipedia-en-simple" selected>wikipedia-en-simple — 120 MB, recommended</option>
              <option value="openwebtext-10pct">openwebtext-10pct — 2 GB, web language (needs HF library)</option>
              <option value="cc-news">cc-news — 1 GB, news articles (needs HF library)</option>
              <option value="wikipedia-en">wikipedia-en — 20 GB, full English Wikipedia</option>
              <option value="pile-10pct">pile-10pct — 8 GB, diverse high-quality (needs HF library)</option>
            </select>
          </div>

          <div id="dtab-hf" class="dtab-pane" style="display:none">
            <div style="display:flex;gap:8px;margin-bottom:8px">
              <input type="text" id="hf-search-input" placeholder="Search HuggingFace (e.g. wikipedia, openwebtext, bookcorpus)" style="flex:1">
              <button class="btn secondary" onclick="hfSearch()">Search</button>
            </div>
            <div id="hf-results" style="display:flex;flex-direction:column;gap:4px;max-height:200px;overflow-y:auto"></div>
            <div id="hf-selected-display" style="display:none;margin-top:8px;padding:6px 10px;background:var(--bg);border-radius:6px;font-size:0.78rem;color:var(--green)">
              Selected: <b id="hf-selected-id"></b>
            </div>
          </div>

          <div id="dtab-hfid" class="dtab-pane" style="display:none">
            <div class="form-col">
              <label>HuggingFace dataset ID (e.g. <code>wikipedia</code>, <code>allenai/c4</code>, <code>bookcorpus</code>)</label>
              <input type="text" id="hf-direct-id" placeholder="dataset-owner/dataset-name">
            </div>
          </div>
        </div>

        <!-- 3. Config -->
        <div class="remote-section">
          <h2>3. Training Config</h2>
          <div class="form-row">
            <div class="form-col">
              <label>Model size</label>
              <select id="rc-config">
                <option value="nano">nano  — ~3M params,  &lt;1GB VRAM</option>
                <option value="small" selected>small — ~15M params, ~4GB VRAM</option>
                <option value="medium">medium — ~85M params, ~12GB VRAM</option>
                <option value="large">large  — ~350M params, ~40GB VRAM</option>
              </select>
            </div>
            <div class="form-col">
              <label>Batch size</label>
              <input type="number" id="rc-batch" value="8" min="1" max="128" style="width:80px">
            </div>
            <div class="form-col">
              <label>Seq length</label>
              <input type="number" id="rc-seqlen" value="512" min="64" max="4096" step="64" style="width:90px">
            </div>
            <div class="form-col">
              <label>Learning rate</label>
              <input type="text" id="rc-lr" value="2e-4" style="width:80px">
            </div>
            <div class="form-col">
              <label>Epochs</label>
              <input type="number" id="rc-epochs" value="3" min="1" max="100" style="width:70px">
            </div>
            <div class="form-col">
              <label>Max chars</label>
              <input type="number" id="rc-maxchars" value="30000000" min="100000" step="1000000" style="width:120px">
            </div>
          </div>
        </div>

        <!-- 4. Controls -->
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <button class="btn" id="rt-start-btn" onclick="remoteStart()" disabled style="background:var(--accent)">▶ Start Training</button>
          <button class="btn secondary" id="rt-stop-btn" onclick="remoteStop()" disabled>⏹ Stop &amp; Save Checkpoint</button>
          <button class="btn secondary" id="rt-ckpts-btn" onclick="loadCheckpoints()" disabled>📥 List Checkpoints</button>
          <span id="rt-status" style="font-size:0.75rem;color:var(--muted)"></span>
        </div>

        <!-- Live metrics -->
        <div class="metrics-bar" id="rt-metrics" style="display:none">
          <span>Step <b id="rm-step">—</b></span>
          <span>LM loss <b id="rm-lm">—</b></span>
          <span>PPL <b id="rm-ppl">—</b></span>
          <span>Phase <b id="rm-phase">—</b></span>
          <span>Grad norm <b id="rm-gn">—</b></span>
          <span>LR <b id="rm-lr">—</b></span>
        </div>

        <!-- Log output -->
        <div class="log-output" id="rt-log">Waiting to connect…</div>

        <!-- Checkpoints -->
        <div class="remote-section" id="rt-ckpts-section" style="display:none">
          <h2>Checkpoints</h2>
          <div class="ckpt-list" id="rt-ckpts-list"></div>
        </div>

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
  const tabs = ['chat','think','memory','learn','remote'];
  const idx = tabs.indexOf(name);
  document.querySelectorAll('.tab')[idx].classList.add('active');
  const el = document.getElementById('tab-'+name);
  el.classList.add('active'); el.style.display='flex';
  if (name === 'memory') loadStatus();
  if (name === 'remote') remoteStatusPoll();
}

// ── Remote Train ─────────────────────────────────────────────────────────────
let _remoteWs = null;
let _hfSelected = '';
let _datasetMode = 'builtin';
let _rtLogLines = 0;
let _rtPollInterval = null;

function switchDTab(name) {
  ['builtin','hf','hfid'].forEach(n => {
    document.getElementById('dtab-'+n).style.display = n===name ? '' : 'none';
    document.getElementById('dtab-'+n+'-btn').classList.toggle('active', n===name);
  });
  _datasetMode = name;
}

function remoteConnect() {
  const btn = document.getElementById('connect-btn');
  const msg = document.getElementById('connect-msg');
  btn.disabled = true; msg.textContent = 'Connecting…';
  fetch('/api/remote/connect', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      host:     document.getElementById('ssh-host').value.trim(),
      user:     document.getElementById('ssh-user').value.trim(),
      port:     parseInt(document.getElementById('ssh-port').value) || 22,
      password: document.getElementById('ssh-password').value,
      key_data: document.getElementById('ssh-key').value,
    })
  }).then(r=>r.json()).then(d => {
    btn.disabled = false;
    if (d.ok) {
      msg.textContent = `✓ ${d.hostname} — ${d.gpu}`;
      msg.style.color = 'var(--green)';
      setBadge('connected');
      document.getElementById('setup-btn').disabled = false;
      document.getElementById('rt-start-btn').disabled = false;
      document.getElementById('rt-ckpts-btn').disabled = false;
      rtLog('Connected to ' + d.hostname + ' — GPU: ' + d.gpu);
    } else {
      msg.textContent = '✗ ' + d.error;
      msg.style.color = 'var(--red)';
      setBadge('disconnected');
    }
  }).catch(e => { btn.disabled=false; msg.textContent='Error: '+e; msg.style.color='var(--red)'; });
}

function remoteSetup() {
  const btn = document.getElementById('setup-btn');
  btn.disabled = true;
  setBadge('setting_up');
  rtLog('Setting up repo on remote machine…');
  fetch('/api/remote/setup', {method:'POST'})
    .then(r=>r.json()).then(d => {
      btn.disabled = false;
      if (d.ok) {
        rtLog('✓ Setup complete');
        setBadge('connected');
      } else {
        rtLog('✗ Setup failed: ' + d.error);
        setBadge('connected');
      }
      (d.log||[]).forEach(l => rtLog(l));
    });
}

function remoteStart() {
  const dataset = _datasetMode==='hf' ? '' : (_datasetMode==='hfid' ? '' : document.getElementById('builtin-dataset').value);
  const hfDataset = _datasetMode==='hf' ? _hfSelected : (_datasetMode==='hfid' ? document.getElementById('hf-direct-id').value.trim() : '');
  if (_datasetMode!=='builtin' && !hfDataset) { rtLog('⚠ Select or enter a HuggingFace dataset first'); return; }

  document.getElementById('rt-start-btn').disabled = true;
  document.getElementById('rt-stop-btn').disabled = false;
  document.getElementById('rt-metrics').style.display = 'flex';
  setBadge('training');
  rtLog('Starting training…');

  fetch('/api/remote/start', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      dataset:    dataset,
      hf_dataset: hfDataset,
      config:     document.getElementById('rc-config').value,
      batch:      parseInt(document.getElementById('rc-batch').value),
      seq_len:    parseInt(document.getElementById('rc-seqlen').value),
      lr:         parseFloat(document.getElementById('rc-lr').value),
      epochs:     parseInt(document.getElementById('rc-epochs').value),
      max_chars:  parseInt(document.getElementById('rc-maxchars').value),
    })
  }).then(r=>r.json()).then(d => {
    if (d.ok) {
      rtLog('✓ Training started in tmux on remote GPU');
      remoteOpenWs();
    } else {
      rtLog('✗ Start failed: ' + d.error);
      document.getElementById('rt-start-btn').disabled = false;
      document.getElementById('rt-stop-btn').disabled = true;
      setBadge('connected');
    }
  });
}

function remoteStop() {
  rtLog('Stopping… (saving checkpoint)');
  document.getElementById('rt-stop-btn').disabled = true;
  fetch('/api/remote/stop', {method:'POST'}).then(r=>r.json()).then(d => {
    rtLog(d.ok ? '✓ Stopped — checkpoint saved' : '✗ ' + d.error);
    document.getElementById('rt-start-btn').disabled = false;
    setBadge('connected');
    loadCheckpoints();
  });
}

function remoteOpenWs() {
  if (_remoteWs) _remoteWs.close();
  const proto = location.protocol==='https:' ? 'wss:' : 'ws:';
  _remoteWs = new WebSocket(`${proto}//${location.host}/ws/remote`);
  _remoteWs.onmessage = (e) => {
    const d = JSON.parse(e.data);
    (d.lines||[]).forEach(l => rtLog(l));
    if (d.metrics && d.metrics.step) updateMetrics(d.metrics);
    if (d.status === 'done') {
      rtLog('✓ Training complete');
      document.getElementById('rt-start-btn').disabled = false;
      document.getElementById('rt-stop-btn').disabled = true;
      setBadge('connected');
      loadCheckpoints();
    }
  };
  _remoteWs.onclose = () => {};
}

function remoteStatusPoll() {
  fetch('/api/remote/status').then(r=>r.json()).then(d => {
    setBadge(d.status);
    if (d.last_metrics && d.last_metrics.step) updateMetrics(d.last_metrics);
  }).catch(()=>{});
}

function updateMetrics(m) {
  if (m.step  !== undefined) document.getElementById('rm-step').textContent  = m.step;
  if (m.lm    !== undefined) document.getElementById('rm-lm').textContent    = (+m.lm).toFixed(4);
  if (m.ppl   !== undefined) document.getElementById('rm-ppl').textContent   = (+m.ppl).toFixed(1);
  if (m.phase !== undefined) document.getElementById('rm-phase').textContent = m.phase;
  if (m.gn    !== undefined) document.getElementById('rm-gn').textContent    = (+m.gn).toFixed(3);
  if (m.lr    !== undefined) document.getElementById('rm-lr').textContent    = m.lr;
  document.getElementById('rt-metrics').style.display = 'flex';
}

function rtLog(line) {
  const el = document.getElementById('rt-log');
  if (el.textContent === 'Waiting to connect…') el.textContent = '';
  el.textContent += line + '\n';
  el.scrollTop = el.scrollHeight;
}

function setBadge(status) {
  const b = document.getElementById('conn-badge');
  b.className = 'conn-badge ' + status;
  b.textContent = status;
  document.getElementById('rt-status').textContent = status;
}

function hfSearch() {
  const q = document.getElementById('hf-search-input').value.trim();
  if (!q) return;
  const res = document.getElementById('hf-results');
  res.innerHTML = '<span style="color:var(--muted);font-size:0.75rem">Searching…</span>';
  fetch('/api/hf/search?q=' + encodeURIComponent(q) + '&limit=8')
    .then(r=>r.json()).then(d => {
      if (!d.results || !d.results.length) { res.innerHTML = '<span style="color:var(--muted);font-size:0.75rem">No results</span>'; return; }
      res.innerHTML = '';
      d.results.forEach(ds => {
        const div = document.createElement('div');
        div.className = 'hf-result';
        div.innerHTML = `<span class="hf-id">${ds.id}</span><span class="hf-desc">${ds.description || ''} · ${(ds.downloads||0).toLocaleString()} downloads</span>`;
        div.onclick = () => {
          res.querySelectorAll('.hf-result').forEach(x => x.classList.remove('selected'));
          div.classList.add('selected');
          _hfSelected = ds.id;
          document.getElementById('hf-selected-display').style.display = '';
          document.getElementById('hf-selected-id').textContent = ds.id;
        };
        res.appendChild(div);
      });
    });
}

function loadCheckpoints() {
  fetch('/api/remote/checkpoints').then(r=>r.json()).then(d => {
    const sec = document.getElementById('rt-ckpts-section');
    const list = document.getElementById('rt-ckpts-list');
    if (!d.checkpoints || !d.checkpoints.length) { list.innerHTML = '<span style="color:var(--muted);font-size:0.78rem">No checkpoints found</span>'; sec.style.display=''; return; }
    list.innerHTML = '';
    d.checkpoints.forEach(ck => {
      const row = document.createElement('div');
      row.className = 'ckpt-row';
      row.innerHTML = `<span>${ck.name}</span><span style="color:var(--muted)">${ck.size_mb} MB</span>`;
      const btn = document.createElement('button');
      btn.className = 'btn secondary';
      btn.textContent = '⬇ Download';
      btn.onclick = () => { btn.textContent='Downloading…'; btn.disabled=true; window.location='/api/remote/download/'+ck.name; setTimeout(()=>{btn.textContent='⬇ Download';btn.disabled=false;},3000); };
      row.appendChild(btn);
      list.appendChild(row);
    });
    sec.style.display = '';
  });
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


@app.get("/", response_class=FileResponse)
async def root():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return HTMLResponse(_HTML)  # fallback to built-in UI


# ─────────────────────────────────────────────────────────────────────────────
# Web Explorer & Test-Time Learning (TTL) routes
# ─────────────────────────────────────────────────────────────────────────────

try:
    from interface.web_explorer import WebExplorer as _WebExplorerClass
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
        from training.online_learner import OnlineLearner
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


# ─────────────────────────────────────────────────────────────────────────────
# Remote Training — SSH + HuggingFace integration
# ─────────────────────────────────────────────────────────────────────────────

class RemoteConnectRequest(BaseModel):
    host:     str
    user:     str
    port:     int   = 22
    password: str   = ""
    key_data: str   = ""

class RemoteStartRequest(BaseModel):
    dataset:    str   = "wikipedia-en-simple"
    hf_dataset: str   = ""
    config:     str   = "small"
    batch:      int   = 8
    seq_len:    int   = 512
    lr:         float = 2e-4
    epochs:     int   = 3
    max_chars:  int   = 30_000_000


@app.post("/api/remote/connect")
async def remote_connect(req: RemoteConnectRequest):
    result = await asyncio.to_thread(
        _remote_trainer.connect,
        req.host, req.user, req.port, req.password, req.key_data,
    )
    return JSONResponse(result)


@app.post("/api/remote/setup")
async def remote_setup():
    logs: List[str] = []
    result = await asyncio.to_thread(_remote_trainer.setup, logs.append)
    return JSONResponse({**result, "log": logs})


@app.post("/api/remote/start")
async def remote_start(req: RemoteStartRequest):
    result = await asyncio.to_thread(
        _remote_trainer.start,
        req.dataset, req.hf_dataset, req.config,
        req.batch, req.seq_len, req.lr, req.epochs, req.max_chars,
    )
    return JSONResponse(result)


@app.post("/api/remote/stop")
async def remote_stop():
    result = await asyncio.to_thread(_remote_trainer.stop)
    return JSONResponse(result)


@app.get("/api/remote/status")
async def remote_status():
    return JSONResponse({
        "status":       _remote_trainer.status,
        "gpu":          _remote_trainer.gpu_info,
        "last_metrics": _remote_trainer.last_metrics,
    })


@app.get("/api/remote/logs")
async def remote_logs(since: int = 0):
    return JSONResponse({"lines": _remote_trainer.get_logs(since)})


@app.get("/api/remote/checkpoints")
async def remote_checkpoints():
    result = await asyncio.to_thread(_remote_trainer.list_checkpoints)
    return JSONResponse({"checkpoints": result})


@app.get("/api/remote/download/{name}")
async def remote_download(name: str):
    from fastapi.responses import FileResponse as _FR
    dest = str(ROOT / "checkpoints" / name)
    result = await asyncio.to_thread(_remote_trainer.download_checkpoint, name, dest)
    if not result.get("ok"):
        return JSONResponse(result, status_code=500)
    return _FR(dest, filename=name)


@app.get("/api/hf/search")
async def hf_search(q: str = "", limit: int = 8):
    results = await asyncio.to_thread(RemoteTrainer.search_hf, q, limit)
    return JSONResponse({"results": results})


@app.websocket("/ws/remote")
async def ws_remote(ws: WebSocket):
    """Stream remote training logs to the browser in real time."""
    await ws.accept()
    sent = 0
    try:
        while True:
            lines = _remote_trainer.get_logs(sent)
            if lines:
                await ws.send_json({
                    "lines":   lines,
                    "metrics": _remote_trainer.last_metrics,
                    "status":  _remote_trainer.status,
                })
                sent += len(lines)
            if _remote_trainer.status in ("done", "error", "disconnected"):
                await ws.send_json({"status": _remote_trainer.status, "lines": [], "metrics": {}})
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
