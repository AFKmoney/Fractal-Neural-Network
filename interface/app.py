"""
NFN Interface — FastAPI server.

Endpoints:
  GET  /                    → web UI
  GET  /api/status          → model info
  POST /api/chat            → single-turn chat (JSON)
  POST /api/generate        → raw text generation (JSON)
  POST /api/code            → code completion (JSON)
  POST /api/train/start     → start background training
  POST /api/train/stop      → stop training
  GET  /api/train/status    → training metrics
  POST /api/load_model      → load checkpoint
  POST /api/save_model      → save checkpoint
  WS   /ws/stream           → streaming generation (WebSocket)
  WS   /ws/train            → live training metrics (WebSocket)
  POST /api/agent/run       → run reasoning agent step
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
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Adjust path so we can import top-level packages ──────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nfn.config import NFNConfig
from nfn.network import NFNLanguageModel
from nfn.tokenizer import NFNTokenizer, load_tokenizer
from inference.engine import NFNInferenceEngine
from training.trainer import NFNTrainer
from interface.agents import ChatAgent, CodeAgent, ReasoningAgent


app = FastAPI(title="Neural Fractal Network", version="1.0.0")

# ── Static files ──────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Global state ─────────────────────────────────────────────────────────────

class AppState:
    model: Optional[NFNLanguageModel] = None
    tokenizer: Optional[NFNTokenizer] = None
    engine: Optional[NFNInferenceEngine] = None
    trainer: Optional[NFNTrainer] = None
    training_thread: Optional[threading.Thread] = None
    training_metrics: List[Dict] = []
    device: torch.device = torch.device("cpu")
    model_path: Optional[str] = None
    training_clients: List[WebSocket] = []

state = AppState()


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_default_model(config_name: str = "nano"):
    """Load (or create) the default NFN model."""
    cfg_path = ROOT / "configs" / f"{config_name}.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = NFNConfig.from_dict(json.load(f))
    else:
        cfg = NFNConfig()
        cfg.n_levels = 3
        cfg.n_blocks = 2
        cfg.d_model = 128
        cfg.d_ff = 512
        cfg.max_seq_len = 256
        cfg.n_time_steps = 4

    tok = NFNTokenizer()
    cfg.vocab_size = tok.vocab_size
    cfg.pad_token_id = tok.pad_token_id
    cfg.bos_token_id = tok.bos_token_id
    cfg.eos_token_id = tok.eos_token_id

    device = _detect_device()
    model = NFNLanguageModel(cfg).to(device)

    ckpt_path = ROOT / "checkpoints" / "nfn_final.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint from {ckpt_path}")

    engine = NFNInferenceEngine(model, tok, device)

    state.model = model
    state.tokenizer = tok
    state.engine = engine
    state.device = device

    print(f"NFN ready | params={model.param_summary()} | device={device}")
    return model


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    try:
        _load_default_model()
    except Exception as e:
        print(f"Warning: could not load model at startup: {e}")


# ── Root page ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = STATIC_DIR / "index.html"
    return html_path.read_text(encoding="utf-8")


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    system: str = "Tu es NFN, un assistant IA basé sur le Neural Fractal Network."
    max_tokens: int = 400
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    strategy: str = "top_p"

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 300
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    strategy: str = "top_p"

class CodeRequest(BaseModel):
    code: str
    instruction: str = ""
    language: str = "python"
    max_tokens: int = 500
    temperature: float = 0.4

class TrainRequest(BaseModel):
    text: str
    n_epochs: int = 3
    seq_len: int = 128
    batch_size: int = 2
    lr: float = 3e-4
    config_name: str = "nano"

class AgentRequest(BaseModel):
    goal: str
    history: List[Dict] = []
    max_steps: int = 5

class LoadModelRequest(BaseModel):
    path: str
    config_name: Optional[str] = None


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    if state.model is None:
        return {"status": "no_model", "model": None}
    return {
        "status": "ready",
        "model": {
            "params": state.model.param_summary(),
            "n_params": state.model.n_params(),
            "vocab_size": state.model.cfg.vocab_size,
            "d_model": state.model.cfg.d_model,
            "n_levels": state.model.cfg.n_levels,
            "n_blocks": state.model.cfg.n_blocks,
            "motifs": state.model.cfg.motifs,
            "max_seq_len": state.model.cfg.max_seq_len,
            "device": str(state.device),
        },
        "training": _training_status(),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if state.engine is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})
    try:
        reply = state.engine.chat(
            req.messages,
            system=req.system,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            strategy=req.strategy,
        )
        return {"reply": reply, "tokens": len(reply)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if state.engine is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})
    try:
        text = state.engine.generate(
            req.prompt, max_new_tokens=req.max_tokens,
            temperature=req.temperature, top_k=req.top_k,
            top_p=req.top_p, strategy=req.strategy,
        )
        return {"text": text, "prompt": req.prompt}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/code")
async def code_complete(req: CodeRequest):
    if state.engine is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})
    agent = CodeAgent(state.engine)
    result = agent.complete(req.code, req.instruction, req.language, req.max_tokens, req.temperature)
    return result


@app.post("/api/train/start")
async def train_start(req: TrainRequest):
    if state.model is None:
        # Create a fresh model
        _load_default_model(req.config_name)

    if state.training_thread and state.training_thread.is_alive():
        return {"error": "Training already running"}

    state.training_metrics = []

    def step_cb(metrics: Dict):
        state.training_metrics.append(metrics)
        # Broadcast to WS clients
        msg = json.dumps({"type": "metrics", "data": metrics})
        for ws in list(state.training_clients):
            asyncio.run_coroutine_threadsafe(
                _safe_ws_send(ws, msg), asyncio.get_event_loop()
            )

    trainer = NFNTrainer(
        state.model, state.tokenizer, state.model.cfg,
        lr=req.lr,
        output_dir=str(ROOT / "checkpoints"),
        step_callback=step_cb,
    )
    state.trainer = trainer

    def run_training():
        try:
            trainer.train(
                req.text,
                n_epochs=req.n_epochs,
                seq_len=req.seq_len,
                batch_size=req.batch_size,
            )
        except Exception as e:
            print(f"Training error: {e}")
            traceback.print_exc()

    state.training_thread = threading.Thread(target=run_training, daemon=True)
    state.training_thread.start()
    return {"status": "started"}


@app.post("/api/train/stop")
async def train_stop():
    if state.trainer:
        state.trainer.stop()
    return {"status": "stopping"}


@app.get("/api/train/status")
async def train_status():
    return _training_status()


@app.post("/api/load_model")
async def load_model(req: LoadModelRequest):
    try:
        device = _detect_device()
        if not os.path.isabs(req.path):
            req.path = str(ROOT / req.path)
        ckpt = torch.load(req.path, map_location=device)
        cfg = NFNConfig.from_dict(ckpt["cfg"])
        model = NFNLanguageModel(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        tok = NFNTokenizer()
        engine = NFNInferenceEngine(model, tok, device)
        state.model = model
        state.tokenizer = tok
        state.engine = engine
        state.device = device
        state.model_path = req.path
        return {"status": "ok", "params": model.param_summary()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/save_model")
async def save_model():
    if state.model is None:
        return JSONResponse(status_code=503, content={"error": "No model"})
    path = ROOT / "checkpoints" / "nfn_manual_save.pt"
    path.parent.mkdir(exist_ok=True)
    torch.save({
        "model_state": state.model.state_dict(),
        "cfg": state.model.cfg.to_dict(),
        "step": getattr(state.trainer, "step", 0),
    }, path)
    return {"path": str(path)}


@app.post("/api/agent/run")
async def agent_run(req: AgentRequest):
    if state.engine is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})
    agent = ReasoningAgent(state.engine)
    result = await agent.run(req.goal, req.history, req.max_steps)
    return result


# ── WebSocket streaming ───────────────────────────────────────────────────────

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            req = json.loads(data)
            prompt = req.get("prompt", "")
            max_tokens = req.get("max_tokens", 300)
            temperature = req.get("temperature", 0.8)
            top_k = req.get("top_k", 50)
            top_p = req.get("top_p", 0.95)
            strategy = req.get("strategy", "top_p")
            mode = req.get("mode", "generate")   # "generate" | "chat"

            if state.engine is None:
                await ws.send_text(json.dumps({"type": "error", "text": "Model not loaded"}))
                continue

            await ws.send_text(json.dumps({"type": "start"}))
            generated = ""

            if mode == "chat":
                messages = req.get("messages", [{"role": "user", "content": prompt}])
                system = req.get("system", "Tu es NFN, un assistant IA Neural Fractal Network.")
                # Build prompt
                parts = [f"<sys>{system}</sys>\n"]
                for msg in messages:
                    r, c = msg["role"], msg["content"]
                    tag = "usr" if r == "user" else "ast"
                    parts.append(f"<{tag}>{c}</{tag}>\n")
                parts.append("<ast>")
                full_prompt = "".join(parts)
            else:
                full_prompt = prompt

            async for token in state.engine.astream(
                full_prompt, max_new_tokens=max_tokens,
                temperature=temperature, top_k=top_k,
                top_p=top_p, strategy=strategy,
            ):
                generated += token
                await ws.send_text(json.dumps({"type": "token", "text": token}))

            await ws.send_text(json.dumps({"type": "end", "full": generated}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "text": str(e)}))
        except:
            pass


@app.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    await ws.accept()
    state.training_clients.append(ws)
    try:
        # Send existing metrics
        for m in state.training_metrics[-50:]:
            await ws.send_text(json.dumps({"type": "metrics", "data": m}))
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        state.training_clients.discard(ws) if hasattr(state.training_clients, 'discard') else None
        if ws in state.training_clients:
            state.training_clients.remove(ws)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _training_status() -> Dict:
    if not state.trainer:
        return {"running": False, "step": 0, "metrics": []}
    running = bool(state.training_thread and state.training_thread.is_alive())
    last = state.training_metrics[-1] if state.training_metrics else {}
    return {
        "running": running,
        "step": last.get("step", 0),
        "loss": last.get("loss"),
        "metrics": state.training_metrics[-100:],
    }


async def _safe_ws_send(ws: WebSocket, msg: str):
    try:
        await ws.send_text(msg)
    except:
        pass
