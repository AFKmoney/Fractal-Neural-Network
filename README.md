# Neural Fractal Network (NFN)

> **Language model architecture — fractal geometry, phase dynamics, autonomous learning**

**Author:** Philippe-Antoine Robert  
**Version:** 5.0  
**License:** Proprietary — Philippe-Antoine Robert, all rights reserved

---

## Honest description

NFN is a **research architecture** for language models. It is not a pre-trained model ready to use — it is the architecture and training system. To get a model capable of conversation, you need to train it on a real corpus (gigabytes of text) with GPU compute (hours to days depending on model size).

What makes the architecture different from a standard transformer:

| Component | What it replaces | Advantage |
|-----------|-----------------|-----------|
| Fractal linear attention | O(L²) softmax attention | O(L·d²) complexity, 255× faster at L=32 768 |
| Kuramoto phase dynamics | Position embeddings | Learned oscillator synchrony as a similarity measure |
| Analytic token embedding | Learned embedding table | 0 parameters — no cold-start |
| Zipf-initialised decoder | Random head init | Matches natural word frequency distribution from step 0 |
| Spectral condensate (SVD) | Nothing (transformer has no memory) | Persistent knowledge without catastrophic forgetting |

**v5.0 adds a complete AGI training system** with 5 simultaneous signals that go beyond next-token prediction.

---

## Installation

### Requirements

- **Python 3.10+** — [download](https://www.python.org/downloads/)
- **pip** (included with Python)
- **Git** — [download](https://git-scm.com/)
- NVIDIA GPU recommended for training (CPU works for testing)

### 1. Clone the repository

```bash
git clone https://github.com/AFKmoney/FNN.git
cd FNN
```

### 2. Install dependencies

```bash
pip install -e .
```

This automatically installs: `torch`, `fastapi`, `uvicorn`, `numpy`, `tqdm`, etc.

### 3. Optional dependencies

```bash
# Faster tokenizer (recommended)
pip install tiktoken

# Native desktop window instead of browser tab
pip install pywebview

# CUDA 12.1 (NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon (M1/M2/M3) — MPS included since torch 2.0
pip install torch
```

### 4. Verify installation

```bash
python -m pytest tests/ -q
# → 67 tests pass
```

---

## Running the app

### Windows — double-click

Double-click **`start.bat`** in the project folder.

A console window opens, the server starts, and the browser opens automatically at `http://127.0.0.1:8000`.

### Command line (all platforms)

```bash
# Quick start (nano model, port 8000)
python run.py

# Larger model
python run.py --config small

# Load a trained checkpoint
python run.py --model checkpoints/agi_nfn_final.pt

# Enable test-time learning (TTL) at startup
python run.py --ttl --adapter-rank 8

# Different port, no auto-open
python run.py --port 8080 --no-open

# Expose on local network (access from another machine)
python run.py --host 0.0.0.0 --port 8000
```

### Linux / Mac

```bash
chmod +x start.sh
./start.sh                    # equivalent to python run.py
./start.sh --config small
```

---

## Web interface

The interface opens at `http://127.0.0.1:8000` and has 6 tabs:

| Tab | Function |
|-----|----------|
| **Chat** | Conversation with the model, token-by-token streaming |
| **Code** | Code completion, explanation, refactoring |
| **Agent** | Multi-step reasoning with tools (calculate, search, analyse) |
| **Training** | Start training from the UI, watch live metrics |
| **Explore the Web** | Give a URL → model reads the page and adapts. Auto mode: autonomous navigation by keywords |
| **TTL Adaptation** | LoRA adapter controls — enable/disable, live stats, reset |

---

## Training the model

### AGI mode (recommended)

5 simultaneous training signals:

```bash
# Minimal training to test (built-in corpus, no external data needed)
python train_agi.py --config nano --epochs 3

# With your own data
python train_agi.py --text data/corpus.txt --config nano --epochs 10

# Medium model with test-time learning during training
python train_agi.py \
    --text data/corpus.txt \
    --config medium \
    --epochs 10 \
    --batch 4 \
    --lr 3e-4 \
    --ttl --adapter-rank 8 \
    --sample-every 200

# Without self-play (faster)
python train_agi.py --text data/corpus.txt --no-self-play --no-critique

# Resume from checkpoint
python train_agi.py --resume checkpoints/agi_nfn_step500.pt --epochs 5
```

### All training flags

```
--text PATH          Training text file (UTF-8)
--config NAME        nano | small | medium | large
--epochs N           Number of epochs (default: 3)
--batch N            Batch size (default: 4)
--lr FLOAT           Learning rate (default: 3e-4)
--seq-len N          Context length (default: from config)
--output DIR         Checkpoint directory (default: checkpoints/)
--save-every N       Save every N steps (default: 500)
--log-every N        Log every N steps (default: 10)
--device auto|cpu|cuda|mps
--fp16               Mixed precision fp16 (CUDA only)
--grad-accum N       Gradient accumulation steps (default: 1)

AGI signals:
--no-self-play       Disable self-play DPO-lite
--no-critique        Disable constitutional critique
--no-sleep           Disable WAKE/SLEEP cycle
--no-curiosity       Disable curiosity weighting
--agi-start N        Step where AGI losses start ramping in (default: 200)
--agi-ramp N         Ramp-up steps (default: 100)

Test-time learning:
--ttl                Enable LoRA adapters during training
--adapter-rank N     LoRA rank (default: 8)
--online-lr FLOAT    Adapter learning rate (default: 2e-4)
--online-steps N     Gradient steps per adapt() call (default: 4)
--ppl-gate FLOAT     Skip update if model ppl < threshold (default: 30)
--save-adapters PATH Save adapters at end of training
--load-adapters PATH Load adapters at startup

Evaluation:
--eval-text PATH     Validation text for perplexity
--eval-every N       Evaluate every N steps
--sample-every N     Generate a sample every N steps
--sample-prompt STR  Prompt for samples
```

### Standard LM mode

```bash
python train.py --text data/corpus.txt --config nano --epochs 5
```

---

## AGI training system (v5.0)

A standard LLM = minimise cross-entropy on the next token. NFN v5.0 trains with 5 signals in parallel:

| Signal | Mechanism |
|--------|-----------|
| **LM + curiosity** | Cross-entropy weighted by per-token entropy — surprising examples receive stronger gradient signal |
| **Multi-objective AGI loss** | Causal DAG sparsity + goal alignment + phase coherence + ACT halting + predictive coding + free energy + self-consistency |
| **Self-play DPO-lite** | Generate N candidates → rank by −LM loss → DPO preference gradient + distillation toward the winner |
| **Constitutional critique** | Generate → append `[CRITIQUE]` → generate critique → append `[REVISION]` → train on revision at 2× weight |
| **WAKE/SLEEP cycle** | Every step: write to episodic memory. Every N steps: episodic→semantic consolidation + replay |

Curriculum: steps 0→200 LM only (stable base), then ramp to full AGI losses over 100 steps.

---

## Test-time learning (adapt without retraining)

The `--ttl` flag enables **LoRA fast-weight adapters** — a thin overlay on attention projections that updates at inference time:

```
Base model weights  →  FROZEN  (result of training, never touched)
LoRA adapters       →  UPDATE  (O(rank × d) params, ~0.1% of model)
```

How it works:
1. Model reads new context → computes perplexity
2. If ppl > threshold (content is surprising): N gradient steps on adapters only
3. Exponential decay of adapters after each update (controlled forgetting)
4. Adapters can be saved and reloaded between sessions

```bash
# Enable TTL in the web interface
python run.py --ttl

# Save/load adapters
python train_agi.py --save-adapters checkpoints/adapters.pt
python run.py --model checkpoints/agi_nfn_final.pt --ttl
# then load via the "TTL Adaptation" tab in the interface
```

---

## Internet exploration

The model can read web pages and adapt in real time, with no extra dependencies (stdlib only — `urllib.request` + `html.parser`):

**Via the web interface:** "Explore the Web" tab → paste a URL → the model reads the page, shows perplexity before/after.

**Auto mode:** Give a seed URL + keywords → the model navigates page by page autonomously.

**Via Python:**

```python
from nfn.web_explorer import WebExplorer
from nfn.online_learner import OnlineLearner

explorer = WebExplorer()
learner  = OnlineLearner(model, tokenizer, adapter_rank=8)

# Read a single page
page = explorer.fetch("https://en.wikipedia.org/wiki/Artificial_intelligence")
print(f"Title: {page['title']} — {page['n_chars']} chars")

# Adapt from the page
stats = learner.adapt_from_text(page['text'])
print(f"ppl: {stats['ppl']:.1f} → loss: {stats['loss']:.4f}")

# Autonomous navigation
for page in explorer.explore(
    "https://en.wikipedia.org/wiki/Neural_network",
    n_pages=10,
    keywords=["learning", "architecture", "attention"],
):
    stats = learner.adapt_from_text(page['text'])
    print(f"  {page['title']}: ppl {stats['ppl']:.0f}")
```

---

## Python API

```python
from nfn.agi_model import build_agi_model
from nfn.tokenizer import NFNTokenizer
from nfn.online_learner import OnlineLearner

tok   = NFNTokenizer()
model = build_agi_model(vocab_size=tok.vocab_size, d_model=512, n_blocks=8)

# Goal-directed generation
model.set_goal(tok.encode("Explain step by step"))
ids = tok.encode("The key concept is", add_bos=True)
out = model.generate(ids, max_new_tokens=200, temperature=0.8)
print(tok.decode(out[0].tolist()))

# Reset goal
model.reset_goal()

# Test-time adaptation
learner = OnlineLearner(model, tok, adapter_rank=8, ppl_gate=30.0)
learner.adapt_from_text("New text the model has not seen during training...")
print(learner.stats())
# → {'n_adapters': 12, 'adapter_params': 13824, 'adapter_ratio': '0.43%', ...}

# Persist adapters
learner.save_adapters("session_adapters.pt")
learner.load_adapters("session_adapters.pt")
learner.reset()          # full forgetting
learner.decay(steps=10)  # manual decay
```

---

## HTTP API

The server exposes a REST + WebSocket API at `http://127.0.0.1:8000`:

```
POST /api/chat              → Multi-turn chat (JSON, blocking)
WS   /ws/chat               → Streaming chat (WebSocket)
POST /api/generate          → Raw text generation
POST /api/think             → Reasoning + answer (N rounds)
POST /api/agent/run         → Tool-calling agent
POST /api/learn             → Learn from text (episodic memory)
POST /api/explore/url       → Fetch URL and adapt
POST /api/explore/text      → Adapt from raw text
WS   /ws/explore            → Autonomous exploration stream
GET  /api/ttl/stats         → LoRA adapter statistics
POST /api/ttl/enable        → Enable TTL
POST /api/ttl/disable       → Disable TTL
POST /api/ttl/reset         → Zero all adapter weights
POST /api/train/start       → Start background training
POST /api/train/stop        → Stop training
GET  /api/train/status      → Live training metrics
WS   /ws/train              → Live metrics (WebSocket)
GET  /api/status            → Model info, active modules
```

---

## Architecture

```
NFN v5.0
┌──────────────────────────────────────────────────────────┐
│  Token Input [B, L]                                      │
│       │                                                  │
│  AnalyticTokenEmbedding  [0 parameters]                  │
│  Fourier fractal + character-class geometry              │
│       │                                                  │
│  AGIBlock × n_blocks                                     │
│  ├─ EfficientNFNBlock                                    │
│  │   ├─ FractalLinearAttention  O(L·d²)                  │
│  │   ├─ PhaseSoliton                                     │
│  │   └─ PhaseRoutedMoE                                   │
│  ├─ TwoTierMemory  (episodic ring + semantic SVD)        │
│  ├─ CausalGraphLayer  (DAG + do-calculus)                │
│  ├─ PhaseGoalPredictor  (λ·sin(θ*−θ) forcing)           │
│  ├─ RecursiveReasoner  (ACT halting)                     │
│  └─ PredictiveCodingBlock                                │
│       │                                                  │
│  BayesianZipfianDecoder                                  │
│  Logits [B, L, V]                                        │
└──────────────────────────────────────────────────────────┘
  Optional LoRA adapters on all attention projections
  (TTL — base weights stay frozen)
```

---

## Model sizes

| Config | Parameters | CPU RAM | GPU VRAM | Use case |
|--------|-----------|---------|----------|----------|
| `nano` | ~3M | ~100 MB | ~200 MB | Fast tests, CI |
| `small` | ~15M | ~500 MB | ~800 MB | Experiments |
| `medium` | ~85M | ~2 GB | ~3 GB | Serious training |
| `large` | ~350M | ~8 GB | ~12 GB | Production |

---

## Efficiency

| Metric | Dense Transformer | NFN v5.0 |
|--------|------------------|----------|
| Attention FLOPs (L=512) | 33.6M | 8.4M **(4×)** |
| Attention FLOPs (L=4 096) | 2.15B | 134M **(16×)** |
| Attention FLOPs (L=32 768) | 137B | 537M **(255×)** |
| Embedding parameters | standard | **0** (analytic) |
| Cross-session memory | none | episodic ring + semantic SVD |
| Inference adaptation | none | LoRA fast weights (~0.1%) |

---

## File structure

```
FNN/
├── nfn/                        Core architecture
│   ├── config.py               NFNConfig — all hyperparameters
│   ├── analytic_embed.py       Zero-parameter analytic embedding
│   ├── condensate.py           FractalRFF + SpectralCondensate
│   ├── moe.py                  PhaseRoutedMoE + FractalLinearAttention
│   ├── efficient_block.py      EfficientNFNBlock
│   ├── episodic_memory.py      TwoTierMemory (episodic + semantic)
│   ├── causal.py               CausalGraphLayer — DAG + do-calculus
│   ├── goal.py                 PhaseGoalPredictor — Kuramoto forcing
│   ├── reasoning.py            RecursiveReasoner (ACT halting)
│   ├── predictive.py           PredictiveCodingBlock
│   ├── hyper.py                ContextHyperNet
│   ├── ssm.py                  FractalSSM (Mamba-style, opt-in)
│   ├── online_learner.py       OnlineLearner — LoRA test-time adaptation
│   ├── web_explorer.py         WebExplorer — stdlib internet navigation
│   ├── agi_block.py            AGIBlock — full v5.0 block
│   ├── agi_model.py            AGINFNModel — complete stack
│   ├── network.py              NFNLanguageModel (standard LM)
│   └── tokenizer.py            3-tier tokenizer
│
├── training/
│   ├── agi_trainer.py          AGITrainer — all 5 training signals
│   ├── trainer.py              NFNTrainer — standard LM training
│   └── losses.py               NFNLoss + AGILoss
│
├── inference/
│   └── engine.py               Streaming, beam, speculative decode
│
├── interface/
│   ├── app.py                  FastAPI server + WebSocket
│   ├── agents.py               Chat / Code / Reasoning agents
│   └── static/
│       ├── index.html          Web UI (6 tabs)
│       ├── style.css           Dark theme
│       └── app.js              Frontend logic
│
├── configs/
│   ├── nano.json               ~3M parameters — fast tests
│   ├── small.json              ~15M parameters
│   ├── medium.json             ~85M parameters
│   └── large.json              ~350M parameters
│
├── tests/                      67 unit tests
├── docs/
│   ├── ARCHITECTURE.md         Mathematical formalism
│   ├── CHANGELOG.md            Version history
│   ├── API.md                  API reference
│   └── THEORY.md               Theory: fractals, Kuramoto, NFMC, Zipf
│
├── train_agi.py                AGI training entry point (v5.0)
├── train.py                    Standard LM training entry point
├── run.py                      Web interface launcher
├── start.bat                   Windows launcher (double-click)
├── start.sh                    Linux / Mac launcher
└── requirements.txt            Dependencies
```

---

## Tests

```bash
python -m pytest tests/ -q      # 67 tests, ~80s on CPU
python -m pytest tests/ -v      # verbose output
python -m pytest tests/test_agi.py -v    # AGI tests only
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'nfn'`**
```bash
pip install -e .    # install in editable mode from project root
```

**`CUDA out of memory`**
```bash
python train_agi.py --config nano --batch 1 --seq-len 64   # reduce size
```

**Interface won't open**
```bash
python run.py --no-open    # disable auto-open
# then navigate manually to http://127.0.0.1:8000
```

**Error on Mac (MPS)**
```bash
python train_agi.py --device cpu    # MPS has limitations with some ops
```

**Model generates noise**  
Normal for an untrained model. Train it on a real corpus first.

---

## What this is and what it is not

**Is:** A well-structured research architecture with a principled training system. All components are differentiable and tested (67 unit tests pass). The training signals (DPO-lite, constitutional critique, WAKE/SLEEP) are grounded in published research.

**Is not:** A pre-trained model. You cannot have a real conversation with it out of the box. It needs gigabytes of text and GPU hours to become capable. The architectural innovations give structural advantages but cannot substitute for data and compute.

**On the "AGI" label:** It describes the training *objective* (multi-signal, self-correcting, goal-directed) — not a claim that general intelligence exists in the untrained weights.

---

*Philippe-Antoine Robert*  
*"Intelligence is not a matter of size. It is a matter of structure."*
