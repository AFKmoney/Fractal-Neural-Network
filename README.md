# Neural Fractal Network (NFN)

> **Language model architecture — fractal geometry, phase dynamics, genuine AGI training system**

**Author:** Philippe-Antoine Robert · **Version:** 5.1 · **License:** Proprietary

---

## What is this?

NFN is a **research-grade AGI architecture** — a language model built from the ground up with
cognitive principles absent from standard transformers.

It is not a pre-trained model. It is the architecture, training system, and tools.
To get a capable model, you train it on real data with GPU compute.

**v5.1 ships a complete cloud training system** — one command, any cloud GPU, automatic dataset download.

---

## What makes NFN different?

| Component | Standard Transformer | NFN v5.1 |
|-----------|---------------------|----------|
| Attention | O(L²) softmax | O(L·d²) fractal linear — **255× faster at L=32 768** |
| Position encoding | Learned embeddings | Kuramoto oscillator phases — emergent synchrony |
| Token embedding | Learned lookup table | Analytic Fourier geometry — **0 parameters** |
| Output head | Random init | Zipf distribution init — matches word frequency from step 0 |
| Memory | None | Two-tier episodic ring + semantic SVD — no catastrophic forgetting |
| Training objective | Cross-entropy only | 5+ simultaneous AGI signals |
| Planning | None | MCTS-guided hierarchical sub-goal pursuit |
| Self-improvement | None | Constitutional critique + self-play DPO |
| Curiosity | None | Forward model curiosity + state novelty + learning progress |
| Value learning | None | Advantage-Weighted Regression (AWR) with goal-conditioned value |
| Theory of Mind | None | Agent belief encoding + perspective taking |
| Causal reasoning | None | Differentiable DAG + NOTEARS + counterfactual queries |

---

## ⚡ Cloud Training — Quick Start

**Rent any cloud GPU → SSH in → one command:**

```bash
# Auto-setup + auto-train (detects your GPU, picks the right preset)
curl -fsSL https://raw.githubusercontent.com/AFKmoney/FNN/main/setup_cloud.sh | bash
```

Or if you're already in the repo:

```bash
bash setup_cloud.sh
```

**Training presets (auto-selects based on VRAM):**

```bash
python cloud_train.py --preset nano_shakespeare      # < 1 GB VRAM,  ~5 min
python cloud_train.py --preset small_wikipedia       # ~4 GB VRAM,   ~2 h
python cloud_train.py --preset medium_wikipedia      # ~12 GB VRAM,  ~12 h
python cloud_train.py --preset medium_openwebtext    # ~12 GB VRAM,  ~18 h
python cloud_train.py --preset large_pile            # ~40 GB VRAM,  ~48 h
```

```bash
python cloud_train.py --list-presets    # show all 7 presets
python cloud_train.py --list-datasets   # show all 7 datasets
```

**Survive SSH disconnects with tmux:**

```bash
# Training inside tmux — survives disconnect
tmux new -s nfn
python cloud_train.py --preset small_wikipedia

# Detach: Ctrl+B then D
# Reconnect later:
tmux attach -t nfn

# Or resume from auto-saved checkpoint:
python cloud_train.py --resume checkpoints/agi_nfn_latest.pt
```

→ Full cloud guide: [docs/CLOUD_TRAINING.md](docs/CLOUD_TRAINING.md)

---

## Installation (local)

### Requirements

- Python 3.10+
- pip
- NVIDIA GPU recommended (CPU works for testing)

```bash
git clone https://github.com/AFKmoney/FNN.git
cd FNN
pip install -e .
```

Optional extras:

```bash
# CUDA 12.x (NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon
pip install torch   # MPS included since PyTorch 2.0

# HuggingFace datasets (for openwebtext, wikipedia-en, pile)
pip install datasets
```

Verify:

```bash
python -m pytest tests/ -q
# → 67 tests pass
```

---

## Running the web interface

```bash
# Quick start (nano model, opens browser automatically)
python run.py

# Larger model
python run.py --config small

# Load a trained checkpoint
python run.py --model checkpoints/agi_nfn_final.pt

# Different port
python run.py --port 8080
```

Windows: double-click **`start.bat`**.
Linux/Mac: `./start.sh`

The interface opens at `http://127.0.0.1:8000` with 6 tabs:

| Tab | What it does |
|-----|-------------|
| **Chat** | Token-streaming conversation |
| **Code** | Code completion and explanation |
| **Agent** | Multi-step reasoning with tools |
| **Training** | Start training from the UI, live metrics |
| **Explore the Web** | Read a URL → model adapts in real time |
| **TTL Adaptation** | LoRA adapter controls and stats |

---

## Training

### Cloud training (recommended)

```bash
# Simplest — downloads everything automatically
python cloud_train.py --preset small_wikipedia

# Custom dataset
python cloud_train.py --dataset openwebtext-10pct --config medium

# Your own data
python cloud_train.py --data-file my_corpus.txt --config small

# Resume after disconnect
python cloud_train.py --resume checkpoints/agi_nfn_latest.pt
```

### Local training (AGI mode)

```bash
# With built-in demo text (instant, no download)
python train_agi.py --config nano --epochs 3

# With your own data
python train_agi.py --text data/corpus.txt --config small --epochs 10

# Full options
python train_agi.py \
    --text data/corpus.txt \
    --config medium \
    --epochs 5 \
    --batch 8 \
    --lr 2e-4 \
    --seq-len 1024 \
    --fp16 \
    --sample-every 500 \
    --eval-text data/val.txt

# Resume from checkpoint
python train_agi.py --resume checkpoints/agi_nfn_step1000.pt --epochs 5
```

---

## AGI Training System (v5.1)

A standard LLM minimises one loss: predict the next token.
NFN v5.1 trains with **10+ simultaneous signals**:

### v4.0 signals (stable)

| Signal | Mechanism |
|--------|-----------|
| **LM + curiosity** | Cross-entropy, up-weighted by per-token entropy (surprising tokens → stronger gradient) |
| **Causal DAG** | Sparse differentiable DAG over memory slots + NOTEARS acyclicity + counterfactual loss |
| **Goal alignment** | Phase-goal cosine alignment loss (Kuramoto forcing toward θ*) |
| **Predictive coding** | N-step probabilistic predictions between layers (multi-horizon NLL) |
| **Free energy** | KL[q(z\|h) \|\| p(z)] + reconstruction — Friston's active inference |
| **Self-consistency** | Causal graph agreement across noisy candidates |
| **ACT halting** | Ponder cost: λ · mean_steps (encourages efficient reasoning) |
| **Self-play DPO** | Generate N → rank by quality → DPO gradient + winner distillation + offline replay |
| **Constitutional critique** | Generate → `[CRITIQUE]` → critique → `[REVISION]` → train on revision at 2× weight |
| **WAKE/SLEEP** | Every step: write episodic. Every N steps: episodic→semantic SVD consolidation + replay |

### v5.1 new signals

| Signal | Mechanism |
|--------|-----------|
| **Value learning (AWR)** | V(s_t) via ValueHead → TD advantages → Advantage-Weighted Regression loss |
| **Intrinsic curiosity** | Forward model error + state novelty (LSH count) + learning progress (EMA) |
| **Theory of Mind** | Agent belief encoding from self-play pairs → perspective modulation → contrastive loss |
| **Goal reward** | `reward_goal_achievement()` → [0,1] scalar for current goal alignment |
| **MCTS planning** | UCB1 tree search over 8 hierarchical sub-goals — non-linear planning trajectories |

### Adaptive curriculum (v5.1)

Replaces the v4.0 linear ramp:

```
Phase 1 [warmup]:   steps 0 → 200   — LM loss only (build stable language base)
Phase 2 [ramp]:     steps 200 → 300 — linear ramp to full AGI losses
Phase 3 [adaptive]: LM stable?       — full AGI + per-signal weight adjustment
                                       (high progress → +1%, increasing loss → -0.5%)
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
ids = tok.encode("The key insight is", add_bos=True)
out = model.generate(ids, max_new_tokens=200, temperature=0.8)
print(tok.decode(out[0].tolist()))
model.reset_goal()

# Test-time adaptation (no retraining needed)
learner = OnlineLearner(model, tok, adapter_rank=8, ppl_gate=30.0)
learner.adapt_from_text("New domain text the model hasn't seen …")
learner.save_adapters("session.pt")

# Dataset downloading
from datasets.downloader import DatasetDownloader
dl   = DatasetDownloader("data")
text = dl.get("wikipedia-en-simple", max_chars=10_000_000)
# → 10M characters, cached to data/wikipedia-en-simple.txt
```

---

## Model sizes

| Config | Parameters | VRAM | Best preset |
|--------|-----------|------|-------------|
| `nano` | ~3M | < 1 GB | `nano_shakespeare` |
| `small` | ~15M | ~4 GB | `small_wikipedia` |
| `medium` | ~85M | ~12 GB | `medium_openwebtext` |
| `large` | ~350M | ~40 GB | `large_pile` |

---

## File structure

```
FNN/
├── cloud_train.py              ← Cloud training CLI (one command)
├── setup_cloud.sh              ← One-command cloud GPU setup
├── train_agi.py                ← Local AGI training entry point
├── train.py                    ← Standard LM training
├── run.py                      ← Web interface launcher
├── start.bat / start.sh        ← Double-click launchers
│
├── nfn/                        Core architecture
│   ├── config.py               NFNConfig — 100+ hyperparameters
│   ├── analytic_embed.py       Zero-parameter analytic embedding
│   ├── efficient_block.py      EfficientNFNBlock (fractal attn + MoE + phase)
│   ├── agi_block.py            AGIBlock — all v5.1 modules integrated
│   ├── agi_model.py            AGINFNModel — full stack
│   ├── episodic_memory.py      TwoTierMemory (episodic ring + semantic SVD)
│   ├── working_memory.py       FractalWorkingMemory (NTM-style slots)
│   ├── causal.py               CausalGraphLayer — NOTEARS DAG + counterfactual
│   ├── goal.py                 PhaseGoalPredictor + HierarchicalGoalDecomposer
│   ├── reasoning.py            RecursiveReasoner (ACT) + MCTS PlanExecutor
│   ├── predictive.py           N-step probabilistic predictive coding
│   ├── value.py                ← NEW: ValueHead + RewardModel + AWR
│   ├── intrinsic.py            ← NEW: Forward curiosity + novelty + LP tracker
│   ├── theory_of_mind.py       ← NEW: AgentBeliefEncoder + PerspectiveTaker
│   ├── hyper.py                ContextHyperNet (in-context weight adaptation)
│   ├── online_learner.py       OnlineLearner — LoRA test-time adaptation
│   ├── web_explorer.py         WebExplorer — stdlib internet navigation
│   └── tokenizer.py            3-tier tokenizer
│
├── datasets/                   ← NEW: Public dataset downloader
│   └── downloader.py           DatasetDownloader — 7 datasets, auto-cache
│
├── training/
│   ├── agi_trainer.py          AGITrainer v5.1 — adaptive curriculum + offline replay
│   └── losses.py               AGILoss v5.1 — 12+ signals + EMA tracking
│
├── inference/
│   └── engine.py               Streaming, beam, speculative decode
│
├── interface/
│   ├── app.py                  FastAPI server + WebSocket streaming
│   └── static/                 Web UI (6 tabs)
│
├── configs/
│   ├── nano.json               ~3M parameters
│   ├── small.json              ~15M parameters
│   ├── medium.json             ~85M parameters
│   └── large.json              ~350M parameters
│
├── tests/                      67 unit tests
└── docs/
    ├── CLOUD_TRAINING.md       ← NEW: Complete cloud training guide
    ├── CHANGELOG.md            Version history
    ├── ARCHITECTURE.md         Mathematical formalism
    ├── API.md                  REST + WebSocket API reference
    └── THEORY.md               Theory: fractals, Kuramoto, NFMC, Zipf
```

---

## Tests

```bash
python -m pytest tests/ -q      # 67 tests
python -m pytest tests/test_agi.py -v
```

---

## Troubleshooting

**CUDA out of memory**
```bash
python cloud_train.py --preset medium_wikipedia --batch 4 --seq-len 512
```

**`No module named 'nfn'`**
```bash
pip install -e .    # from project root
```

**Training loss not decreasing after 1000 steps**
```bash
# Try lower LR and disable AGI signals to stabilise first
python cloud_train.py --data-file corpus.txt --config small --lr 1e-4 \
    --no-self-play --no-critique
```

**Interface won't open**
```bash
python run.py --no-open    # then open http://127.0.0.1:8000 manually
```

---

## What this is and what it is not

**Is:** A well-structured research architecture with a principled multi-signal training system.
All components are differentiable and tested (67 unit tests pass).

**Is not:** A pre-trained model. You cannot have a conversation out of the box.
It needs gigabytes of text and GPU hours to become capable.

**On "AGI":** Describes the training *objective* (multi-signal, self-improving, goal-directed,
value-learning, theory-of-mind) — not a claim about the untrained weights.

---

*Philippe-Antoine Robert*  
*"Intelligence is not a matter of size. It is a matter of structure."*
