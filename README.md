# Neural Fractal Network (NFN)

> **Language model architecture — fractal geometry, phase dynamics, genuine AGI training system**

**Author:** Philippe-Antoine Robert · **Version:** 5.1 · **License:** Proprietary

---

## What is this?

NFN is a **research-grade AGI architecture** — a language model built from the ground up with
cognitive principles absent from standard transformers.

It is not a pre-trained model. It is the architecture, training system, and tools.
To get a capable model, you train it on real data with GPU compute.

**v5.1 ships a complete training system** — free cloud notebooks, one-command SSH setup, automatic dataset download.

---

## What makes NFN different?

| Component | Standard Transformer | NFN v5.1 |
|-----------|---------------------|----------|
| Attention | O(L²) softmax | O(L·d²) fractal linear — **255× faster at L=32 768** |
| Position encoding | Learned embeddings | Kuramoto oscillator phases — emergent synchrony |
| Token embedding | Learned lookup table | Analytic Fourier geometry — **0 parameters** |
| Output head | Random init | Zipf distribution init — matches word frequency from step 0 |
| Memory | None | Two-tier episodic ring + semantic SVD — no catastrophic forgetting |
| Training objective | Cross-entropy only | 12+ simultaneous AGI signals |
| Planning | None | MCTS-guided hierarchical sub-goal pursuit |
| Self-improvement | None | Constitutional critique + self-play DPO |
| Curiosity | None | Forward model curiosity + state novelty + learning progress |
| Value learning | None | Advantage-Weighted Regression (AWR) with goal-conditioned value |
| Theory of Mind | None | Agent belief encoding + perspective taking |
| Causal reasoning | None | Differentiable DAG + NOTEARS + counterfactual queries |

---

## Training — Pick your path

| Path | Cost | GPU | Time | Best for |
|------|------|-----|------|----------|
| [Kaggle notebook](#-kaggle-free-30-hweek) | **Free** | T4 16 GB | ~12 h | Best free option |
| [Google Colab](#-google-colab-free-t4) | **Free** | T4 15 GB | ~2 h | Easiest setup |
| [Lightning.ai](#-lightningai-free-22-hmonth) | **Free** | T4 16 GB | ~12 h | Persistent disk |
| [Cloud GPU via SSH](#-cloud-gpu-ssh-paid) | ~$5–15 | Any | 2–48 h | Most control |
| [Local machine](#-local-training) | Your hardware | CPU/GPU | Varies | Development |

---

## Free GPU Training

### Kaggle — Free, 30 h/week

> Best option: most free GPU time, checkpoints download automatically.

**Step 1** — Go to [kaggle.com](https://www.kaggle.com) → sign in (free account)

**Step 2** — `+ New Notebook` → upload [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb)

**Step 3** — `Settings → Accelerator → GPU T4 x1`

**Step 4** — `Run All`

That's it. The notebook will:
- Clone the repo and install dependencies
- Download Simple Wikipedia (~120 MB, auto-cached)
- Train the `medium` config (~85M params) for ~12 hours
- Save checkpoints to `/kaggle/working/checkpoints/` (download before session ends)

**Resume after session ends:**
Just re-run the training cell — it detects the checkpoint automatically.

---

### Google Colab — Free T4

> Easiest setup. Checkpoints saved to Google Drive — survive session resets.

Click to open directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AFKmoney/FNN/blob/main/notebooks/colab_train.ipynb)

Or manually:
1. [colab.research.google.com](https://colab.research.google.com) → `File → Open notebook → GitHub`
2. Paste `AFKmoney/FNN` → select `notebooks/colab_train.ipynb`
3. `Runtime → Change runtime type → T4 GPU`
4. `Runtime → Run all`

The notebook mounts Google Drive automatically so checkpoints persist between sessions.
Trains the `small` config (~15M params) — fits easily in 15 GB, completes in ~2 hours.

**Resume after disconnect:** Run cells 1–2 (mount + install) then the resume cell at the bottom.

---

### Lightning.ai — Free, 22 h/month

> Best persistence: your files and datasets stay on disk between sessions.

1. Go to [lightning.ai](https://lightning.ai) → sign up free
2. `New Studio → Jupyter Notebook`
3. Upload [`notebooks/lightning_train.ipynb`](notebooks/lightning_train.ipynb)
4. Enable GPU (top-right selector → T4)
5. `Run All`

Because the studio disk persists, the dataset downloads once and stays. Checkpoint from session 1
is automatically picked up in session 2.

---

### Cloud GPU SSH (paid)

> Most control. Cheapest option: Vast.ai RTX 3090 at ~$0.20/h.

**Recommended providers:**

| Provider | GPU | VRAM | Price | Preset |
|----------|-----|------|-------|--------|
| [Vast.ai](https://vast.ai) | RTX 3090 | 24 GB | ~$0.20/h | `medium_wikipedia` |
| [RunPod](https://runpod.io) | RTX 4090 | 24 GB | ~$0.44/h | `medium_wikipedia` |
| [Lambda Labs](https://lambdalabs.com) | A10 | 24 GB | ~$0.60/h | `medium_wikipedia` |
| [Paperspace](https://paperspace.com) | A100 | 40 GB | ~$1.10/h | `large_pile` |

**One command — everything automated:**

```bash
# SSH into your instance, then:
curl -fsSL https://raw.githubusercontent.com/AFKmoney/FNN/main/setup_cloud.sh | bash
```

This script:
- Detects your GPU and VRAM
- Installs PyTorch with the right CUDA version
- Clones the repo and installs all dependencies
- Auto-selects the best training preset for your VRAM
- Starts training (add `--tmux` to survive SSH disconnects)

**Manual preset selection:**

```bash
python cloud_train.py --preset nano_shakespeare      # < 1 GB VRAM,  ~5 min
python cloud_train.py --preset small_wikipedia       # ~4 GB VRAM,   ~2 h
python cloud_train.py --preset medium_wikipedia      # ~12 GB VRAM,  ~12 h
python cloud_train.py --preset medium_openwebtext    # ~12 GB VRAM,  ~18 h
python cloud_train.py --preset large_pile            # ~40 GB VRAM,  ~48 h
```

**Survive SSH disconnects with tmux:**

```bash
tmux new -s nfn
python cloud_train.py --preset small_wikipedia
# Detach: Ctrl+B then D
# Reconnect: tmux attach -t nfn

# Or resume from checkpoint if session was lost:
python cloud_train.py --resume checkpoints/agi_nfn_latest.pt
```

**Cost estimates:**

| Preset | GPU | Time | Cost |
|--------|-----|------|------|
| `nano_shakespeare` | Any | 0.1 h | $0.02 |
| `small_wikipedia` | RTX 3090 | 6 h | $1.20 |
| `medium_wikipedia` | RTX 4090 | 12 h | $5.28 |
| `large_pile` | A100 80GB | 48 h | $52.80 |

→ Full guide: [docs/CLOUD_TRAINING.md](docs/CLOUD_TRAINING.md)

---

### Local Training

```bash
git clone https://github.com/AFKmoney/FNN.git
cd FNN
pip install -e .

# Quick test — built-in Shakespeare text, no download needed
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
    --sample-every 500

# Resume from checkpoint
python train_agi.py --resume checkpoints/agi_nfn_step1000.pt --epochs 5
```

---

## Datasets

All datasets download automatically and are cached locally.

```bash
python cloud_train.py --list-datasets   # show all available datasets

# Download manually
python -m datasets.downloader wikipedia-en-simple

# Use in Python
from datasets.downloader import DatasetDownloader
dl   = DatasetDownloader("data")
text = dl.get("wikipedia-en-simple", max_chars=10_000_000)
```

| Dataset | Size | Notes |
|---------|------|-------|
| `tiny-shakespeare` | 1 MB | Quick smoke tests |
| `gutenberg-top100` | 20 MB | Classic literature |
| `wikipedia-en-simple` | 120 MB | General knowledge, recommended |
| `openwebtext-10pct` | 2 GB | Web-style language |
| `cc-news` | 1 GB | News articles |
| `wikipedia-en` | 20 GB | Full English Wikipedia |
| `pile-10pct` | 8 GB | Diverse high-quality text |

---

## What to expect during training

```
step     1 | lm 6.89 | ppl   987.3 | agi_w 0.00 [warmup]  | gn 1.23 | lr 3.0e-04
step   100 | lm 4.21 | ppl    67.8 | agi_w 0.12 [ramp]    | gn 0.65 | lr 2.8e-04
step   500 | lm 3.54 | ppl    34.5 | agi_w 1.00 [adaptive] | gn 0.72 | lr 2.5e-04
💾 Checkpoint saved → checkpoints/agi_nfn_latest.pt
```

**Healthy signs:** LM loss decreasing, gradient norm 0.5–3.0, perplexity below 50 after a few hours.

**3-phase adaptive curriculum:**
```
Phase 1 [warmup]:   steps 0 → 200   — LM loss only (build stable language base)
Phase 2 [ramp]:     steps 200 → 300 — linear ramp to full AGI losses
Phase 3 [adaptive]: LM stable?       — full AGI + per-signal weight adjustment
```

---

## After training — use the model

```python
import torch
from nfn.agi_model import build_agi_model
from nfn.tokenizer import NFNTokenizer

tok   = NFNTokenizer()
ckpt  = torch.load("checkpoints/agi_nfn_final.pt", map_location="cpu")
model = build_agi_model(vocab_size=tok.vocab_size, d_model=512, n_blocks=8)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Generate text
ids = tok.encode("The future of intelligence is", add_bos=True)
out = model.generate(ids, max_new_tokens=200, temperature=0.8)
print(tok.decode(out[0].tolist()))

# Goal-directed generation
model.set_goal(tok.encode("Explain step by step"))
out = model.generate(ids, max_new_tokens=200, temperature=0.8)
model.reset_goal()

# Web interface
# python run.py --model checkpoints/agi_nfn_final.pt
```

**Run the web interface:**

```bash
python run.py --model checkpoints/agi_nfn_final.pt
# Opens at http://127.0.0.1:8000
```

| Tab | What it does |
|-----|-------------|
| **Chat** | Token-streaming conversation |
| **Code** | Code completion and explanation |
| **Agent** | Multi-step reasoning with tools |
| **Training** | Start training from the UI, live metrics |
| **Explore the Web** | Read a URL → model adapts in real time |
| **TTL Adaptation** | LoRA adapter controls and stats |

---

## AGI Training System (v5.1)

A standard LLM minimises one loss: predict the next token.
NFN v5.1 trains with **12+ simultaneous signals**:

### Core signals (v4.0)

| Signal | Mechanism |
|--------|-----------|
| **LM + curiosity** | Cross-entropy, up-weighted by per-token entropy (surprising tokens → stronger gradient) |
| **Causal DAG** | Sparse differentiable DAG over memory slots + NOTEARS acyclicity + counterfactual loss |
| **Goal alignment** | Phase-goal cosine alignment loss (Kuramoto forcing toward θ*) |
| **Predictive coding** | N-step probabilistic predictions between layers (multi-horizon NLL) |
| **Free energy** | KL[q(z\|h) ‖ p(z)] + reconstruction — Friston's active inference |
| **Self-consistency** | Causal graph agreement across noisy candidates |
| **ACT halting** | Ponder cost: λ · mean_steps (encourages efficient reasoning) |
| **Self-play DPO** | Generate N → rank by quality → DPO gradient + winner distillation + offline replay |
| **Constitutional critique** | Generate → `[CRITIQUE]` → critique → `[REVISION]` → train on revision at 2× weight |
| **WAKE/SLEEP** | Every step: write episodic. Every N steps: episodic→semantic SVD consolidation + replay |

### New signals (v5.1)

| Signal | Mechanism |
|--------|-----------|
| **Value learning (AWR)** | V(s_t) via ValueHead → TD advantages → Advantage-Weighted Regression loss |
| **Intrinsic curiosity** | Forward model error + state novelty (LSH count) + learning progress (EMA) |
| **Theory of Mind** | Agent belief encoding from self-play pairs → perspective modulation → contrastive loss |
| **Goal reward** | `reward_goal_achievement()` → [0,1] scalar for current goal alignment |
| **MCTS planning** | UCB1 tree search over 8 hierarchical sub-goals — non-linear planning trajectories |

---

## Model sizes

| Config | Parameters | VRAM | Free tier | Paid preset |
|--------|-----------|------|-----------|-------------|
| `nano` | ~3M | < 1 GB | Any | `nano_shakespeare` |
| `small` | ~15M | ~4 GB | Colab T4 | `small_wikipedia` |
| `medium` | ~85M | ~12 GB | Kaggle T4 | `medium_openwebtext` |
| `large` | ~350M | ~40 GB | — | `large_pile` |

---

## Installation (local)

```bash
git clone https://github.com/AFKmoney/FNN.git
cd FNN
pip install -e .

# CUDA (NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon
pip install torch   # MPS included since PyTorch 2.0

# HuggingFace datasets (openwebtext, wikipedia-en, pile)
pip install datasets
```

Verify:

```bash
python -m pytest tests/ -q
# → 67 tests pass
```

---

## File structure

```
FNN/
├── cloud_train.py              Cloud training CLI (one command)
├── setup_cloud.sh              One-command cloud GPU setup script
├── train_agi.py                Local AGI training entry point
├── run.py                      Web interface launcher
│
├── notebooks/                  Free GPU training notebooks
│   ├── kaggle_train.ipynb      Kaggle T4 — 30h/week free
│   ├── colab_train.ipynb       Google Colab T4 — Drive checkpoint sync
│   ├── lightning_train.ipynb   Lightning.ai T4 — persistent disk
│   └── README.md               Platform comparison and quick-start
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
│   ├── value.py                ValueHead + RewardModel + AWR (v5.1)
│   ├── intrinsic.py            Forward curiosity + novelty + LP tracker (v5.1)
│   ├── theory_of_mind.py       AgentBeliefEncoder + PerspectiveTaker (v5.1)
│   ├── hyper.py                ContextHyperNet (in-context weight adaptation)
│   ├── online_learner.py       OnlineLearner — LoRA test-time adaptation
│   ├── web_explorer.py         WebExplorer — stdlib internet navigation
│   └── tokenizer.py            3-tier tokenizer
│
├── datasets/                   Public dataset downloader
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
    ├── CLOUD_TRAINING.md       Complete cloud training guide
    ├── CHANGELOG.md            Version history
    ├── ARCHITECTURE.md         Mathematical formalism
    ├── API.md                  REST + WebSocket API reference
    └── THEORY.md               Theory: fractals, Kuramoto, NFMC, Zipf
```

---

## Troubleshooting

**CUDA out of memory**
```bash
# Reduce batch size and/or sequence length
python cloud_train.py --preset medium_wikipedia --batch 4 --seq-len 512
```

**`No module named 'nfn'`**
```bash
pip install -e .    # run from the FNN project root
```

**Training loss not decreasing after 1000 steps**
```bash
# Reduce LR and disable AGI signals to stabilise first
python cloud_train.py --preset small_wikipedia --lr 1e-4 --no-self-play --no-critique
```

**Colab session keeps disconnecting**
```
Paste this in your browser console (F12 → Console):
function k(){document.querySelector('colab-connect-button')?.click();setTimeout(k,60000)}k()
```

**Interface won't open**
```bash
python run.py --no-open    # then open http://127.0.0.1:8000 manually
```

---

## Tests

```bash
python -m pytest tests/ -q          # 67 tests
python -m pytest tests/test_agi.py -v
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
