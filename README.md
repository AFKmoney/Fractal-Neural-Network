# Neural Fractal Network (NFN)

> **Research language model architecture based on fractal geometry, phase dynamics, and structured inductive biases**

---

**Author:** Philippe-Antoine Robert  
**Version:** 5.0 — AGI Training System  
**License:** Proprietary — Philippe-Antoine Robert, all rights reserved

---

## Honest description

NFN is an experimental language model architecture. It is **not** a trained model you can talk to today — it is an architecture and training system. To get a capable model you need to train it on a real corpus (gigabytes of text) with adequate compute (GPU hours).

What makes the architecture different from a standard transformer:

| Component | What it replaces | Why |
|-----------|-----------------|-----|
| Fractal linear attention | O(L²) softmax attention | O(L·d²) complexity, scales to long sequences |
| Kuramoto phase dynamics | Position embeddings | Learned oscillator synchrony as a similarity measure |
| Analytic token embedding | Learned embedding table | Zero-parameter representation — no cold-start |
| Zipf-initialised decoder | Random head init | Matches natural word frequency from step 0 |
| Spectral condensate (SVD) | None | Cross-context persistent knowledge without catastrophic forgetting |

**v5.0 adds a full AGI training system** with five concurrent training signals that go beyond next-token prediction.

---

## Training system (v5.0)

Standard LLM training = minimise cross-entropy on next token. That produces fluent text but no self-correction, no goal-directed behaviour, no adaptation.

NFN v5.0 trains with five signals in parallel:

| Signal | What it does |
|--------|-------------|
| **LM + curiosity** | Cross-entropy weighted by per-token entropy — surprising examples get higher gradient signal |
| **Multi-objective AGI loss** | causal DAG sparsity + goal alignment + phase coherence + ACT halting + predictive coding + free energy + self-consistency |
| **Self-play DPO-lite** | Generate N candidates from same prompt → rank by −LM loss → DPO preference gradient + distillation toward best |
| **Constitutional critique** | Generate → append `[CRITIQUE]` → generate critique → append `[REVISION]` → train on revision at 2× weight |
| **WAKE/SLEEP cycle** | Every step: write to episodic memory. Every N steps: consolidate episodic→semantic + replay last batches |

Curriculum schedule: steps 0→200 LM only (stable base), then ramp to full multi-objective loss over 100 steps.

---

## Test-time learning (self-adaptation without retraining)

The `--ttl` flag enables **LoRA fast-weight adapters** — a thin overlay on the attention layers that updates at inference time:

```
Base model weights  →  FROZEN  (result of training, never touched)
LoRA adapters       →  UPDATE  (O(rank × d) params, ~0.1% of model)
```

How it works:
1. Model reads new context → computes perplexity
2. If ppl > threshold (model finds it surprising): run N gradient steps on adapter only
3. Apply exponential decay to adapters after each update (controlled forgetting)
4. Adapters can be saved/loaded between sessions

This is test-time adaptation (TTA), not online learning of the base weights. The main weights remain frozen. Adapters are tiny and reset-able.

```bash
# Train with test-time learning enabled
python train_agi.py --text data/corpus.txt --ttl --adapter-rank 8 --online-lr 2e-4

# Load saved adapters from a previous session
python train_agi.py --resume checkpoints/agi_nfn_final.pt \
    --load-adapters checkpoints/adapters_final.pt
```

---

## Installation

```bash
git clone <repo>
cd flow
pip install -e .

# Optional: faster tokenizer
pip install tiktoken

# CUDA / mixed precision
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Quick start

### Train (AGI mode)

```bash
# Nano model, 3 epochs, all AGI signals on, test-time learning
python train_agi.py \
    --text data/corpus.txt \
    --config nano \
    --epochs 3 \
    --batch 4 \
    --lr 3e-4 \
    --ttl --adapter-rank 8

# Medium model, disable self-play for speed
python train_agi.py \
    --text data/corpus.txt \
    --config medium \
    --epochs 10 \
    --no-self-play \
    --sample-every 500

# Resume from checkpoint
python train_agi.py --resume checkpoints/agi_nfn_step1000.pt --epochs 2
```

### Train (standard LM mode)

```bash
python train.py --text data/corpus.txt --config nano --epochs 5
```

### Chat interface

```bash
python run.py --model checkpoints/agi_nfn_final.pt
# → http://localhost:8000
```

### Python API

```python
from nfn.agi_model import build_agi_model
from nfn.tokenizer import NFNTokenizer
from nfn.online_learner import OnlineLearner

model     = build_agi_model(vocab_size=32000, d_model=512, n_blocks=8)
tokenizer = NFNTokenizer()

# Goal-directed generation
model.set_goal(tokenizer.encode("Write a step-by-step explanation"))
out = model.generate(tokenizer.encode("The key insight is"), max_new_tokens=200)
print(tokenizer.decode(out[0].tolist()))

# Test-time adaptation (learns from new context without touching base weights)
learner = OnlineLearner(model, tokenizer, adapter_rank=8)
learner.adapt_from_text("New facts the model hasn't seen during training...")
out = model.generate(tokenizer.encode("Question about new facts"))
print(learner.stats())
```

---

## Architecture

```
NFN v5.0
┌──────────────────────────────────────────────────────┐
│  Token Input [B, L]                                  │
│       │                                              │
│  AnalyticTokenEmbedding  [0 params]                  │
│  Fourier fractal + char-class geometry               │
│       │                                              │
│  AGIBlock × n_blocks                                 │
│  ├─ EfficientNFNBlock                                │
│  │   ├─ FractalLinearAttention  O(L·d²)              │
│  │   ├─ PhaseSoliton                                 │
│  │   └─ PhaseRoutedMoE                               │
│  ├─ TwoTierMemory  (episodic ring + semantic SVD)    │
│  ├─ CausalGraphLayer  (DAG + do-calculus)            │
│  ├─ PhaseGoalPredictor  (λ·sin(θ*−θ) forcing)       │
│  ├─ RecursiveReasoner  (ACT halting)                 │
│  └─ PredictiveCodingBlock                            │
│       │                                              │
│  BayesianZipfianDecoder                              │
│  Logits [B, L, V]                                    │
└──────────────────────────────────────────────────────┘
  Optional LoRA adapters on attention projections
  (test-time learning, --ttl flag, base weights stay frozen)
```

---

## Efficiency

| Metric | Dense Transformer | NFN v5.0 |
|--------|-------------------|----------|
| Attention FLOPs (L=512) | 33.6M | 8.4M (4×) |
| Attention FLOPs (L=4096) | 2.15B | 134M (16×) |
| Attention FLOPs (L=32768) | 137B | 537M (255×) |
| Embedding params | standard | **0** (analytic) |
| Cross-session memory | none | episodic ring + semantic SVD |
| Inference adaptation | none | LoRA fast weights (~0.1% params) |

---

## What this is and what it is not

**Is:** A well-structured research architecture with a principled training system. All components are differentiable and tested (67 unit tests pass). The training signals (DPO-lite, constitutional critique, WAKE/SLEEP) are grounded in published research.

**Is not:** A trained model. You cannot have a conversation with it out of the box. It needs gigabytes of text and GPU hours to become capable. The architecture innovations give structural advantages but cannot substitute for data and compute.

**On the AGI label:** Used to describe the training *objective* (multi-signal, self-correcting, goal-directed) — not a claim that general intelligence exists in the untrained weights.

---

## File structure

```
flow/
├── nfn/
│   ├── config.py             # NFNConfig — all hyperparameters
│   ├── analytic_embed.py     # Zero-parameter analytic embedding
│   ├── condensate.py         # FractalRFF + SpectralCondensate
│   ├── moe.py                # PhaseRoutedMoE + FractalLinearAttention
│   ├── efficient_block.py    # EfficientNFNBlock
│   ├── episodic_memory.py    # TwoTierMemory (episodic + semantic)
│   ├── causal.py             # CausalGraphLayer — DAG + do-calculus
│   ├── goal.py               # PhaseGoalPredictor — Kuramoto goal forcing
│   ├── reasoning.py          # RecursiveReasoner (ACT)
│   ├── predictive.py         # PredictiveCodingBlock
│   ├── hyper.py              # ContextHyperNet
│   ├── ssm.py                # FractalSSM (Mamba-style, opt-in)
│   ├── online_learner.py     # OnlineLearner — LoRA test-time adaptation
│   ├── agi_block.py          # AGIBlock — full v5.0 block
│   ├── agi_model.py          # AGINFNModel — complete stack
│   ├── network.py            # NFNLanguageModel (standard LM)
│   └── tokenizer.py          # 3-tier tokenizer
├── training/
│   ├── agi_trainer.py        # AGITrainer — all 5 training signals
│   ├── trainer.py            # NFNTrainer — standard LM training
│   └── losses.py             # NFNLoss + AGILoss
├── inference/
│   └── engine.py             # Streaming, beam, speculative decode
├── interface/
│   ├── app.py                # FastAPI + WebSocket chat
│   └── agents.py             # Chat / Code / Reasoning agents
├── configs/
│   ├── nano.json             # 3M params — fast experiments
│   ├── small.json            # ~15M params
│   ├── medium.json           # ~85M params
│   └── large.json            # ~350M params
├── train_agi.py              # AGI training entry point (v5.0)
├── train.py                  # Standard LM training entry point
└── run.py                    # Web interface
```

---

## Tests

```bash
python -m pytest tests/ -q     # 67 tests
```

---

*Philippe-Antoine Robert*  
*"Intelligence is not a matter of size. It is a matter of structure."*
