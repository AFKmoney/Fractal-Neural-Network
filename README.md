# Neural Fractal Network (NFN)

> **Efficient language model architecture grounded in fractal geometry and phase dynamics**  
> *Structured inductive biases that let small models punch above their weight*

---

**Author:** Philippe-Antoine Robert  
**Version:** 4.0 — AGI Module Stack  
**License:** Proprietary — Philippe-Antoine Robert, all rights reserved

---

## What NFN actually is

NFN is a language model architecture that replaces brute-force scale with structural priors:

- **Fractal multi-scale attention** — O(L·d²) linear attention over hierarchical token groups instead of O(L²)
- **Kuramoto phase dynamics** — oscillator synchrony as a learned similarity measure, replacing softmax attention in the inner loop
- **Analytic embedding** — zero-parameter token representation using Fourier + char-class geometry; warm-start without a learned embedding table
- **Zipf-initialized decoder** — output head seeded to match the natural frequency distribution of language, converging faster from step 0
- **Spectral condensate** — one-shot SVD compression of the fractal kernel, no gradient required; updated online via incremental SVD (Brand 2002)

The AGI v4.0 module stack adds four capabilities that work together:

| Module | What it does | Why it matters |
|--------|-------------|----------------|
| **TwoTierMemory** | Episodic ring-buffer (fast exact recall) + semantic SVD condensate (slow compressed knowledge) | Cross-context persistence without catastrophic forgetting |
| **CausalGraphLayer** | Learns a lower-triangular DAG over memory slots; supports do-calculus hard interventions | Counterfactual reasoning: "what would change if X were different?" |
| **PhaseGoalPredictor** | Adds λ·sin(θ\*−θ) forcing to Kuramoto dynamics; goal θ\* encoded from prompt | Steerable generation without RLHF |
| **BayesianZipfianDecoder** | Logit uncertainty derived from condensate singular values S; Thompson sampling in training | Calibrated confidence; exploration without temperature hacks |

---

## Architecture

```
NFN v4.0 — AGI Stack
┌─────────────────────────────────────────────────┐
│  Token Input [B, L]                             │
│       │                                         │
│  ┌────▼────────────────────────────────────┐    │
│  │  AnalyticTokenEmbedding  [0 params]     │    │
│  │  Fourier fractal + char-class geometry  │    │
│  └────────────────────────────────────────┘    │
│       │                                         │
│  ┌────▼────────────────────────────────────┐    │
│  │  AGIBlock × n_blocks                    │    │
│  │  ├─ EfficientNFNBlock                   │    │
│  │  │   ├─ FractalLinearAttention O(L·d²)  │    │
│  │  │   ├─ PhaseSoliton                    │    │
│  │  │   └─ PhaseRoutedMoE                  │    │
│  │  ├─ TwoTierMemory (episodic + semantic) │    │
│  │  ├─ CausalGraphLayer (DAG + do-calculus)│    │
│  │  └─ PhaseGoalPredictor (λ·sin(θ*−θ))   │    │
│  └────────────────────────────────────────┘    │
│       │                                         │
│  ┌────▼────────────────────────────────────┐    │
│  │  [MultimodalFractalRFF]  (optional)     │    │
│  │  Shared phase space: text/image/audio   │    │
│  └────────────────────────────────────────┘    │
│       │                                         │
│  ┌────▼────────────────────────────────────┐    │
│  │  BayesianZipfianDecoder                 │    │
│  │  Uncertainty from condensate S values   │    │
│  └────────────────────────────────────────┘    │
│       │                                         │
│  Logits [B, L, V]                               │
└─────────────────────────────────────────────────┘
```

---

## Installation

```bash
git clone <repo>
cd flow
pip install -e .

# Optional: fast tokenizer
pip install tiktoken

# Optional: CUDA / Flash Attention
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Quick Start

### AGI model (v4.0)

```python
from nfn.agi_model import build_agi_model
from nfn.tokenizer import load_tokenizer

model = build_agi_model(
    vocab_size  = 32000,
    d_model     = 512,
    n_blocks    = 8,
    use_memory  = True,   # episodic + semantic memory
    use_causal  = True,   # causal DAG + interventions
    use_goal    = True,   # goal-directed generation
    use_bayesian= True,   # calibrated uncertainty
)
tok = load_tokenizer()

# Goal-directed generation
model.set_goal(tok.encode("Write a step-by-step explanation"))
out = model.generate(tok.encode("The key insight is"), max_new_tokens=200)
print(tok.decode(out[0].tolist()))

# Counterfactual reasoning
h = model.blocks[0].counterfactual(hidden, slot_idx=3, value=alt_concept)
```

### NFMC condensate (zero-shot warm-start)

```python
from nfn.config import NFNConfig
from nfn.nfmc import ZeroShotNFMC

cfg = NFNConfig(d_model=256, vocab_size=32000)
model = ZeroShotNFMC(cfg)
model.condense_from_text(open("corpus.txt").read(), tok)

out = model.generate(tok.encode("Hello"), max_new_tokens=100)
```

### Training

```bash
python train.py --config configs/small.json --data corpus.txt
torchrun --nproc_per_node=4 train.py --config configs/medium.json --distributed ddp
```

### Web interface

```bash
python run.py --config configs/small.json
# → http://localhost:8000
```

---

## Efficiency

| Metric | Dense Transformer | EfficientNFN v4.0 | Gain |
|--------|-------------------|--------------------|------|
| Attention FLOPs (L=512) | 33.6M | 8.4M | **4×** |
| Attention FLOPs (L=4096) | 2.15B | 134M | **16×** |
| Attention FLOPs (L=32768) | 137B | 537M | **255×** |
| Embedding params | standard | **0** (analytic) | ∞ |
| Catastrophic forgetting | full | **none** (SVD merge) | ✓ |
| Cross-context memory | none | **episodic ring + semantic SVD** | ✓ |

---

## File Structure

```
flow/
├── nfn/                      # Architecture core
│   ├── config.py             # NFNConfig — all hyperparameters
│   ├── analytic_embed.py     # Zero-parameter analytic embedding
│   ├── condensate.py         # FractalRFF + SpectralCondensate + HelmholtzPhaseLocking
│   ├── hopfield.py           # Hopfield memory + Mandelbrot priors + BayesianZipfianDecoder
│   ├── moe.py                # PhaseRoutedMoE + FractalLinearAttention
│   ├── efficient_block.py    # EfficientNFNBlock + EfficientNFNLM
│   ├── episodic_memory.py    # TwoTierMemory (EpisodicStore + SemanticConsolidator)
│   ├── causal.py             # CausalGraphLayer — DAG + do-calculus interventions
│   ├── goal.py               # PhaseGoalPredictor — Kuramoto goal forcing
│   ├── multimodal.py         # MultimodalFractalRFF — shared phase space
│   ├── agi_block.py          # AGIBlock — full v4.0 block
│   ├── agi_model.py          # AGINFNModel — complete AGI stack
│   ├── network.py            # NFNLanguageModel v2.0
│   ├── nfmc.py               # ZeroShotNFMC
│   ├── rope.py               # RoPE + NTK long-context
│   ├── kv_cache.py           # Fractal KV-Cache
│   ├── memory.py             # Persistent working memory
│   └── tokenizer.py          # 3-tier tokenizer
├── training/
│   ├── trainer.py            # Trainer (grad accum, torch.compile)
│   ├── losses.py             # NFNLoss — LM + phase + freq + spectral
│   └── distributed.py        # DDP / FSDP
├── inference/
│   └── engine.py             # Streaming, beam, mirostat
├── interface/
│   ├── app.py                # FastAPI + WebSocket
│   └── agents.py             # Chat / Code / Reasoning agents
├── docs/
│   ├── ARCHITECTURE.md       # Mathematical formalism
│   ├── API.md                # API reference
│   ├── THEORY.md             # Theoretical foundations
│   └── CHANGELOG.md          # Version history
├── configs/                  # nano / small / medium / large
├── train.py
└── run.py
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Mathematical formalism for every component |
| [docs/API.md](docs/API.md) | Python & HTTP API reference |
| [docs/THEORY.md](docs/THEORY.md) | Fractal kernels, Kuramoto, NFMC, Zipf |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Version history v1.0 → v4.0 |

---

*Philippe-Antoine Robert*  
*"Intelligence is not a matter of size. It is a matter of structure."*
