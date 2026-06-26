# FNN — Fractal Neural Network

> A neuro-symbolic language model where representation is built from **fractal recursion**, **phase synchronization**, and **causal structure** — not just stacked attention.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-160%20passing-brightgreen.svg)](#tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Version 6.0.0**

---

## What is FNN?

FNN is a research framework for a different kind of language model. Instead of the standard Transformer recipe (token embedding → self-attention blocks → softmax head), FNN combines several ideas from physics, topology, and cognitive science into a single trainable architecture:

| Idea | Where it lives | Why |
|------|----------------|-----|
| **Fractal topology** | `topology.py`, `kv_cache.py` | Sequence positions are organized into a self-similar hierarchy (binary tree / Cantor / Sierpinski). Higher levels are recomputed exponentially less often. |
| **Phase synchronization (Kuramoto)** | `phase_ode.py` | Tokens are oscillators; meaning emerges when phases lock. Differentiable RK4 integration. |
| **Causal reasoning (SCM)** | `causal.py` | A learnable DAG with NOTEARS acyclicity + do-calculus — the model can reason about interventions and counterfactuals. |
| **Self-model / introspection** | `self_model.py` | A global workspace where the network represents its own state. |
| **Mixture-of-Experts** | `moe.py` | Phase-routed MoE (Von Mises distribution), experts are sparse. |
| **Analytic embeddings** | `analytic_embed.py`, `gematria.py` | Fourier + character-class + 5 crossed gematria systems — most of the embedding is **zero-parameter**. |
| **Optional advanced layers** | `ads_cft.py`, `tensor_network.py`, `godel_loop.py`, `rg_flow.py`, `hyperbolic_gematria.py` | Holographic attention (AdS/CFT), MERA O(log L), Gödel self-reference, renormalization-group flow, hyperbolic geometry. All toggleable. |
| **Streaming / long context** | `ssm.py`, `rope.py`, `streaming.py` | Mamba-style recurrence, NTK-aware RoPE, chunked generation. |

The result is a model where **regularization comes from structure** (phase coherence, DAG sparsity, spectral stability) rather than just dropout and weight decay.

---

## PRISM — alternative backbone (integrated from PRISM-KB)

FNN also ships **PRISM** (`nfn.prism`), a complete alternative architecture integrated from the [PRISM-KB](https://github.com/AFKmoney/prism-kb) project. It unifies four paradigms under one abstraction and adds several mechanisms FNN's core does not have:

| PRISM mechanism | Module | What it does |
|-----------------|--------|--------------|
| **Multi-Rate Bus** | `mrb.py` | Sub-quadratic `O(n)` backbone: a bank of recurrent filters at logarithmically-spaced decay rates, with a learned per-token scale gate. Replaces self-attention. |
| **Polymorphic MoE** | `router.py`, `experts.py` | A router picks, **per token**, which *kind* of computation it needs: Neural (MLP), Memory (read/write head), or Symbolic (differentiable primitives) — heterogeneous experts, not homogeneous MLPs. |
| **Shared memory bus** | `memory.py` | One memory tape `(S × d_mem)` flows through all layers and time steps — the Global Workspace through which experts communicate. |
| **Holographic memory (PRISM-Holo)** | `holo.py` | An algebraic Vector-Symbolic-Architecture tape that **binds and retrieves facts with zero training**. `tape.bind(key, value)` is instant CPU algebra; retrieval lands the right fact at rank 1/20 in testing. |
| **Differentiable symbolic reasoning** | `symbolic.py` | 6 differentiable primitives (compare, gate, select, shift, threshold, count) soft-selected and composed end-to-end inside the router. |
| **Progressive Capacity Stacking** | `pcs.py` | Grow a model 350M → 700M → 1B with weight inheritance (`grow_model()`) — ~40-50% wall-clock savings. |
| **CogLoop** | `cogloop.py` | PERCEIVE → REFLECT → RESPOND → CONSOLIDATE cognitive loop with persistent two-tier (working + long-term) memory. |
| **Curriculum + token recycling** | `curriculum.py` | Re-weight dataset mix across training (neural → memory → symbolic); inverse-frequency token weighting for hard tokens. |
| **Modular pretraining** | `modular.py` | Train each expert kind separately on its optimal data, then assemble. |

```python
from nfn.prism import Prism, PrismConfig
from nfn.prism.holo import HoloTape

# PRISM model (multi-rate bus + polymorphic MoE)
cfg = PrismConfig(vocab_size=256, d_model=128, num_layers=4)
model = Prism(cfg)

# Holographic zero-training fact binding
import torch
tape = HoloTape(D=4096)
tape.bind(torch.randn(4096), torch.randn(4096))   # pure algebra, instant
retrieved = tape.unbind(torch.randn(4096))         # same op = self-inverse
```

PRISM and the FNN core are independent backbones that share the FNN training infrastructure.

---

## Quick start

```bash
pip install torch
git clone https://github.com/AFKmoney/Fractal-Neural-Network.git
cd Fractal-Neural-Network
pip install -e .          # optional, installs the `nfn` package
```

### Build a model and run a forward pass

```python
import torch
from nfn import build_fnn_model

# nano preset: ~11.5M params, runs on CPU
model = build_fnn_model(vocab_size=110, preset="nano")
x = torch.randint(0, 110, (2, 64))
logits, losses = model(x, targets=x)

print(logits.shape)           # torch.Size([2, 64, 110])
print(losses["lm"].item())    # cross-entropy
print(losses["total"].item()) # lm + all structural regularizers
```

### Generate text

```python
prompt = torch.randint(0, 110, (1, 8))
out = model.generate(prompt, max_new_tokens=64, temperature=0.8, top_k=40)
```

### Train on raw text

```bash
# Minimal CPU training run
python examples/quickstart.py
```

```python
from training.trainer import NFNTrainer
from nfn import build_fnn_model
from nfn.tokenizer import NFNTokenizer

model = build_fnn_model(vocab_size=110, preset="nano")
trainer = NFNTrainer(model, NFNTokenizer(), lr=3e-4, output_dir="checkpoints")
trainer.train(your_text_corpus, n_epochs=5, seq_len=128, batch_size=8)
```

---

## Presets

| Preset | `d_model` | Blocks | Advanced layers | Params | Target |
|--------|-----------|--------|-----------------|--------|--------|
| `nano` | 256 | 4 | — | ~11.5M | CPU experimentation |
| `small` | 512 | 8 | — | ~88M | Single GPU |
| `medium` | 1024 | 12 | — | ~517M | Multi-GPU |
| `large` | 512 | 12 | ✓ (AdS/CFT, MERA, Gödel, RG, hyperbolic) | ~180M | Research, all features |

Advanced layers are enabled per-preset but every feature is independently toggleable through `FNNConfig` flags (`use_ads_cft`, `use_mera`, `use_godel_loop`, `use_rg_flow`, `use_hyperbolic_gematria`, `use_ssm`, `use_mixture_of_depths`, …).

---

## Architecture

```
Token IDs
   │
   ├─ AnalyticTokenEmbedding (zero-parameter: Fourier + char-class + n-gram)
   ├─ GematriaEmbedding (5 crossed systems: ordinal · prime · Fibonacci · digital-root · learned)
   │
   ├─ [FNNBlock × N] ─────────────────────────────────────────────────────
   │     ├─ FractalLinearAttention   O(L·d²)    Katharopoulos linear kernel
   │     ├─ PhaseSoliton             O(L·n_p)   coherent phase amplification
   │     ├─ PhaseRoutedMoE           O(L·K·d·d_ff/E)   Von Mises routing
   │     ├─ [CausalGraphLayer]       O(n_slots²·d)     NOTEARS + do-calculus
   │     └─ [SelfModel]              O(L·n_slots·d)    global workspace
   │
   ├─ [Optional advanced layers]
   │     ├─ AdS/CFT Attention        holographic bulk projector
   │     ├─ MERA Attention           O(log L), disentangler + isometry
   │     ├─ Gödel Fixed Point        self-reference operator
   │     ├─ RG Flow Scheduler        UV evaporation · IR condensation
   │     └─ Hyperbolic Gematria      Poincaré ball · sheaf cohomology
   │
   ├─ ZipfianDecoder (power-law prior, optional Bayesian recalibration)
   └─ Spectral Condensate + Helmholtz Phase Locking
```

The **master equation** governing the phase field:

```
∂Θ/∂t = Ω + λ·sin(Θ*−Θ) + K·sin(Θ̄−Θ) + ∇_DAG L_causal + α·∂W/∂F
```

where `Θ*` is the goal attractor, `Θ̄` the neighborhood mean, and the last two terms couple phase dynamics to causal-graph gradients and free-energy flow.

---

## Project structure

```
nfn/                  # the model engine (importable as `import nfn`)
├── config.py         # FNNConfig — 80+ hyperparameters, 4 presets
├── model.py          # FNNModel + build_fnn_model()
├── block.py          # FNNBlock (the core repeated unit)
├── tokenizer.py      # char / BPE / tiktoken tokenizers
├── lifecycle.py      # continuous WAKE/SLEEP/META training cycle
│
├── # core mechanisms
├── moe.py            # phase-routed MoE + fractal linear attention + soliton
├── phase_ode.py      # Kuramoto dynamics + goal forcing + hierarchical goals
├── causal.py         # causal DAG (NOTEARS + do-calculus + counterfactuals)
├── self_model.py     # global workspace + introspection
├── gematria.py       # 5-system gematric embedding
├── analytic_embed.py # zero-parameter Fourier embedding
├── hopfield.py       # Zipfian decoder + modern Hopfield memory
├── condensate.py     # spectral condensate (fractal RFF + SVD + phase locking)
├── episodic_memory.py    # ring-buffer + SVD semantic consolidation
├── working_memory.py     # differentiable fractal DNC scratchpad
├── auto_genesis.py   # facade for self-developing math engine
│
├── # optional advanced layers
├── ads_cft.py            # AdS/CFT holographic attention
├── tensor_network.py     # MERA O(log L)
├── godel_loop.py         # Gödel self-reference
├── rg_flow.py            # renormalization-group flow
├── hyperbolic_gematria.py# Poincaré ball + sheaves
│
├── # recovered research modules (Phase 2)
├── ssm.py                # Mamba-style state-space model
├── mixture_of_depths.py  # token-skipping (compute allocation)
├── reasoning.py          # adaptive computation time + self-consistency
├── predictive.py         # predictive coding + free-energy minimizer
├── multi_token_pred.py   # multi-token prediction heads
├── hyper.py              # context-conditioned hypernetwork / LoRA
├── program_synthesis.py  # neuro-symbolic program synthesis
├── multimodal.py         # cross-modal phase sync
│
├── # long-context infrastructure
├── rope.py           # NTK-aware rotary embeddings
├── kv_cache.py       # fractal KV cache for generation
├── topology.py       # motif builders (binary tree / Cantor / Sierpinski)
└── connections.py    # sinusoidal gates & aggregators

training/             # training loops
├── trainer.py        # standard trainer (AdamW + cosine + warmup, AMP, grad-accum)
├── agi_trainer.py    # full-cycle trainer (WAKE/SLEEP + self-play + critique)
├── losses.py         # multi-objective loss aggregator
├── distributed.py    # FSDP wrapper
├── intrinsic.py      # curiosity / novelty rewards (Pathak + Oudeyer)
├── value.py          # value head + reward model + advantage (AWR)
├── continual.py      # continual learning + EWC + knowledge store (RAG)
└── online_learner.py # test-time LoRA adaptation

inference/            # generation
├── engine.py         # production engine (sampling, streaming, tools, RAG)
├── streaming.py      # chunked long-context generation
└── fast_numpy.py     # pure-numpy reference implementation

interface/            # agent / app layer
├── app.py            # web app (FastAPI + live training UI)
├── agents.py         # chat / think / tool / learn / code / reasoning agents
├── tools.py          # tool-calling framework (registry + parser + dispatch)
├── web_explorer.py   # stdlib web crawler
└── remote_trainer.py # remote training client

examples/             # runnable demos
├── quickstart.py         # build → train → generate in <2 min on CPU
└── train_shakespeare.py  # train on Shakespeare text

tests/                # 160 passing tests (57 FNN + 103 PRISM)
└── prism/            # PRISM test suite
```

---

## Continuous life-cycle

FNN supports a **continuous training mode** where the model does not learn by epochs but by a cyclical process:

```
WAKE   generate → verify → weight by curiosity → self-critique
SLEEP  episodic → semantic consolidation (SVD rank-r)
META   test-time LoRA (0.1% params, 3-5 steps) if perplexity high
EVOL   architectural darwinism (propose mutation → measure fitness → accept/reject)
```

```python
from nfn import FNNModel, FNNConfig, FNNLifecycle

model = build_fnn_model(vocab_size=110, preset="small")
lifecycle = FNNLifecycle(model, model.cfg, train_seq_len=128)
lifecycle.live(n_cycles=10000, log_every=100)
```

---

## Tests

```bash
pytest tests/ -q
```

57 tests cover the FNN full stack: model forward/backward, every sub-module (memory, causal, phase-goal, reasoning, predictive, MoD, MTP, hyper, streaming), tool-calling, continual learning, and the trainer. An additional 103 tests cover the integrated PRISM backbone (MRB, polymorphic router, memory bus, symbolic library, holographic VSA tape, PCS scaling, CogLoop, curriculum).

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — full architectural walkthrough
- [`docs/MATHEMATICS.md`](docs/MATHEMATICS.md) — derivations of each mechanism
- [`docs/THEORY.md`](docs/THEORY.md) — theoretical motivation
- [`docs/API.md`](docs/API.md) — API reference
- [`PAPER.md`](PAPER.md) — research paper (compact form)

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy (for `fast_numpy.py`)
- Optional: `fastapi`, `uvicorn` (for the web app in `interface/app.py`)

See [`requirements.txt`](requirements.txt).

---

## License

MIT
