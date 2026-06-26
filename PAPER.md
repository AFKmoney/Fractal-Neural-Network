# Fractal Neural Networks: Structured Representation through Phase, Causality, and Self-Similarity

**FNN v6.0 — Research Summary**

> This document is the compact research companion to the FNN codebase. Full derivations live in `docs/MATHEMATICS.md`; this paper motivates the design, states the key equations, and reports what is implemented and verified.

---

## Abstract

Modern language models achieve strong performance through scale, but their internal representations are learned from scratch and carry little built-in structure. We propose the **Fractal Neural Network (FNN)**, an architecture that injects three forms of structure into the representation: (1) a *self-similar fractal topology* over sequence positions, (2) *phase synchronization dynamics* (a differentiable Kuramoto model) that bind related tokens into coherent assemblies, and (3) an explicit *causal graph* that lets the model reason about interventions and counterfactuals rather than only correlations. We further introduce a family of optional layers drawn from physics — holographic (AdS/CFT) attention, MERA tensor networks, Gödel self-reference, and renormalization-group flow — that provide inductive biases for long-range structure, hierarchy, and criticality. The entire embedding pathway is analytic (zero-parameter Fourier + character-class + gematria features), so most of the representational capacity comes from *structure* rather than *parameters*. We implement the full system in PyTorch and verify it trains end-to-end on character-level language modeling.

---

## 1. Motivation

A standard Transformer tokenizes, embeds via a learned lookup table, and processes through identical self-attention blocks. Three observations motivate FNN:

1. **Embeddings are wasteful.** A learned `nn.Embedding(V, d)` costs `V·d` parameters to encode information (character class, frequency, positional periodicity) that is *computable analytically*. FNN replaces most of this with a parameter-free analytic embedding.

2. **Attention is correlation-only.** Self-attention learns what co-occurs, not what *causes* what. This limits reasoning about interventions ("what if X were different?"). FNN adds an explicit causal DAG with differentiable structure learning.

3. **There is no notion of coherent state.** Tokens are processed independently until attention mixes them. FNN treats tokens as *oscillators* and uses phase synchronization (Kuramoto dynamics) as a first-class mechanism: tokens that should belong together lock into the same phase.

---

## 2. Architecture

### 2.1 Fractal topology

Sequence positions are organized into a hierarchy of `n_levels` over a branching factor `b`. Each level-k node aggregates `b^k` leaves. Three motif generators are supported (`topology.py`):

- **Binary tree** — balanced `b`-ary tree
- **Cantor** — middle-third pruning (Cantor set connectivity)
- **Sierpinski** — triangle connectivity

The payoff is computational: higher levels are recomputed exponentially less often. The KV cache (`kv_cache.py`) exploits this — a level-k representation is updated every `b^k` tokens during generation, giving sublinear amortized cost.

### 2.2 Phase synchronization (Kuramoto dynamics)

Each token position `i` carries a phase `θ_i`. Phases evolve under the Kuramoto ODE, integrated by differentiable RK4 (`phase_ode.py`):

$$
\frac{d\theta_i}{dt} = \omega_i + \frac{1}{N}\sum_j K_{ji}\,\sin(\theta_j - \theta_i + \varphi_{ji})
$$

where `ω` are natural frequencies (seeded from the Mandelbrot set's natural resonances), `K` is a low-rank learnable coupling, and `φ` are phase offsets. Synchronized tokens (small `|θ_i - θ_j|`) are amplified; desynchronized tokens are suppressed. This is the model's notion of **coherent context**.

**Goal-directed forcing.** A target phase `Θ*` (the "goal") adds a forcing term `λ·sin(Θ* - Θ)`, biasing generation toward a desired state. A hierarchical decomposer splits a goal into a binary tree of sub-goals.

### 2.3 Causal reasoning

A `CausalGraphLayer` (`causal.py`) maintains a differentiable directed acyclic graph over `n_slots` latent variables. Structure is learned via the NOTEARS acycularity constraint:

$$
\text{Tr}(e^{W \odot W}) - d = 0
$$

where `W` is the weighted adjacency matrix. The layer supports:
- **do-calculus** — intervene by setting a slot to a value and propagating
- **Counterfactuals** — "what would the output have been if slot `k` had value `v`?"
- **Acyclicity penalty** — keeps the learned graph a valid DAG

This gives the model a mechanism for causal, not merely associational, reasoning.

### 2.4 Self-model and global workspace

A `SelfModel` (`self_model.py`) maintains a compressed representation of the network's own state in a global workspace of `n_slots` signals. This is introspection: the model can attend to and report on its own internal configuration. It draws on Global Workspace Theory from cognitive science.

### 2.5 Phase-routed mixture of experts

The feed-forward component is a `PhaseRoutedMoE` (`moe.py`): experts are selected by phase coherence via a Von Mises distribution. Routing weight for expert `e` is proportional to `exp(κ·cos(θ_e - θ_token))`, giving a smooth, differentiable top-k routing that respects the phase structure.

### 2.6 Analytic embedding

The token embedding (`analytic_embed.py`) is **zero-parameter**: it concatenates
- Fourier features `sin(ω·t), cos(ω·t)` at Mandelbrot-derived frequencies
- Character-class one-hots (letter / digit / punctuation / whitespace)
- An n-gram hash projection

A complementary `GematriaEmbedding` (`gematria.py`) encodes each token under **five crossed numerical systems**: ordinal, prime-indexed, Fibonacci-indexed, digital-root, and a small learned residual. These give arithmetic and symbolic structure for free.

### 2.7 Spectral condensate

A `SpectralCondensate` (`condensate.py`) approximates a universal fractal kernel via Random Fourier Features drawn from a Mandelbrot-inspired 1/f spectral measure, then condenses to rank-`r` via a one-shot SVD (no gradient). A Helmholtz phase-locking layer then solves a coupled Kuramoto system on the kernel graph.

### 2.8 Optional advanced layers

Five research layers are available behind config flags, enabled together in the `large` preset:

| Layer | File | Inductive bias |
|-------|------|----------------|
| **AdS/CFT attention** | `ads_cft.py` | Holographic duality: boundary tokens project to a bulk representation and back. Models long-range dependencies as geometric proximity in the bulk. |
| **MERA tensor network** | `tensor_network.py` | Multi-scale Entanglement Renormalization Ansatz: `O(log L)` hierarchical attention via disentanglers + isometries. |
| **Gödel fixed point** | `godel_loop.py` | A self-reference operator that iterates `h ← F(h)` to a fixed point — a mathematical model of introspective stability. |
| **RG flow** | `rg_flow.py` | Renormalization-group-inspired scheduler that evaporates fine-grained (UV) features and condenses coarse (IR) features toward criticality. |
| **Hyperbolic gematria** | `hyperbolic_gematria.py` | Embeds tokens in the Poincaré ball where hierarchical relationships are distance-efficient; a sheaf-cohomology layer detects "hallucination" as a gluing defect. |

### 2.9 Recovered research modules

The framework also integrates several standalone research components (`ssm.py`, `mixture_of_depths.py`, `reasoning.py`, `predictive.py`, `multi_token_pred.py`, `hyper.py`, `program_synthesis.py`, `multimodal.py`) that can wrap or extend a block. These cover Mamba-style recurrence, adaptive computation time, predictive coding, and neuro-symbolic synthesis.

---

## 3. Training

The training signal is multi-objective. The total loss is:

$$
\mathcal{L} = \mathcal{L}_{\text{lm}} + \lambda_{\text{phase}}\mathcal{L}_{\text{phase}} + \lambda_{\text{causal}}\mathcal{L}_{\text{DAG}} + \lambda_{\text{notears}}\mathcal{L}_{\text{acyc}} + \lambda_{\text{goal}}\mathcal{L}_{\text{goal}} + \lambda_{\text{spectral}}\mathcal{L}_{\text{spec}} + \ldots
$$

Every regularizer is differentiable and computed in a single forward pass. This means **structure is learned, not imposed**: the model discovers which causal edges matter, which phases should synchronize, and which frequencies to keep.

A continuous life-cycle mode (`lifecycle.py`) alternates:
- **WAKE** — generate, verify, weight by curiosity, self-critique
- **SLEEP** — consolidate episodic memory into semantic (SVD rank-r)
- **META** — test-time LoRA adaptation (0.1% of params, a few steps) when perplexity is high
- **EVOL** — architectural darwinism: propose a mutation, measure fitness delta, accept/reject

---

## 4. Implementation & Verification

The system is implemented in ~6000 lines of PyTorch across the `nfn/`, `training/`, `inference/`, and `interface/` packages.

**Verified:**
- All 4 presets (`nano`–`large`) build and run forward/backward.
- 57 unit tests pass (`pytest tests/`), covering the model, every core sub-module, tool-calling, continual learning, and the trainer.
- The `quickstart.py` example trains end-to-end on CPU: a nano model (11.5M params) trains 125 steps on a small corpus and generates coherent text.
- The standard trainer runs with AdamW + cosine LR + warmup, mixed precision, gradient accumulation, and gradient checkpointing.

**Complexity notes:**
- Fractal linear attention: `O(L·d²)` vs standard `O(L²·d)`.
- MERA attention: `O(log L)` per token.
- Fractal KV cache: level-k updated every `b^k` tokens → sublinear amortized.

---

## 5. Relation to prior work

- **Katharopoulos et al. (2020)** — linear attention kernel trick (FNN's `FractalLinearAttention`).
- **Kuramoto (1984)** — oscillator synchronization (FNN's phase dynamics).
- **NOTEARS (Zheng et al. 2018)** — continuous acyclicity constraint for DAG learning.
- **Baars (1988)** — Global Workspace Theory (FNN's self-model).
- **MERA (Vidal 2008)** — entanglement renormalization (FNN's tensor network).
- **Mamba (Gu & Dao 2023)** — selective state-space models (FNN's `ssm.py`).
- **Graves (2016)** — Adaptive Computation Time (FNN's `reasoning.py`).

FNN's contribution is **not** any single mechanism but their integration into a single trainable whole where regularization arises from physically- and cognitively-motivated structure.

---

## 6. Limitations & future work

- The advanced layers (AdS/CFT, Gödel, RG) are implemented and importable but their empirical benefit over baselines is not yet benchmarked at scale — they are research directions, not validated wins.
- Associative scan in `ssm.py` is a pure-PyTorch implementation (correct but slower than a CUDA kernel).
- No large-scale pretraining run has been completed; results are on character-level corpora.

Future work: benchmark each structural component against ablations on a standard LM benchmark, implement a fused CUDA associative scan, and evaluate the causal-graph layer on intervention tasks.

---

## References

See `docs/MATHEMATICS.md` for full derivations and `docs/THEORY.md` for extended theoretical discussion.

---

*FNN is released under the MIT license. It is a research framework, not a production language model.*
