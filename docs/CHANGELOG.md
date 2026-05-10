# CHANGELOG — Neural Fractal Network (NFN)

> **Author:** Philippe-Antoine Robert  
> **License:** Proprietary — Philippe-Antoine Robert, all rights reserved

---

## [5.0.0] — 2026-05-10 — All-in-one app, autonomous learning, TTL adaptation

### New files

#### `nfn/online_learner.py` — Test-time learning (TTL)
- `LoRALinear`: wraps `nn.Linear` with rank-r A/B matrices (B init=0 → adapter starts silent)
- `OnlineLearner`: injects adapters into all attention/MLP projections in the model
- **Base weights stay frozen** — only adapters (~0.1% of params) update
- `adapt(ids)`: N gradient steps on adapters for a new context sequence
- `adapt_from_text(text)`: tokenise then adapt
- Confidence gate `ppl_gate`: skip update if model already knows the content (ppl < threshold)
- `decay(steps)`: exponential decay of adapters (controlled forgetting)
- `save_adapters / load_adapters`: persistence across sessions
- Proxy `.weight / .bias / .in_features / .out_features` for `nn.MultiheadAttention` compatibility

#### `nfn/web_explorer.py` — Internet navigation (stdlib only)
- `WebExplorer.fetch(url)`: fetch + clean text extraction via `html.parser`
  - Strips `<script>`, `<style>`, `<nav>`, `<footer>`, `<aside>`, `<header>`
  - Returns `{url, title, text, n_chars, error?}`
- `extract_links(html, base_url)`: extract and normalise absolute links
- `score_link(url, text, keywords)`: keyword overlap score for guided navigation
- `explore(seed_url, n_pages, keywords)`: autonomous navigation generator
- Zero extra dependencies — `urllib.request` + `html.parser` only

#### `train_agi.py` — AGI training entry point
- 5 simultaneous training signals with curriculum
- `--ttl`, `--adapter-rank`, `--online-lr`, `--online-steps`, `--ppl-gate` flags
- `--eval-every` for periodic held-out perplexity
- `--save-adapters / --load-adapters` for session continuity

#### `start.bat` / `start.sh` — Windows / Linux-Mac launchers
- Windows double-click → checks Python, starts server, opens browser
- Guided error messages if Python is missing

### Improvements

#### `training/agi_trainer.py` — AGITrainer v5.0 (full rewrite)
- `AGITextDataset`: tokenisation + sliding window
- `SelfPlayBuffer`: deque of (prompt, winner, loser, score_delta)
- `CuriosityWeighter`: per-token entropy weighting via softmax
- `_forward_step()`: WAKE forward with curiosity weighting
- `_self_play_step()`: generate N candidates, DPO-lite + distillation toward winner
- `_critique_step()`: generate → critique → revision → train on revision at 2× weight
- `_sleep_cycle()`: `maybe_consolidate()` + replay buffer
- Curriculum: LM only → progressive ramp → full AGI losses

#### `interface/app.py` — New routes
- `POST /api/explore/url`: fetch URL + OnlineLearner adaptation, returns ppl before/after
- `POST /api/explore/text`: adapt from raw text
- `WS /ws/explore`: autonomous exploration stream (asyncio.to_thread for blocking fetch)
- `GET /api/ttl/stats`: adapter statistics
- `POST /api/ttl/enable`: create OnlineLearner on the live model
- `POST /api/ttl/disable`: remove adapters
- `POST /api/ttl/reset`: zero all adapter weights

#### `interface/static/index.html + app.js` — 2 new tabs
- **Explore the Web**: URL fetch, autonomous exploration, adapt from text
- **TTL Adaptation**: enable/disable toggle, rank/lr/steps controls, live stats, reset

#### `run.py` — Improved launcher
- `--ttl`, `--adapter-rank` flags
- Tries `pywebview` for native window; falls back to browser if not installed
- Startup banner with active features

### Bug fixes
- `tests/test_agi.py`: `consolidate_every` → `sleep_every` (renamed in AGITrainer)
- `LoRALinear`: proxy `.weight`, `.bias`, `.in_features`, `.out_features` for `nn.MultiheadAttention` compatibility

---

## [4.0.0] — 2026-05-03 — Full AGI stack

### New modules
- `nfn/episodic_memory.py` — `TwoTierMemory`: episodic ring + semantic SVD condensate
- `nfn/causal.py` — `CausalGraphLayer`: DAG + do-calculus interventions
- `nfn/goal.py` — `PhaseGoalPredictor`: Kuramoto forcing λ·sin(θ*−θ)
- `nfn/reasoning.py` — `RecursiveReasoner`: differentiable ACT halting
- `nfn/predictive.py` — `PredictiveCodingBlock`: hierarchical prediction error
- `nfn/hyper.py` — `ContextHyperNet`: context-conditioned weight generation
- `nfn/ssm.py` — `FractalSSM`: Mamba-style SSM with Mandelbrot frequencies
- `nfn/agi_block.py` — `AGIBlock`: unified v4.0 block
- `nfn/agi_model.py` — `AGINFNModel`: full stack + speculative decode
- `training/losses.py` — `AGILoss`: multi-objective aggregator
- `training/agi_trainer.py` — initial `AGITrainer` (replaced in v5.0)
- `inference/engine.py` — `AGIInferenceEngine`: streaming, beam, speculative
- `interface/app.py` — full FastAPI interface
- `interface/agents.py` — Chat/Code/Reasoning agents with tools

---

## [3.2.0] — 2026-05-03 — EfficientNFN: Architecture of 2099

### New Features

#### EfficientNFNBlock Architecture
- **FractalLinearAttention** — Attention $O(L \cdot d^2)$ via linear kernel trick, multi-scale feature map using Mandelbrot frequencies per level. Speedup vs standard attention: 4× at L=512, 16× at L=4096, **255× at L=32768**.
- **PhaseSoliton** — Long-range phase coherence preservation: amplifies coherent patterns, suppresses noise. Replaces LSTM/RNN dependencies with $O(L \cdot n_p)$ complexity.
- **PhaseRoutedMoE** — Expert routing using von Mises distribution over angular distance. Top-K sparse, self-balancing without auxiliary loss. Expert phases initialized from the Farey/Mandelbrot sequence.

#### EfficientNFNLanguageModel
- Complete stack: `AnalyticTokenEmbedding → EfficientNFNBlocks → NFMC condensate → ZipfianDecoder`
- Initial Mandelbrot condensation without corpus (`_seed_condensate_mandelbrot`)
- `condense_from_text()`: one-shot refinement from a corpus (no SGD)
- `param_summary()`: breakdown by component (embed, attn, MoE, soliton, head)

#### New file: `nfn/moe.py`
- `PhaseEncoder`: differentiable phase encoding
- `PhaseRoutedMoE`: sparse Top-K dispatch with von Mises gating
- `FractalLinearAttention`: linear causal attention via cumulative sum
- `PhaseSoliton`: phase coherence with adaptive gate

#### New file: `nfn/efficient_block.py`
- `EfficientNFNBlock`: unified block FLA + Soliton + MoE
- `EfficientNFNLanguageModel`: full v3.2 model

### Performance Metrics
| L | Standard Attention | FractalLinearAttn | Gain |
|---|--------------------|-------------------|------|
| 512 | 33.6M FLOPs | 8.4M FLOPs | **4×** |
| 4096 | 2.15B FLOPs | 134M FLOPs | **16×** |
| 32768 | 137B FLOPs | 537M FLOPs | **255×** |

### Hyperparameters added to `NFNConfig`
- `moe_n_experts` (default: 8)
- `moe_top_k` (default: 2)
- `moe_d_ff_per_expert` (default: 256)

---

## [3.1.0] — 2026-04-28 — ZeroShotNFMC: Analytic Intelligence Without Training

### New Features

#### Zero-parameter Analytic Embedding
- **FractalCodepointEmbedding**: pre-computed Fourier table on codepoints, fixed buffer (0 learned params).
- **CharClassEmbedding**: 16 morphological features (vowel, consonant, digit, MD5 hash, etc.), fixed buffer.
- **AnalyticTokenEmbedding**: fusion via fixed orthogonal QR projection + LayerNorm without parameters. Result: **0 learned parameters** in embedding.

#### New file: `nfn/analytic_embed.py`

#### Mandelbrot Frequencies and Farey Sequence
- `farey_sequence(n)`: Farey mediant algorithm
- `mandelbrot_frequencies(n)`: external angles of Mandelbrot bulbs, sorted by increasing period

#### Modern Hopfield Memory
- **ModernHopfieldMemory**: Fourier patterns (Mandelbrot) + Zipf + QR. Capacity $O(e^{d/2})$.
- **ZipfianDecoder**: optimal Zipf prior for an unknown vocabulary.
- **CausalPhasePredictor**: strictly causal phase prediction.

#### New file: `nfn/hopfield.py`

#### ZeroShotNFMC (complete rewrite of `nfn/nfmc.py`)

### Convergence Results
| Model | Initial Loss | After 100 steps | Improvement |
|-------|--------------|-----------------|-------------|
| Transformer baseline | 4.70 | 2.88 | −38.7% |
| ZeroShotNFMC v3.1 | 5.96 | **1.47** | **−75.3%** |

### Bug Fix
- `AnalyticTokenEmbedding`: `nn.LayerNorm` was creating 128 hidden parameters. Fixed with `elementwise_affine=False`.

---

## [3.0.0] — 2026-04-20 — NFMC: Condensed Multidimensional Fractal Kernel

### New Features

#### New file: `nfn/condensate.py`
- **FractalRFF**: multi-scale random Fourier features, fixed buffers.
- **SpectralCondensate**: one-shot SVD via `torch.linalg.svd`.
- **HelmholtzPhaseLocking**: gradient descent on XY energy.
- **NFMCKernelLayer**: plug-in layer for `NFNBlock`.

#### NFMCLanguageModel (`nfn/nfmc.py`)
- `condense_from_text()`: one-shot SVD, no SGD
- `condense_vocabulary()`: initialization via bigrams

#### New config: `configs/large.json`
- d=1024, 12 blocks, use_nfmc=true, rank r=128, context 128K

### Hyperparameters added
- `use_nfmc`, `nfmc_n_rff`, `nfmc_n_scales`, `nfmc_rank`
- `nfmc_n_phases`, `nfmc_lock_iter`, `nfmc_eta`, `nfmc_lambda_phase`

---

## [2.0.0] — 2026-04-10 — Flash Attention, RoPE/NTK, KV-Cache, Kuramoto, Memory, Multi-GPU

### Major New Features

- **Flash Attention** via `torch.nn.functional.scaled_dot_product_attention`
- **RoPE + NTK scaling**: context extension 4K → 32K+ without retraining
- **Fractal KV-Cache**: `nfn/kv_cache.py` — O(1) per token
- **Kuramoto ODE RK4**: `nfn/phase_ode.py` — BPTP compatible
- **Persistent Memory**: `nfn/memory.py` — M slots, EMA gated
- **3-tier Tokenizer**: `nfn/tokenizer.py` — tiktoken/BPE/char
- **Multi-GPU DDP/FSDP**: `training/distributed.py`
- **Gradient accumulation** + `torch.compile` + grad checkpointing
- New API endpoints: `/api/condense`, `/api/memory/reset`, `/api/memory/state`

---

## [1.0.0] — 2026-04-01 — Initial NFN Implementation

- Fractal topology: binary tree + Cantor
- SinusoidalAggregator / SinusoidalBroadcast
- NFNBlock: MotifBranches + CausalSelfAttention + TemporalRefinement
- NFNLanguageModel, NFNTrainer, NFNInferenceEngine
- FastAPI server + WebSocket streaming
- nano/small/medium configs

---

## Planned Future Versions

### [4.0.0] — Planned
- **Continuous-Time Fractal Network**: fractal neural ODE integrated via adaptive RK
- **Sparse Fractal Attention**: FractalLinearAttention + windowed local attention (>1M tokens)
- **Fractal Meta-learning**: adaptive topology based on data type

---

*Philippe-Antoine Robert — 2026-05-03 07:22:48 UTC*  
*"Every version is a leap, not a step."*
