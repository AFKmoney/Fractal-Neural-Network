# CHANGELOG — Neural Fractal Network (NFN)

> **Auteur :** Philippe-Antoine Robert  
> **Licence :** Propriétaire — Philippe-Antoine Robert, tous droits réservés

---

## [5.0.0] — 2026-05-10 — App tout-en-un, apprentissage autonome, adaptation TTL

### Nouveaux fichiers

#### `nfn/online_learner.py` — Adaptation test-time (TTL)
- `LoRALinear` : wrapper sur `nn.Linear` avec matrices A/B rang-r (init B=0 → adaptateu muet au démarrage)
- `OnlineLearner` : injecte les adaptateurs dans toutes les projections attention/MLP du modèle
- **Poids de base gelés** — seuls les adaptateurs (~0.1% des params) se mettent à jour
- `adapt(ids)` : N steps de gradient sur les adaptateurs pour un nouveau contexte
- `adapt_from_text(text)` : tokenise puis adapte
- Confidence gate `ppl_gate` : skip si le modèle connaît déjà (ppl < seuil)
- `decay(steps)` : décroissance exponentielle des adaptateurs (oubli contrôlé)
- `save_adapters / load_adapters` : persistance entre sessions
- Proxy `.weight / .bias` pour compatibilité avec `nn.MultiheadAttention`

#### `nfn/web_explorer.py` — Navigation internet (stdlib only)
- `WebExplorer.fetch(url)` : fetch + extraction texte propre via `html.parser`
  - Supprime `<script>`, `<style>`, `<nav>`, `<footer>`, `<aside>`, `<header>`
  - Retourne `{url, title, text, n_chars, error?}`
- `extract_links(html, base_url)` : extraction + normalisation des liens absolus
- `score_link(url, text, keywords)` : score par overlap de mots-clés pour navigation guidée
- `explore(seed_url, n_pages, keywords)` : générateur de navigation autonome
- Aucune dépendance externe — `urllib.request` + `html.parser` uniquement

#### `train_agi.py` — Point d'entrée entraînement AGI
- 5 signaux d'entraînement simultanés avec curriculum
- Flags `--ttl`, `--adapter-rank`, `--online-lr`, `--online-steps`, `--ppl-gate`
- `--eval-every` pour perplexité validation périodique
- `--save-adapters / --load-adapters` pour continuité des sessions

#### `start.bat` / `start.sh` — Lanceurs Windows / Linux-Mac
- Double-clic Windows → vérifie Python, démarre le serveur, ouvre le navigateur
- Messages d'erreur guidés si Python manquant

### Améliorations

#### `training/agi_trainer.py` — AGITrainer v5.0 (réécriture complète)
- `AGITextDataset` : tokenisation + fenêtre glissante
- `SelfPlayBuffer` : deque (prompt, gagnant, perdant, delta_score)
- `CuriosityWeighter` : pondération par entropie par token via softmax
- `_forward_step()` : forward WAKE avec curiosité
- `_self_play_step()` : génère N candidats, DPO-lite + distillation
- `_critique_step()` : génère → critique → révision → entraîne sur révision ×2
- `_sleep_cycle()` : `maybe_consolidate()` + replay buffer
- Curriculum : LM seulement → montée progressive → pertes AGI complètes

#### `interface/app.py` — Nouvelles routes
- `POST /api/explore/url` : fetch URL + adaptation OnlineLearner, retourne ppl avant/après
- `POST /api/explore/text` : adaptation depuis texte brut
- `WS /ws/explore` : stream d'exploration autonome (asyncio.to_thread pour les appels bloquants)
- `GET /api/ttl/stats` : statistiques des adaptateurs
- `POST /api/ttl/enable` : créer OnlineLearner sur le modèle courant
- `POST /api/ttl/disable` : supprimer les adaptateurs
- `POST /api/ttl/reset` : remettre tous les adaptateurs à zéro

#### `interface/static/index.html + app.js` — 2 nouveaux onglets
- **Explorer le Web** : URL fetch, exploration autonome, adaptation depuis texte
- **Adaptation TTL** : toggle activation, contrôles rank/lr/steps, stats live, reset

#### `run.py` — Lanceur amélioré
- Flags `--ttl`, `--adapter-rank`
- Tente `pywebview` pour fenêtre native ; se rabat sur le navigateur si absent
- Bannière de démarrage avec features actives

### Corrections
- `tests/test_agi.py` : `consolidate_every` → `sleep_every` (renommé dans AGITrainer)
- `LoRALinear` : proxy `.weight`, `.bias`, `.in_features`, `.out_features` pour compatibilité `nn.MultiheadAttention`

---

## [4.0.0] — 2026-05-03 — Stack AGI complet

### Nouveaux modules
- `nfn/episodic_memory.py` — `TwoTierMemory` : anneau épisodique + condensat sémantique SVD
- `nfn/causal.py` — `CausalGraphLayer` : DAG + interventions do-calculus
- `nfn/goal.py` — `PhaseGoalPredictor` : forçage Kuramoto λ·sin(θ*−θ)
- `nfn/reasoning.py` — `RecursiveReasoner` : ACT halting différentiable
- `nfn/predictive.py` — `PredictiveCodingBlock` : erreur de prédiction hiérarchique
- `nfn/hyper.py` — `ContextHyperNet` : génération de poids par contexte
- `nfn/ssm.py` — `FractalSSM` : SSM style Mamba avec fréquences Mandelbrot
- `nfn/agi_block.py` — `AGIBlock` : bloc unifié v4.0
- `nfn/agi_model.py` — `AGINFNModel` : stack complet + décodage spéculatif
- `training/losses.py` — `AGILoss` : agrégateur multi-objectif
- `training/agi_trainer.py` — `AGITrainer` : entraîneur initial (remplacé en v5.0)
- `inference/engine.py` — `AGIInferenceEngine` : streaming, beam, spéculatif
- `interface/app.py` — Interface FastAPI complète
- `interface/agents.py` — Agents Chat/Code/Raisonnement avec outils

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
