# CHANGELOG — Neural Fractal Network (NFN)

> **Auteur :** Philippe-Antoine Robert  
> **Licence :** Propriétaire — Philippe-Antoine Robert, tous droits réservés

---

## [3.2.0] — 2026-05-03 — EfficientNFN : Architecture de 2099

### Nouveautés

#### Architecture EfficientNFNBlock
- **FractalLinearAttention** — Attention $O(L \cdot d^2)$ via trick du noyau linéaire, feature map multi-échelles par fréquences de Mandelbrot par niveau. Speedup vs attention standard : 4× à L=512, 16× à L=4096, **255× à L=32768**.
- **PhaseSoliton** — Préservation de cohérence de phase long-range : amplifie les patterns cohérents, supprime le bruit. Remplace les dépendances LSTM/RNN avec $O(L \cdot n_p)$ complexité.
- **PhaseRoutedMoE** — Routage d'experts par distribution de von Mises sur la distance angulaire. Top-K sparse, auto-équilibré sans perte auxiliaire. Phases d'experts initialisées depuis la suite de Farey/Mandelbrot.

#### EfficientNFNLanguageModel
- Stack complet : `AnalyticTokenEmbedding → EfficientNFNBlocks → NFMC condensate → ZipfianDecoder`
- Condensation Mandelbrot initiale sans corpus (`_seed_condensate_mandelbrot`)
- `condense_from_text()` : raffinement one-shot depuis un corpus (pas de SGD)
- `param_summary()` : décomposition par composant (embed, attn, MoE, soliton, head)

#### Nouveau fichier : `nfn/moe.py`
- `PhaseEncoder` : encodage de phase différentiable
- `PhaseRoutedMoE` : dispatch sparse Top-K avec gating von Mises
- `FractalLinearAttention` : attention causale linéaire par somme cumulée
- `PhaseSoliton` : cohérence de phase avec gate adaptative

#### Nouveau fichier : `nfn/efficient_block.py`
- `EfficientNFNBlock` : bloc unifié FLA + Soliton + MoE
- `EfficientNFNLanguageModel` : modèle complet v3.2

### Métriques de performance
| L | Attention standard | FractalLinearAttn | Gain |
|---|--------------------|-------------------|------|
| 512 | 33.6M FLOPs | 8.4M FLOPs | **4×** |
| 4096 | 2.15B FLOPs | 134M FLOPs | **16×** |
| 32768 | 137B FLOPs | 537M FLOPs | **255×** |

### Hyperparamètres ajoutés à `NFNConfig`
- `moe_n_experts` (défaut : 8)
- `moe_top_k` (défaut : 2)
- `moe_d_ff_per_expert` (défaut : 256)

---

## [3.1.0] — 2026-04-28 — ZeroShotNFMC : Intelligence Analytique Sans Entraînement

### Nouveautés

#### Embedding analytique zéro-paramètre
- **FractalCodepointEmbedding** : table pré-calculée par Fourier sur les codepoints, buffer fixe (0 params appris).
- **CharClassEmbedding** : 16 features morphologiques (voyelle, consonne, chiffre, hash MD5, etc.), buffer fixe.
- **AnalyticTokenEmbedding** : fusion via projection QR orthogonale + LayerNorm sans paramètres. Résultat : **0 paramètre appris** dans l'embedding.

#### Nouveau fichier : `nfn/analytic_embed.py`

#### Fréquences de Mandelbrot et suite de Farey
- `farey_sequence(n)` : algorithme de la médiante de Farey
- `mandelbrot_frequencies(n)` : angles externes des bulbes de Mandelbrot, triés par période croissante

#### Mémoire de Hopfield moderne
- **ModernHopfieldMemory** : patterns Fourier (Mandelbrot) + Zipf + QR. Capacité $O(e^{d/2})$.
- **ZipfianDecoder** : prior de Zipf optimal pour un vocabulaire inconnu.
- **CausalPhasePredictor** : prédiction de phase strictement causale.

#### Nouveau fichier : `nfn/hopfield.py`

#### ZeroShotNFMC (réécriture complète de `nfn/nfmc.py`)

### Résultats de convergence
| Modèle | Perte initiale | Après 100 pas | Amélioration |
|--------|----------------|---------------|--------------|
| Transformer baseline | 4.70 | 2.88 | −38.7% |
| ZeroShotNFMC v3.1 | 5.96 | **1.47** | **−75.3%** |

### Correction de bug
- `AnalyticTokenEmbedding` : `nn.LayerNorm` créait 128 paramètres cachés. Corrigé avec `elementwise_affine=False`.

---

## [3.0.0] — 2026-04-20 — NFMC : Noyau Fractal Multidimensionnel Condensé

### Nouveautés

#### Nouveau fichier : `nfn/condensate.py`
- **FractalRFF** : random Fourier features multi-échelles, buffers fixes.
- **SpectralCondensate** : SVD one-shot via `torch.linalg.svd`.
- **HelmholtzPhaseLocking** : descente de gradient sur l'énergie XY.
- **NFMCKernelLayer** : couche plug-in pour `NFNBlock`.

#### NFMCLanguageModel (`nfn/nfmc.py`)
- `condense_from_text()` : SVD one-shot, aucun SGD
- `condense_vocabulary()` : initialisation par bigrammes

#### Nouveau config : `configs/large.json`
- d=1024, 12 blocs, use_nfmc=true, rang r=128, contexte 128K

### Hyperparamètres ajoutés
- `use_nfmc`, `nfmc_n_rff`, `nfmc_n_scales`, `nfmc_rank`
- `nfmc_n_phases`, `nfmc_lock_iter`, `nfmc_eta`, `nfmc_lambda_phase`

---

## [2.0.0] — 2026-04-10 — Flash Attention, RoPE/NTK, KV-Cache, Kuramoto, Mémoire, Multi-GPU

### Nouveautés majeures

- **Flash Attention** via `torch.nn.functional.scaled_dot_product_attention`
- **RoPE + NTK scaling** : extension contexte 4K → 32K+ sans réentraînement
- **KV-Cache fractal** : `nfn/kv_cache.py` — O(1) par token
- **Kuramoto ODE RK4** : `nfn/phase_ode.py` — BPTP compatible
- **Mémoire persistante** : `nfn/memory.py` — M slots, EMA gated
- **Tokenizer 3 niveaux** : `nfn/tokenizer.py` — tiktoken/BPE/char
- **Multi-GPU DDP/FSDP** : `training/distributed.py`
- **Gradient accumulation** + `torch.compile` + grad checkpointing
- Nouveaux endpoints API : `/api/condense`, `/api/memory/reset`, `/api/memory/state`

---

## [1.0.0] — 2026-04-01 — Implémentation initiale NFN

- Topologie fractale : arbre binaire + Cantor
- SinusoidalAggregator / SinusoidalBroadcast
- NFNBlock : MotifBranches + CausalSelfAttention + TemporalRefinement
- NFNLanguageModel, NFNTrainer, NFNInferenceEngine
- FastAPI server + WebSocket streaming
- Configs nano/small/medium

---

## Versions futures prévues

### [4.0.0] — Planifié
- **Continuous-Time Fractal Network** : ODE neurale fractale intégrée par RK adaptatif
- **Sparse Fractal Attention** : FractalLinearAttention + attention locale fenêtrée (>1M tokens)
- **Meta-learning fractal** : topologie adaptative selon le type de données

---

*Philippe-Antoine Robert — 2026-05-03 07:22:48 UTC*  
*"Chaque version est un saut, pas un pas."*
