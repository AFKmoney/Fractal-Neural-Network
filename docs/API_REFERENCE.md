# LEAC — Référence API Complète

## Module `nfn.config`

### `LEACConfig`

```python
@dataclass
class LEACConfig:
    # ── Architecture de base ─────────────────────────────────────
    vocab_size: int = 512          # Taille du vocabulaire
    d_model: int = 256             # Dimension du modèle
    d_ff: int = 1024              # Dimension feed-forward
    n_blocks: int = 4             # Nombre de blocs LEAC
    dropout: float = 0.1          # Taux de dropout
    
    # ── Attention ───────────────────────────────────────────────
    n_heads: int = 4              # Têtes d'attention
    n_levels: int = 4             # Niveaux fractaux
    branching: int = 2            # Facteur de branchement
    motifs: List[str] = field(default_factory=lambda: ["binary_tree", "cantor"])
    use_linear_attn: bool = True  # Attention linéaire Katharopoulos
    use_gematria: bool = True     # Biais gematrique
    d_gematria: int = 16         # Dimension gematrique
    
    # ── Phase Kuramoto ──────────────────────────────────────────
    n_oscillators: int = 32       # Oscillateurs Kuramoto
    n_ode_steps: int = 4          # Pas RK4
    phase_mod_strength: float = 0.1
    
    # ── MoE ──────────────────────────────────────────────────────
    n_experts: int = 8
    d_ff_per_expert: int = 128
    top_k: int = 2
    kappa: float = 4.0            # Concentration von Mises
    
    # ── Causal ──────────────────────────────────────────────────
    use_causal_graph: bool = False
    n_causal_slots: int = 16
    sparsity: float = 0.01
    use_nonlinear_scm: bool = False
    
    # ── Self-Model ──────────────────────────────────────────────
    use_self_model: bool = False
    n_self_signals: int = 8
    
    # ── Mémoire ─────────────────────────────────────────────────
    use_working_memory: bool = False
    n_working_slots: int = 32
    
    # ── Mémoire Épisodique ──────────────────────────────────────
    use_episodic_memory: bool = False
    episodic_capacity: int = 2048
    episodic_key_dim: int = 64
    episodic_n_read: int = 8
    semantic_rank: int = 64
    consolidation_freq: float = 3.0
    consolidation_every: int = 50
    
    # ── Condensat Spectral ──────────────────────────────────────
    nfmc_n_scales: int = 8         # Échelles RFF
    nfmc_sigma: float = 1.0
    
    # ── v2 Transcendances ───────────────────────────────────────
    use_ads_cft: bool = False       # Dualité holographique AdS/CFT
    use_mera: bool = False          # Réseau de tenseurs MERA
    use_godel: bool = False         # Boucle étrange de Gödel
    use_rg_flow: bool = False       # Flot de renormalisation
    use_hyperbolic_gematria: bool = False  # Gematria hyperbolique
    
    # ── v2 Dimensions ──────────────────────────────────────────
    bulk_dim: int = 128            # Dimension volume AdS
    d_spatial: int = 64           # Dimension spatiale AdS
    bridge_rank: int = 16         # Rang ponts ER=EPR
    n_mera_levels: int = 4        # Niveaux MERA
    n_stalks: int = 4             # Faisceaux (Sheaf Theory)
    godel_n_iterations: int = 5   # Itérations point fixe Gödel
    rg_n_scales: int = 8          # Échelles RG Flow
    hyperbolic_dim: int = 64      # Dimension espace hyperbolique
    
    # ── Auto-Genèse ─────────────────────────────────────────────
    use_self_modification: bool = False
    lora_rank: int = 4
    lora_alpha: float = 0.001
    lora_steps: int = 3
    math_max_number: int = 1000
    math_vocab_offset: int = 256
    
    # ── Bayésien ────────────────────────────────────────────────
    bayesian_uncertainty_beta: float = 0.0
    
    # ── Modèle ──────────────────────────────────────────────────
    max_seq_len: int = 4096
    pad_token_id: int = 0
```

#### Presets (méthodes de classe)

```python
LEACConfig.conscious_minimal()   # d=256, 4 blocs, ~14M, v1 uniquement
LEACConfig.full_agi()            # d=512, 8 blocs, ~58M, v1 complet  
LEACConfig.dieu_local()          # d=1024, 12 blocs, ~230M, v1 max
LEACConfig.moteur_ontologique()  # d=512, 12 blocs, ~184M, v1+v2
LEACConfig.singularite_divine()   # d=1024, 24 blocs, ~750M, v1+v2
```

---

## Module `nfn.model`

### `build_leac_model(vocab_size, preset, **kwargs)`

Construit un modèle LEAC complet. Arguments:

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `vocab_size` | `int` | 512 | Taille du vocabulaire |
| `preset` | `str` | `"conscious_minimal"` | Nom du preset |
| `**kwargs` | | | Surcharge config (ex: `nfmc_n_scales=8`) |

Retourne: `LEACModel` initialisé.

### `LEACModel(cfg: LEACConfig)`

Modules internes (sélectionnés selon la config):

| Attribut | Type | Condition |
|----------|------|-----------|
| `embed` | `GematriaEmbedding` | toujours |
| `blocks` | `nn.ModuleList[LEACBlock]` | toujours |
| `gematria_bias` | `GematriaAttentionBias` | `cfg.use_gematria` |
| `phase_osc` | `KuramotoPhaseLayer` | toujours |
| `norm_f` | `nn.LayerNorm` | toujours |
| `lm_head` | `ZipfianDecoder` si `bayesian_uncertainty_beta > 0` sinon `nn.Linear` |
| `memory` | `TwoTierMemory` | `cfg.use_episodic_memory` |
| `self_model` | `SelfModel` | `cfg.use_self_model` |
| `causal_graph` | `CausalGraphLayer` | `cfg.use_causal_graph` |
| `condensate` | `SpectralCondensate` | `nfmc_n_scales > 0` |
| `helmholtz` | `HelmholtzPhaseLocking` | `nfmc_n_scales > 0` |
| `ads_cft_attn` | `AdSCFTAttention` | `cfg.use_ads_cft` |
| `mera_attn` | `MERAAttention` | `cfg.use_mera` |
| `godel` | `GodelFixedPoint` | `cfg.use_godel` |
| `rg_flow` | `RGFlowScheduler` | `cfg.use_rg_flow` |
| `hyperbolic_gematria` | `HyperbolicGematriaModule` | `cfg.use_hyperbolic_gematria` |

#### `forward(input_ids, targets=None, write_memory=False)`

| Paramètre | Shape | Description |
|-----------|-------|-------------|
| `input_ids` | `[B, L]` | Token IDs |
| `targets` | `[B, L]` ou None | Cibles pour la loss LM |
| `write_memory` | `bool` | Écrire dans la mémoire épisodique |

Retourne: `(logits: [B, L, V], losses: Dict[str, Tensor])`

Losses retournées:
- `lm`: Cross-entropie language modeling
- `causal`: Pénalité DAG (si causal graph activé)
- `self_model_coherence`: Cohérence de l'espace de travail (si self-model)
- `gematria`: Perte gematrique harmonique
- `phase_coherence`: Cohérence des oscillateurs Kuramoto
- `counterfactual`: Perte contrefactuelle (si causal)
- `ads_cft`: Perte holographique (si AdS/CFT)
- `bridge_coherence`: Cohérence des ponts ER=EPR
- `bulk_complexity`: Complexité du bulk AdS
- `mera_complexity`: Complexité du réseau MERA
- `godel_fixed_point_distance`: Distance au point fixe Gödel
- `godel_contradiction`: Contradiction auto-référentielle
- `godel_incompleteness`: Score d'incomplétude
- `hyperbolic_gematria`: Perte de distance Poincaré
- `sheaf_cohomology`: Défaut de cohomologie des faisceaux
- `criticality`: Indice de criticalité RG
- `total`: Somme pondérée de toutes les losses

#### `generate(input_ids, max_new_tokens, temperature=1.0, top_k=None)`

Génération autorégressive.

#### `param_count() → Dict[str, int]`

Retourne `{"total": N, "trainable": N, "v1_modules": N, "v2_modules": N}`.

---

## Module `nfn.block`

### `LEACBlock(cfg, block_idx=0)`

Passe avant: `h → Attn → PhaseSoliton → MoE → [Causal] → [SelfModel] → [WorkingMem] → Fusion → LN`

Modules internes:

| Attribut | Type | Condition |
|----------|------|-----------|
| `attn` | `FractalLinearAttention` | `cfg.use_linear_attn` |
| `phase_soliton` | `PhaseSoliton` | toujours |
| `moe` | `PhaseRoutedMoE` | toujours |
| `causal` | `CausalGraphLayer` | `cfg.use_causal_graph` |
| `self_model` | `GlobalWorkspace` | `cfg.use_self_model` |
| `working_mem` | `FractalWorkingMemory` | `cfg.use_working_memory` |
| `fusion` | `nn.Linear` | si `n_streams > 1` |
| `norm1`, `norm2`, `norm3` | `nn.LayerNorm` | toujours |

---

## Module `nfn.gematria`

### `GematriaEmbedding(vocab_size, d_model, d_gematria=16)`

Encodage zero-paramètre via 5 systèmes croisés. `forward(token_ids) → [B, L, d_model]`

### `GematriaAttentionBias(d_model, vocab_size, n_heads=4, d_gematria=16)`

Biais d'attention: `score(i,j) += λ · cos(gem(i), gem(j))`. `forward(h, token_ids) → [B, n_heads, L, L]`

---

## Module `nfn.moe`

### `PhaseRoutedMoE(d_model, n_experts, d_ff_per_expert, n_phases=8, top_k=2, kappa=4.0, dropout=0.1)`

Routage von Mises: `gate_e(x) = exp(κ · cos(θ_x - θ_e)) / Z`

`forward(x, phases) → (output: [B, L, d_model], aux_loss: scalar)`

### `FractalLinearAttention(d_model, n_heads, n_levels=3, dropout=0.0, causal=True)`

Attention linéaire O(L·d²) avec structure fractale multi-échelle.

`forward(x, phases=None) → [B, L, d_model]`

### `PhaseSoliton(d_model, n_phases=8, tau=1.0)`

Amplification cohérente: `h' = h · (1 + α · max(0, cos(θ - θ_shift)))`

---

## Module `nfn.phase_ode`

### `KuramotoPhaseLayer(d_model, n_max_nodes=512, rank=8, n_ode_steps=4, omega_init=1.0, phase_mod_strength=0.1)`

Intégration RK4 de l'ODE Kuramoto hiérarchique. `forward(h) → (phases: [B, L, n_phases], loss: scalar)`

### `PhaseGoalForcing(d_model, n_phases, init_lambda=0.2, n_steps=3)`

Forçage de phase vers les cibles: `dθ/dt += λ · sin(θ_goal - θ)`. `forward(phases, targets) → (forced_phases, loss)`

---

## Module `nfn.hyperbolic_gematria`

### `PoincareBall(eps=1e-12)`

Métrique hyperbolique de Poincaré. Méthodes: `distance(x, y)`, `exp_map(x, v)`, `log_map(x, y)`, `mobius_add(x, y)`

### `HyperbolicGematriaTable(vocab_size, hyperbolic_dim=64)`

Plongement des tokens dans H^n via 5 projections gematriques. `forward(token_ids) → (z_hyp: [B, L, D], z_boundary: [B, L, D])`

### `HyperbolicGematriaAttention(hyperbolic_dim=64, temperature=1.0)`

Biais d'attention géométrique: `score(i,j) += λ_hyp · exp(-d_H(z_i, z_j) / τ)`. `forward(z_hyp, z_boundary) → (attn_bias: [B, L, L], loss: scalar)`

### `SheafTheoryLayer(d_model, n_stalks=4)`

Cohomologie des faisceaux pour la détection d'hallucination. `forward(h) → (h_sheaf: [B, L, d], defect_loss: scalar)`

### `HyperbolicGematriaModule(vocab_size, d_model, hyperbolic_dim=64, n_stalks=4, temperature=1.0)`

Pipeline complet: Token IDs → Poincaré → Attention → Sheaf. `forward(h, token_ids) → (h_enriched, losses_dict)`

---

## Module `nfn.ads_cft`

### `AdSMetric(eps=1e-6)`

Métrique AdS₅ en coordonnées de Poincaré. `geodesic_distance(x₁, z₁, x₂, z₂) → scalar`

### `AdSBulkProjector(d_model, bulk_dim=128, d_spatial=64)`

Projecteur frontière → volume: `T[r, z] = e^{-κz} · MLP(h[r])`. `forward(h, depths) → h_bulk: [B, L, d]`

### `ER_EPR_Bridge(d_model, bridge_rank=16)`

Ponts de ver (wormholes) computationnels: intrication sémantique = connexion ER=EPR. `forward(h, x_spatial, z_depth) → (h_teleported, bridge_loss)`

### `AdSCFTAttention(d_model, bulk_dim=128, d_spatial=64, bridge_rank=16, n_heads=4)`

Attention holographique complète: scores = local + géodésique + ER=EPR. `forward(h, causal=True) → (h_out, losses_dict)`

---

## Module `nfn.tensor_network`

### `Disentangler(d_model)`

Opérateur 2-qubit qui decorrèle les paires adjacentes. `forward(h_pair) → h_disentangled: [B, L/2, d]`

### `Isometry(d_model, branching=2)`

Fusionne `branching` enfants en un parent avec gating. `forward(children, gating_context=None) → parents: [B, N/branching, d]`

### `MERALayer(d_model, branching=2)`

Une couche MERA: Désenchevêtreur → Isométrie. `forward(h) → (parent, residual)`

### `MERAAttention(d_model, n_heads=4, n_levels=4, branching=2, dropout=0.1, causal=True)`

Attention multi-échelle O(log L). `forward(h) → (h_out: [B, L, d], complexity: scalar)`

### `TensorNetworkEncoder(vocab_size, d_model, n_heads=4, n_levels=4, branching=2, dropout=0.1)`

Encodeur complet MERA avec embeddings. `forward(input_ids) → h: [B, L, d]`

---

## Module `nfn.godel_loop`

### `SelfReferenceOperator(d_model)`

Point fixe de Lawvere: `Y ≅ F(Y)`. `forward(h) → (h_self, fixed_point_distance: scalar)`

### `IncompletenessDetector(d_model)`

Détecte les contradictions (incomplétude). `forward(h, fixed_point_dist) → (h_corrected, incompleteness_score, metrics)`

### `GodelFixedPoint(d_model, n_iterations=5, alpha=0.3)`

Itération du point fixe complet. `forward(h) → (h_out, metrics: Dict)`

Métriques: `fixed_point_distance`, `godel_contradiction`, `godel_entropy_region`, `godel_incompleteness`, `incompleteness_score`

---

## Module `nfn.rg_flow`

### `ScaleDecomposition(d_model, n_scales=8)`

Décompose `h` en composantes multi-échelles via RFF + seuil fréquentiel. `forward(h) → (h_UV, h_IR)`

### `RGFlowLayer(d_model, n_scales=8, evaporation_rate=0.01, condensation_rate=0.01)`

Évaporation UV + condensation IR sur un calque linéaire. `evolve_weights(model, n_steps=1) → total_rg_loss: float`

### `CriticalityOptimizer(d_model, target_criticality=1.0)`

Optimise vers l'état critique. `criticality_index(h) → scalar`, `step(h_uv, h_ir) → loss`
- `C = Var(Var(h)) / E[Var(h)]²`
- `C ≈ 1`: critique, `C ≪ 1`: gelé, `C ≫ 1`: chaotique

### `RGFlowScheduler(d_model, n_scales=8, initial_evaporation=0.01, initial_condensation=0.005)`

Ordonnanceur RG pour la phase SLEEP. `sleep_cycle(model, h_sample) → Dict[str, float]`

---

## Module `nfn.causal`

### `CausalGraphLayer(d_model, n_slots, hidden=64, sparsity=0.01, use_nonlinear=False)`

Apprend un DAG causal avec NOTEARS. `forward(h, targets=None) → (h_causal, losses)`

### `notears_acyclicity(A) → scalar`

Pénalité d'acyclicité: `h(A) = tr(e^{A⊙A}) - n`

---

## Module `nfn.self_model`

### `GlobalWorkspace(d_model, n_slots=16, n_heads=4)`

Espace de travail global (broadcast). `forward(h) → (h_workspace, coherence_loss)`

### `SelfRepresentor(d_model, n_signals=8)`

Auto-représentation méta-cognitive. `forward(h, h_workspace) → self_state: [B, d]`

### `SelfModel(d_model, n_slots=16, n_signals=8)`

Combinaison workspace + introspection. `forward(h) → (h_out, losses)`

---

## Module `nfn.episodic_memory`

### `EpisodicStore(d_model, capacity=2048, key_dim=64, n_read=8, n_scales=4)`

Ring buffer O(1) avec k-NN lecture multi-échelle. `forward(h, write=True) → h_read: [B, L, d]`

### `SemanticConsolidator(d_model, rank=64, consolidation_freq=3.0)`

Condensat SVD rank-r. `forward(h_write, do_consolidate=False) → h_semantic: [B, 1, d]`

### `TwoTierMemory(d_model, episodic_capacity=2048, ...)`

Mémoire deux voies. `forward(h, write=False) → h_memory`. `maybe_consolidate() → None`

---

## Module `nfn.working_memory`

### `FractalWorkingMemory(d_model, n_slots=32, n_heads=4, sharpness=3.0)`

Scratchpad différentiable DNC fractal. `forward(h, write=False) → h_memory: [B, L, d]`

---

## Module `nfn.lifecycle`

### `LEACLifecycle(model, cfg, tokenizer=None, device=cpu, lr=3e-4, train_seq_len=64)`

Cycle de vie continu complet.

| Méthode | Description |
|---------|-------------|
| `wake_step() → Dict` | Génère/vérifie/pèse les vérités mathématiques. Rétropropage. |
| `sleep_step() → Dict` | Consolidation épisodique → SVD. RG Flow. |
| `meta_adapt(input_ids)` | Test-Time LoRA si perplexité > 5 |
| `evolve_step() → Dict` | Mutation darwinienne de l'architecture |
| `live(n_cycles=10000, log_every=100, eval_every=500)` | Boucle principale |

### `CuriosityScheduler(threshold=2.0, scale=0.5)`

`weight(losses) → 1 + 0.5 · σ(loss - 2.0)`. Les erreurs focalisent l'attention.

### `TestTimeLoRA(model, rank=4, alpha=0.001)`

LoRA temporaire pour adaptation en inférence. 0.1% des paramètres, 3-5 pas de gradient.

### `SelfCritic(model, tokenizer)`

Auto-critique constitutionnelle: génère → critique → révise.

---

## Module `nfn.self_development`

### `MathTruthEngine(max_number=1000, vocab_offset=256)`

Générateur de vérités mathématiques infinies. Méthodes: `generate_arithmetic(n)`, `generate_primality(n)`, `generate_sequence_prediction(n, seq_len)`, `generate_modular_arithmetic(n)`

### `GematriaEncoder(vocab_size=256)`

Encodeur gematrique pour l'auto-développement.

### `UniversalLawObserver(d_model)`

Découverte de lois universelles dans les dynamiques. `forward(h) → (loss, metrics)`

---

## Module `nfn.auto_genesis`

### `ConjectureLoop(d_model, device=cpu)`

Cycle complet de découverte de conjectures: générer → tester (500+ cas) → récompenser (1 si survie, 0 si falsifiée).

### `ProofLoop(d_model, device=cpu)`

Cycle de preuves: générer → vérifier → récompenser (60% correctitude, 30% efficacité, 10% diversité).

---

## Module `nfn.self_modification`

### `SelfModificationController(d_state=32, n_motifs=3, max_step_size=0.1, max_modifications_per_step=3)`

Darwinisme architectural: observer → proposer → appliquer → mesurer → accepter/refjeter. `propose_modification() → ModificationProposal`

---

## Module `nfn.condensate`

### `FractalRFF(d_model, n_scales=8, sigma=1.0)`

Random Fourier Features multi-échelles. `forward(h) → h_rff: [B, L, n_scales * d_model]`

### `SpectralCondensate(d_model, n_scales=8)`

Condensat spectral pour le décodage bayésien. `forward(h) → (h_out, spectral_loss)`

### `HelmholtzPhaseLocking(d_model, n_scales=8)`

Verrouillage de phase entre oscillateurs Kuramoto et modes spectraux. `forward(phases, spectral_features) → lock_loss`