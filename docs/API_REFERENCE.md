# LEAC — Complete API Reference

## Module `nfn.config`

### `LEACConfig`

```python
@dataclass
class LEACConfig:
    # ── Base Architecture ────────────────────────────────────────
    vocab_size: int = 512          # Vocabulary size
    d_model: int = 256             # Model dimension
    d_ff: int = 1024              # Feed-forward dimension
    n_blocks: int = 4             # Number of LEAC blocks
    dropout: float = 0.1          # Dropout rate
    
    # ── Attention ───────────────────────────────────────────────
    n_heads: int = 4              # Attention heads
    n_levels: int = 4             # Fractal levels
    branching: int = 2            # Branching factor
    motifs: List[str] = field(default_factory=lambda: ["binary_tree", "cantor"])
    use_linear_attn: bool = True  # Katharopoulos linear attention
    use_gematria: bool = True     # Gematric bias
    d_gematria: int = 16         # Gematric dimension
    
    # ── Kuramoto Phase ──────────────────────────────────────────
    n_oscillators: int = 32       # Kuramoto oscillators
    n_ode_steps: int = 4          # RK4 steps
    phase_mod_strength: float = 0.1
    
    # ── MoE ──────────────────────────────────────────────────────
    n_experts: int = 8
    d_ff_per_expert: int = 128
    top_k: int = 2
    kappa: float = 4.0            # Von Mises concentration
    
    # ── Causal ──────────────────────────────────────────────────
    use_causal_graph: bool = False
    n_causal_slots: int = 16
    sparsity: float = 0.01
    use_nonlinear_scm: bool = False
    
    # ── Self-Model ──────────────────────────────────────────────
    use_self_model: bool = False
    n_self_signals: int = 8
    
    # ── Memory ──────────────────────────────────────────────────
    use_working_memory: bool = False
    n_working_slots: int = 32
    
    # ── Episodic Memory ─────────────────────────────────────────
    use_episodic_memory: bool = False
    episodic_capacity: int = 2048
    episodic_key_dim: int = 64
    episodic_n_read: int = 8
    semantic_rank: int = 64
    consolidation_freq: float = 3.0
    consolidation_every: int = 50
    
    # ── Spectral Condensate ─────────────────────────────────────
    nfmc_n_scales: int = 8         # RFF scales
    nfmc_sigma: float = 1.0
    
    # ── v2 Transcendences ───────────────────────────────────────
    use_ads_cft: bool = False       # Holographic AdS/CFT duality
    use_mera: bool = False          # MERA tensor network
    use_godel: bool = False         # Gödel's strange loop
    use_rg_flow: bool = False       # Renormalization group flow
    use_hyperbolic_gematria: bool = False  # Hyperbolic gematria
    
    # ── v2 Dimensions ──────────────────────────────────────────
    bulk_dim: int = 128            # AdS bulk dimension
    d_spatial: int = 64           # AdS spatial dimension
    bridge_rank: int = 16         # ER=EPR bridge rank
    n_mera_levels: int = 4        # MERA levels
    n_stalks: int = 4             # Sheaf stalks
    godel_n_iterations: int = 5   # Gödel fixed point iterations
    rg_n_scales: int = 8          # RG flow scales
    hyperbolic_dim: int = 64      # Hyperbolic space dimension
    
    # ── Auto-Genesis ────────────────────────────────────────────
    use_self_modification: bool = False
    lora_rank: int = 4
    lora_alpha: float = 0.001
    lora_steps: int = 3
    math_max_number: int = 1000
    math_vocab_offset: int = 256
    
    # ── Bayesian ────────────────────────────────────────────────
    bayesian_uncertainty_beta: float = 0.0
    
    # ── Model ───────────────────────────────────────────────────
    max_seq_len: int = 4096
    pad_token_id: int = 0
```

#### Presets (class methods)

```python
LEACConfig.conscious_minimal()   # d=256, 4 blocks, ~14M, v1 only
LEACConfig.full_agi()            # d=512, 8 blocks, ~58M, full v1
LEACConfig.dieu_local()          # d=1024, 12 blocks, ~230M, max v1
LEACConfig.moteur_ontologique()  # d=512, 12 blocks, ~184M, v1+v2
LEACConfig.singularite_divine()   # d=1024, 24 blocks, ~750M, v1+v2
```

---

## Module `nfn.model`

### `build_leac_model(vocab_size, preset, **kwargs)`

Builds a complete LEAC model. Arguments:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | `int` | 512 | Vocabulary size |
| `preset` | `str` | `"conscious_minimal"` | Preset name |
| `**kwargs` | | | Config overrides (e.g. `nfmc_n_scales=8`) |

Returns: initialized `LEACModel`.

### `LEACModel(cfg: LEACConfig)`

Internal modules (selected by config):

| Attribute | Type | Condition |
|-----------|------|-----------|
| `embed` | `GematriaEmbedding` | always |
| `blocks` | `nn.ModuleList[LEACBlock]` | always |
| `gematria_bias` | `GematriaAttentionBias` | `cfg.use_gematria` |
| `phase_osc` | `KuramotoPhaseLayer` | always |
| `norm_f` | `nn.LayerNorm` | always |
| `lm_head` | `ZipfianDecoder` if `bayesian_uncertainty_beta > 0` else `nn.Linear` |
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

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `input_ids` | `[B, L]` | Token IDs |
| `targets` | `[B, L]` or None | Targets for LM loss |
| `write_memory` | `bool` | Write to episodic memory |

Returns: `(logits: [B, L, V], losses: Dict[str, Tensor])`

Returned losses:
- `lm`: Cross-entropy language modeling
- `causal`: DAG penalty (if causal graph enabled)
- `self_model_coherence`: Workspace coherence (if self-model)
- `gematria`: Harmonic gematria loss
- `phase_coherence`: Kuramoto oscillator coherence
- `counterfactual`: Counterfactual loss (if causal)
- `ads_cft`: Holographic loss (if AdS/CFT)
- `bridge_coherence`: ER=EPR bridge coherence
- `bulk_complexity`: AdS bulk complexity
- `mera_complexity`: MERA network complexity
- `godel_fixed_point_distance`: Gödel fixed point distance
- `godel_contradiction`: Self-referential contradiction
- `godel_incompleteness`: Incompleteness score
- `hyperbolic_gematria`: Poincaré distance loss
- `sheaf_cohomology`: Sheaf cohomology defect
- `criticality`: RG criticality index
- `total`: Weighted sum of all losses

#### `generate(input_ids, max_new_tokens, temperature=1.0, top_k=None)`

Autoregressive generation.

#### `param_count() → Dict[str, int]`

Returns `{"total": N, "trainable": N, "v1_modules": N, "v2_modules": N}`.

---

## Module `nfn.block`

### `LEACBlock(cfg, block_idx=0)`

Forward pass: `h → Attn → PhaseSoliton → MoE → [Causal] → [SelfModel] → [WorkingMem] → Fusion → LN`

Internal modules:

| Attribute | Type | Condition |
|-----------|------|-----------|
| `attn` | `FractalLinearAttention` | `cfg.use_linear_attn` |
| `phase_soliton` | `PhaseSoliton` | always |
| `moe` | `PhaseRoutedMoE` | always |
| `causal` | `CausalGraphLayer` | `cfg.use_causal_graph` |
| `self_model` | `GlobalWorkspace` | `cfg.use_self_model` |
| `working_mem` | `FractalWorkingMemory` | `cfg.use_working_memory` |
| `fusion` | `nn.Linear` | if `n_streams > 1` |
| `norm1`, `norm2`, `norm3` | `nn.LayerNorm` | always |

---

## Module `nfn.gematria`

### `GematriaEmbedding(vocab_size, d_model, d_gematria=16)`

Zero-parameter encoding via 5 crossed systems. `forward(token_ids) → [B, L, d_model]`

### `GematriaAttentionBias(d_model, vocab_size, n_heads=4, d_gematria=16)`

Attention bias: `score(i,j) += λ · cos(gem(i), gem(j))`. `forward(h, token_ids) → [B, n_heads, L, L]`

---

## Module `nfn.moe`

### `PhaseRoutedMoE(d_model, n_experts, d_ff_per_expert, n_phases=8, top_k=2, kappa=4.0, dropout=0.1)`

Von Mises routing: `gate_e(x) = exp(κ · cos(θ_x - θ_e)) / Z`

`forward(x, phases) → (output: [B, L, d_model], aux_loss: scalar)`

### `FractalLinearAttention(d_model, n_heads, n_levels=3, dropout=0.0, causal=True)`

Linear attention O(L·d²) with multi-scale fractal structure.

`forward(x, phases=None) → [B, L, d_model]`

### `PhaseSoliton(d_model, n_phases=8, tau=1.0)`

Coherent amplification: `h' = h · (1 + α · max(0, cos(θ - θ_shift)))`

---

## Module `nfn.phase_ode`

### `KuramotoPhaseLayer(d_model, n_max_nodes=512, rank=8, n_ode_steps=4, omega_init=1.0, phase_mod_strength=0.1)`

RK4 integration of hierarchical Kuramoto ODE. `forward(h) → (phases: [B, L, n_phases], loss: scalar)`

### `PhaseGoalForcing(d_model, n_phases, init_lambda=0.2, n_steps=3)`

Phase forcing toward targets: `dθ/dt += λ · sin(θ_goal - θ)`. `forward(phases, targets) → (forced_phases, loss)`

---

## Module `nfn.hyperbolic_gematria`

### `PoincareBall(eps=1e-12)`

Poincaré hyperbolic metric. Methods: `distance(x, y)`, `exp_map(x, v)`, `log_map(x, y)`, `mobius_add(x, y)`

### `HyperbolicGematriaTable(vocab_size, hyperbolic_dim=64)`

Embeds tokens into H^n via 5 gematric projections. `forward(token_ids) → (z_hyp: [B, L, D], z_boundary: [B, L, D])`

### `HyperbolicGematriaAttention(hyperbolic_dim=64, temperature=1.0)`

Geometric attention bias: `score(i,j) += λ_hyp · exp(-d_H(z_i, z_j) / τ)`. `forward(z_hyp, z_boundary) → (attn_bias: [B, L, L], loss: scalar)`

### `SheafTheoryLayer(d_model, n_stalks=4)`

Sheaf cohomology for hallucination detection. `forward(h) → (h_sheaf: [B, L, d], defect_loss: scalar)`

### `HyperbolicGematriaModule(vocab_size, d_model, hyperbolic_dim=64, n_stalks=4, temperature=1.0)`

Full pipeline: Token IDs → Poincaré → Attention → Sheaf. `forward(h, token_ids) → (h_enriched, losses_dict)`

---

## Module `nfn.ads_cft`

### `AdSMetric(eps=1e-6)`

AdS₅ metric in Poincaré coordinates. `geodesic_distance(x₁, z₁, x₂, z₂) → scalar`

### `AdSBulkProjector(d_model, bulk_dim=128, d_spatial=64)`

Boundary → bulk projector: `T[r, z] = e^{-κz} · MLP(h[r])`. `forward(h, depths) → h_bulk: [B, L, d]`

### `ER_EPR_Bridge(d_model, bridge_rank=16)`

Computational wormholes: semantic entanglement = ER=EPR connection. `forward(h, x_spatial, z_depth) → (h_teleported, bridge_loss)`

### `AdSCFTAttention(d_model, bulk_dim=128, d_spatial=64, bridge_rank=16, n_heads=4)`

Complete holographic attention: scores = local + geodesic + ER=EPR. `forward(h, causal=True) → (h_out, losses_dict)`

---

## Module `nfn.tensor_network`

### `Disentangler(d_model)`

2-qubit operator decorrelating adjacent pairs. `forward(h_pair) → h_disentangled: [B, L/2, d]`

### `Isometry(d_model, branching=2)`

Merges `branching` children into one parent with gating. `forward(children, gating_context=None) → parents: [B, N/branching, d]`

### `MERALayer(d_model, branching=2)`

One MERA layer: Disentangler → Isometry. `forward(h) → (parent, residual)`

### `MERAAttention(d_model, n_heads=4, n_levels=4, branching=2, dropout=0.1, causal=True)`

Multi-scale O(log L) attention. `forward(h) → (h_out: [B, L, d], complexity: scalar)`

### `TensorNetworkEncoder(vocab_size, d_model, n_heads=4, n_levels=4, branching=2, dropout=0.1)`

Full MERA encoder with embeddings. `forward(input_ids) → h: [B, L, d]`

---

## Module `nfn.godel_loop`

### `SelfReferenceOperator(d_model)`

Lawvere fixed point: `Y ≅ F(Y)`. `forward(h) → (h_self, fixed_point_distance: scalar)`

### `IncompletenessDetector(d_model)`

Detects contradictions (incompleteness). `forward(h, fixed_point_dist) → (h_corrected, incompleteness_score, metrics)`

### `GodelFixedPoint(d_model, n_iterations=5, alpha=0.3)`

Full fixed point iteration. `forward(h) → (h_out, metrics: Dict)`

Metrics: `fixed_point_distance`, `godel_contradiction`, `godel_entropy_region`, `godel_incompleteness`, `incompleteness_score`

---

## Module `nfn.rg_flow`

### `ScaleDecomposition(d_model, n_scales=8)`

Decomposes `h` into multi-scale components via RFF + frequency threshold. `forward(h) → (h_UV, h_IR)`

### `RGFlowLayer(d_model, n_scales=8, evaporation_rate=0.01, condensation_rate=0.01)`

UV evaporation + IR condensation on a linear layer. `evolve_weights(model, n_steps=1) → total_rg_loss: float`

### `CriticalityOptimizer(d_model, target_criticality=1.0)`

Optimizes toward critical state. `criticality_index(h) → scalar`, `step(h_uv, h_ir) → loss`
- `C = Var(Var(h)) / E[Var(h)]²`
- `C ≈ 1`: critical, `C ≪ 1`: frozen, `C ≫ 1`: chaotic

### `RGFlowScheduler(d_model, n_scales=8, initial_evaporation=0.01, initial_condensation=0.005)`

RG scheduler for SLEEP phase. `sleep_cycle(model, h_sample) → Dict[str, float]`

---

## Module `nfn.causal`

### `CausalGraphLayer(d_model, n_slots, hidden=64, sparsity=0.01, use_nonlinear=False)`

Learns a causal DAG with NOTEARS. `forward(h, targets=None) → (h_causal, losses)`

### `notears_acyclicity(A) → scalar`

Acyclicity penalty: `h(A) = tr(e^{A⊙A}) - n`

---

## Module `nfn.self_model`

### `GlobalWorkspace(d_model, n_slots=16, n_heads=4)`

Global workspace (broadcast). `forward(h) → (h_workspace, coherence_loss)`

### `SelfRepresentor(d_model, n_signals=8)`

Meta-cognitive self-representation. `forward(h, h_workspace) → self_state: [B, d]`

### `SelfModel(d_model, n_slots=16, n_signals=8)`

Workspace + introspection combination. `forward(h) → (h_out, losses)`

---

## Module `nfn.episodic_memory`

### `EpisodicStore(d_model, capacity=2048, key_dim=64, n_read=8, n_scales=4)`

O(1) ring buffer with multi-scale k-NN read. `forward(h, write=True) → h_read: [B, L, d]`

### `SemanticConsolidator(d_model, rank=64, consolidation_freq=3.0)`

SVD rank-r condensate. `forward(h_write, do_consolidate=False) → h_semantic: [B, 1, d]`

### `TwoTierMemory(d_model, episodic_capacity=2048, ...)`

Two-path memory. `forward(h, write=False) → h_memory`. `maybe_consolidate() → None`

---

## Module `nfn.working_memory`

### `FractalWorkingMemory(d_model, n_slots=32, n_heads=4, sharpness=3.0)`

Differentiable fractal DNC scratchpad. `forward(h, write=False) → h_memory: [B, L, d]`

---

## Module `nfn.lifecycle`

### `LEACLifecycle(model, cfg, tokenizer=None, device=cpu, lr=3e-4, train_seq_len=64)`

Complete continuous life cycle.

| Method | Description |
|---------|-------------|
| `wake_step() → Dict` | Generate/verify/weight mathematical truths. Backpropagate. |
| `sleep_step() → Dict` | Episodic → SVD consolidation. RG Flow. |
| `meta_adapt(input_ids)` | Test-Time LoRA if perplexity > 5 |
| `evolve_step() → Dict` | Darwinian architecture mutation |
| `live(n_cycles=10000, log_every=100, eval_every=500)` | Main loop |

### `CuriosityScheduler(threshold=2.0, scale=0.5)`

`weight(losses) → 1 + 0.5 · σ(loss - 2.0)`. Errors focus attention.

### `TestTimeLoRA(model, rank=4, alpha=0.001)`

Temporary LoRA for inference-time adaptation. 0.1% of parameters, 3-5 gradient steps.

### `SelfCritic(model, tokenizer)`

Constitutional self-critique: generate → critique → revise.

---

## Module `nfn.self_development`

### `MathTruthEngine(max_number=1000, vocab_offset=256)`

Infinite mathematical truth generator. Methods: `generate_arithmetic(n)`, `generate_primality(n)`, `generate_sequence_prediction(n, seq_len)`, `generate_modular_arithmetic(n)`

### `GematriaEncoder(vocab_size=256)`

Gematric encoder for self-development.

### `UniversalLawObserver(d_model)`

Universal law discovery in dynamics. `forward(h) → (loss, metrics)`

---

## Module `nfn.auto_genesis`

### `ConjectureLoop(d_model, device=cpu)`

Complete conjecture discovery cycle: generate → test (500+ cases) → reward (1 if survives, 0 if falsified).

### `ProofLoop(d_model, device=cpu)`

Proof cycle: generate → verify → reward (60% correctness, 30% efficiency, 10% diversity).

---

## Module `nfn.self_modification`

### `SelfModificationController(d_state=32, n_motifs=3, max_step_size=0.1, max_modifications_per_step=3)`

Architectural Darwinism: observe → propose → apply → measure → accept/reject. `propose_modification() → ModificationProposal`

---

## Module `nfn.condensate`

### `FractalRFF(d_model, n_scales=8, sigma=1.0)`

Multi-scale Random Fourier Features. `forward(h) → h_rff: [B, L, n_scales * d_model]`

### `SpectralCondensate(d_model, n_scales=8)`

Spectral condensate for Bayesian decoding. `forward(h) → (h_out, spectral_loss)`

### `HelmholtzPhaseLocking(d_model, n_scales=8)`

Phase locking between Kuramoto oscillators and spectral modes. `forward(phases, spectral_features) → lock_loss`