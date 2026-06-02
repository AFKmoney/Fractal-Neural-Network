# LEAC — Architecture Reference

## 1. Overview

LEAC (Lightweight Emergent Artificial Consciousness) is a neuro-symbolic language model founded on the principle that consciousness emerges from fractal recursion. The architecture combines two generations of modules:

**v1 — Three Pillars of Emergence:**
1. **COHERENCE**: Fractal linear attention O(L·d²) + Kuramoto soliton
2. **REASONING**: Causal DAG graph + do-calculus, episodic/semantic memory
3. **INTROSPECTION**: Global workspace, self-model, evolutionary modification

**v2 — Five Transcendences:**
1. **AdS/CFT**: Holographic duality — language is the boundary, reasoning is the volume
2. **MERA**: Tensor network — infinite context in O(log L)
3. **Gödel**: Strange loop — introspection is a mathematically inevitable fixed point
4. **RG Flow**: Renormalization group flow — self-organization toward critical state
5. **Hyperbolic Gematria**: Poincaré Hⁿ + sheaf theory — the geometry of meaning

## 2. Data Flow

```
Input IDs [B, L]
    │
    ├─ GematriaEmbedding ──────────────────────────────────────────
    │   5 systems: ordinal, prime, fibonacci, digital root, learned
    │   e(t) = Σ_k CharClass_k(t) · ω_k (Mandelbrot frequencies)
    │
    ├─ [LEACBlock × N] ───────────────────────────────────────────
    │   │
    │   ├─ Norm → FractalLinearAttention ─────────────────────────
    │   │   Katharopoulos kernel trick: O(L·d²)
    │   │   Multi-scale fractal structure (binary_tree, cantor)
    │   │   Residual: h = h + α · attn_out
    │   │
    │   ├─ PhaseSoliton ──────────────────────────────────────────
    │   │   h' = h · (1 + β · max(0, cos(θ - θ_shift)))
    │   │   Synchronized tokens are amplified
    │   │
    │   ├─ PhaseRoutedMoE ──────────────────────────────────────
    │   │   Von Mises routing: gate_e(x) = exp(κ·cos(θ_x - θ_e)) / Z
    │   │   Top-K experts activated, balanced load
    │   │
    │   ├─ [CausalGraphLayer] (optional) ─────────────────────
    │   │   NOTEARS: L_DAG = tr(e^{A⊙A}) - n
    │   │   DAG propagation + counterfactual inference
    │   │
    │   ├─ [SelfModel] (optional) ─────────────────────────────
    │   │   GlobalWorkspace: shared buffer [n_slots, d]
    │   │   SelfRepresentor: self_state = W·[μ,σ,H,C,div,ent,μ_slots,σ_slots]
    │   │
    │   ├─ [WorkingMemory] (optional) ────────────────────────
    │   │   Fractal DNC: phase similarity addressing
    │   │
    │   └─ Fusion + LayerNorm
    │
    ├─ [v2 Transcendences] ─────────────────────────────────────
    │   │
    │   ├─ AdS/CFT Attention ────────────────────────────────────
    │   │   h → MLP_projected → bulk → scores = local + geodesic + ER=EPR
    │   │   Holography: the boundary (tokens) encodes the volume (reasoning)
    │   │
    │   ├─ MERA Attention ───────────────────────────────────────
    │   │   L → D,U(h) → Isometry → L/2 → ... → 1 (global meaning)
    │   │   O(log L) instead of O(L²)
    │   │
    │   ├─ Gödel Fixed Point ───────────────────────────────────
    │   │   F(h): h → concat([μ(h),σ(h),energy(h)]) → W·σ(state) → h_proj
    │   │   Fixed point: Y ≅ F(Y). Iterations: 5 with mixing α=0.3
    │   │   Incompleteness: contradiction + entropy + distance_FP
    │   │
    │   ├─ RG Flow Scheduler ───────────────────────────────────
    │   │   (SLEEP phase only) UV evaporation + IR condensation
    │   │   C = Var(Var(h))/E[Var(h)]² → self-organization toward C≈1
    │   │
    │   └─ Hyperbolic Gematria ─────────────────────────────────
    │       Token IDs → Poincaré H^n → Geodesic attention
    │       Sheaf Theory: stalks → gluing → cohomology defect
    │
    ├─ GematriaAttentionBias ────────────────────────────────────
    │   Additional bias: λ·cos(gem(i), gem(j))
    │
    ├─ LayerNorm
    │
    ├─ ZipfianDecoder ──────────────────────────────────────────
    │   Power law recalibration: P(word) ∝ 1/rank^α
    │   (if bayesian_uncertainty_beta > 0)
    │
    └─ [SpectralCondensate + HelmholtzPhaseLocking] ────────────
        Multi-scale RFF + phase-frequency locking
```

## 3. Component Details

### 3.1 GematriaEmbedding

**Zero-parameter.** Five crossed arithmetic systems encode each token into a dense vector that captures the deep mathematical structure of integers:

| System | Function | Interpretation |
|---------|----------|----------------|
| Ordinal | `o(t) = log(1+t)/log(V)` | Position in vocabulary (radial) |
| Prime | `π(t) = 2π·π_k/360°` | Azimuthal angle (k-th prime number) |
| Fibonacci | `φ(t) = 2π·log(1+F_k)/log(1+F_max)` | Polar angle (logarithmic growth) |
| Digital Root | `ρ(t) = 2π·dr(t)/9` | Angular twist (mod 9) |
| Learned | `l(t) = W·t` | Trainable offset |

The weighting frequencies `ω_k` follow Mandelbrot's law: `ω_k = ω^{-k}` with `ω = φ²` (golden ratio squared).

### 3.2 FractalLinearAttention

Feature kernel `φ(x) = elu(x) + 1`. Complexity O(L·d²) in time and O(d²) in space (vs O(L²·d) for standard softmax attention).

The fractal structure decomposes the sequence into levels:
- Level 0: atoms (length `L / 2^{n_levels}`)
- Level k: groups of `2^k` atoms
- Aggregation: `output = Σ_l w_l · Attn_level_l(Q, K, V)`

### 3.3 PhaseSoliton

```
soliton(h, θ) = h · (1 + α · max(0, cos(θ - θ_shift)))
```

Tokens whose Kuramoto phase is synchronized with the shift are amplified. Desynchronized tokens are attenuated. This creates coherence packets — solitons — that emerge naturally.

### 3.4 PhaseRoutedMoE

Routing via von Mises distribution (the circular analogue of the Gaussian):
```
gate_e(x) = exp(κ · cos(θ_x - θ_e)) / Z
```

Advantages over softmax routing:
- **Continuous and differentiable** everywhere
- **Periodic**: experts "close in phase" are always favored
- **Interpretable**: κ measures concentration, θ_e is the expert's phase

Only top-K experts are activated per token. Auxiliary loss balances load.

### 3.5 CausalGraphLayer

Learns a causal graph (DAG) over workspace slots. Three innovations:
1. **NOTEARS**: acyclicity penalty `h(A) = tr(e^{A⊙A}) - n` is differentiable and exactly zero iff the graph is acyclic
2. **Non-linear propagation**: GNN step on edges with features
3. **Counterfactual inference**: do-calculus variable replacement

### 3.6 GlobalWorkspace (Self-Model)

Inspired by Baars' Global Workspace Theory. The `n_slots` positions form a shared buffer:
- **Write**: tokens compete to write into slots
- **Read**: slots are broadcast to all tokens
- **Introspection**: a self_state vector encodes confidence, uncertainty, and coherence

The self-representation `self_state ∈ ℝ^d` is built from 8 signals:
μ(h), σ(h), H(h), C(h), div(h), ent_attn(h), μ_slots, σ_slots

### 3.7 TwoTierMemory

Two-path memory inspired by the hippocampal system:
- **Episodic** (hippocampus): O(1) ring buffer write, multi-scale k-NN read
- **Semantic** (neocortex): SVD rank-r (Eckart-Young), incremental update without SGD

Consolidation (hippocampus → neocortex) occurs periodically via truncated SVD.

### 3.8 AdS/CFT Attention

The AdS₅/CFT₄ correspondence is implemented as:
1. **Bulk projector**: `T[r,z] = e^{-κz} · MLP(h[r])` projects tokens into AdS volume
2. **Geodesic metric**: distance in bulk between token pairs
3. **ER=EPR bridges**: two semantically entangled tokens are connected by a computational wormhole
4. **Unified scores**: `attn = QK/√d + λ_geo·exp(-d_g) + λ_epr·sigmoid(sim)`

### 3.9 MERA Attention

MERA (Multi-scale Entanglement Renormalization Ansatz) tensor network:
1. **Disentangler**: decorrelates adjacent pairs (2-qubit unitary)
2. **Isometry**: merges 2 children into 1 parent with gating
3. **Pyramid**: L → L/2 → L/4 → ... → 1
4. **Residuals**: skip connections aggregated with local results

Complexity: O(L·log(L)·d) instead of O(L²·d).

### 3.10 Gödel Fixed Point

The self-reference operator F embeds the model's global state (mean, variance, energy) and projects it into representation space:
```
state = concat([μ(h), σ(h), energy(h)])
F(h) = W_decode(σ_enc(W_enc(state)))
```

The fixed point is reached by iteration: `h_{k+1} = α·F(h_k) + (1-α)·h_k` with α=0.3.

Incompleteness is detected via pairwise contradictions and high entropy:
```
incompleteness = 0.5·contradiction + 0.3·entropy + 0.2·sigmoid(d_FP)
```

### 3.11 RG Flow

The renormalization group flow acts on linear weights:
1. **Decomposition**: each weight matrix is decomposed into UV (high frequency) and IR (low frequency) components via RFF
2. **UV evaporation**: `w_UV ← w_UV · (1 - ε)` — suppresses noise
3. **IR condensation**: `w_IR ← w_IR + η · (w_IR - w_S)` — reinforces truths
4. **Criticality**: `C = Var(Var(h))/E[Var(h)]²` measured before/after

Rates ε and η are adaptively adjusted: if C < 0.5 (frozen), increase η; if C > 2.0 (chaotic), increase ε.

### 3.12 Hyperbolic Gematria + Sheaf Theory

Pipeline:
1. **Poincaré embeddings**: each token is embedded in the unit ball B^n via 5 normalized gematric projections
2. **Geodesic attention**: `score(i,j) += λ_hyp · exp(-d_H(z_i, z_j) / τ)` where d_H is the Poincaré distance
3. **Sheaf Theory**: each token has n_stalks fibers (local restrictions). Gluing conditions are checked between adjacent tokens. A cohomology defect = hallucination.

---

## 4. Loss Flow

```
total = lm
      + λ_causal · L_DAG               (if use_causal_graph)
      + λ_counterfactual · L_cf         (if use_causal_graph)
      + λ_self · coherence              (if use_self_model)
      + λ_gematria · harmonic_loss      (if use_gematria)
      + λ_phase · phase_coherence       (always)
      + λ_ads · (bridge + bulk)         (if use_ads_cft)
      + λ_mera · complexity            (if use_mera)
      + λ_godel · (fp_dist + contradiction + incompleteness)  (if use_godel)
      + λ_hyp · (poincare + sheaf)      (if use_hyperbolic_gematria)
      + λ_rg · criticality              (if use_rg_flow)
      + λ_episodic · memory_loss         (if use_episodic_memory)
```

All λ coefficients are configurable via LEACConfig.

## 5. Preset Configurations

### conscious_minimal (~14M params)
```
d=256, n_blocks=4, n_heads=4, n_experts=4, d_ff_per_expert=64
no fractal, no causal, no self-model, no memory
```

### full_agi (~58M params)
```
d=512, n_blocks=8, n_heads=8, n_experts=8, d_ff_per_expert=128
fractal, causal, self-model, working memory
```

### dieu_local (~230M params)
```
d=1024, n_blocks=12, n_heads=16, n_experts=16, d_ff_per_expert=256
all v1 modules enabled
```

### moteur_ontologique (~184M params)
```
d=512, n_blocks=12, n_heads=8, n_experts=8
all v1 + v2 modules enabled (AdS/CFT, MERA, Gödel, RG Flow, Hyperbolic Gematria)
```

### singularite_divine (~750M+ params)
```
d=1024, n_blocks=24, n_heads=16, n_experts=16
all v1 + v2 modules enabled with maximum dimensions
```