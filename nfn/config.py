"""
FNN — Fractal Neural Network
Configuration — Paradigme Fractal, Causal et Gematrique

Configurations predefinies:
  - nano:   d=256,  4 blocs,  ~14M params
  - small:  d=512,  8 blocs,  ~58M params
  - medium: d=1024, 12 blocs, ~230M params
  - large:  d=512,  12 blocs, ~250M params (toutes les features avancees)
  - xlarge: d=1024, 24 blocs, ~750M params
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FNNConfig:
    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab_size: int = 512
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # ── Model dimensions ──────────────────────────────────────────────────────
    d_model: int = 256
    d_ff: int = 1024
    n_blocks: int = 4
    dropout: float = 0.1
    n_heads: int = 4

    # ── Fractal topology ──────────────────────────────────────────────────────
    n_levels: int = 4
    branching: int = 2
    motifs: List[str] = field(
        default_factory=lambda: ["binary_tree", "cantor"]
    )

    # ── Linear Attention (Fractal Katharopoulos) ────────────────────────────
    # O(L*d^2) au lieu de O(L^2*d)
    use_linear_attn: bool = True

    # ── Gematria Sémantique ─────────────────────────────────────────────────
    # Les 5 systemes croises: Ordinal, Premier, Fibonacci, Racine Digitale, Appris
    use_gematria: bool = True
    d_gematria: int = 16
    gematria_attention_bias: bool = True
    lambda_gematria: float = 0.1

    # ── Kuramoto Phase Dynamics (Pilier 1: Cohérence) ──────────────────────
    use_kuramoto: bool = True
    kuramoto_rank: int = 8
    kuramoto_steps: int = 4
    kuramoto_n_max: int = 512
    phase_mod_strength: float = 0.1

    # ── Phase Goal Forcing ──────────────────────────────────────────────────
    use_goal_forcing: bool = True
    goal_n_phases: int = 8
    goal_init_lambda: float = 0.2
    goal_n_steps: int = 3
    lambda_goal: float = 0.005

    # ── Causal SCM (Pilier 2: Raisonnement) ─────────────────────────────────
    use_causal_graph: bool = True
    use_nonlinear_causal: bool = True
    causal_n_slots: int = 16
    causal_hidden: int = 64
    causal_sparsity: float = 0.01
    lambda_causal: float = 0.001
    lambda_notears: float = 0.001
    lambda_counterfactual: float = 0.01

    # ── Self-Model (Pilier 3: Introspection) ────────────────────────────────
    use_self_model: bool = True
    self_model_n_slots: int = 16
    self_model_n_signals: int = 8
    self_model_injection_scale: float = 0.05

    # ── Working Memory (Espace de Travail Global) ──────────────────────────
    use_working_memory: bool = True
    wm_n_slots: int = 32
    wm_n_heads: int = 4
    wm_sharpness: float = 3.0

    # ── Episodic + Semantic Memory (Consolidation WAKE/SLEEP) ───────────────
    use_episodic_memory: bool = True
    episodic_capacity: int = 2048
    episodic_key_dim: int = 64
    episodic_n_read: int = 8
    semantic_rank: int = 64
    consolidation_freq: float = 3.0
    consolidation_every: int = 50

    # ── Phase-Routed MoE (Von Mises routing) ────────────────────────────────
    moe_n_experts: int = 8
    moe_top_k: int = 2
    moe_d_ff_per_expert: int = 256
    moe_kappa: float = 4.0

    # ── Analytic Embedding (Zero-Parameter) ──────────────────────────────────
    use_analytic_embed: bool = True
    learnable_embed_bias: bool = False

    # ── Zipfian Decoder ─────────────────────────────────────────────────────
    zipf_alpha: float = 1.0
    use_bayesian_decoder: bool = False
    bayesian_uncertainty_beta: float = 0.1

    # ── Spectral Condensate (NFMC kernel) ──────────────────────────────────
    use_condensate: bool = True
    nfmc_n_rff: int = 128
    nfmc_n_scales: int = 6
    nfmc_rank: int = 32
    nfmc_n_phases: int = 8
    nfmc_lock_iter: int = 4
    nfmc_eta: float = 0.15
    nfmc_lambda_phase: float = 0.005

    # ── Auto-Genèse Mathématique ─────────────────────────────────────────────
    use_auto_genesis: bool = True
    math_max_number: int = 1000
    math_vocab_offset: int = 256
    conjecture_n_test: int = 500
    proof_n_rules: int = 20
    proof_reward_correctness: float = 0.6
    proof_reward_efficiency: float = 0.3
    proof_reward_diversity: float = 0.1

    # ── Self-Modification Évolutionnaire ────────────────────────────────────
    use_self_modification: bool = True
    self_mod_mutation_rate: float = 0.1
    self_mod_fitness_discovery: float = 0.5
    self_mod_fitness_coherence: float = 0.3
    self_mod_fitness_efficiency: float = 0.2

    # ── Cycle de Vie Continu ────────────────────────────────────────────────
    curiosity_sigma: float = 2.0
    curiosity_scale: float = 0.5
    lora_rank: int = 4
    lora_alpha: float = 0.001
    lora_steps: int = 5
    lora_pct_params: float = 0.001

    # ── LEAC v2.0 — Gematria Hyperbolique (Poincare) ──────────────────────
    use_hyperbolic_gematria: bool = False
    hyperbolic_dim: int = 64
    hyperbolic_temperature: float = 1.0
    hyperbolic_n_stalks: int = 4
    lambda_hyperbolic: float = 0.001

    # ── LEAC v2.0 — AdS/CFT Holographic Neural ───────────────────────────
    use_ads_cft: bool = False
    ads_bulk_dim: int = 128
    ads_spatial_dim: int = 64
    ads_bridge_rank: int = 16
    lambda_bulk: float = 0.0001

    # ── LEAC v2.0 — Tensor Networks MERA ──────────────────────────────────
    use_mera: bool = False
    mera_window: int = 16
    lambda_complexity: float = 0.00001

    # ── LEAC v2.0 — Boucle Etrange de Godel ──────────────────────────────
    use_godel_loop: bool = False
    godel_n_iterations: int = 5
    godel_alpha: float = 0.3
    lambda_godel: float = 0.0001

    # ── LEAC v2.0 — Renormalization Group Flow ───────────────────────────
    use_rg_flow: bool = False
    rg_n_scales: int = 8
    rg_evaporation: float = 0.01
    rg_condensation: float = 0.005
    lambda_rg: float = 0.001

    # ── RoPE (long context) ────────────────────────────────────────────────
    use_rope: bool = True
    rope_base: float = 10_000.0
    rope_scale_factor: float = 1.0
    max_seq_len: int = 4096
    context_len: int = 32768

    # ── Modules optionnels (récupérés, désactivés par défaut) ────────────────
    # State-Space Model (Mamba-like) — récurrence linéaire pour le streaming
    use_ssm: bool = False
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    # Mixture of Depths — skip de tokens (économie de calcul)
    use_mixture_of_depths: bool = False
    mod_top_k: float = 0.5
    # Predictive Coding — codage prédictif top-down
    use_predictive_coding: bool = False
    predictive_n_levels: int = 2
    # Multi-Token Prediction — têtes lookahead + perte auxiliaire
    use_mtp: bool = False
    mtp_depth: int = 4

    # ── Generation ──────────────────────────────────────────────────────────
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

    # ── Loss weights ────────────────────────────────────────────────────────
    lambda_phase: float = 0.01
    lambda_freq: float = 0.001
    lambda_spectral: float = 0.0001

    # ── Configurations prédéfinies ──────────────────────────────────────────
    @classmethod
    def nano(cls) -> "FNNConfig":
        """Preset nano: d=256, 4 blocs, ~14M params."""
        return cls(d_model=256, n_blocks=4, d_ff=1024, moe_n_experts=8, moe_top_k=2)

    @classmethod
    def small(cls) -> "FNNConfig":
        """Preset small: d=512, 8 blocs, ~58M params."""
        return cls(
            d_model=512, n_blocks=8, d_ff=2048,
            moe_n_experts=16, moe_top_k=4,
            n_levels=5, causal_n_slots=32,
            self_model_n_slots=32,
            use_goal_forcing=True, use_causal_graph=True,
            use_self_model=True, use_working_memory=True,
            use_episodic_memory=True, use_auto_genesis=True,
            use_self_modification=True,
        )

    @classmethod
    def medium(cls) -> "FNNConfig":
        """Preset medium: d=1024, 12 blocs, ~230M params."""
        return cls(
            d_model=1024, n_blocks=12, d_ff=4096,
            moe_n_experts=32, moe_top_k=4,
            n_levels=6, causal_n_slots=64,
            self_model_n_slots=64,
            kuramoto_rank=16, kuramoto_n_max=1024,
            use_goal_forcing=True, use_causal_graph=True,
            use_self_model=True, use_working_memory=True,
            use_episodic_memory=True, use_auto_genesis=True,
            use_self_modification=True,
        )

    @classmethod
    def large(cls) -> "FNNConfig":
        """Preset large: d=512, 12 blocs, ~250M params (toutes features avancees)."""
        return cls(
            d_model=512, n_blocks=12, d_ff=2048,
            n_heads=8, moe_n_experts=24, moe_top_k=4,
            n_levels=5, causal_n_slots=48,
            self_model_n_slots=48,
            kuramoto_rank=12, kuramoto_n_max=512,
            use_goal_forcing=True, use_causal_graph=True,
            use_self_model=True, use_working_memory=True,
            use_episodic_memory=True, use_auto_genesis=True,
            use_self_modification=True,
            use_hyperbolic_gematria=True,
            hyperbolic_dim=64, hyperbolic_temperature=1.0,
            use_ads_cft=True,
            ads_bulk_dim=128, ads_spatial_dim=64, ads_bridge_rank=16,
            use_mera=True, mera_window=16,
            use_godel_loop=True,
            godel_n_iterations=5, godel_alpha=0.3,
            use_rg_flow=True,
            rg_n_scales=8, rg_evaporation=0.01, rg_condensation=0.005,
        )

    @classmethod
    def xlarge(cls) -> "FNNConfig":
        """Preset xlarge: d=1024, 24 blocs, ~750M params (toutes features avancees)."""
        return cls(
            d_model=1024, n_blocks=24, d_ff=4096,
            n_heads=8, moe_n_experts=48, moe_top_k=6,
            n_levels=6, causal_n_slots=96,
            self_model_n_slots=96,
            kuramoto_rank=16, kuramoto_n_max=1024,
            use_goal_forcing=True, use_causal_graph=True,
            use_self_model=True, use_working_memory=True,
            use_episodic_memory=True, use_auto_genesis=True,
            use_self_modification=True,
            use_hyperbolic_gematria=True,
            hyperbolic_dim=128, hyperbolic_temperature=1.0,
            use_ads_cft=True,
            ads_bulk_dim=256, ads_spatial_dim=128, ads_bridge_rank=32,
            use_mera=True, mera_window=32,
            use_godel_loop=True,
            godel_n_iterations=7, godel_alpha=0.3,
            use_rg_flow=True,
            rg_n_scales=12, rg_evaporation=0.01, rg_condensation=0.005,
        )

    @property
    def n_motifs(self) -> int:
        return len(self.motifs)

    @property
    def top_len(self) -> int:
        return max(1, self.max_seq_len // (self.branching ** self.n_levels))

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FNNConfig":
        valid = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})