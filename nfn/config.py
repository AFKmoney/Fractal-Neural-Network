from dataclasses import dataclass, field
from typing import List


@dataclass
class NFNConfig:
    # ── Vocabulary ──────────────────────────────────────────────────────────────────────────────
    vocab_size: int = 512           # overridden by tokenizer at build time
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # ── Model dimensions ──────────────────────────────────────────────────────────────────────
    d_model: int = 256
    d_ff: int = 1024
    n_blocks: int = 4
    dropout: float = 0.1

    # ── Fractal topology ────────────────────────────────────────────────────────────────────────
    n_levels: int = 4               # fractal depth K
    branching: int = 2              # branching factor b
    motifs: List[str] = field(
        default_factory=lambda: ["binary_tree", "cantor"]
    )

    # ── Sinusoidal connections ─────────────────────────────────────────────────────────────────────
    rank: int = 8
    omega_base: float = 10_000.0    # matches RoPE base for coherence
    lambda_scale: float = 2.0
    damping: bool = True
    gamma_init: float = 0.1

    # ── RoPE (long context) ───────────────────────────────────────────────────────────────────────
    use_rope: bool = True
    rope_base: float = 10_000.0
    rope_scale_factor: float = 1.0  # > 1 enables NTK long-context extension
    max_seq_len: int = 4096         # training context window
    context_len: int = 32768        # inference context via NTK scaling

    # ── Flash Attention ───────────────────────────────────────────────────────────────────────────
    use_flash_attn: bool = True     # uses torch SDPA (auto Flash when on GPU)
    n_heads: int = 4

    # ── Kuramoto ODE phase dynamics ────────────────────────────────────────────────────────────────
    use_kuramoto: bool = True
    kuramoto_rank: int = 8          # low-rank coupling matrix rank
    kuramoto_steps: int = 4         # RK4 integration steps
    kuramoto_n_max: int = 512       # max oscillators per level

    # ── Persistent working memory ───────────────────────────────────────────────────────────────
    use_memory: bool = True
    memory_slots: int = 64          # M memory slots
    memory_heads: int = 4
    memory_per_level: bool = True   # FractalMemoryBank vs single bank

    # ── Temporal dynamics ────────────────────────────────────────────────────────────────────────
    n_time_steps: int = 4           # P temporal integration steps
    alpha: float = 0.9              # state decay

    # ── Loss weights ─────────────────────────────────────────────────────────────────────────────
    lambda_phase: float = 0.01
    lambda_freq: float = 0.001
    lambda_spectral: float = 0.0001

    # ── Generation ──────────────────────────────────────────────────────────────────────────────
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

    # ── NFMC v3.0 — Condensed Multidimensional Fractal Kernel ─────────────────────────────
    use_nfmc: bool = False           # enable NFMC kernel layer in NFNBlocks
    nfmc_n_rff: int = 256            # random fractal Fourier features
    nfmc_n_scales: int = 8           # octave bands in frequency lattice
    nfmc_rank: int = 64              # condensate rank r
    nfmc_n_phases: int = 8           # phase oscillators per token
    nfmc_lock_iter: int = 8          # Helmholtz phase-locking gradient steps
    nfmc_eta: float = 0.15           # phase-locking step size
    nfmc_lambda_phase: float = 0.005 # weight of phase coherence loss

    # ── NFMC v3.1 — ZeroShotNFMC (Mandelbrot + Hopfield + Zipf) ─────────────────────
    nfmc_hopfield_n: int = 256       # number of Hopfield analytic patterns
    nfmc_zipf_alpha: float = 1.0     # Zipf exponent for decoder initialization

    # ── EfficientNFN v3.2 — MoE + Linear Attention ────────────────────────────────
    moe_n_experts: int = 8           # total number of MoE experts
    moe_top_k: int = 2               # experts activated per token (sparse)
    moe_d_ff_per_expert: int = 256   # FFN width inside each expert

    # ── AGI v4.0 — Two-Tier Episodic + Semantic Memory ────────────────────────────
    use_episodic_memory: bool = False
    episodic_capacity: int = 2048    # ring-buffer size (episodes)
    episodic_key_dim: int = 64       # FractalRFF key dimension
    episodic_n_read: int = 8         # top-k neighbours per query
    semantic_rank: int = 64          # condensate rank for neocortex
    consolidation_freq: float = 3.0  # repeat threshold for consolidation
    consolidation_every: int = 50    # steps between consolidation checks

    # ── AGI v4.0 — Causal Graph Layer ───────────────────────────────────────────────
    use_causal_graph: bool = False
    causal_n_slots: int = 16         # number of causal memory slots
    causal_hidden: int = 64          # edge MLP hidden dim
    causal_sparsity: float = 0.01    # L1 penalty on adjacency matrix
    lambda_causal: float = 0.001     # weight of causal sparsity loss

    # ── AGI v4.0 — Goal-Directed Phase Predictor ──────────────────────────────────────
    use_goal_predictor: bool = False
    goal_n_phases: int = 8           # phases in goal attractor
    goal_init_lambda: float = 0.2    # initial goal attraction strength
    goal_n_steps: int = 3            # Euler steps of goal forcing
    lambda_goal: float = 0.005       # weight of goal alignment loss

    # ── AGI v4.0 — Multimodal Fractal Phase Space ───────────────────────────────────
    use_multimodal: bool = False
    multimodal_d_image: int = 768    # ViT patch embedding dim (e.g. ViT-B/16)
    multimodal_d_audio: int = 128    # audio frame embedding dim
    multimodal_patch_h: int = 14     # image patch grid height
    multimodal_patch_w: int = 14     # image patch grid width
    lambda_coherence: float = 0.01   # cross-modal phase coherence loss weight

    # ── AGI v4.0 — Bayesian Zipfian Decoder ───────────────────────────────────────────
    use_bayesian_decoder: bool = False
    bayesian_uncertainty_beta: float = 0.1   # uncertainty penalty at inference
    bayesian_thompson_sampling: bool = False # enable Thompson sampling in train

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
    def from_dict(cls, d: dict) -> "NFNConfig":
        valid = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})
