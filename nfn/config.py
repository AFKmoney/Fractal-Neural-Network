from dataclasses import dataclass, field
from typing import List


@dataclass
class NFNConfig:
    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab_size: int = 512           # character-level default
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # ── Model dimensions ──────────────────────────────────────────────────────
    d_model: int = 256              # hidden dimension d
    d_ff: int = 1024                # feed-forward inner dimension
    n_blocks: int = 4               # stacked NFN blocks (≈ depth)
    dropout: float = 0.1

    # ── Fractal topology ──────────────────────────────────────────────────────
    n_levels: int = 4               # fractal depth K (levels above leaf)
    branching: int = 2              # branching factor b
    motifs: List[str] = field(
        default_factory=lambda: ["binary_tree", "cantor"]
    )                               # fractal patterns to superpose

    # ── Sinusoidal connections ─────────────────────────────────────────────────
    rank: int = 8                   # R for low-rank sinusoidal factorisation
    omega_base: float = 1.0         # ω₀ — base frequency
    lambda_scale: float = 2.0       # λ — inter-level scale ratio
    damping: bool = True            # use amortised sinusoids (exp decay)
    gamma_init: float = 0.1         # initial damping rate γ

    # ── Temporal (phase) dynamics ─────────────────────────────────────────────
    n_time_steps: int = 8           # P discrete time steps
    alpha: float = 0.9              # state decay α
    phase_coupling: float = 0.1     # strength of inter-node phase coupling

    # ── Top-level attention ───────────────────────────────────────────────────
    n_heads: int = 4                # attention heads on compressed sequence
    max_seq_len: int = 512          # L — context window

    # ── Loss weights ──────────────────────────────────────────────────────────
    lambda_phase: float = 0.01      # phase smoothness
    lambda_freq: float = 0.001      # frequency sparsity
    lambda_spectral: float = 0.0001 # Jacobian stability

    # ── Generation ────────────────────────────────────────────────────────────
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

    @property
    def n_motifs(self) -> int:
        return len(self.motifs)

    @property
    def top_len(self) -> int:
        """Length of the compressed sequence at the top level."""
        return max(1, self.max_seq_len // (self.branching ** self.n_levels))

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NFNConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
