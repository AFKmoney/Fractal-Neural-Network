"""
Modern Hopfield Memory + Mandelbrot Phase Initialization (NFN v3.1)

═══════════════════════════════════════════════════════════════════

I. Modern Hopfield Network (Ramsauer et al., NeurIPS 2020)
─────────────────────────────────────────────────────────
Classical Hopfield (1982): stores N patterns in d neurons, capacity O(d).
Modern Hopfield (2020): uses softmax update, capacity O(exp(d/2)).

Update rule:
    ξ^new = Xᵀ · softmax(β · X · ξ)

where X ∈ R^{N×d} is the pattern matrix, ξ ∈ R^d is the query.
This is exactly scaled dot-product attention with Q=ξ, K=V=X.

Memory is pre-loaded from analytic patterns (no training):
  • Zipf-weighted random Fourier features
  • Mandelbrot basis vectors
  • Universal syntactic templates

II. Mandelbrot Phase Initialization
────────────────────────────────────
The Mandelbrot set M = {c ∈ ℂ : |z_{n+1} = z_n² + c| stays bounded}.
Its boundary is parameterized by external angle θ ∈ [0, 1) via the
Böttcher coordinate: Φ_M(c) = e^{2πiθ}.

Key resonances (Douady-Hubbard theory):
  The landing point of external angle p/q (in lowest terms) is a
  parabolic fixed point of period q. These are:
    1/2 → period-2 (bulb at left of cardioid)
    1/3, 2/3 → period-3 bulbs
    1/4, 3/4 → period-4 bulbs
    Farey sequence: 1/2, 1/3, 2/3, 1/4, 3/4, 2/5, 3/5, ...

These angles give a natural hierarchy of time-scales — analogous to the
hierarchical structure of language (syllable < word < phrase < sentence).

We use these as the natural frequencies ω_k of the Kuramoto oscillators,
giving a fractal but principled phase space a priori.

═══════════════════════════════════════════════════════════════════
"""

import math
from fractions import Fraction
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Farey / Stern-Brocot frequency generator (Mandelbrot angles)
# ─────────────────────────────────────────────────────────────────────────────

def farey_sequence(n: int) -> List[Fraction]:
    """
    Farey sequence F_n: all fractions p/q in [0,1] with q ≤ n, in order.
    These are the external angles of the Mandelbrot set's main bulbs.
    """
    seq = [Fraction(0, 1)]
    a, b, c, d = 0, 1, 1, n
    while c <= n:
        seq.append(Fraction(c, d))
        k = (n + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
    return seq


def mandelbrot_frequencies(n: int) -> torch.Tensor:
    """
    Return n frequencies derived from the Mandelbrot set's Farey structure.
    Frequencies ω_k = 2π · p_k/q_k where p_k/q_k is the k-th Farey fraction.

    These represent natural resonance frequencies at multiple time-scales:
    ω=π (period-2), ω=2π/3 (period-3), ω=π/2 (period-4), etc.
    """
    fareys = farey_sequence(max(3, int(math.sqrt(n)) + 2))
    # Remove 0 and 1 (DC and Nyquist), deduplicate, sort by period (denominator)
    fareys = sorted(
        {f for f in fareys if 0 < float(f) < 1},
        key=lambda f: (f.denominator, f.numerator),
    )
    # Take n values, cycling if needed
    freqs = []
    for i in range(n):
        frac = fareys[i % len(fareys)]
        freqs.append(float(frac) * 2 * math.pi)
    return torch.tensor(freqs, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Modern Hopfield Memory  (fixed patterns, no gradient)
# ─────────────────────────────────────────────────────────────────────────────

class ModernHopfieldMemory(nn.Module):
    """
    Associative memory with exponential capacity O(exp(d/2)).

    Stored patterns X ∈ R^{N×d} are fixed buffers seeded analytically.
    Retrieval via ONE softmax update step:
        ξ_out = Xᵀ · softmax(β · X · ξ_in / √d)

    This is equivalent to one-step cross-attention (query=input, key=value=patterns).
    The advantage over standard attention: patterns X are ANALYTIC (no training).

    Pattern seeding strategies:
      1. Mandelbrot basis: Fourier vectors at Farey-sequence frequencies
      2. Zipf vectors: random unit vectors weighted by Zipf's law
      3. Uniform sphere: maximal coverage of d-dimensional ball
    """

    def __init__(
        self,
        d_model: int,
        n_patterns: int,
        beta: float = 8.0,       # inverse temperature (higher = harder retrieval)
        n_mandelbrot: int = 0,   # patterns from Mandelbrot frequencies (0 = auto)
        seed: int = 42,
    ):
        super().__init__()
        self.d = d_model
        self.N = n_patterns
        self.beta = beta

        # Build the pattern matrix analytically
        patterns = self._build_patterns(d_model, n_patterns, n_mandelbrot, seed)
        self.register_buffer("X", patterns)   # [N, d]

    def _build_patterns(
        self,
        d: int,
        N: int,
        n_mandelbrot: int,
        seed: int,
    ) -> torch.Tensor:
        patterns = []

        # ── 1. Mandelbrot-frequency Fourier vectors ──
        n_mb = n_mandelbrot if n_mandelbrot > 0 else N // 3
        n_mb = min(n_mb, N)
        freqs = mandelbrot_frequencies(n_mb)          # [n_mb] natural resonances
        t = torch.linspace(0, 1, d)                   # [d] time axis
        for omega in freqs:
            v = torch.cat([
                torch.cos(omega * t[:d // 2]),
                torch.sin(omega * t[d - d // 2:]),
            ])[:d]
            patterns.append(F.normalize(v, dim=0))

        # ── 2. Zipf-weighted random unit vectors ──
        n_zipf = (N - n_mb) // 2
        g = torch.Generator()
        g.manual_seed(seed)
        for k in range(1, n_zipf + 1):
            v = torch.randn(d, generator=g)
            # Scale by Zipf weight (most frequent patterns dominate)
            weight = 1.0 / (k ** 0.5)
            patterns.append(F.normalize(v, dim=0) * weight)

        # ── 3. Orthogonal complement (maximum coverage) ──
        n_ortho = N - n_mb - n_zipf
        for i in range(n_ortho):
            g.manual_seed(seed + 10000 + i)
            v = torch.randn(d, generator=g)
            patterns.append(F.normalize(v, dim=0))

        X = torch.stack(patterns[:N], dim=0)          # [N, d]
        # Re-normalise rows to unit sphere
        return F.normalize(X, dim=1)

    def retrieve(
        self,
        query: torch.Tensor,    # [..., d]
        n_steps: int = 1,
    ) -> torch.Tensor:
        """
        Hopfield retrieval: one or more softmax update steps.
        Works like cross-attention with learned key/value = self.X.
        """
        xi = query
        X = self.X                                             # [N, d]
        scale = self.beta / math.sqrt(self.d)
        for _ in range(n_steps):
            # Attention weights: softmax(β · X · ξ / √d)
            scores = xi @ X.T * scale                          # [..., N]
            attn   = F.softmax(scores, dim=-1)                 # [..., N]
            xi = attn @ X                                      # [..., d]  ← retrieved
        return xi

    def forward(
        self,
        x: torch.Tensor,       # [B, L, d]
        n_steps: int = 1,
    ) -> torch.Tensor:
        """Enrich each token's representation with Hopfield retrieval."""
        retrieved = self.retrieve(x, n_steps)
        return F.normalize(retrieved, dim=-1) * x.norm(dim=-1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
# Zipfian decoder initialization
# ─────────────────────────────────────────────────────────────────────────────

class ZipfianDecoder(nn.Module):
    """
    Decoder head initialized from Zipf's law — no training required for
    reasonable marginal token probability estimates.

    Zipf's law: P(rank=k) ∝ k^{-α},  α ≈ 1.0 for natural language.

    The decoder weight is initialized with the Zipf spectral structure:
        W[k, :] = singular_vector_k * k^{-α/2}
    where the singular vectors come from the token's analytic geometry.

    This gives:
      • Correct marginal P(t) ∝ t^{-α} (without any corpus)
      • Coherent geometry (tokens close in embedding space → similar logits)
      • Fast convergence if fine-tuned (warm start from correct distribution)
    """

    def __init__(
        self,
        in_dim: int,
        vocab_size: int,
        alpha: float = 1.0,
        seed: int = 27182,
    ):
        super().__init__()
        V = vocab_size
        g = torch.Generator()
        g.manual_seed(seed)

        # Random orthonormal basis for the projection
        W_rand = torch.randn(in_dim, V, generator=g)
        U, _, Vh = torch.linalg.svd(W_rand, full_matrices=False)

        # Zipf singular value scaling: σ_k ∝ k^{-α/2}
        k = torch.arange(1, min(in_dim, V) + 1, dtype=torch.float32)
        sigma = k ** (-alpha / 2)
        sigma = sigma / sigma[0]   # normalise to [0, 1]

        # W = U · diag(σ) · Vh   [in_dim, V]
        W = (U * sigma.unsqueeze(0)) @ Vh
        # Zipf bias: log P(rank=k) ∝ -α·log(k)
        ranks = torch.arange(1, V + 1, dtype=torch.float32)
        bias_init = -alpha * torch.log(ranks)
        bias_init = bias_init - bias_init.mean()    # zero-mean

        self.proj = nn.Linear(in_dim, V, bias=True)
        with torch.no_grad():
            self.proj.weight.copy_(W.T)             # [V, in_dim]
            self.proj.bias.copy_(bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# Causal Phase Predictor — autoregressive model in phase space
# ─────────────────────────────────────────────────────────────────────────────

class CausalPhasePredictor(nn.Module):
    """
    Autoregressive prediction in phase space rather than token space.

    Key innovation: instead of predicting P(token_t | context), we predict
    the *phase trajectory* of the Kuramoto oscillators, then decode phases
    to tokens via Hopfield retrieval.

    The phase dynamics are governed by an input-conditioned Kuramoto equation:
        dθ_t/dt = Ω(x_t) + Σⱼ K(x_t, x_j) · sin(θ_j - θ_t)

    where Ω(x_t) = W_Ω · x_t  maps the current context to natural frequencies.

    This is implemented as a recurrent map over the sequence:
        θ_0 = init_phases(x_0)
        θ_t = θ_{t-1} + η · f(θ_{t-1}, x_t, history)

    The advantages:
    1. Phase space is continuous → smoother, more stable predictions
    2. Kuramoto coupling captures long-range dependencies geometrically
    3. Phases can be decoded to tokens via analytic inverse mapping
    4. Causality is built-in (only past phases couple to current)

    Parameters: W_Ω (d→n_phases), W_K (d→n_phases²) — very small.
    """

    def __init__(
        self,
        d_model: int,
        n_phases: int,
        n_hopfield_patterns: int = 128,
        eta: float = 0.2,
        n_steps: int = 4,
        mandelbrot_init: bool = True,
    ):
        super().__init__()
        self.n_phases = n_phases
        self.eta = eta
        self.n_steps = n_steps

        # Natural frequency predictor: x_t → Ω_t ∈ R^{n_phases}
        self.W_omega = nn.Linear(d_model, n_phases, bias=False)
        nn.init.normal_(self.W_omega.weight, std=0.01)

        # Coupling strength predictor: x_t → K_t ∈ R^{n_phases} (diagonal coupling)
        self.W_coupling = nn.Linear(d_model, n_phases, bias=False)
        nn.init.constant_(self.W_coupling.weight, 0.0)

        # Phase-to-feature projection (for decoding)
        self.phase_to_feat = nn.Linear(2 * n_phases, d_model, bias=False)
        nn.init.normal_(self.phase_to_feat.weight, std=0.02)

        # Mandelbrot initial frequencies (fixed — no gradient)
        if mandelbrot_init:
            omega_mb = mandelbrot_frequencies(n_phases)
        else:
            omega_mb = torch.linspace(0, 2 * math.pi, n_phases + 1)[:-1]
        self.register_buffer("omega_init", omega_mb)

        # Hopfield memory for phase-to-token retrieval
        self.hopfield = ModernHopfieldMemory(
            d_model, n_hopfield_patterns, beta=4.0
        )

        self.norm = nn.LayerNorm(d_model)

    def init_phases(self, x: torch.Tensor) -> torch.Tensor:
        """
        Initialise phase from context + Mandelbrot base frequencies.
        x: [B, d]  →  theta: [B, n_phases]
        """
        # Base: Mandelbrot angles
        theta = self.omega_init.unsqueeze(0).expand(x.shape[0], -1)  # [B, n_phases]
        # Modulate by input (gives different starting point per token)
        delta = self.W_omega(x)  # [B, n_phases]
        return theta + 0.1 * delta

    def step_phases(
        self,
        theta: torch.Tensor,   # [B, n_phases] — current phases
        x: torch.Tensor,       # [B, d]         — current token embedding
        theta_ctx: Optional[torch.Tensor] = None,  # [B, L_past, n_phases]
    ) -> torch.Tensor:
        """
        One step of the input-conditioned Kuramoto map.

        dθ/dt = Ω(x) + K(x) ⊙ Σⱼ sin(θⱼ - θ)   [causal: j < t only]
        """
        Omega = self.W_omega(x)       # [B, n_phases]  natural freq
        K     = torch.sigmoid(self.W_coupling(x)) * 2.0  # [B, n_phases]  coupling

        if theta_ctx is not None and theta_ctx.shape[1] > 0:
            # Causal coupling: mean over all past phases
            diff = theta_ctx - theta.unsqueeze(1)  # [B, L_past, n_phases]
            coupling = K * torch.sin(diff).mean(dim=1)   # [B, n_phases]
        else:
            coupling = torch.zeros_like(theta)

        return theta + self.eta * (Omega + coupling)

    def forward(
        self,
        x: torch.Tensor,          # [B, L, d_model] — embedded tokens
        return_phases: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run causal phase prediction over the sequence.

        Returns:
            h_out : [B, L, d_model] — phase-enriched representations
            phases : [B, L, n_phases] — phase trajectories (if return_phases)
        """
        B, L, d = x.shape

        # Initialise from first token
        theta = self.init_phases(x[:, 0])        # [B, n_phases]
        all_phases = [theta]

        for t in range(1, L):
            # Stack past phases [B, t, n_phases]
            theta_ctx = torch.stack(all_phases, dim=1)
            theta = self.step_phases(theta, x[:, t], theta_ctx)
            all_phases.append(theta)

        phase_traj = torch.stack(all_phases, dim=1)   # [B, L, n_phases]

        # Encode phases → feature space
        cos_theta = torch.cos(phase_traj)
        sin_theta = torch.sin(phase_traj)
        h_phase = self.phase_to_feat(
            torch.cat([cos_theta, sin_theta], dim=-1)
        )                                             # [B, L, d_model]

        # Hopfield enrichment: retrieve nearest analytic patterns
        h_retrieved = self.hopfield(h_phase)          # [B, L, d_model]

        h_out = self.norm(x + h_phase + h_retrieved)

        if return_phases:
            return h_out, phase_traj
        return h_out, None


# ─────────────────────────────────────────────────────────────────────────────
# BayesianZipfianDecoder — uncertainty-aware LM head
# ─────────────────────────────────────────────────────────────────────────────

class BayesianZipfianDecoder(nn.Module):
    """
    Uncertainty-aware LM head that replaces ZipfianDecoder.

    Uncertainty is derived analytically from the condensate singular values S:
        σ_k (large) → well-defined direction → low uncertainty
        σ_k (small) → diffuse direction → high uncertainty

    The logit distribution for each token is modelled as:

        logit(v | h) = W · h  +  b_zipf  +  ε·N(0, σ_decoder²)

    where:
        σ_decoder²  = diag(W · Σ_condensate · Wᵀ)
        Σ_condensate = diag(1 - S²)   — complement of condensate certainty

    At inference:
        • Greedy/top-p: use mean logit (standard)
        • Uncertainty-weighted: subtract β·σ to penalise uncertain tokens
        • Beam search: use logit ± σ for optimistic/pessimistic bounds

    This is equivalent to Thompson Sampling in logit space:
        logit ← logit + σ · ξ,  ξ ~ N(0, 1)
    which is natural exploration without any external temperature parameter.
    """

    def __init__(
        self,
        in_dim:         int,
        vocab_size:     int,
        alpha:          float = 1.0,
        seed:           int   = 27182,
        uncertainty_beta: float = 0.1,   # weight of uncertainty penalty
    ):
        super().__init__()
        self.in_dim   = in_dim
        self.vocab_size = vocab_size
        self.uncertainty_beta = uncertainty_beta

        # ── Zipf-initialised projection (same as ZipfianDecoder) ──────────
        V = vocab_size
        g = torch.Generator()
        g.manual_seed(seed)
        W_rand = torch.randn(in_dim, V, generator=g)
        U, _, Vh = torch.linalg.svd(W_rand, full_matrices=False)
        k     = torch.arange(1, min(in_dim, V) + 1, dtype=torch.float32)
        sigma = k ** (-alpha / 2)
        sigma = sigma / sigma[0]
        W     = (U * sigma.unsqueeze(0)) @ Vh
        ranks = torch.arange(1, V + 1, dtype=torch.float32)
        bias_init = -alpha * torch.log(ranks)
        bias_init = bias_init - bias_init.mean()

        self.proj = nn.Linear(in_dim, V, bias=True)
        with torch.no_grad():
            self.proj.weight.copy_(W.T)
            self.proj.bias.copy_(bias_init)

        # ── Condensate singular values (updated externally) ───────────────
        # S: [rank] — set by attach_condensate() from SpectralCondensate.S
        self.register_buffer("condensate_S", torch.ones(in_dim))

        # ── Uncertainty scale: learnable temperature ───────────────────────
        self.log_sigma_scale = nn.Parameter(torch.tensor(0.0))

    def attach_condensate(self, S: torch.Tensor):
        """
        Update uncertainty prior from condensate singular values.
        S: [rank]  — normalised singular values in [0, 1].
        Call after each condensate.condense() or update_online().
        """
        r = min(S.shape[0], self.in_dim)
        with torch.no_grad():
            self.condensate_S.fill_(1e-4)    # small baseline uncertainty
            self.condensate_S[:r] = S[:r].clamp(0, 1)

    @property
    def uncertainty_var(self) -> torch.Tensor:
        """Per-dimension variance from condensate: 1 - S² ∈ (0, 1]."""
        return (1.0 - self.condensate_S ** 2).clamp(min=1e-6)   # [in_dim]

    def forward(
        self,
        x:           torch.Tensor,              # [..., in_dim]
        sample_noise: bool = False,             # True → Thompson sampling
        return_sigma: bool = False,             # True → also return logit σ
    ) -> torch.Tensor:
        """
        Returns logits [..., vocab_size].
        If sample_noise=True, adds N(0, σ²) to each logit (exploration).
        """
        logits = self.proj(x)   # [..., V]

        # Propagate input uncertainty through projection
        # σ²_logit[v] = Σ_d  W[v,d]² · var_d
        W = self.proj.weight                          # [V, in_dim]
        var_d = self.uncertainty_var                  # [in_dim]
        sigma_sq = (W ** 2 * var_d.unsqueeze(0)).sum(-1)  # [V]
        sigma = sigma_sq.sqrt() * self.log_sigma_scale.exp()  # [V]

        if sample_noise and self.training:
            # Thompson sampling: perturb logits by noise proportional to σ
            noise = torch.randn_like(logits) * sigma
            logits = logits + noise
        else:
            # Deterministic: apply uncertainty penalty (pessimistic inference)
            logits = logits - self.uncertainty_beta * sigma

        if return_sigma:
            return logits, sigma
        return logits

    def expected_entropy(self) -> torch.Tensor:
        """
        Expected entropy of the output distribution due to condensate uncertainty.
        Useful as a diagnostic: high entropy → model is genuinely uncertain.
        """
        W   = self.proj.weight          # [V, in_dim]
        var = self.uncertainty_var      # [in_dim]
        # Avg logit variance = mean over vocab of Σ_d W[v,d]² · var_d
        return (W ** 2 * var.unsqueeze(0)).sum(-1).mean().sqrt()

    def loss_calibration(
        self,
        logits: torch.Tensor,   # [B, L, V]
        targets: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        """
        NLL loss that also penalises over-confidence (Brier-style regulariser).
        Encourages calibrated uncertainty — not just minimum perplexity.
        """
        nll = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        # Entropy regulariser: reward higher entropy on non-target tokens
        probs = F.softmax(logits, dim=-1)                        # [B, L, V]
        entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean() # scalar
        # We want moderate entropy — penalise both extremes lightly
        return nll + 0.01 * (entropy - math.log(self.vocab_size) * 0.5).abs()
