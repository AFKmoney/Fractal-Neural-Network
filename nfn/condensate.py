"""
NFMC — Neural Fractal Multidimensional Condensate (NFN v3.0)

Universal Fractal Kernel approximated via Random Fractal Fourier Features:

    K(x, y) = ∫_Ω exp(i·Φ_ω(x,y)) dμ(ω)
             ≈ φ(x)ᵀφ(y)

where φ uses a self-similar frequency lattice drawn from a Mandelbrot-inspired
spectral measure (1/f spectrum — dense low-freq, sparse high-freq).

Condensation:
    K̃_r = U Σ Uᵀ  via one-shot SVD on a reference feature set.
    No gradient updates. Fixed buffers.

Phase Locking (Helmholtz / XY-model):
    E(θ) = -½ Σᵢⱼ K̃(xᵢ,xⱼ) cos(θᵢ - θⱼ)
    dθᵢ/dt = Σⱼ K̃(xᵢ,xⱼ) sin(θⱼ - θᵢ)   [Kuramoto coupling on kernel graph]

This is fully differentiable — can backprop through phase locking for
end-to-end training with the decoder.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Random Fractal Fourier Features  (fixed — no gradient)
# ─────────────────────────────────────────────────────────────────────────────

class FractalRFF(nn.Module):
    """
    Random Fourier Features with a fractal (self-similar) frequency distribution.

    Frequencies are drawn in n_scales octave bands:
        ω_{k,j} = base^(k/n_scales) · ξ_{k,j},   ξ ~ N(0, I_d)

    This gives a 1/f-like power spectrum over scale — structures at every
    resolution from fine-grained (token) to coarse (discourse) are captured.

    All weights are registered as buffers (no gradient, deterministic seed).
    Output dimension: 2 * n_features  (cos + sin components).
    """

    def __init__(
        self,
        d_in: int,
        n_features: int,
        n_scales: int = 8,
        base: float = 2.0,
        seed: int = 1337,
    ):
        super().__init__()
        assert n_features % n_scales == 0, "n_features must be divisible by n_scales"
        feat_per_scale = n_features // n_scales

        g = torch.Generator()
        g.manual_seed(seed)

        W_parts, b_parts = [], []
        for k in range(n_scales):
            scale = base ** (k / n_scales)   # 1.0 … base^((n_scales-1)/n_scales)
            Wk = torch.randn(d_in, feat_per_scale, generator=g) * scale
            bk = torch.rand(feat_per_scale, generator=g) * (2 * math.pi)
            W_parts.append(Wk)
            b_parts.append(bk)

        self.register_buffer("W", torch.cat(W_parts, dim=1))  # [d_in, n_features]
        self.register_buffer("b", torch.cat(b_parts, dim=0))  # [n_features]
        self.n_features = n_features
        # RFF normalisation: sqrt(2/n_features) so K ≈ φᵀφ
        self._norm = math.sqrt(2.0 / n_features)

    @property
    def out_dim(self) -> int:
        return 2 * self.n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_in]  →  [..., 2*n_features]
        proj = x @ self.W + self.b             # [..., n_features]
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) * self._norm


# ─────────────────────────────────────────────────────────────────────────────
# 2. Spectral Condensate  (fixed after one-shot SVD — no gradient)
# ─────────────────────────────────────────────────────────────────────────────

class SpectralCondensate(nn.Module):
    """
    Rank-r spectral projection of the fractal kernel operator.

    Stores:
        U  ∈ R^{rff_dim × rank}  — top-r right singular vectors
        S  ∈ R^rank              — normalised singular values

    Initialised by one of:
      condense(X)              — SVD of feature matrix X [N, rff_dim]
      condense_cooccurrence(C) — SVD of character co-occurrence matrix
      (default)                — identity (first `rank` dims pass-through)
    """

    def __init__(self, rff_dim: int, rank: int):
        super().__init__()
        # Default: identity pass-through on first `rank` dimensions
        U_init = torch.zeros(rff_dim, rank)
        r0 = min(rank, rff_dim)
        U_init[:r0, :r0] = torch.eye(r0)
        self.register_buffer("U", U_init)           # [rff_dim, rank]
        self.register_buffer("S", torch.ones(rank)) # [rank]
        self.rff_dim = rff_dim
        self.rank = rank

    @torch.no_grad()
    def condense(self, features: torch.Tensor):
        """
        One-shot condensation. No gradient, no SGD.
        features: [N, rff_dim]
        """
        features = features - features.mean(dim=0, keepdim=True)
        try:
            _, S, Vh = torch.linalg.svd(features, full_matrices=False)
        except Exception:
            _, S, Vh = torch.svd(features)
        r = min(self.rank, S.shape[0], Vh.shape[0])
        self.U.zero_()
        self.U[:, :r] = Vh[:r].T           # [rff_dim, r]
        self.S.fill_(1e-8)
        self.S[:r] = S[:r] / (S[0] + 1e-8)  # normalise to [0, 1]

    @torch.no_grad()
    def condense_cooccurrence(self, cooccur: torch.Tensor):
        """Seed from a vocabulary co-occurrence matrix [V, V]."""
        try:
            _, S, Vh = torch.linalg.svd(cooccur.float(), full_matrices=False)
        except Exception:
            _, S, Vh = torch.svd(cooccur.float())
        # Vh is [min(V,V), V]; we need [rff_dim, rank] — use as mixing vectors
        r = min(self.rank, S.shape[0])
        Vh_pad = torch.zeros(r, self.rff_dim)
        cols = min(Vh.shape[1], self.rff_dim)
        Vh_pad[:r, :cols] = Vh[:r, :cols]
        self.U.zero_()
        self.U[:, :r] = Vh_pad.T
        self.S.fill_(1e-8)
        self.S[:r] = S[:r] / (S[0] + 1e-8)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        # phi: [..., rff_dim]  →  [..., rank]
        return (phi @ self.U) * self.S


# ─────────────────────────────────────────────────────────────────────────────
# 3. Helmholtz Phase Locking  (differentiable Kuramoto on kernel graph)
# ─────────────────────────────────────────────────────────────────────────────

class HelmholtzPhaseLocking(nn.Module):
    """
    Minimises the Helmholtz free energy via gradient descent:

        E(θ) = -½ Σᵢⱼ K̃(xᵢ,xⱼ) cos(θᵢ - θⱼ)

    Update rule (attractive Kuramoto coupling):
        θᵢ ← θᵢ + η Σⱼ K̃(xᵢ,xⱼ) sin(θⱼ - θᵢ)

    This is fully differentiable — gradients flow back to the condensate
    and embedding via phase trajectories.

    Phases are initialised deterministically from the condensate geometry
    (no random noise), ensuring reproducible phase attractors.
    """

    def __init__(
        self,
        n_phases: int = 8,
        n_iter: int = 8,
        eta: float = 0.15,
        sparse_k: int = 0,        # if > 0, keep only top-k connections per node
    ):
        super().__init__()
        self.n_phases = n_phases
        self.n_iter = n_iter
        self.eta = eta
        self.sparse_k = sparse_k

    def forward(
        self,
        z: torch.Tensor,                              # [B, L, rank]
        init_phases: Optional[torch.Tensor] = None,  # [B, L, n_phases] or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        theta : [B, L, n_phases]  — converged phases
        K_sim : [B, L, L]         — kernel similarity matrix
        """
        B, L, rank = z.shape

        # Kernel similarity matrix from normalised condensate features
        z_n = F.normalize(z, dim=-1)
        K_sim = z_n @ z_n.transpose(-2, -1)   # [B, L, L]  in [-1, 1]

        # Optional sparse k-NN masking
        if self.sparse_k > 0 and self.sparse_k < L:
            topk_vals, _ = torch.topk(K_sim, self.sparse_k, dim=-1)
            thresh = topk_vals[..., -1:].detach()
            K_sim = K_sim * (K_sim >= thresh).float()

        # Phase initialisation from condensate geometry (deterministic)
        if init_phases is not None:
            theta = init_phases
        else:
            c0 = z[..., 0:1]
            c1 = z[..., 1:2] if rank > 1 else torch.zeros_like(c0)
            base_angle = torch.atan2(c1, c0)                              # [B, L, 1]
            offsets = torch.linspace(
                0, 2 * math.pi, self.n_phases + 1, device=z.device
            )[:-1]                                                          # [n_phases]
            theta = base_angle + offsets.view(1, 1, -1)                   # [B, L, n_phases]

        # Kuramoto-style gradient descent on Helmholtz energy
        for _ in range(self.n_iter):
            # diff[b,i,j,p] = θ_j[p] - θ_i[p]
            diff = theta.unsqueeze(2) - theta.unsqueeze(1)                # [B, L, L, n_phases]
            coupling = (K_sim.unsqueeze(-1) * torch.sin(diff)).mean(dim=2)  # [B, L, n_phases]
            theta = theta + self.eta * coupling

        return theta, K_sim


# ─────────────────────────────────────────────────────────────────────────────
# 4. Condensate decoder  (maps phases + condensate → logits)
# ─────────────────────────────────────────────────────────────────────────────

class CondensateDecoder(nn.Module):
    """
    Sinusoidal inverse decoding:

        h = [cos(θ), sin(θ), z_condensate]      [..., 2*n_phases + rank]
        logits = h @ W_decode

    Optionally frozen (for zero-training experiments) or learnable.
    """

    def __init__(
        self,
        n_phases: int,
        rank: int,
        vocab_size: int,
        learnable: bool = True,
    ):
        super().__init__()
        in_dim = 2 * n_phases + rank
        self.proj = nn.Linear(in_dim, vocab_size, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        if not learnable:
            for p in self.proj.parameters():
                p.requires_grad_(False)

    def forward(self, theta: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # theta: [..., n_phases],  z: [..., rank]
        h = torch.cat([torch.cos(theta), torch.sin(theta), z], dim=-1)
        return self.proj(h)


# ─────────────────────────────────────────────────────────────────────────────
# 5. NFMCKernelLayer  (drop-in enrichment for NFNBlock integration)
# ─────────────────────────────────────────────────────────────────────────────

class NFMCKernelLayer(nn.Module):
    """
    Drop-in layer that enriches NFNBlock hidden states with fractal kernel
    features and phase-locking dynamics.

    Can be inserted after self-attention in each NFNBlock as:
        x = x + nfmc_layer(x)

    The kernel (RFF + condensate + phase locking) is parameter-free once
    initialised. Only the output projection is learnable.
    """

    def __init__(
        self,
        d_model: int,
        n_rff: int = 128,
        n_scales: int = 6,
        rank: int = 32,
        n_phases: int = 4,
        n_iter: int = 4,
        eta: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.rff = FractalRFF(d_model, n_rff, n_scales=n_scales, seed=seed)
        rff_out = self.rff.out_dim
        self.condensate = SpectralCondensate(rff_out, rank)
        self.phase_lock = HelmholtzPhaseLocking(n_phases, n_iter=n_iter, eta=eta)
        self.out_proj = nn.Linear(2 * n_phases + rank, d_model, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.01)
        self.norm = nn.LayerNorm(d_model)

    def condense_from_hidden(self, h: torch.Tensor):
        """Seed condensate from a hidden-state matrix [N, d_model] (one-shot)."""
        with torch.no_grad():
            phi = self.rff(h.view(-1, h.shape[-1]))
            self.condensate.condense(phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        phi = self.rff(x)
        z = self.condensate(phi)
        theta, _ = self.phase_lock(z)
        h = torch.cat([torch.cos(theta), torch.sin(theta), z], dim=-1)
        return self.norm(x + self.out_proj(h))
