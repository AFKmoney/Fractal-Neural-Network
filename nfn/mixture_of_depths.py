"""
NFN v4.0 — Mixture of Depths (MoD)

Tokens that the model is already confident about don't need to pass
through every block.  A lightweight router decides, per token, whether
to run it through the full block or skip it with a residual bypass.

Theory (Raposo et al. 2024 — "Mixture of Depths"):
  For each block b and each token position i:
    if router_score(h_i) ≥ threshold:
        h_i ← block(h_i)          # full computation
    else:
        h_i ← h_i                 # free pass-through

The router is a single linear layer that produces a scalar confidence.
In NFN we ground the router in phase coherence:
  - High phase coherence  → token is already well-represented → skip
  - Low phase coherence   → token needs more processing → compute

This has two effects:
  1. Efficiency: easy tokens (punctuation, stopwords) skip most blocks.
     Empirically ~40-60% of tokens can be skipped in later layers.
  2. Quality: computation concentrates on hard tokens (rare words, logical
     connectors, ambiguous references) — exactly where it is needed.

The routing decision is differentiable via a soft gate during training
(straight-through estimator for the discrete skip decision at inference).

Capacity constraint:
  To ensure at least k_min tokens per block, we use top-k routing:
  the top k tokens (by router score) always get processed.
  k = capacity_factor × sequence_length  (default: 0.5 → 50% of tokens)
"""

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoDRouter(nn.Module):
    """
    Lightweight token router for Mixture of Depths.

    Score = sigmoid(W · h + b)  ∈ (0, 1)
    High score  → process (token needs work)
    Low score   → skip  (token is already good)

    Uses phase coherence as a prior: tokens with high local phase
    agreement get a score bias toward skipping.
    """

    def __init__(self, d_model: int, n_phases: int = 8):
        super().__init__()
        self.n_phases = n_phases

        # Primary router: learns content-based routing
        self.router = nn.Linear(d_model, 1)
        nn.init.zeros_(self.router.weight)
        nn.init.constant_(self.router.bias, 0.5)   # start at 50% processing

        # Phase coherence extractor (no params — pure geometry)
        # coherence = mean |e^{i·θ}| across phases = mean cos(Δθ)

    def _phase_coherence(self, h: torch.Tensor) -> torch.Tensor:
        """
        Estimate local phase coherence from hidden state geometry.
        h: [B, L, d]  →  coherence: [B, L]  ∈ [0, 1]

        Treat pairs (h[2k], h[2k+1]) as real/imag of complex number.
        Coherence = mean magnitude of normalised complex phasors.
        """
        d = h.shape[-1]
        n_pairs = min(self.n_phases, d // 2)
        real = h[..., :n_pairs]
        imag = h[..., n_pairs:2 * n_pairs]
        magnitude = (real ** 2 + imag ** 2).sqrt().mean(-1) + 1e-6
        coherence  = (real ** 2 + imag ** 2).sqrt() / magnitude.unsqueeze(-1)
        return coherence.mean(-1)   # [B, L]

    def forward(
        self,
        h: torch.Tensor,   # [B, L, d]
    ) -> torch.Tensor:
        """Returns routing scores [B, L] ∈ (0, 1).  Higher = process."""
        base_score  = self.router(h).squeeze(-1)         # [B, L]
        coherence   = self._phase_coherence(h).detach()  # [B, L]  no grad through coherence
        # Low coherence → needs processing (high score)
        # High coherence → already good (low score)
        return torch.sigmoid(base_score - coherence)


class MixtureOfDepths(nn.Module):
    """
    Wraps any h→h block with token-level routing.

    During training: soft gating via straight-through estimator
    During inference: hard top-k selection (exact skip)

    Capacity: capacity_factor controls fraction of tokens processed.
    capacity_factor=1.0 → all tokens (no skip, standard block)
    capacity_factor=0.5 → top 50% tokens processed, rest bypassed
    capacity_factor=0.25 → top 25% tokens processed

    The router loss (load balance) ensures routing doesn't collapse
    to always skipping or always processing.
    """

    def __init__(
        self,
        d_model:         int,
        block:           nn.Module,
        capacity_factor: float = 0.5,   # fraction of tokens to process
        n_phases:        int   = 8,
        lambda_router:   float = 0.01,  # load-balance penalty
    ):
        super().__init__()
        self.block           = block
        self.capacity_factor = capacity_factor
        self.lambda_router   = lambda_router

        self.router = MoDRouter(d_model, n_phases)

    def forward(
        self,
        h:   torch.Tensor,                  # [B, L, d]
        **block_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          h_out      : [B, L, d]  — processed tokens merged with bypassed
          router_loss: scalar     — load balance penalty
        """
        B, L, d = h.shape
        k = max(1, int(L * self.capacity_factor))

        scores = self.router(h)                    # [B, L] ∈ (0,1)

        # Select top-k token indices per batch item
        topk_scores, topk_idx = torch.topk(scores, k, dim=1)  # [B, k]

        # Gather selected tokens
        idx_expanded = topk_idx.unsqueeze(-1).expand(B, k, d)
        h_selected   = h.gather(1, idx_expanded)   # [B, k, d]

        # Run block on selected tokens only
        if hasattr(self.block, 'forward'):
            result = self.block(h_selected, **block_kwargs)
            h_processed = result[0] if isinstance(result, tuple) else result
            block_losses = result[1] if isinstance(result, tuple) and len(result) > 1 else {}
        else:
            h_processed = self.block(h_selected)
            block_losses = {}

        # Scatter processed tokens back; unselected tokens keep original h
        # Straight-through: forward=hard scatter, backward=soft gradient
        h_out = h.clone()

        if self.training:
            # Soft gate: use router scores as weights for gradient flow
            gate = topk_scores.unsqueeze(-1)                   # [B, k, 1]
            h_gated = h_selected + gate * (h_processed - h_selected)
            h_out.scatter_(1, idx_expanded, h_gated)
        else:
            h_out.scatter_(1, idx_expanded, h_processed)

        # Load balance loss: encourage uniform routing (not all-skip or all-process)
        # Target: capacity_factor of tokens should be selected on average
        target = self.capacity_factor
        actual = scores.mean()
        router_loss = self.lambda_router * (actual - target) ** 2

        return h_out, router_loss

    @property
    def effective_flop_fraction(self) -> float:
        """Fraction of full-block FLOPs used (lower = more efficient)."""
        return self.capacity_factor
