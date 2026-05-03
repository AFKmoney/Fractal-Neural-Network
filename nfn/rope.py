"""
Rotary Position Embeddings (RoPE) with NTK-aware long-context scaling.

References:
  - Su et al. (2021) RoFormer: Enhanced Transformer with Rotary Position Embedding
  - bloc97/NTK-Aware Scaled RoPE (2023)
  - LongRoPE / YaRN extensions

Supports contexts up to 128K tokens via dynamic NTK frequency scaling.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Core RoPE math
# ─────────────────────────────────────────────────────────────────────────────

def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    base: float = 10_000.0,
    scale_factor: float = 1.0,   # NTK scaling: > 1 extends context
) -> torch.Tensor:
    """
    Precompute the complex exponentials e^{i θ_k} for positions 0..max_seq_len-1.
    Returns freqs_cis : [max_seq_len, dim//2]  (complex64)
    """
    # Apply NTK-aware frequency scaling
    if scale_factor != 1.0:
        base = base * (scale_factor ** (dim / (dim - 2)))

    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # [dim/2]
    positions = torch.arange(max_seq_len).float()                     # [L]
    angles = torch.outer(positions, theta)                             # [L, dim/2]
    return torch.polar(torch.ones_like(angles), angles)               # complex [L, dim/2]


def apply_rotary_emb(
    xq: torch.Tensor,           # [B, H, T, d_head]
    xk: torch.Tensor,           # [B, H, T, d_head]
    freqs_cis: torch.Tensor,    # [T, d_head//2]   complex
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to query and key tensors."""
    # View as complex: [B, H, T, d_head//2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis: [T, d//2] → [1, 1, T, d//2] for broadcasting
    f = freqs_cis.unsqueeze(0).unsqueeze(0)

    xq_rot = torch.view_as_real(xq_ * f).flatten(-2).to(xq.dtype)
    xk_rot = torch.view_as_real(xk_ * f).flatten(-2).to(xk.dtype)
    return xq_rot, xk_rot


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic NTK scaling for inference beyond training length
# ─────────────────────────────────────────────────────────────────────────────

class RoPECache(nn.Module):
    """
    Manages RoPE frequencies with dynamic NTK scaling.

    At training time: use static frequencies for max_seq_len.
    At inference with longer sequences: automatically rescale via NTK.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10_000.0,
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scale_factor = scale_factor

        freqs = precompute_freqs_cis(dim, max_seq_len, base, scale_factor)
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def get_freqs(self, seq_len: int, offset: int = 0) -> torch.Tensor:
        """
        Return frequencies for positions [offset, offset+seq_len).
        Extends dynamically if seq_len+offset exceeds precomputed range.
        """
        end = offset + seq_len
        if end > self.freqs_cis.shape[0]:
            # Dynamic NTK extension
            scale = end / self.max_seq_len
            freqs = precompute_freqs_cis(
                self.dim, end, self.base,
                scale_factor=self.scale_factor * scale
            ).to(self.freqs_cis.device)
            return freqs[offset:end]
        return self.freqs_cis[offset:end]

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = xq.shape[2]
        freqs = self.get_freqs(T, offset)
        return apply_rotary_emb(xq, xk, freqs)


# ─────────────────────────────────────────────────────────────────────────────
# Fractal-level RoPE: position-aware sinusoidal frequencies per level
# ─────────────────────────────────────────────────────────────────────────────

def fractal_level_positions(
    level: int,
    n_nodes: int,
    branching: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns the effective sequence positions for level-k nodes.
    Level-k node i corresponds to original positions [i*b^k, (i+1)*b^k).
    We use the centre: i*b^k + b^k/2.
    """
    stride = branching ** level
    centres = torch.arange(n_nodes, device=device).float() * stride + stride / 2
    return centres
