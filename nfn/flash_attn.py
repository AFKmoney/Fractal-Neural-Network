"""
NFN v5.0 — Unified Flash Attention Utility

Wraps torch.nn.functional.scaled_dot_product_attention (SDPA) for
maximum performance on GPU (Flash Attention / Memory-Efficient Attention)
with automatic CPU fallback.

All NFN sub-modules should use this instead of nn.MultiheadAttention
or manual bmm attention for consistent O(L) memory usage.

SDPA auto-selects the best kernel:
  - FlashAttention-2 (Ampere+ GPUs, fp16/bf16)
  - Memory-efficient attention (xformers-style)
  - Math fallback (CPU / old GPUs)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashAttention(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention that uses SDPA.
    
    Advantages over nn.MultiheadAttention:
      - 2-4x faster on GPU via Flash Attention kernels
      - O(L) memory instead of O(L²) — no materialized attention matrix
      - Automatic kernel selection (Flash > memory-efficient > math)
      - Simpler API — no num_heads confusion, no in_proj_weight gymnastics
    
    Usage:
        attn = FlashAttention(d_model=256, n_heads=4)
        out = attn(query, key, value)  # [B, L, d] → [B, L, d]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout
        self.causal = causal

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)

        self._has_sdpa = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        query:  torch.Tensor,
        key:    torch.Tensor,
        value:  torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        query/key/value: [B, L, d_model]
        mask: optional additive mask [B, 1, L_q, L_k] or None
        Returns: [B, L, d_model]
        """
        B, L, _ = query.shape
        H, d_h = self.n_heads, self.d_head

        Q = self.W_q(query).view(B, L, H, d_h).transpose(1, 2)  # [B, H, L, d_h]
        K = self.W_k(key).view(B, key.shape[1], H, d_h).transpose(1, 2)
        V = self.W_v(value).view(B, value.shape[1], H, d_h).transpose(1, 2)

        if self._has_sdpa and mask is None:
            out = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=self.causal,
            )
        else:
            scale = d_h ** -0.5
            scores = (Q @ K.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            if self.causal:
                causal_mask = torch.triu(
                    torch.ones(L, K.shape[2], device=query.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = F.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0:
                attn = F.dropout(attn, p=self.dropout_p)
            out = attn @ V

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_out(out)


def flash_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    causal: bool = False,
) -> torch.Tensor:
    """
    Functional flash attention for inline use (no module needed).
    
    q/k/v: [B, H, L, d_head]
    Returns: [B, H, L, d_head]
    """
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout_p,
            is_causal=causal,
        )
    
    d_h = q.shape[-1]
    scale = d_h ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        L = q.shape[2]
        causal_mask = torch.triu(
            torch.ones(L, k.shape[2], device=q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ v
