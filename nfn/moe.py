"""
Phase-Routed Mixture of Experts (PR-MoE) — NFN v3.2

Phase-routed MoE + fractal linear attention.
Speed path (2026-09-12): top-k expert GEMM only; chunked causal
linear attn with carry (S, z). Same parameter names as before.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hopfield import mandelbrot_frequencies


class PhaseEncoder(nn.Module):
    """Maps d-dimensional hidden state to n_phases phase angles in (-pi, pi)."""

    def __init__(self, d_model: int, n_phases: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_phases * 2, bias=False)
        nn.init.normal_(self.proj.weight, std=0.01)
        self.n_phases = n_phases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        re = h[..., :self.n_phases]
        im = h[..., self.n_phases:]
        return torch.atan2(im, re)


class PhaseRoutedMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        d_ff_per_expert: int,
        n_phases: int = 8,
        top_k: int = 2,
        kappa: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.E = n_experts
        self.K = top_k
        self.kappa = kappa
        self.n_phases = n_phases
        self.d_ff = d_ff_per_expert
        self.W1 = nn.Parameter(torch.empty(n_experts, d_ff_per_expert, d_model))
        self.W2 = nn.Parameter(torch.empty(n_experts, d_model, d_ff_per_expert))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        self.phase_enc = PhaseEncoder(d_model, n_phases)
        expert_phases = mandelbrot_frequencies(n_experts * n_phases)
        expert_phases = expert_phases[:n_experts * n_phases].view(n_experts, n_phases)
        self.register_buffer("expert_phases", expert_phases)
        self.norm = nn.LayerNorm(d_model)

    def routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        theta_x = self.phase_enc(x)
        theta_e = self.expert_phases
        diff = theta_x.unsqueeze(-2) - theta_e
        energy = self.kappa * torch.cos(diff).mean(dim=-1)
        return F.softmax(energy, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        N = B * L
        weights = self.routing_weights(x)
        topk_vals, topk_idx = torch.topk(weights, self.K, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(-1, keepdim=True).clamp(min=1e-6)
        x_flat = x.reshape(N, d)
        idx = topk_idx.reshape(N, self.K)
        val = topk_vals.reshape(N, self.K)
        out = x_flat.new_zeros(N, d)
        for e in range(self.E):
            hit = idx == e
            token_mask = hit.any(dim=-1)
            if not token_mask.any():
                continue
            w = (val * hit.float()).sum(dim=-1)[token_mask].unsqueeze(-1)
            xe = x_flat[token_mask]
            h = F.gelu(xe @ self.W1[e].t())
            out[token_mask] = out[token_mask] + w * (h @ self.W2[e].t())
        return self.norm(x + out.view(B, L, d))

    def load_balance_loss(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.routing_weights(x)
        mean_load = weights.mean(dim=[0, 1])
        target = torch.ones_like(mean_load) / self.E
        return F.mse_loss(mean_load, target)


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


class FractalLinearAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_levels: int = 3,
        dropout: float = 0.0,
        causal: bool = True,
        chunk_size: int = 32,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.H = n_heads
        self.d_head = d_model // n_heads
        self.n_levels = n_levels
        self.causal = causal
        self.chunk_size = chunk_size
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        freqs = mandelbrot_frequencies(n_levels * self.d_head).view(n_levels, self.d_head)
        self.register_buffer("level_freqs", freqs)
        self.norm_q = nn.LayerNorm(d_model)

    def _feature_map(self, x: torch.Tensor, level: int) -> torch.Tensor:
        freq = self.level_freqs[level % self.n_levels]
        return elu_feature_map(x + freq.unsqueeze(0).unsqueeze(0).unsqueeze(0))

    def _causal_linear_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        level: int = 0,
    ) -> torch.Tensor:
        q = self._feature_map(q, level)
        k = self._feature_map(k, level)
        B, H, L, D = q.shape
        C = max(1, int(self.chunk_size))
        out = q.new_empty(B, H, L, D)
        S = q.new_zeros(B, H, D, D)
        z = q.new_zeros(B, H, D)
        for s in range(0, L, C):
            e = min(s + C, L)
            qc, kc, vc = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e]
            kv = kc.unsqueeze(-1) * vc.unsqueeze(-2)
            kv_cs = torch.cumsum(kv, dim=2) + S.unsqueeze(2)
            z_cs = torch.cumsum(kc, dim=2) + z.unsqueeze(2)
            num = torch.einsum("bhcd,bhcdv->bhcv", qc, kv_cs)
            den = torch.einsum("bhcd,bhcd->bhc", qc, z_cs).unsqueeze(-1).clamp(min=1e-6)
            out[:, :, s:e] = num / den
            S = S + kv.sum(dim=2)
            z = z + kc.sum(dim=2)
        return out

    def forward(self, x: torch.Tensor, level: int = 0) -> torch.Tensor:
        B, L, d = x.shape
        x_n = self.norm_q(x)
        qkv = self.qkv(x_n)
        q, k, v = qkv.split(d, dim=-1)

        def mh(t):
            return t.view(B, L, self.H, self.d_head).transpose(1, 2)

        q, k, v = mh(q), mh(k), mh(v)
        if self.causal:
            attn_out = self._causal_linear_attn(q, k, v, level=level)
        else:
            q = self._feature_map(q, level)
            k = self._feature_map(k, level)
            kv = k.transpose(-2, -1) @ v
            attn_out = q @ kv
            norm = (q * k.sum(dim=-2, keepdim=True)).sum(dim=-1, keepdim=True).clamp(1e-6)
            attn_out = attn_out / norm
        out = attn_out.transpose(1, 2).contiguous().view(B, L, d)
        return self.drop(self.out(out))


class PhaseSoliton(nn.Module):
    def __init__(self, d_model: int, n_phases: int = 8, tau: float = 1.0):
        super().__init__()
        self.phase_enc = PhaseEncoder(d_model, n_phases)
        self.gain = nn.Linear(n_phases, 1, bias=True)
        nn.init.constant_(self.gain.bias, -2.0)
        self.tau = tau
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.phase_enc(x)
        theta_mean = theta.mean(dim=1, keepdim=True)
        coherence = torch.cos(theta - theta_mean).mean(dim=-1, keepdim=True)
        gain = torch.sigmoid(self.gain(theta) + coherence / self.tau)
        return self.norm(x + gain * x)
