"""
Sinusoidal parametric connections — core innovation of the NFN.

Each connection j → i has dynamic weight:
    Γ_{j→i}(t) = A_{ji} · exp(-γ_{ji}·t) · sin(ω_{ji}·t + φ_{ji})

Low-rank factorisation for efficiency:
    Γ_{j→i}(t) = Σ_r α_r(i) · β_r(j) · sin(ω_r·t + φ_{i,r})

This module provides:
  - SinusoidalGate   : scalar gate for a single (position, rank) query
  - SinusoidalAggregator : groups b children → 1 parent with sinusoidal weights
  - InterMotifCoupler: connects two distinct fractal motifs
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SinusoidalGate(nn.Module):
    """
    Learnable sinusoidal gate over a sequence of positions.

    gate(t) = Σ_r A_r · [exp(-γ_r·t)]? · sin(ω_r·t + φ_r)

    Returns a scalar (or vector of size out_channels) per position.
    """

    def __init__(self, out_channels: int, rank: int = 4,
                 damping: bool = True, gamma_init: float = 0.1):
        super().__init__()
        self.rank = rank
        self.damping = damping

        # Per-channel, per-rank parameters
        self.A = nn.Parameter(torch.randn(out_channels, rank) * 0.02)
        self.omega = nn.Parameter(torch.rand(rank) * math.pi + 0.1)   # > 0
        self.phi = nn.Parameter(torch.zeros(out_channels, rank))
        if damping:
            self.log_gamma = nn.Parameter(
                torch.full((rank,), math.log(gamma_init + 1e-6))
            )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions : [N] long or float tensor of discrete time / position indices
        returns   : [N, out_channels] gate values in (-1, 1)

        Shapes:
          omega : [rank]
          phi   : [out_channels, rank]
          A     : [out_channels, rank]
          angle : [N, out_channels, rank]  (full broadcast)
        """
        t = positions.float()   # [N]

        # [N, 1, 1] * [1, 1, rank] + [1, out_channels, rank] → [N, out_channels, rank]
        angle = (
            t.view(-1, 1, 1) * self.omega.view(1, 1, -1)
            + self.phi.unsqueeze(0)
        )

        if self.damping:
            gamma = F.softplus(self.log_gamma) + 1e-6          # [rank]
            decay = torch.exp(-gamma.view(1, 1, -1) * t.view(-1, 1, 1))  # [N, 1, rank]
            sin_val = decay * torch.sin(angle)                  # [N, out_channels, rank]
        else:
            sin_val = torch.sin(angle)                          # [N, out_channels, rank]

        # [N, out_channels] = sum_r A[c,r] * sin_val[n,c,r]
        gate = (self.A.unsqueeze(0) * sin_val).sum(-1)          # [N, out_channels]
        return torch.tanh(gate)


class SinusoidalAggregator(nn.Module):
    """
    Aggregates b child node representations into 1 parent representation
    using position-dependent sinusoidal gating.

    For each parent at position p:
        gated_child_j = Γ_j(p) · child_j
        parent = MLP(concat(gated_children))

    Phase tracking:
        θ_parent = θ_parent_prev + Ω · Δt + g(h_parent)
    """

    def __init__(self, d_model: int, branching: int,
                 rank: int = 4, omega_level: float = 1.0,
                 damping: bool = True, gamma_init: float = 0.1,
                 dropout: float = 0.1):
        super().__init__()
        self.b = branching
        self.d = d_model
        self.omega_level = omega_level  # Ω_i = ω_0 / λ^s for this level

        # Sinusoidal gate: one gate per child slot, per position
        self.gate = SinusoidalGate(
            out_channels=branching, rank=rank,
            damping=damping, gamma_init=gamma_init
        )

        # Feature mixing (gated linear unit style)
        self.W_v = nn.Linear(d_model * branching, d_model)
        self.W_g = nn.Linear(d_model * branching, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Phase update network: maps h → phase delta
        self.phase_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(
        self,
        children: torch.Tensor,          # [B, N*b, d]
        positions: torch.Tensor,          # [N] parent position indices
        prev_phase: Optional[torch.Tensor] = None,  # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h     : [B, N, d]  parent hidden states
        phase : [B, N]     parent phases in [-π, π]
        """
        B, Nb, d = children.shape
        N = len(positions)
        assert Nb == N * self.b, f"children shape mismatch: {Nb} ≠ {N}×{self.b}"

        # Reshape children: [B, N, b, d]
        c = children.view(B, N, self.b, d)

        # Sinusoidal gates: [N, b] → [1, N, b, 1]
        gates = self.gate(positions)              # [N, b]
        gates = gates.unsqueeze(0).unsqueeze(-1)  # [1, N, b, 1]

        # Gate and flatten: [B, N, b*d]
        gated = (c * gates).view(B, N, self.b * d)

        # Gated linear unit
        v = self.W_v(gated)                       # [B, N, d]
        g = torch.sigmoid(self.W_g(gated))
        h = self.dropout(self.norm(v * g))        # [B, N, d]

        # Phase update
        delta_phase = self.phase_net(h).squeeze(-1)   # [B, N]
        if prev_phase is not None:
            phase = prev_phase + self.omega_level * 1.0 + delta_phase
        else:
            phase = self.omega_level * positions.float().unsqueeze(0) + delta_phase
        # Wrap to [-π, π]
        phase = torch.atan2(torch.sin(phase), torch.cos(phase))

        return h, phase


class SinusoidalBroadcast(nn.Module):
    """
    Top-down: broadcasts 1 parent representation back to b children.
    Adds context from the level above to each child via a residual.
    """

    def __init__(self, d_model: int, branching: int, rank: int = 4,
                 damping: bool = True, dropout: float = 0.1):
        super().__init__()
        self.b = branching

        # Gate: inject parent into each child slot
        self.gate = SinusoidalGate(
            out_channels=branching, rank=rank, damping=damping
        )
        self.W_ctx = nn.Linear(d_model, d_model * branching)
        self.W_mix = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        children: torch.Tensor,   # [B, N*b, d]  child states (to update)
        parents: torch.Tensor,     # [B, N, d]    parent context
        positions: torch.Tensor,   # [N] parent positions
    ) -> torch.Tensor:
        """Returns updated children [B, N*b, d]."""
        B, N, d = parents.shape

        # Broadcast parent context to b slots: [B, N, b, d]
        ctx = self.W_ctx(parents).view(B, N, self.b, d)

        # Sinusoidal gates [N, b] → [1, N, b, 1]
        gates = self.gate(positions).unsqueeze(0).unsqueeze(-1)
        ctx = ctx * gates                              # [B, N, b, d]

        # Flatten context to match children layout
        ctx_flat = ctx.reshape(B, N * self.b, d)       # [B, N*b, d]

        # Mix with existing children
        mixed = self.W_mix(torch.cat([children, ctx_flat], dim=-1))
        return self.dropout(self.norm(mixed))


class InterMotifCoupler(nn.Module):
    """
    Cross-frequency coupling between two fractal motifs at the same level.

    Implements phase-amplitude coupling analogous to hippocampal
    theta-gamma cross-frequency interaction.
    """

    def __init__(self, d_model: int, rank: int = 4,
                 damping: bool = True, dropout: float = 0.1):
        super().__init__()
        self.gate_ab = SinusoidalGate(d_model, rank, damping)
        self.gate_ba = SinusoidalGate(d_model, rank, damping)
        self.W_ab = nn.Linear(d_model, d_model)
        self.W_ba = nn.Linear(d_model, d_model)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        ha: torch.Tensor,   # [B, N, d]  motif A
        hb: torch.Tensor,   # [B, N, d]  motif B
        positions: torch.Tensor,  # [N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns updated (ha, hb) with cross-motif information."""
        # Gates: [N, d]
        g_ab = self.gate_ab(positions).unsqueeze(0)  # [1, N, d]
        g_ba = self.gate_ba(positions).unsqueeze(0)

        msg_ab = g_ab * self.W_ab(hb)   # B → A message
        msg_ba = g_ba * self.W_ba(ha)   # A → B message

        ha_new = self.dropout(self.norm_a(ha + msg_ab))
        hb_new = self.dropout(self.norm_b(hb + msg_ba))
        return ha_new, hb_new
