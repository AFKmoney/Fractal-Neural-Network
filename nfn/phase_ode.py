"""
True Kuramoto ODE phase dynamics — the core AGI innovation of NFN.

The Kuramoto model on a fractal graph:
    dθᵢ/dt = Ωᵢ + Σⱼ∈N(i) Kⱼᵢ · sin(θⱼ - θᵢ + φⱼᵢ)

Extensions:
  - Hierarchical coupling: nodes at level k couple with their
    parent (level k+1) and children (level k-1)
  - Cross-motif coupling: phase-amplitude coupling between motifs
  - Amortised sinusoidal weights: Kⱼᵢ(t) = Aⱼᵢ · exp(-γt)

This is fully differentiable — gradients flow through the ODE steps
back to the coupling strengths K, phases φ, and natural frequencies Ω.

Solver: adaptive RK4 with S fixed steps (S=4 in practice, good quality/cost).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# RK4 ODE integrator (fully differentiable)
# ─────────────────────────────────────────────────────────────────────────────

def rk4_step(f, y: torch.Tensor, t: float, dt: float) -> torch.Tensor:
    """Single 4th-order Runge-Kutta step. f(t, y) → dy/dt."""
    k1 = f(t,            y)
    k2 = f(t + dt / 2,   y + dt * k1 / 2)
    k3 = f(t + dt / 2,   y + dt * k2 / 2)
    k4 = f(t + dt,       y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_ode(f, y0: torch.Tensor, T: float = 1.0, n_steps: int = 4) -> torch.Tensor:
    """Integrate dy/dt = f(t,y) from 0 to T with n_steps RK4 steps."""
    dt = T / n_steps
    y = y0
    for i in range(n_steps):
        y = rk4_step(f, y, i * dt, dt)
    # Wrap phases to [-π, π]
    return torch.atan2(torch.sin(y), torch.cos(y))


# ─────────────────────────────────────────────────────────────────────────────
# Local Kuramoto coupling: N fully-connected oscillators
# ─────────────────────────────────────────────────────────────────────────────

class KuramotoCoupling(nn.Module):
    """
    Learnable Kuramoto coupling for N oscillators at one fractal level.

    dθᵢ/dt = Ωᵢ + (1/N) Σⱼ Kⱼᵢ · sin(θⱼ - θᵢ + φⱼᵢ)

    Parameters
    ----------
    n_max  : maximum number of oscillators (pads/truncates dynamically)
    rank   : low-rank factorisation of K matrix (K = U Vᵀ)
    """

    def __init__(self, n_max: int, rank: int = 8, omega_init: float = 1.0):
        super().__init__()
        self.n_max = n_max
        self.rank = rank

        # Natural frequencies Ωᵢ — one per oscillator position
        self.omega = nn.Parameter(torch.ones(n_max) * omega_init)
        # Low-rank coupling strength: K_{ji} = U_j · V_i (dot product)
        self.U = nn.Parameter(torch.randn(n_max, rank) * 0.1)
        self.V = nn.Parameter(torch.randn(n_max, rank) * 0.1)
        # Phase offsets φⱼᵢ (low-rank)
        self.phi_U = nn.Parameter(torch.zeros(n_max, rank))
        self.phi_V = nn.Parameter(torch.zeros(n_max, rank))

    def coupling_matrix(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return K [N, N] and φ [N, N] for the first N oscillators."""
        U = self.U[:N]
        V = self.V[:N]
        K = torch.sigmoid(U @ V.t())          # [N, N], in (0,1)
        phi = (self.phi_U[:N] @ self.phi_V[:N].t()) * math.pi  # [N, N]
        return K, phi

    def dtheta_dt(self, t: float, theta: torch.Tensor) -> torch.Tensor:
        """
        theta : [B, N]
        returns dθ/dt : [B, N]
        """
        B, N = theta.shape
        N_eff = min(N, self.n_max)
        theta = theta[:, :N_eff]

        omega = self.omega[:N_eff]           # [N_eff]
        K, phi = self.coupling_matrix(N_eff) # [N_eff, N_eff]

        # diff[b, i, j] = θⱼ - θᵢ + φⱼᵢ
        # [B, N, N] = [B, N, 1] broadcast
        diff = theta.unsqueeze(2) - theta.unsqueeze(1)  # θⱼ - θᵢ = θ[:,j] - θ[:,i]
        # Actually we want θⱼ - θᵢ:
        # diff[b,i,j] = theta[b,j] - theta[b,i]
        diff = theta.unsqueeze(1) - theta.unsqueeze(2)   # [B, N, N]  diff[b,i,j] = θⱼ-θᵢ

        # Add phase offset
        diff = diff + phi.unsqueeze(0)       # [B, N, N]

        # Coupling term: (1/N) Σⱼ K[i,j] · sin(diff[b,i,j])
        coupling = (K.unsqueeze(0) * torch.sin(diff)).mean(dim=2)  # [B, N]

        return omega.unsqueeze(0) + coupling   # [B, N]

    def forward(
        self,
        theta: torch.Tensor,   # [B, N]
        n_steps: int = 4,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Integrate phases over n_steps RK4 steps, return final phases [B, N]."""
        N_eff = min(theta.shape[1], self.n_max)
        theta_eff = theta[:, :N_eff]
        result = integrate_ode(
            lambda t, y: self.dtheta_dt(t, y),
            theta_eff, T=dt, n_steps=n_steps
        )
        if theta.shape[1] > N_eff:
            # Pad back if input was larger than n_max
            result = torch.cat([result, theta[:, N_eff:]], dim=1)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical Kuramoto: parent-child coupling across fractal levels
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalKuramoto(nn.Module):
    """
    Cross-level phase coupling in the fractal hierarchy.

    Each child θᵢ at level k is driven by its parent θ_p at level k+1:
        dθᵢ/dt += K_up · sin(θ_p(i) - θᵢ + φ_up)

    Each parent is driven by the mean of its children:
        dθ_p/dt += K_down · sin(mean_children(θᵢ) - θ_p + φ_down)
    """

    def __init__(self, branching: int, n_steps: int = 2):
        super().__init__()
        self.b = branching
        self.n_steps = n_steps

        self.K_up = nn.Parameter(torch.tensor(0.3))
        self.K_down = nn.Parameter(torch.tensor(0.3))
        self.phi_up = nn.Parameter(torch.zeros(1))
        self.phi_down = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        child_phases: torch.Tensor,    # [B, N*b]
        parent_phases: torch.Tensor,   # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Nb = child_phases.shape
        N = parent_phases.shape[1]
        b = self.b

        # Reshape children: [B, N, b]
        ch = child_phases.view(B, N, b)

        K_up = torch.sigmoid(self.K_up)
        K_down = torch.sigmoid(self.K_down)

        def deriv(t, state):
            ch_s, par_s = state[..., :Nb].view(B, N, b), state[..., Nb:]
            # Child ← parent
            d_child = K_up * torch.sin(par_s.unsqueeze(-1) - ch_s + self.phi_up)  # [B,N,b]
            # Parent ← mean children
            mean_ch = ch_s.mean(dim=-1)   # [B, N]
            d_parent = K_down * torch.sin(mean_ch - par_s + self.phi_down)
            return torch.cat([d_child.view(B, Nb), d_parent], dim=-1)

        state0 = torch.cat([child_phases, parent_phases], dim=-1)
        state = integrate_ode(deriv, state0, T=1.0, n_steps=self.n_steps)

        ch_new = torch.atan2(
            torch.sin(state[..., :Nb]), torch.cos(state[..., :Nb])
        )
        par_new = torch.atan2(
            torch.sin(state[..., Nb:]), torch.cos(state[..., Nb:])
        )
        return ch_new, par_new


# ─────────────────────────────────────────────────────────────────────────────
# Kuramoto layer: drop-in replacement for the simple phase MLP
# ─────────────────────────────────────────────────────────────────────────────

class KuramotoPhaseLayer(nn.Module):
    """
    Replaces the simple 'phase = MLP(h)' with:
      1. Project h → raw_phase
      2. Integrate Kuramoto ODE for n_ode_steps
      3. Modulate h by phase: h' = h * (1 + α·cos(θ))

    This makes the hidden state phase-aware: representations that
    are in-phase are amplified; out-of-phase are suppressed.
    """

    def __init__(
        self,
        d_model: int,
        n_max_nodes: int = 512,
        rank: int = 8,
        n_ode_steps: int = 4,
        omega_init: float = 1.0,
        phase_mod_strength: float = 0.1,
    ):
        super().__init__()
        self.n_ode_steps = n_ode_steps
        self.alpha = phase_mod_strength

        # Project hidden state → initial phase estimate
        self.h_to_phase = nn.Linear(d_model, 1)
        # Kuramoto ODE
        self.kuramoto = KuramotoCoupling(n_max_nodes, rank, omega_init)
        # Project phase → modulation of hidden state
        self.phase_to_mod = nn.Linear(2, d_model)   # [cos θ, sin θ] → d

    def forward(
        self,
        h: torch.Tensor,              # [B, N, d]
        prev_phase: Optional[torch.Tensor] = None,  # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, d = h.shape

        # Initial phase: from hidden state + previous
        raw = self.h_to_phase(h).squeeze(-1)   # [B, N]
        if prev_phase is not None:
            theta0 = prev_phase + raw
        else:
            theta0 = raw

        # Integrate Kuramoto ODE
        theta = self.kuramoto(theta0, n_steps=self.n_ode_steps)  # [B, N]

        # Phase modulation: enrich hidden state with phase info
        phase_feat = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # [B,N,2]
        mod = self.alpha * torch.tanh(self.phase_to_mod(phase_feat))            # [B,N,d]
        h_modulated = h + mod

        return h_modulated, theta
