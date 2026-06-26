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
from typing import List, Optional, Tuple

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


# ─────────────────────────────────────────────────────────────────────────────
# Phase Goal Forcing (Forçage de Phase Causal)
# ─────────────────────────────────────────────────────────────────────────────

class PhaseGoalForcing(nn.Module):
    """
    Forçage de phase par but (goal-directed Kuramoto forcing).

    Equation maitresse LEAC:
        dθ/dt = ω + λ·sin(θ* − θ) + K·sin(θ̄ − θ)

    Le but est encode comme un attracteur de phase θ*.
    Le forçage pousse la dynamique vers la cible:
        θ_forced = θ + λ_goal · sin(θ* − θ)
    """

    def __init__(
        self,
        d_model: int,
        n_phases: int,
        init_lambda: float = 0.2,
        n_steps: int = 3,
    ):
        super().__init__()
        self.n_phases = n_phases
        self.n_steps = n_steps

        from .hopfield import mandelbrot_frequencies
        freqs = mandelbrot_frequencies(n_phases)
        base = torch.tensor(freqs[:n_phases], dtype=torch.float32) if not isinstance(freqs, torch.Tensor) else freqs[:n_phases].clone().float()
        self.register_buffer("base_angles", base)

        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_phases),
            nn.Tanh(),
        )
        nn.init.normal_(self.encoder[-2].weight, std=0.01)
        nn.init.zeros_(self.encoder[-2].bias)

        self.log_lambda = nn.Parameter(torch.tensor(math.log(init_lambda)))

        self.goal_bias_proj = nn.Linear(n_phases, d_model, bias=False)
        nn.init.normal_(self.goal_bias_proj.weight, std=0.01)

        self._goal_phase: Optional[torch.Tensor] = None

    @property
    def lambda_goal(self) -> float:
        return self.log_lambda.exp()

    def set_goal(self, h_prompt: torch.Tensor):
        """Encode le prompt comme attracteur de phase."""
        summary = h_prompt.mean(1)
        self._goal_phase = (self.base_angles.unsqueeze(0) + self.encoder(summary) * math.pi).detach()

    def reset_goal(self):
        self._goal_phase = None

    def forward(
        self,
        theta: torch.Tensor,
        h_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applique le forçage de phase vers le but.
        theta: [B, L, n_phases] or [B, n_phases]
        Returns: (theta_forced, alignment_score)
        """
        if self._goal_phase is None:
            return theta, torch.ones(theta.shape[:-1], device=theta.device)

        goal = self._goal_phase
        if goal.dim() < theta.dim():
            goal = goal.unsqueeze(1)
        lam = self.lambda_goal

        for _ in range(self.n_steps):
            forcing = lam * torch.sin(goal - theta)
            theta = theta + forcing

        alignment = torch.cos(theta - goal).mean(-1)
        return theta, alignment

    def loss_goal(self, theta: torch.Tensor) -> torch.Tensor:
        """Perte d'alignement de phase vers le but."""
        if self._goal_phase is None:
            return torch.tensor(0.0, device=theta.device)
        goal = self._goal_phase
        if goal.dim() < theta.dim():
            goal = goal.unsqueeze(1)
        alignment = torch.cos(theta - goal).mean(-1)
        return -alignment.mean()

    def goal_logit_bias(self, goal_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Biais dans l'espace des logits conditionne par le but.
        goal_phase: [B, n_phases] (utilise le but courant si None).
        Returns [B, d_model] a ajouter a l'entree du LM head.
        """
        gp = goal_phase if goal_phase is not None else self._goal_phase
        if gp is None:
            return None
        return self.goal_bias_proj(gp) * 0.05

    def reward_goal_achievement(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Recompense d'atteinte du but ∈ [0, 1] (haute quand les phases
        sont alignees avec θ*). Utilisable comme recompense intrinseque RL.
        theta: [B, L, n_phases] ou [B, n_phases].
        """
        if self._goal_phase is None:
            return torch.zeros(theta.shape[0], device=theta.device)
        goal = self._goal_phase
        if goal.dim() < theta.dim():
            goal = goal.unsqueeze(1)
        alignment = torch.cos(theta - goal).mean(-1)  # [B, L] ou [B]
        if alignment.dim() > 1:
            alignment = alignment.mean(-1)
        return alignment.clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Decomposition Hierarchique de Buts (sub-goals en arbre binaire)
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalGoalDecomposer(nn.Module):
    """
    Decompose un but de haut niveau en une hierarchie de sous-buts.

    Arbre binaire a n_levels: un but → 2^level sous-buts fins.
    Le modele apprend quelle branche poursuivre a chaque etape via les
    scores d'alignement.

    Usage:
        decomposer.set_goal(goal_phase)             # [B, n_phases]
        subgoal = decomposer.get_subgoal(alignment)  # [B, n_phases]
        theta = attractor(theta, subgoal)
    """

    def __init__(self, n_phases: int, n_levels: int = 3):
        super().__init__()
        self.n_phases = n_phases
        self.n_levels = n_levels
        self.n_leaves = 2 ** n_levels

        self.splitter = nn.Sequential(
            nn.Linear(n_phases, n_phases * 2),
            nn.SiLU(),
            nn.Linear(n_phases * 2, n_phases * 2),
        )
        nn.init.zeros_(self.splitter[-1].weight)
        nn.init.zeros_(self.splitter[-1].bias)

        self._goal_tree: Optional[List[torch.Tensor]] = None
        self._current_leaf: int = 0

    def _build_tree(self, goal: torch.Tensor) -> List[torch.Tensor]:
        tree = [goal]
        for _ in range(self.n_levels):
            new_tree = []
            for node in tree:
                split = self.splitter(node)
                left = node + split[:, :self.n_phases] * 0.3
                right = node + split[:, self.n_phases:] * 0.3
                new_tree.extend([left, right])
            tree = new_tree
        return tree

    def set_goal(self, goal_phase: torch.Tensor):
        """Construit l'arbre des sous-buts depuis goal_phase [B, n_phases]."""
        self._goal_tree = self._build_tree(goal_phase)
        self._current_leaf = 0

    def get_subgoal(
        self,
        alignment: Optional[torch.Tensor] = None,
        advance_threshold: float = 0.75,
    ) -> Optional[torch.Tensor]:
        """
        Retourne le sous-but courant [B, n_phases].
        Avance a la feuille suivante si l'alignement depasse le seuil.
        """
        if self._goal_tree is None:
            return None
        if (alignment is not None
                and alignment.mean().item() >= advance_threshold
                and self._current_leaf < self.n_leaves - 1):
            self._current_leaf += 1
        return self._goal_tree[self._current_leaf]

    def hierarchy_loss(self, theta: torch.Tensor) -> torch.Tensor:
        """Perte encourageant chaque feuille de la hierarchie a etre atteignable."""
        if self._goal_tree is None:
            return torch.tensor(0.0, device=theta.device)
        total = torch.tensor(0.0, device=theta.device)
        for leaf in self._goal_tree:
            leaf_d = leaf.to(theta.device)
            if leaf_d.dim() < theta.dim():
                leaf_d = leaf_d.unsqueeze(1)
            total = total + (1.0 - torch.cos(theta - leaf_d)).mean()
        return total / max(len(self._goal_tree), 1)

    def reset(self):
        self._goal_tree = None
        self._current_leaf = 0
