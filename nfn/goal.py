"""
NFN v4.0 — Phase Goal Predictor (Agency Module)

Transforms NFN from a passive next-token predictor into a goal-directed
generator by maintaining a target phase attractor and biasing the Kuramoto
dynamics toward it.

Theory
------
In Kuramoto dynamics, the system naturally settles into one of several phase
attractors. By specifying a target attractor θ* (the "goal"), we add an
external forcing term to the ODE:

    dθᵢ/dt = Ωᵢ + Σⱼ Kⱼᵢ sin(θⱼ − θᵢ) + λ_goal · sin(θ*ᵢ − θᵢ)
                                                  ────────────────────
                                                  goal attraction term

This is equivalent to adding a virtual "goal oscillator" with coupling λ_goal
to every real oscillator.

Semantics of the Goal Phase
  • θ* encodes the desired semantic state (e.g. "generate a question")
  • λ_goal controls how strongly generation is biased toward the goal
  • Setting λ_goal=0 recovers unconstrained generation
  • Setting λ_goal→∞ forces generation to match the goal exactly

Goal Encoding
  Goals can be specified as:
    1. Natural language prompt → GoalEncoder → θ*   (text goal)
    2. Task one-hot vector     → GoalEncoder → θ*   (classification goal)
    3. Direct phase vector     → θ*                  (expert specification)

Architecture
------------
  GoalEncoder      : text/task → θ* [n_phases]
  GoalAttractor    : adds sin(θ* − θ) forcing to any phase tensor
  PhaseGoalPredictor: full module — infers goal from context, applies forcing
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Goal Encoder  (maps goal specification → target phase vector)
# ─────────────────────────────────────────────────────────────────────────────

class GoalEncoder(nn.Module):
    """
    Maps a goal representation (hidden state summary) to a target phase θ*.

    θ* is parameterised as:
        θ*[k] = base_angle[k] + δ[k],  δ = tanh(MLP(h_goal)) · π
    where base_angle are the Mandelbrot fractal frequencies (stable priors).
    """

    def __init__(self, d_model: int, n_phases: int, n_mandelbrot: int = 32):
        super().__init__()
        self.n_phases = n_phases

        # Mandelbrot base angles (fixed prior — seed the goal space with natural frequencies)
        from .hopfield import mandelbrot_frequencies
        freqs = mandelbrot_frequencies(n_phases)
        base  = torch.tensor(freqs[:n_phases], dtype=torch.float32)
        self.register_buffer("base_angles", base)

        # Goal projection
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_phases),
            nn.Tanh(),
        )
        nn.init.normal_(self.proj[-2].weight, std=0.01)
        nn.init.zeros_(self.proj[-2].bias)

    def forward(self, h_goal: torch.Tensor) -> torch.Tensor:
        """
        h_goal: [B, d_model]  — goal context summary (e.g. mean of prompt)
        Returns θ*: [B, n_phases]  — target phase vector in (-π, π)
        """
        delta = self.proj(h_goal) * math.pi    # [B, n_phases] ∈ (-π, π)
        return self.base_angles.unsqueeze(0) + delta   # [B, n_phases]


# ─────────────────────────────────────────────────────────────────────────────
# Goal Attractor  (adds forcing term to Kuramoto dynamics)
# ─────────────────────────────────────────────────────────────────────────────

class GoalAttractor(nn.Module):
    """
    Applies goal-directed forcing to a phase tensor θ [B, L, n_phases]:

        θ_forced = θ + λ_goal · sin(θ* − θ)

    where θ* is broadcast from [B, n_phases] to [B, L, n_phases].

    This is a single Euler step of the forced Kuramoto ODE.
    Multiple steps can be applied for stronger goal attraction.
    """

    def __init__(self, n_phases: int, init_lambda: float = 0.2, n_steps: int = 3):
        super().__init__()
        self.n_phases = n_phases
        self.n_steps  = n_steps
        # λ_goal is learnable — the model learns how strongly to pursue goals
        self.log_lambda = nn.Parameter(torch.tensor(math.log(init_lambda)))

    @property
    def lambda_goal(self) -> float:
        return self.log_lambda.exp()

    def forward(
        self,
        theta:      torch.Tensor,           # [B, L, n_phases]
        goal_phase: torch.Tensor,           # [B, n_phases]
        mask:       Optional[torch.Tensor] = None,  # [B, L] bool — where to apply goal
    ) -> torch.Tensor:
        """
        Returns goal-forced phase tensor [B, L, n_phases].
        """
        goal = goal_phase.unsqueeze(1)    # [B, 1, n_phases]
        lam  = self.lambda_goal

        for _ in range(self.n_steps):
            forcing = lam * torch.sin(goal - theta)   # [B, L, n_phases]
            if mask is not None:
                forcing = forcing * mask.unsqueeze(-1).float()
            theta = theta + forcing

        return theta

    def goal_alignment(self, theta: torch.Tensor, goal_phase: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment score between current phases and goal.
        Returns [B, L] cosine similarity in phase space (1.0 = perfect alignment).
        """
        goal = goal_phase.unsqueeze(1)
        return torch.cos(theta - goal).mean(-1)    # [B, L]


# ─────────────────────────────────────────────────────────────────────────────
# PhaseGoalPredictor  (full agency module)
# ─────────────────────────────────────────────────────────────────────────────

class PhaseGoalPredictor(nn.Module):
    """
    Full goal-directed generation module.

    Workflow:
      1. Infer goal θ* from the prompt / instruction prefix
      2. During generation, apply goal attractor to each phase update
      3. Optionally apply goal-conditioned logit bias (GoalLogitBias)

    Integration with EfficientNFNBlock:
      In the block's forward(), after phase_lock:
        if goal_predictor is not None:
            theta = goal_predictor(theta, h_prompt)

    Goal tracking:
      The predictor exposes goal_alignment() to measure how well the current
      generation matches the goal — useful for beam search / mirostat.
    """

    def __init__(
        self,
        d_model:     int,
        n_phases:    int,
        init_lambda: float = 0.2,
        n_steps:     int   = 3,
    ):
        super().__init__()
        self.encoder  = GoalEncoder(d_model, n_phases)
        self.attractor = GoalAttractor(n_phases, init_lambda, n_steps)

        # Logit bias: goal-aligned tokens get a small bonus
        self.goal_bias_proj = nn.Linear(n_phases, d_model, bias=False)
        nn.init.normal_(self.goal_bias_proj.weight, std=0.01)

        # Internal goal state (set once per prompt)
        self._goal_phase: Optional[torch.Tensor] = None

    def set_goal(self, h_prompt: torch.Tensor):
        """
        h_prompt: [B, L_prompt, d_model]
        Encodes the prompt into a goal phase and stores it.
        Call once before generation begins.
        """
        summary = h_prompt.mean(1)                 # [B, d]
        self._goal_phase = self.encoder(summary)   # [B, n_phases]

    def forward(
        self,
        theta:      torch.Tensor,                   # [B, L, n_phases]
        h_context:  Optional[torch.Tensor] = None,  # [B, L_ctx, d_model]  for auto-goal
        mask:       Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          theta_forced  : [B, L, n_phases]  — goal-attracted phases
          alignment     : [B, L]            — goal alignment scores
        """
        if self._goal_phase is None and h_context is not None:
            self.set_goal(h_context)

        if self._goal_phase is None:
            return theta, torch.zeros(theta.shape[:2], device=theta.device)

        goal = self._goal_phase.to(theta.device)
        theta_forced = self.attractor(theta, goal, mask)
        alignment    = self.attractor.goal_alignment(theta_forced, goal)
        return theta_forced, alignment

    def goal_logit_bias(self, goal_phase: torch.Tensor) -> torch.Tensor:
        """
        goal_phase: [B, n_phases]
        Returns a logit-space bias [B, d_model] to add to the LM head input.
        Encourages generation of tokens aligned with the goal.
        """
        return self.goal_bias_proj(goal_phase) * 0.05   # small bias, stable

    def loss_goal(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary goal-alignment loss: encourage phases to match θ*.
        Only meaningful during fine-tuning with explicit goal supervision.
        """
        if self._goal_phase is None:
            return torch.tensor(0.0, device=theta.device)
        goal = self._goal_phase.to(theta.device).unsqueeze(1)
        return (1.0 - torch.cos(theta - goal)).mean()

    def reset_goal(self):
        """Call between generation episodes."""
        self._goal_phase = None
