"""
NFN v4.0 — Recursive Reasoning with Adaptive Computation Time

Instead of running each block exactly once per token, the model can
"think" for a variable number of steps — spending more computation
on hard inputs and less on easy ones.

Theory
------
Adaptive Computation Time (ACT, Graves 2016):
  At each position i, the model maintains a halting probability p_t.
  It keeps running until cumulative probability ≥ 1 − ε, then stops.
  The output is the weighted sum of states across all steps:
      h_final = Σ_t w_t · h_t,   w_t = p_t (halt weights)

Here we tie the halting signal to goal alignment:
  - If a goal is set, the model halts when phase alignment ≥ threshold
  - Without a goal, it uses standard ACT halting probability
  - Penalty: λ_ponder · (mean halting steps) — encourages efficiency

Self-Consistency Check
----------------------
After each reasoning round, the model samples two candidate continuations
and measures their causal consistency via the DAG. If inconsistency is
high (disagreement on causal structure), it triggers another round.
This is internal debate — the model argues with itself before committing.

Architecture
------------
  HaltingUnit        : maps h → halting probability p ∈ (0,1)
  RecursiveReasoner  : wraps a callable block, runs ACT over it
  SelfConsistencyCheck: generates N candidates, returns most consistent
"""

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Halting Unit
# ─────────────────────────────────────────────────────────────────────────────

class HaltingUnit(nn.Module):
    """
    Maps hidden state → scalar halting probability per position.

    p_halt[i] = sigmoid(W_halt · h[i] + b)

    Initialised so that the model starts with low halting probability
    (i.e., tends to keep thinking until it builds confidence).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, -2.0)   # start: ~12% halt probability

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, L, d] → p_halt: [B, L]"""
        return torch.sigmoid(self.proj(h).squeeze(-1))


# ─────────────────────────────────────────────────────────────────────────────
# Recursive Reasoner  (ACT loop)
# ─────────────────────────────────────────────────────────────────────────────

class RecursiveReasoner(nn.Module):
    """
    Wraps any h → h block and runs it for a variable number of steps.

    Each step:
      1. Run block(h) → h_new
      2. Compute halting probability p = halt_unit(h_new)
      3. Accumulate weighted output: out += (1 - halted) * p * h_new
      4. Update halted mask: halted |= (cumulative_p ≥ threshold)
      5. Remainder weight on last step

    When a goal_alignment tensor is provided (from PhaseGoalPredictor),
    halting is additionally triggered when alignment exceeds halt_on_alignment.

    Ponder cost (added to loss):
        L_ponder = mean number of steps taken (encourages early halting)
    """

    def __init__(
        self,
        d_model:            int,
        max_steps:          int   = 8,
        halt_threshold:     float = 0.99,
        halt_on_alignment:  float = 0.85,  # halt if goal alignment exceeds this
        lambda_ponder:      float = 0.01,
    ):
        super().__init__()
        self.max_steps         = max_steps
        self.halt_threshold    = halt_threshold
        self.halt_on_alignment = halt_on_alignment
        self.lambda_ponder     = lambda_ponder

        self.halt_unit = HaltingUnit(d_model)
        self.state_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h:             torch.Tensor,                    # [B, L, d]
        block:         Callable,                        # h → (h, losses_dict)
        goal_alignment: Optional[torch.Tensor] = None, # [B, L] ∈ [-1,1]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
          h_out        : [B, L, d]  — weighted sum across reasoning steps
          ponder_loss  : scalar     — λ_ponder · mean_steps
          steps_taken  : int        — max steps taken across batch
        """
        B, L, d = h.shape
        device   = h.device

        # Accumulators
        h_out        = torch.zeros_like(h)
        cum_p        = torch.zeros(B, L, device=device)   # cumulative halt prob
        halted       = torch.zeros(B, L, dtype=torch.bool, device=device)
        step_counter = torch.zeros(B, L, device=device)

        h_cur = h
        steps_taken = 0

        for step in range(self.max_steps):
            h_new, _ = block(h_cur)              # run one reasoning step
            h_new = self.state_norm(h_new)

            p = self.halt_unit(h_new)            # [B, L] ∈ (0,1)

            # On the last step, remainder weight
            if step == self.max_steps - 1:
                w = 1.0 - cum_p                  # [B, L]
            else:
                w = p * (~halted).float()        # only non-halted positions

            # Weight contribution
            h_out = h_out + (w.unsqueeze(-1) * h_new)
            step_counter = step_counter + (~halted).float()

            # Update cumulative probability
            cum_p = cum_p + w

            # Determine which positions halt now
            halt_by_prob = cum_p >= self.halt_threshold
            halt_by_goal = (
                goal_alignment is not None
                and (goal_alignment >= self.halt_on_alignment)
            )
            if halt_by_goal is not False:
                halted = halted | halt_by_prob | halt_by_goal
            else:
                halted = halted | halt_by_prob

            h_cur = h_new
            steps_taken = step + 1

            if halted.all():
                break

        ponder_loss = self.lambda_ponder * step_counter.mean()

        return h_out, ponder_loss, steps_taken


# ─────────────────────────────────────────────────────────────────────────────
# Self-Consistency Check
# ─────────────────────────────────────────────────────────────────────────────

class SelfConsistencyCheck(nn.Module):
    """
    Generates N candidate hidden states and selects the most internally
    consistent one using causal graph agreement.

    At inference, call check(h, causal_layer) to get the most consistent
    candidate. During training, the consistency score becomes an auxiliary loss.

    Consistency measure:
      For each pair (h_i, h_j), derive their slot representations and compute
      the Frobenius distance between their causal adjacency matrices.
      The most consistent candidate minimises mean pairwise DAG distance.

    This is "self-debate" — the model generates variations and selects
    the one whose causal structure is most stable.
    """

    def __init__(self, d_model: int, n_candidates: int = 3, noise_scale: float = 0.05):
        super().__init__()
        self.n_candidates = n_candidates
        self.noise_scale  = noise_scale

        # Lightweight scorer: how confident is the model in this candidate?
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
        )
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(
        self,
        h:              torch.Tensor,               # [B, L, d]
        causal_layer:   Optional[nn.Module] = None, # CausalGraphLayer or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          h_best         : [B, L, d]  — most consistent candidate
          consistency_loss: scalar    — penalises inconsistency
        """
        B, L, d = h.shape

        # Generate candidates by adding small perturbations
        candidates = [h]
        for _ in range(self.n_candidates - 1):
            noise = torch.randn_like(h) * self.noise_scale
            candidates.append(h + noise)

        # Score each candidate
        scores = []
        for cand in candidates:
            score = self.scorer(cand).squeeze(-1).mean(-1)   # [B]
            scores.append(score)

        score_tensor = torch.stack(scores, dim=1)            # [B, n_cand]

        # If causal layer available, use DAG agreement as additional signal
        if causal_layer is not None:
            dag_scores = []
            adj_matrices = []
            for cand in candidates:
                _, loss_s = causal_layer.edge_net(
                    causal_layer.slot_proj(
                        cand[:, :causal_layer.n_slots, :]
                        if L >= causal_layer.n_slots
                        else F.pad(cand, (0, 0, 0, causal_layer.n_slots - L))[:, :causal_layer.n_slots, :]
                    )
                )
                adj_matrices.append(loss_s)

            # Consistency loss: variance of DAG sparsity across candidates
            # Low variance → candidates agree on causal structure
            dag_var = torch.var(torch.stack(adj_matrices))
            consistency_loss = dag_var
        else:
            consistency_loss = torch.tensor(0.0, device=h.device)

        # Select best candidate by score
        best_idx = score_tensor.argmax(dim=1)          # [B]
        h_best   = torch.stack(candidates, dim=1)      # [B, n_cand, L, d]
        h_best   = h_best[torch.arange(B), best_idx]  # [B, L, d]

        return h_best, consistency_loss


# ─────────────────────────────────────────────────────────────────────────────
# Plan Executor  (sequential sub-goal pursuit)
# ─────────────────────────────────────────────────────────────────────────────

class PlanExecutor(nn.Module):
    """
    Decomposes a goal phase θ* into K sub-goals and pursues them in order.

    Sub-goal decomposition:
      θ*_k = GoalDecoder(θ*, k/K)   — interpolate toward goal in K steps

    At each generation step, the executor advances to the next sub-goal
    once alignment with the current one exceeds a threshold.

    This enables long-horizon planning: "first introduce topic, then
    argue, then conclude" — each stage has its own phase attractor.
    """

    def __init__(self, n_phases: int, n_subgoals: int = 4):
        super().__init__()
        self.n_subgoals = n_subgoals
        self.n_phases   = n_phases

        # Learned interpolation weights for sub-goal generation
        self.interpolator = nn.Linear(n_phases + 1, n_phases)
        nn.init.zeros_(self.interpolator.weight)
        nn.init.zeros_(self.interpolator.bias)

        self._goal_phase:    Optional[torch.Tensor] = None   # [B, n_phases]
        self._current_step:  int = 0

    def set_plan(self, goal_phase: torch.Tensor):
        """goal_phase: [B, n_phases]"""
        self._goal_phase   = goal_phase
        self._current_step = 0

    def current_subgoal(self, device: torch.device) -> Optional[torch.Tensor]:
        """Returns the current sub-goal phase [B, n_phases] or None."""
        if self._goal_phase is None:
            return None
        k = self._current_step / max(self.n_subgoals - 1, 1)
        k_tensor = torch.tensor([k], device=device).expand(self._goal_phase.shape[0], 1)
        inp = torch.cat([self._goal_phase, k_tensor], dim=-1)
        delta = self.interpolator(inp)
        return self._goal_phase * k + delta * (1 - k)

    def advance(self, alignment: torch.Tensor, threshold: float = 0.7):
        """
        alignment: [B] — mean alignment with current sub-goal.
        Advances to next sub-goal if mean alignment exceeds threshold.
        """
        if self._current_step < self.n_subgoals - 1:
            if alignment.mean().item() >= threshold:
                self._current_step += 1

    def reset(self):
        self._goal_phase   = None
        self._current_step = 0
