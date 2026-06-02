"""
NFN AGI v5.0 — Intrinsic Motivation

Curiosity-driven learning goes beyond entropy weighting: the model receives
an intrinsic reward signal for exploring novel states and making learning progress.

Three complementary intrinsic motivation signals:

  1. Forward Model Curiosity (Pathak et al. 2017):
     Predict next hidden state from current state + action.
     Intrinsic reward = prediction error (be curious about unpredictable states).

  2. State Novelty (approximate):
     Track visited state embeddings via a random projection hash table.
     Reward novelty = 1 / (visit_count + 1).

  3. Learning Progress (Oudeyer et al. 2007):
     Track how fast the forward model error decreases on each state cluster.
     High learning progress → high intrinsic reward (competence progress).

These signals are combined into a scalar intrinsic reward r_int that is
added to the AGI training objective.

Architecture
------------
  ForwardDynamicsModel    : (h_t, z_action) → predicted h_{t+1}
  StateNoveltyTracker     : h → novelty_score ∈ [0, 1]
  LearningProgressTracker : tracks prediction error history per cluster
  IntrinsicMotivation     : combines all signals into r_int
"""

from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Forward Dynamics Model
# ─────────────────────────────────────────────────────────────────────────────

class ForwardDynamicsModel(nn.Module):
    """
    Predicts the next hidden state h_{t+1} from the current state h_t.

    This is the classic intrinsic motivation signal: the model is
    "curious" about states where its forward model makes large errors —
    these are novel, not-yet-understood states.

    Architecture:
      Inverse model: (h_t, h_{t+1}) → z_action  (infer what action led here)
      Forward model: (h_t, z_action) → h_{t+1}_predicted

    The inverse model learns a compact action representation without
    requiring explicit action labels.
    """

    def __init__(self, d_model: int, z_action_dim: int = 64):
        super().__init__()
        self.z_action_dim = z_action_dim

        # Inverse model: encode (h_t, h_{t+1}) → z_action
        self.inverse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, z_action_dim),
        )
        nn.init.zeros_(self.inverse[-1].weight)
        nn.init.zeros_(self.inverse[-1].bias)

        # Forward model: (h_t, z_action) → h_{t+1}
        self.forward_net = nn.Sequential(
            nn.Linear(d_model + z_action_dim, d_model * 2),
            nn.SiLU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
        )
        nn.init.zeros_(self.forward_net[-1].weight)
        nn.init.zeros_(self.forward_net[-1].bias)

        # State encoder: compress h to avoid overfitting to surface features
        self.state_enc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        h: torch.Tensor,   # [B, L, d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute forward model prediction error as intrinsic reward.

        Uses consecutive token positions as (state, next_state) pairs.

        Returns:
          curiosity_loss  : scalar  — forward model MSE (training loss)
          curiosity_reward: [B, L]  — per-token intrinsic reward (detached)
        """
        if h.shape[1] < 2:
            z = torch.zeros(1, device=h.device)
            return z, torch.zeros(h.shape[:2], device=h.device)

        # Encode states
        h_enc = self.state_enc(h)                  # [B, L, d]
        h_t   = h_enc[:, :-1, :]                   # [B, L-1, d]
        h_t1  = h_enc[:, 1:,  :]                   # [B, L-1, d]

        # Inverse: infer action
        pair     = torch.cat([h_t, h_t1], dim=-1)  # [B, L-1, 2d]
        z_action = self.inverse(pair)               # [B, L-1, z_action_dim]

        # Forward: predict next state
        inp     = torch.cat([h_t, z_action], dim=-1)   # [B, L-1, d+z]
        h_pred  = self.forward_net(inp)                 # [B, L-1, d]

        # Curiosity = squared prediction error
        pred_error = (h_pred - h_t1.detach()) ** 2     # [B, L-1, d]
        curiosity  = pred_error.mean(-1)               # [B, L-1]

        curiosity_loss = curiosity.mean()

        # Pad back to full length
        pad = torch.zeros(h.shape[0], 1, device=h.device)
        curiosity_reward = torch.cat([curiosity.detach(), pad], dim=1)  # [B, L]

        return curiosity_loss, curiosity_reward


# ─────────────────────────────────────────────────────────────────────────────
# State Novelty Tracker
# ─────────────────────────────────────────────────────────────────────────────

class StateNoveltyTracker(nn.Module):
    """
    Approximate state novelty via random projection hashing.

    Maps hidden states to discrete buckets using random Gaussian projections.
    Novelty = 1 / (visit_count + 1).

    This is an approximation of count-based exploration bonuses (Bellemare 2016)
    that requires no external memory and works online.

    The projection is fixed (not trained) — it encodes which "area" of state
    space the model is currently exploring.
    """

    def __init__(
        self,
        d_model:    int,
        hash_dim:   int = 32,      # projection dimension
        n_buckets:  int = 1024,    # number of hash buckets
        decay:      float = 0.99,  # exponential decay of visit counts
    ):
        super().__init__()
        self.n_buckets = n_buckets
        self.decay     = decay

        # Fixed random projection (not trained)
        proj = torch.randn(d_model, hash_dim) / (d_model ** 0.5)
        self.register_buffer("proj", proj)

        # Visit counts (not a parameter — no grad)
        self.register_buffer("counts", torch.ones(n_buckets))

    def _hash(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, L, d] → bucket indices [B, L] ∈ {0, ..., n_buckets-1}
        Uses binary random projection as a locality-sensitive hash.
        """
        projected = h @ self.proj                          # [B, L, hash_dim]
        bits      = (projected > 0).long()                # [B, L, hash_dim]
        # Convert binary vector to integer index
        powers = 2 ** torch.arange(self.proj.shape[1], device=h.device)
        indices = (bits * powers).sum(-1) % self.n_buckets  # [B, L]
        return indices

    @torch.no_grad()
    def novelty(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, L, d]
        Returns novelty score [B, L] ∈ (0, 1].
        Updates visit counts.
        """
        indices = self._hash(h)                            # [B, L]
        flat    = indices.reshape(-1)                      # [B*L]

        # Fetch counts before update
        visit_counts = self.counts[flat].float().reshape(h.shape[:2])  # [B, L]

        # Update counts
        for idx in flat:
            self.counts[idx] += 1.0
        self.counts *= self.decay                          # exponential decay

        novelty = 1.0 / (visit_counts + 1.0)              # [B, L]
        return novelty


# ─────────────────────────────────────────────────────────────────────────────
# Learning Progress Tracker
# ─────────────────────────────────────────────────────────────────────────────

class LearningProgressTracker:
    """
    Tracks prediction error history to compute learning progress.

    Learning progress for cluster c:
        LP(c) = error_{c, t-1} - error_{c, t}   (positive = improving)

    High LP → model is actively learning about this cluster → high reward.
    Low LP → model already knows this area or can't learn it → low reward.

    Uses exponential moving averages to track error over time.
    Clusters are defined by k-means on random projections (online, approximate).
    """

    def __init__(self, n_clusters: int = 32, d_model: int = 256, decay: float = 0.9):
        self.n_clusters = n_clusters
        self.decay      = decay
        self._prev_errors: Dict[int, float] = defaultdict(lambda: 1.0)
        self._curr_errors: Dict[int, float] = defaultdict(lambda: 1.0)
        self._step = 0

        # Random projection for cluster assignment (not trained)
        self._proj = torch.randn(d_model, 8) / (d_model ** 0.5)

    def _cluster(self, h: torch.Tensor) -> int:
        """Assign hidden state summary to a cluster via random projection."""
        with torch.no_grad():
            h_cpu  = h.mean(0).cpu()                    # [d]
            proj   = h_cpu @ self._proj.cpu()           # [8]
            bits   = (proj > 0).long()
            powers = 2 ** torch.arange(8)
            idx    = int((bits * powers).sum().item() % self.n_clusters)
        return idx

    def update(self, h: torch.Tensor, error: float):
        """Record current prediction error for this state cluster."""
        c = self._cluster(h)
        # EMA of current error
        self._curr_errors[c] = (
            self.decay * self._curr_errors[c] + (1.0 - self.decay) * error
        )

    def learning_progress(self, h: torch.Tensor) -> float:
        """
        Returns learning progress ∈ [0, ∞) for the current state.
        High value → model is improving quickly on this type of state.
        """
        c  = self._cluster(h)
        lp = max(0.0, self._prev_errors[c] - self._curr_errors[c])
        return lp

    def step_epoch(self):
        """Called at the end of each epoch to shift error windows."""
        self._prev_errors = dict(self._curr_errors)
        self._step       += 1


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsic Motivation  (full module)
# ─────────────────────────────────────────────────────────────────────────────

class IntrinsicMotivation(nn.Module):
    """
    Combines forward-model curiosity, state novelty, and learning progress
    into a unified intrinsic reward signal.

    r_int = w_curiosity · r_curiosity
           + w_novelty  · r_novelty
           + w_progress · r_progress

    This reward is:
      1. Added to the training objective as an auxiliary signal
      2. Used to weight token-level loss (high r_int → upweight)
      3. Logged for monitoring what the model is "curious" about

    The signal is normalised by a running mean/std to keep it in a
    consistent range across training.
    """

    def __init__(
        self,
        d_model:       int,
        n_clusters:    int   = 32,
        hash_dim:      int   = 32,
        n_buckets:     int   = 1024,
        w_curiosity:   float = 0.5,
        w_novelty:     float = 0.3,
        w_progress:    float = 0.2,
        decay:         float = 0.99,
    ):
        super().__init__()
        self.w_curiosity = w_curiosity
        self.w_novelty   = w_novelty
        self.w_progress  = w_progress

        self.forward_dynamics = ForwardDynamicsModel(d_model)
        self.novelty_tracker  = StateNoveltyTracker(d_model, hash_dim, n_buckets, decay)
        self.lp_tracker       = LearningProgressTracker(n_clusters, d_model, decay)

        # Running normalisation stats (not parameters)
        self.register_buffer("_reward_mean", torch.tensor(0.0))
        self.register_buffer("_reward_std",  torch.tensor(1.0))
        self._update_count = 0

    def _normalise(self, r: torch.Tensor) -> torch.Tensor:
        """Normalise rewards using running stats (Welford online)."""
        with torch.no_grad():
            self._update_count += 1
            alpha = 1.0 / self._update_count
            self._reward_mean = (1 - alpha) * self._reward_mean + alpha * r.mean()
            self._reward_std  = (1 - alpha) * self._reward_std  + alpha * r.std().clamp(min=1e-6)
        return (r - self._reward_mean) / (self._reward_std + 1e-8)

    def forward(
        self,
        h: torch.Tensor,   # [B, L, d]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Compute intrinsic motivation signals.

        Returns:
          curiosity_loss  : scalar  — forward model loss (train with main loss)
          intrinsic_reward: [B, L]  — per-token reward for weighting
          metrics         : dict    — breakdown for logging
        """
        # 1. Forward dynamics curiosity
        curiosity_loss, curiosity_reward = self.forward_dynamics(h)

        # 2. State novelty
        novelty = self.novelty_tracker.novelty(h)              # [B, L]

        # 3. Learning progress (scalar, same for whole batch)
        h_summary = h.detach().mean(1).mean(0, keepdim=True)  # [1, d]
        lp = self.lp_tracker.learning_progress(h_summary)
        lp_reward = torch.full(h.shape[:2], lp, device=h.device)

        # Update tracker with current error
        self.lp_tracker.update(h_summary, curiosity_loss.item())

        # Combine
        r_combined = (
            self.w_curiosity * curiosity_reward
            + self.w_novelty  * novelty
            + self.w_progress * lp_reward
        )

        r_normalised = self._normalise(r_combined).clamp(0.0, 3.0)

        return curiosity_loss, r_normalised, {
            "curiosity_reward": curiosity_reward.mean().item(),
            "novelty":          novelty.mean().item(),
            "learning_progress": lp,
            "intrinsic_total":   r_normalised.mean().item(),
        }

    def intrinsic_weighted_loss(
        self,
        logits:   torch.Tensor,   # [B, L, V]
        targets:  torch.Tensor,   # [B, L]
        h:        torch.Tensor,   # [B, L, d]
        pad_id:   int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Intrinsic-reward-weighted cross-entropy.

        Tokens with high intrinsic reward (novel, surprising, high-learning-progress)
        get upweighted — the model is trained harder on what it doesn't yet know.

        Returns (curiosity_loss, weighted_lm_loss, metrics).
        """
        curiosity_loss, r_int, metrics = self.forward(h)

        B, L, vocab = logits.shape
        per_tok = F.cross_entropy(
            logits.reshape(B * L, vocab),
            targets.reshape(B * L),
            ignore_index=pad_id,
            reduction="none",
        ).reshape(B, L)

        valid        = (targets != pad_id).float()
        weights      = (1.0 + r_int) * valid
        denom        = weights.sum().clamp(min=1.0)
        weighted_lm  = (weights * per_tok).sum() / denom

        return curiosity_loss, weighted_lm, metrics
