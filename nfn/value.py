"""
NFN AGI v5.0 — Value Function & Reward Learning

Adds RL-style value estimation and preference-based reward learning.

Theory
------
Standard LMs maximise P(token|context) — predicting the training distribution.
True AGI requires goal-directed behaviour maximising expected future reward:

  1. Value function V(s): expected return from state s
     → Enables advantage estimation for credit assignment over long horizons

  2. Reward model R(seq): learned from self-play preference pairs (Bradley-Terry)
     → Goes beyond raw likelihood — captures quality, coherence, goal satisfaction

  3. Advantage-Weighted Regression (AWR):
     Weight each token by advantage A(s_t) = R + γV(s_{t+1}) - V(s_t)
         L_AWR = -Σ_t exp(A(s_t) / β) · log P(a_t | s_t)
     Upweights high-value trajectories; downweights low-value ones.

  4. Goal-Conditioned Value: V(s, g) — how valuable is state s for goal g?
     Enables goal-directed planning without explicit search.

Architecture
------------
  ValueHead        : h [B, L, d] → V [B, L]      (per-token value)
  RewardModel      : seq_h [B, d] → reward scalar  (sequence-level quality)
  GoalValueHead    : (h, goal_phase) → V(s,g) [B, L]
  AdvantageEstimator: computes bootstrapped advantage A(s_t)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Value Head
# ─────────────────────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """
    Per-token value function V(s_t) estimating expected future return.

    Takes the final hidden state and produces a scalar value per token.
    Used to compute advantages for AWR training.

    Architecture: 2-layer MLP → scalar, initialised near zero.
    """

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, L, d_model]
        Returns V: [B, L]
        """
        return self.net(h).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Reward Model  (Bradley-Terry preference learning)
# ─────────────────────────────────────────────────────────────────────────────

class RewardModel(nn.Module):
    """
    Learns a scalar reward signal from self-play preference pairs.

    Implements the Bradley-Terry model:
        P(winner > loser) = sigmoid(r(winner) - r(loser))

    Loss: -log P(winner > loser) = -log sigmoid(r_w - r_l)

    The reward model is trained alongside the main model, providing a
    richer training signal than raw log-likelihood for self-play.

    Architecture:
      Aggregate hidden states → reward scalar via attention pooling.
    """

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        # Attention pooling: learn which tokens matter for quality
        self.attn_pool = nn.Linear(d_model, 1)

        self.reward_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.reward_head[-1].weight)
        nn.init.zeros_(self.reward_head[-1].bias)

    def pool(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Attention-weighted pooling over sequence.
        h: [B, L, d], mask: [B, L] bool
        Returns: [B, d]
        """
        weights = self.attn_pool(h).squeeze(-1)   # [B, L]
        if mask is not None:
            weights = weights.masked_fill(~mask, float("-inf"))
        weights = F.softmax(weights, dim=-1)       # [B, L]
        return (weights.unsqueeze(-1) * h).sum(1)  # [B, d]

    def reward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        h: [B, L, d]
        Returns scalar reward [B] for each sequence.
        """
        pooled = self.pool(h, mask)               # [B, d]
        return self.reward_head(pooled).squeeze(-1)  # [B]

    def preference_loss(
        self,
        h_winner: torch.Tensor,   # [B, L, d]
        h_loser:  torch.Tensor,   # [B, L, d]
        margin:   float = 0.0,
    ) -> torch.Tensor:
        """
        Bradley-Terry loss: push r(winner) > r(loser) + margin.
        Returns scalar loss.
        """
        r_w = self.reward(h_winner)   # [B]
        r_l = self.reward(h_loser)    # [B]
        return -F.logsigmoid(r_w - r_l - margin).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Goal-Conditioned Value Head
# ─────────────────────────────────────────────────────────────────────────────

class GoalValueHead(nn.Module):
    """
    Goal-conditioned value V(s, g): how valuable is state s for achieving goal g?

    Conditions the value estimate on the current goal phase θ*,
    enabling the model to reason about goal-directed utility.

    V(s, g) = MLP([h_pooled ; goal_phase])

    Used for:
      - Goal-directed beam search (prefer high-V tokens)
      - Intrinsic motivation (reward states with high V(s, g))
      - Curriculum selection (target goals with high learning potential)
    """

    def __init__(self, d_model: int, n_phases: int, hidden: int = 128):
        super().__init__()
        self.goal_proj = nn.Linear(n_phases, d_model // 4)
        self.net = nn.Sequential(
            nn.Linear(d_model + d_model // 4, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        h:          torch.Tensor,           # [B, L, d]
        goal_phase: Optional[torch.Tensor], # [B, n_phases]
    ) -> torch.Tensor:
        """Returns goal-conditioned value [B, L]."""
        B, L, d = h.shape
        if goal_phase is None:
            goal_phase = torch.zeros(B, self.goal_proj.in_features, device=h.device)

        g = self.goal_proj(goal_phase)                   # [B, d//4]
        g = g.unsqueeze(1).expand(B, L, -1)              # [B, L, d//4]
        inp = torch.cat([h, g], dim=-1)                  # [B, L, d+d//4]
        return self.net(inp).squeeze(-1)                 # [B, L]


# ─────────────────────────────────────────────────────────────────────────────
# Advantage Estimator
# ─────────────────────────────────────────────────────────────────────────────

class AdvantageEstimator(nn.Module):
    """
    Computes per-token advantages for AWR training.

    A(s_t) = R(sequence) + γ · V(s_{t+1}) - V(s_t)

    Where:
      - R(sequence): sequence-level reward from RewardModel
      - V(s_t): per-token value from ValueHead
      - γ: discount factor

    The advantage tells us: "was this token better or worse than
    what the value function expected?" — positive A = better than expected.

    Advantage-Weighted Regression loss:
        L_AWR = -Σ_t exp(A_t / β) · log P(a_t | s_t)
    where β controls the sharpness of the weighting.
    """

    def __init__(
        self,
        d_model:   int,
        n_phases:  int,
        gamma:     float = 0.99,
        beta:      float = 0.1,     # AWR temperature
        clip_adv:  float = 5.0,     # clamp advantages for stability
    ):
        super().__init__()
        self.gamma    = gamma
        self.beta     = beta
        self.clip_adv = clip_adv

        self.value_head      = ValueHead(d_model)
        self.reward_model    = RewardModel(d_model)
        self.goal_value_head = GoalValueHead(d_model, n_phases)

    def compute_advantages(
        self,
        h:          torch.Tensor,           # [B, L, d]
        reward:     Optional[torch.Tensor] = None,  # [B] sequence reward
        goal_phase: Optional[torch.Tensor] = None,  # [B, n_phases]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
          advantages : [B, L]  — per-token advantage estimates
          value_dict : dict with value, goal_value, reward tensors
        """
        V = self.value_head(h)                              # [B, L]

        if goal_phase is not None:
            V_goal = self.goal_value_head(h, goal_phase)   # [B, L]
        else:
            V_goal = V

        # Bootstrap TD targets: V_target[t] = r + γ * V[t+1]
        V_next = torch.cat([V[:, 1:], V[:, -1:]], dim=1)   # [B, L] shift by 1
        td_target = V_next * self.gamma

        if reward is not None:
            # Add sequence reward to last token, propagate back
            r = reward.unsqueeze(1)                         # [B, 1]
            td_target[:, -1:] = td_target[:, -1:] + r

        advantages = (td_target - V).detach()               # [B, L], no grad
        advantages = advantages.clamp(-self.clip_adv, self.clip_adv)

        return advantages, {
            "value":      V,
            "goal_value": V_goal,
            "td_target":  td_target.detach(),
        }

    def awr_loss(
        self,
        logits:     torch.Tensor,           # [B, L, V]
        targets:    torch.Tensor,           # [B, L]
        h:          torch.Tensor,           # [B, L, d]
        reward:     Optional[torch.Tensor] = None,
        goal_phase: Optional[torch.Tensor] = None,
        pad_id:     int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Advantage-Weighted Regression loss.
        Returns (awr_loss scalar, value_loss scalar, metrics dict).
        """
        advantages, vdict = self.compute_advantages(h, reward, goal_phase)

        # AWR weights: exp(A / β), normalised across tokens
        weights = (advantages / self.beta).exp()            # [B, L]
        weights = weights / (weights.sum() + 1e-8)         # normalise

        # Per-token cross-entropy
        B, L, vocab = logits.shape
        per_tok = F.cross_entropy(
            logits.reshape(B * L, vocab),
            targets.reshape(B * L),
            ignore_index=pad_id,
            reduction="none",
        ).reshape(B, L)                                     # [B, L]

        valid = (targets != pad_id).float()
        awr = (weights * per_tok * valid).sum()

        # Value head loss: MSE toward TD targets
        val_loss = F.mse_loss(vdict["value"], vdict["td_target"])

        return awr, val_loss, {
            "awr": awr.item(),
            "val_loss": val_loss.item(),
            "mean_adv": advantages.mean().item(),
            "mean_weight": weights.mean().item(),
        }

    def value_loss(
        self,
        h:      torch.Tensor,
        reward: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standalone value head loss for pre-training the value function."""
        V = self.value_head(h)
        V_next = torch.cat([V[:, 1:], V[:, -1:]], dim=1)
        td = V_next * self.gamma
        if reward is not None:
            td[:, -1:] = td[:, -1:] + reward.unsqueeze(1)
        return F.mse_loss(V, td.detach())
