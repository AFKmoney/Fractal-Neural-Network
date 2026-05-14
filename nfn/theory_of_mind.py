"""
NFN AGI v5.0 — Theory of Mind

Theory of Mind (ToM) is the ability to attribute mental states — beliefs,
desires, intentions — to others and understand that others have perspectives
different from one's own. It is a cornerstone of human-level general intelligence.

This module implements a computational ToM for NFN:

  1. Agent Belief Encoder:
     Given observations of another "agent's" outputs, infer a compact
     belief state b ∈ R^d representing what that agent likely believes/intends.

  2. Perspective Taker:
     Project the model's own representation into the estimated belief space
     of another agent — "what would I look like from their perspective?"

  3. Mental State Predictor:
     Given an agent's belief state, predict their likely next action/output.
     Loss: how well can we predict the agent's behavior from their inferred beliefs?

  4. Counterfactual Agent Reasoning:
     "If the agent had different beliefs, what would they do?"
     Uses the causal graph to reason counterfactually about agent behavior.

Theory
------
ToM in this context means:
  - The model doesn't just generate text — it models WHO is reading/responding
  - During self-play, the model learns to distinguish "winner" and "loser"
    perspectives as different "agents" with different quality thresholds
  - The ToM loss encourages the model to maintain belief representations
    that accurately predict agent behavior

In practice:
  - "Agent 0" = the model itself (self-modeling)
  - "Agent 1" = the target reader/user (other-modeling)
  - "Agent 2" = a critic agent (quality evaluator)

Architecture
------------
  AgentBeliefEncoder   : observed_h [B, L, d] → belief [B, d]
  PerspectiveTaker     : (own_h, other_belief) → perspective_h [B, L, d]
  MentalStatePredictor : belief [B, d] → predicted_action_logits [B, L, d]
  TheoryOfMindModule   : full module integrating all components
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Agent Belief Encoder
# ─────────────────────────────────────────────────────────────────────────────

class AgentBeliefEncoder(nn.Module):
    """
    Infers a compact belief state from an agent's observed outputs.

    The belief state b ∈ R^d represents:
      - What the agent "knows" (estimated knowledge state)
      - What the agent "wants" (inferred goals/intentions)
      - What the agent "believes" (epistemic state)

    Architecture:
      Self-attention over observations → pooled → compressed belief.
      The self-attention allows the model to find patterns in the agent's
      behavior across multiple timesteps.
    """

    def __init__(self, d_model: int, n_heads: int = 4, belief_dim: int = 128):
        super().__init__()
        self.belief_dim = belief_dim

        # Temporal attention over agent observations
        self.obs_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.obs_norm = nn.LayerNorm(d_model)

        # Compress to belief state
        self.belief_enc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, belief_dim),
            nn.LayerNorm(belief_dim),
        )
        nn.init.zeros_(self.belief_enc[-2].weight)
        nn.init.zeros_(self.belief_enc[-2].bias)

        # Learnable "query" that extracts the most belief-relevant features
        self.belief_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.01)

    def forward(self, obs_h: torch.Tensor) -> torch.Tensor:
        """
        obs_h: [B, L, d]  — agent's observed hidden states
        Returns belief: [B, belief_dim]
        """
        B = obs_h.shape[0]
        query = self.belief_query.expand(B, -1, -1)    # [B, 1, d]

        # Attend to observations with a learned query
        attn_out, _ = self.obs_attn(query, obs_h, obs_h)  # [B, 1, d]
        attn_out     = self.obs_norm(attn_out.squeeze(1))  # [B, d]

        belief = self.belief_enc(attn_out)                 # [B, belief_dim]
        return belief


# ─────────────────────────────────────────────────────────────────────────────
# Perspective Taker
# ─────────────────────────────────────────────────────────────────────────────

class PerspectiveTaker(nn.Module):
    """
    Modulates the model's own representation with another agent's belief state.

    This implements "perspective taking": how would my representation look
    if I were viewing the world from the other agent's belief state?

    The modulation uses FiLM (Feature-wise Linear Modulation):
        h_perspective = γ(b) ⊙ h + β(b)
    where γ, β are generated from the other agent's belief b.

    This allows the model to:
      - Adapt its generation to the estimated reader's knowledge level
      - Generate from the perspective of a character with specific beliefs
      - Model how a critic would evaluate the current output
    """

    def __init__(self, d_model: int, belief_dim: int = 128):
        super().__init__()

        # FiLM generators from belief state
        self.gamma_gen = nn.Sequential(
            nn.Linear(belief_dim, d_model),
            nn.Sigmoid(),
        )
        self.beta_gen = nn.Sequential(
            nn.Linear(belief_dim, d_model),
            nn.Tanh(),
        )

        # Blending gate: how much to apply perspective modulation
        self.blend_gate = nn.Sequential(
            nn.Linear(belief_dim, 1),
            nn.Sigmoid(),
        )

        # Initialise near identity (start with minimal effect)
        nn.init.ones_(self.gamma_gen[-2].bias)
        nn.init.zeros_(self.beta_gen[-2].weight)
        nn.init.zeros_(self.beta_gen[-2].bias)
        nn.init.constant_(self.blend_gate[-2].bias, -2.0)  # start with low blending

    def forward(
        self,
        h:         torch.Tensor,   # [B, L, d]
        belief:    torch.Tensor,   # [B, belief_dim]
        strength:  float = 1.0,
    ) -> torch.Tensor:
        """
        Returns perspective-modulated hidden states [B, L, d].
        """
        gamma = self.gamma_gen(belief).unsqueeze(1)     # [B, 1, d]
        beta  = self.beta_gen(belief).unsqueeze(1)      # [B, 1, d]
        gate  = self.blend_gate(belief).unsqueeze(1)    # [B, 1, 1]

        h_perspective = h * gamma + beta * 0.05
        blend         = gate * strength
        return h + blend * (h_perspective - h)


# ─────────────────────────────────────────────────────────────────────────────
# Mental State Predictor
# ─────────────────────────────────────────────────────────────────────────────

class MentalStatePredictor(nn.Module):
    """
    Predicts an agent's next action distribution from their belief state.

    Given: agent belief b ∈ R^{belief_dim}
    Predict: agent's next hidden state distribution

    Loss (self-supervised):
      Given actual next h_{t+1}, how well can we predict it from belief b?
      L_tom = ||h_{t+1}_predicted - h_{t+1}||²

    This loss trains the belief encoder to capture genuine predictive
    information about agent behavior — not just surface features.
    """

    def __init__(self, d_model: int, belief_dim: int = 128):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(belief_dim, d_model // 2),
            nn.SiLU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, d_model),
        )
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    def forward(
        self,
        belief:     torch.Tensor,   # [B, belief_dim]
        h_actual:   torch.Tensor,   # [B, L, d]  — actual next states
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          h_predicted : [B, L, d]   — predicted next states
          predict_loss: scalar      — prediction error
        """
        h_pred = self.predictor(belief).unsqueeze(1).expand_as(h_actual)
        loss   = F.mse_loss(h_pred, h_actual.detach())
        return h_pred, loss


# ─────────────────────────────────────────────────────────────────────────────
# Theory of Mind Module  (full)
# ─────────────────────────────────────────────────────────────────────────────

class TheoryOfMindModule(nn.Module):
    """
    Full Theory of Mind module for NFN AGI.

    During training, uses self-play pairs (winner/loser) as two "agents"
    with different quality/coherence levels. The model learns:

      - "Winner agent" belief: encodes what high-quality generation looks like
      - "Loser agent" belief:  encodes what low-quality generation looks like
      - Perspective taking:    modulate generation toward winner belief
      - Prediction loss:       predict winner's continuation from their belief

    At inference, the model can:
      - Model the reader's knowledge state (adaptive generation)
      - Generate from specific perspectives (characters, roles)
      - Evaluate its own outputs from an external perspective

    Integration:
      The ToM module is called optionally in AGIBlock's forward() when
      self-play history is available (winner_h, loser_h stored).
    """

    def __init__(
        self,
        d_model:    int,
        n_heads:    int   = 4,
        belief_dim: int   = 128,
        lambda_tom: float = 0.1,
    ):
        super().__init__()
        self.lambda_tom = lambda_tom
        self.belief_dim = belief_dim

        # Two belief encoders: "self" and "other"
        self.self_belief  = AgentBeliefEncoder(d_model, n_heads, belief_dim)
        self.other_belief = AgentBeliefEncoder(d_model, n_heads, belief_dim)

        # Perspective taker: modulate h with other's belief
        self.perspective  = PerspectiveTaker(d_model, belief_dim)

        # Mental state predictor: predict agent's next state from belief
        self.predictor    = MentalStatePredictor(d_model, belief_dim)

        # Belief distance metric: are beliefs different? (contrastive)
        self.belief_proj  = nn.Linear(belief_dim, belief_dim // 2)

        # Stored beliefs (set during self-play, used during forward)
        self._winner_belief: Optional[torch.Tensor] = None
        self._loser_belief:  Optional[torch.Tensor] = None

    def update_beliefs(
        self,
        winner_h: torch.Tensor,   # [B, L, d]  — winner hidden states
        loser_h:  torch.Tensor,   # [B, L, d]  — loser hidden states
    ):
        """
        Update stored belief states from self-play observations.
        Called after each self-play step.
        """
        with torch.no_grad():
            self._winner_belief = self.other_belief(winner_h).detach()
            self._loser_belief  = self.other_belief(loser_h).detach()

    def forward(
        self,
        h:                torch.Tensor,            # [B, L, d]  — current hidden states
        other_h:          Optional[torch.Tensor] = None,  # [B, L, d]
        use_winner_perspective: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Apply ToM modulation to hidden states.

        If winner/loser beliefs are available (from self-play),
        modulates h toward the winner's perspective.

        Returns:
          h_out   : [B, L, d]  — ToM-modulated hidden states
          tom_loss: scalar     — prediction + contrastive loss
          metrics : dict
        """
        tom_loss = torch.tensor(0.0, device=h.device)
        metrics: Dict[str, float] = {}

        # Use stored winner belief if available
        if use_winner_perspective and self._winner_belief is not None:
            B = h.shape[0]
            belief = self._winner_belief

            # Batch size mismatch handling
            if belief.shape[0] != B:
                if belief.shape[0] == 1:
                    belief = belief.expand(B, -1)
                else:
                    belief = belief[:B] if belief.shape[0] > B else belief

            # Modulate toward winner's perspective
            h_out = self.perspective(h, belief)

            # Mental state prediction loss
            _, pred_loss = self.predictor(belief, h)
            tom_loss     = tom_loss + self.lambda_tom * pred_loss

            # Contrastive loss: winner and loser beliefs should differ
            if self._loser_belief is not None:
                loser_belief = self._loser_belief
                if loser_belief.shape[0] != B:
                    loser_belief = loser_belief.expand(B, -1) if loser_belief.shape[0] == 1 else loser_belief[:B]

                w_proj = F.normalize(self.belief_proj(belief), dim=-1)        # [B, belief_dim//2]
                l_proj = F.normalize(self.belief_proj(loser_belief), dim=-1)  # [B, belief_dim//2]
                # Contrastive: push winner and loser beliefs apart
                cosine = (w_proj * l_proj).sum(-1)                            # [B]
                contrastive_loss = F.relu(cosine + 0.3).mean()               # margin = -0.3
                tom_loss = tom_loss + self.lambda_tom * contrastive_loss

                metrics["tom_contrastive"] = contrastive_loss.item()

            metrics["tom_pred_loss"] = pred_loss.item()

        elif other_h is not None:
            # No stored beliefs but have another agent's h — infer on the fly
            belief = self.other_belief(other_h)
            h_out  = self.perspective(h, belief)
            _, pred_loss = self.predictor(belief, h)
            tom_loss = self.lambda_tom * pred_loss
            metrics["tom_pred_loss"] = pred_loss.item()
        else:
            h_out = h

        metrics["tom_total"] = tom_loss.item() if isinstance(tom_loss, torch.Tensor) else 0.0
        return h_out, tom_loss, metrics

    def tom_training_loss(
        self,
        winner_h: torch.Tensor,   # [B, L, d]
        loser_h:  torch.Tensor,   # [B, L, d]
        current_h: torch.Tensor,  # [B, L, d]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Full ToM training loss using self-play pairs.

        1. Encode winner and loser beliefs
        2. Predict current h from winner belief (should be close)
        3. Predict current h from loser belief (should be far)
        4. Contrastive: winner belief should differ from loser belief

        Returns (total_loss, metrics).
        """
        winner_belief = self.other_belief(winner_h)    # [B, belief_dim]
        loser_belief  = self.other_belief(loser_h)     # [B, belief_dim]

        # Prediction losses
        _, win_pred_loss  = self.predictor(winner_belief, current_h)
        _, lose_pred_loss = self.predictor(loser_belief,  current_h)

        # Winner should predict current better (it's the "good" example)
        # Loser predictions should be worse — but we don't penalise that directly
        # Instead, use contrastive loss on the belief states themselves

        w_proj = F.normalize(self.belief_proj(winner_belief), dim=-1)
        l_proj = F.normalize(self.belief_proj(loser_belief), dim=-1)
        cosine = (w_proj * l_proj).sum(-1).mean()      # should be < 0 (diverse beliefs)
        contrastive = F.relu(cosine + 0.2)             # margin = -0.2

        total = self.lambda_tom * (win_pred_loss + contrastive)

        # Update stored beliefs
        with torch.no_grad():
            self._winner_belief = winner_belief.detach()
            self._loser_belief  = loser_belief.detach()

        return total, {
            "tom_win_pred":   win_pred_loss.item(),
            "tom_contrastive": contrastive.item(),
            "tom_total":      total.item(),
        }

    def reset_beliefs(self):
        """Clear stored beliefs (call between episodes)."""
        self._winner_belief = None
        self._loser_belief  = None
