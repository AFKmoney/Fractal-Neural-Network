"""
NFN v4.0 — AGIBlock

Wraps EfficientNFNBlock with all AGI v4.0 modules:

  ┌─────────────────────────────────────────────────────────────────┐
  │  h  ─► EfficientNFNBlock (linear attn + phase soliton + MoE)   │
  │         │                                                        │
  │         ├─► TwoTierMemory  (episodic ring + semantic condensate) │
  │         │        ↕ read / write                                  │
  │         ├─► CausalGraphLayer  (DAG over memory slots)           │
  │         │        ↕ SCM propagation + do-calculus                 │
  │         ├─► PhaseGoalPredictor  (λ·sin(θ*−θ) forcing)          │
  │         │        ↕ goal attraction + alignment loss              │
  │         └─► BayesianZipfianDecoder  (uncertainty-calibrated LM) │
  └─────────────────────────────────────────────────────────────────┘

Losses aggregated from all sub-modules and returned as a dict:
  "lm"      : cross-entropy language model loss
  "phase"   : EfficientNFNBlock internal phase coherence
  "causal"  : DAG sparsity
  "goal"    : phase-goal alignment
  "coherence": optional cross-modal coherence
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NFNConfig
from .efficient_block import EfficientNFNBlock
from .episodic_memory import TwoTierMemory
from .causal import CausalGraphLayer
from .goal import PhaseGoalPredictor
from .hopfield import BayesianZipfianDecoder


class AGIBlock(nn.Module):
    """
    Single AGI-capable transformer block.

    Can be stacked N times in AGINFNModel.  Each block shares the same
    TwoTierMemory and CausalGraphLayer (cross-block memory) but has its own
    EfficientNFNBlock, PhaseGoalPredictor, and output projection.

    The PhaseGoalPredictor is applied to the *phase* output of EfficientNFNBlock
    (if the block exposes it), or derived from the hidden state.
    """

    def __init__(self, cfg: NFNConfig, block_idx: int = 0):
        super().__init__()
        self.cfg       = cfg
        self.block_idx = block_idx

        # ── Core sequence mixer ───────────────────────────────────────────
        self.core = EfficientNFNBlock(cfg)

        # ── AGI modules (conditionally enabled) ───────────────────────────
        self.memory: Optional[TwoTierMemory] = None
        if cfg.use_episodic_memory:
            self.memory = TwoTierMemory(
                d_model             = cfg.d_model,
                episodic_capacity   = cfg.episodic_capacity,
                episodic_key_dim    = cfg.episodic_key_dim,
                episodic_n_read     = cfg.episodic_n_read,
                semantic_rank       = cfg.semantic_rank,
                consolidation_freq  = cfg.consolidation_freq,
                consolidation_every = cfg.consolidation_every,
            )

        self.causal: Optional[CausalGraphLayer] = None
        if cfg.use_causal_graph:
            self.causal = CausalGraphLayer(
                d_model  = cfg.d_model,
                n_slots  = cfg.causal_n_slots,
                hidden   = cfg.causal_hidden,
                sparsity = cfg.causal_sparsity,
            )

        self.goal: Optional[PhaseGoalPredictor] = None
        if cfg.use_goal_predictor:
            self.goal = PhaseGoalPredictor(
                d_model     = cfg.d_model,
                n_phases    = cfg.goal_n_phases,
                init_lambda = cfg.goal_init_lambda,
                n_steps     = cfg.goal_n_steps,
            )

        # Residual gate for combining AGI enrichments with core output
        n_streams = 1  # core always present
        if cfg.use_episodic_memory: n_streams += 1
        if cfg.use_causal_graph:    n_streams += 1
        self.fusion = nn.Linear(cfg.d_model * n_streams, cfg.d_model, bias=False)
        nn.init.zeros_(self.fusion.weight)  # start as passthrough of core

    # ─── goal management ──────────────────────────────────────────────────────

    def set_goal(self, h_prompt: torch.Tensor):
        """Set generation goal from prompt hidden states."""
        if self.goal is not None:
            self.goal.set_goal(h_prompt)

    def reset_goal(self):
        if self.goal is not None:
            self.goal.reset_goal()

    # ─── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        h:              torch.Tensor,                   # [B, L, d]
        mask:           Optional[torch.Tensor] = None,  # [B, L] attention mask
        write_memory:   bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
          h_out   : [B, L, d]
          losses  : dict of scalar auxiliary losses
        """
        losses: Dict[str, torch.Tensor] = {}
        streams = []

        # ── 1. Core EfficientNFNBlock ──────────────────────────────────────
        h_core = self.core(h)
        streams.append(h_core)

        # ── 2. Two-Tier Memory ─────────────────────────────────────────────
        if self.memory is not None:
            h_mem = self.memory(h_core, write=write_memory)
            streams.append(h_mem)

        # ── 3. Causal Graph Layer ──────────────────────────────────────────
        if self.causal is not None:
            h_causal, loss_dag = self.causal(h_core)
            losses["causal"] = loss_dag * self.cfg.lambda_causal
        else:
            h_causal = h_core

        # ── 4. Goal-directed phase forcing ─────────────────────────────────
        # Derive a phase proxy from h_causal (cos+sin of linear projection)
        if self.goal is not None:
            d = h_causal.shape[-1]
            half = min(self.cfg.goal_n_phases, d // 2)
            phase_proxy = torch.atan2(
                h_causal[..., :half],
                h_causal[..., half:half*2] + 1e-6,
            )  # [B, L, half]
            # Pad to goal_n_phases if needed
            if half < self.cfg.goal_n_phases:
                pad = torch.zeros(
                    *phase_proxy.shape[:-1],
                    self.cfg.goal_n_phases - half,
                    device=h.device,
                )
                phase_proxy = torch.cat([phase_proxy, pad], dim=-1)

            theta_forced, alignment = self.goal(phase_proxy, h_context=h_core)

            # Goal loss: encourage alignment
            losses["goal"] = self.goal.loss_goal(phase_proxy) * self.cfg.lambda_goal

            # Mix goal-forced signal back into h_causal via sin+cos features
            goal_feat = torch.cat([
                torch.cos(theta_forced),
                torch.sin(theta_forced),
            ], dim=-1)[:, :, :d]  # [B, L, d] (truncate if needed)
            h_causal = h_causal + goal_feat * 0.05

        # ── 5. Fuse streams ────────────────────────────────────────────────
        if len(streams) > 1:
            h_out = self.fusion(torch.cat(streams, dim=-1))
            h_out = h_causal + h_out   # residual from causal / goal on top
        else:
            h_out = h_causal

        return h_out, losses

    # ─── counterfactual reasoning ─────────────────────────────────────────────

    def counterfactual(
        self,
        h:        torch.Tensor,  # [B, L, d]
        slot_idx: int,
        value:    torch.Tensor,  # [B, d]
    ) -> torch.Tensor:
        """
        What would h look like if causal slot `slot_idx` were set to `value`?
        Requires use_causal_graph=True.
        """
        if self.causal is None:
            raise RuntimeError("use_causal_graph must be True for counterfactuals")
        return self.causal.causal_query(h, slot_idx, value)
