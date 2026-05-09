"""
NFN v4.0 — AGIBlock

Full AGI-capable transformer block. Wraps EfficientNFNBlock with:

  ┌─────────────────────────────────────────────────────────────────┐
  │  h_in                                                           │
  │   ├─► [FreeEnergyMinimiser]     belief state compression        │
  │   │                                                             │
  │   ├─► EfficientNFNBlock         linear attn + soliton + MoE     │
  │   │                                                             │
  │   ├─► [TwoTierMemory]           episodic ring + semantic SVD    │
  │   │                                                             │
  │   ├─► [FractalWorkingMemory]    differentiable scratchpad       │
  │   │                                                             │
  │   ├─► [CausalGraphLayer]        DAG + do-calculus               │
  │   │                                                             │
  │   ├─► [PhaseGoalPredictor]      λ·sin(θ*−θ) Kuramoto forcing   │
  │   │                                                             │
  │   └─► [SelfConsistencyCheck]    internal debate → best cand.   │
  │                                                                  │
  │  h_out  (fused residual of all active streams)                  │
  └─────────────────────────────────────────────────────────────────┘

All modules are opt-in via NFNConfig flags — the block degrades
gracefully to a plain EfficientNFNBlock when all flags are False.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NFNConfig
from .efficient_block import EfficientNFNBlock
from .episodic_memory import TwoTierMemory
from .working_memory import FractalWorkingMemory
from .causal import CausalGraphLayer
from .goal import PhaseGoalPredictor
from .hopfield import BayesianZipfianDecoder
from .reasoning import SelfConsistencyCheck, PlanExecutor
from .predictive import FreeEnergyMinimiser
from .mixture_of_depths import MixtureOfDepths
from .hyper import ContextHyperNet, HyperResidual


class AGIBlock(nn.Module):

    def __init__(self, cfg: NFNConfig, block_idx: int = 0):
        super().__init__()
        self.cfg       = cfg
        self.block_idx = block_idx
        d = cfg.d_model

        # ── Core sequence mixer (optionally wrapped with MoD) ─────────────
        core_raw = EfficientNFNBlock(cfg)
        if cfg.use_mixture_of_depths:
            self.core = MixtureOfDepths(
                d_model         = d,
                block           = core_raw,
                capacity_factor = cfg.mod_capacity_factor,
                n_phases        = cfg.goal_n_phases,
                lambda_router   = cfg.lambda_router,
            )
        else:
            self.core = core_raw

        # ── Free Energy (belief compression, runs before core) ────────────
        self.free_energy: Optional[FreeEnergyMinimiser] = None
        if cfg.use_free_energy:
            self.free_energy = FreeEnergyMinimiser(d, cfg.fe_latent_dim, cfg.lambda_fe)

        # ── Two-Tier Memory ───────────────────────────────────────────────
        self.memory: Optional[TwoTierMemory] = None
        if cfg.use_episodic_memory:
            self.memory = TwoTierMemory(
                d_model             = d,
                episodic_capacity   = cfg.episodic_capacity,
                episodic_key_dim    = cfg.episodic_key_dim,
                episodic_n_read     = cfg.episodic_n_read,
                semantic_rank       = cfg.semantic_rank,
                consolidation_freq  = cfg.consolidation_freq,
                consolidation_every = cfg.consolidation_every,
            )

        # ── Differentiable Working Memory ─────────────────────────────────
        self.working_mem: Optional[FractalWorkingMemory] = None
        if cfg.use_working_memory:
            self.working_mem = FractalWorkingMemory(
                d_model   = d,
                n_slots   = cfg.wm_n_slots,
                n_heads   = cfg.wm_n_heads,
                sharpness = cfg.wm_sharpness,
            )

        # ── Causal Graph Layer ────────────────────────────────────────────
        self.causal: Optional[CausalGraphLayer] = None
        if cfg.use_causal_graph:
            self.causal = CausalGraphLayer(
                d_model  = d,
                n_slots  = cfg.causal_n_slots,
                hidden   = cfg.causal_hidden,
                sparsity = cfg.causal_sparsity,
            )

        # ── Goal-Directed Phase Forcing ───────────────────────────────────
        self.goal: Optional[PhaseGoalPredictor] = None
        if cfg.use_goal_predictor:
            self.goal = PhaseGoalPredictor(
                d_model     = d,
                n_phases    = cfg.goal_n_phases,
                init_lambda = cfg.goal_init_lambda,
                n_steps     = cfg.goal_n_steps,
            )

        # ── Self-Consistency Check ────────────────────────────────────────
        self.consistency: Optional[SelfConsistencyCheck] = None
        if cfg.use_self_consistency:
            self.consistency = SelfConsistencyCheck(
                d_model      = d,
                n_candidates = cfg.sc_n_candidates,
                noise_scale  = cfg.sc_noise_scale,
            )

        # ── Plan Executor ─────────────────────────────────────────────────
        self.planner: Optional[PlanExecutor] = None
        if cfg.use_plan_executor and cfg.use_goal_predictor:
            self.planner = PlanExecutor(cfg.goal_n_phases, cfg.plan_n_subgoals)

        # ── Hyper-network (in-context weight adaptation) ──────────────────
        self.hyper: Optional[HyperResidual] = None
        if cfg.use_hyper_net:
            self.hyper_net = ContextHyperNet(
                d_model  = d,
                n_phases = cfg.goal_n_phases,
                rank     = cfg.hyper_rank,
                z_dim    = cfg.hyper_z_dim,
                scale    = cfg.hyper_scale,
            )
            self.hyper = HyperResidual(d, cfg.hyper_z_dim)
        else:
            self.hyper_net = None

        # ── Stream fusion ─────────────────────────────────────────────────
        n_streams = 1
        if cfg.use_episodic_memory: n_streams += 1
        if cfg.use_working_memory:  n_streams += 1
        self.fusion = nn.Linear(d * n_streams, d, bias=False)
        nn.init.zeros_(self.fusion.weight)

    # ─── goal / plan management ──────────────────────────────────────────────

    def set_goal(self, h_prompt: torch.Tensor):
        if self.goal is not None:
            self.goal.set_goal(h_prompt)
            if self.planner is not None:
                self.planner.set_plan(self.goal._goal_phase)

    def reset_goal(self):
        if self.goal is not None:
            self.goal.reset_goal()
        if self.planner is not None:
            self.planner.reset()

    def reset_working_memory(self):
        if self.working_mem is not None:
            self.working_mem.reset()

    # ─── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        h:            torch.Tensor,
        mask:         Optional[torch.Tensor] = None,
        write_memory: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns (h_out [B, L, d], losses dict).
        """
        losses: Dict[str, torch.Tensor] = {}

        # ── 1. Free Energy — belief compression before core ───────────────
        if self.free_energy is not None:
            h, fe_loss = self.free_energy(h)
            losses["free_energy"] = losses.get("free_energy", 0.0) + fe_loss

        # ── 2. Core EfficientNFNBlock (+ optional MoD routing) ────────────
        core_result = self.core(h)
        if isinstance(core_result, tuple):
            h_core, router_loss = core_result
            losses["router"] = losses.get("router", 0.0) + router_loss * self.cfg.lambda_router
        else:
            h_core = core_result
        streams = [h_core]

        # ── 3. Two-Tier Memory ─────────────────────────────────────────────
        if self.memory is not None:
            h_mem = self.memory(h_core, write=write_memory)
            streams.append(h_mem)

        # ── 4. Working Memory ──────────────────────────────────────────────
        if self.working_mem is not None:
            h_wm = self.working_mem(h_core, write=write_memory)
            streams.append(h_wm)

        # ── 5. Fuse memory streams ─────────────────────────────────────────
        if len(streams) > 1:
            h_fused = self.fusion(torch.cat(streams, dim=-1))
        else:
            h_fused = h_core

        # ── 6. Causal Graph ────────────────────────────────────────────────
        if self.causal is not None:
            h_fused, loss_dag = self.causal(h_fused)
            losses["causal"] = losses.get("causal", 0.0) + loss_dag * self.cfg.lambda_causal

        # ── 7. Goal Forcing ────────────────────────────────────────────────
        if self.goal is not None:
            d = h_fused.shape[-1]
            half = min(self.cfg.goal_n_phases, d // 2)
            phase_proxy = torch.atan2(
                h_fused[..., :half],
                h_fused[..., half:half * 2] + 1e-6,
            )
            if half < self.cfg.goal_n_phases:
                pad = torch.zeros(*phase_proxy.shape[:-1],
                                  self.cfg.goal_n_phases - half, device=h.device)
                phase_proxy = torch.cat([phase_proxy, pad], dim=-1)

            # Use sub-goal if planner is active
            if self.planner is not None:
                subgoal = self.planner.current_subgoal(h.device)
                if subgoal is not None:
                    self.goal._goal_phase = subgoal

            theta_forced, alignment = self.goal(phase_proxy, h_context=h_core)
            losses["goal"] = losses.get("goal", 0.0) + \
                self.goal.loss_goal(phase_proxy) * self.cfg.lambda_goal

            # Advance plan if alignment is high enough
            if self.planner is not None:
                self.planner.advance(alignment.mean(-1))

            goal_feat = torch.cat([
                torch.cos(theta_forced), torch.sin(theta_forced)
            ], dim=-1)                                          # [B, L, 2*n_phases]
            # Pad or truncate to d_model
            gf_d = goal_feat.shape[-1]
            if gf_d < d:
                goal_feat = F.pad(goal_feat, (0, d - gf_d))
            elif gf_d > d:
                goal_feat = goal_feat[..., :d]
            h_fused = h_fused + goal_feat * 0.05

        # ── 8. Self-Consistency ────────────────────────────────────────────
        if self.consistency is not None:
            h_fused, cons_loss = self.consistency(h_fused, self.causal)
            losses["consistency"] = losses.get("consistency", 0.0) + cons_loss * 0.1

        # ── 9. Hyper-network residual modulation ───────────────────────────
        if self.hyper is not None and self.hyper_net is not None:
            goal_phase = None
            if self.goal is not None and self.goal._goal_phase is not None:
                goal_phase = self.goal._goal_phase
            z, _ = self.hyper_net(h_fused, goal_phase)
            h_fused = self.hyper(h_fused, z)

        return h_fused, losses

    # ─── counterfactual reasoning ─────────────────────────────────────────────

    def counterfactual(self, h: torch.Tensor, slot_idx: int, value: torch.Tensor) -> torch.Tensor:
        if self.causal is None:
            raise RuntimeError("use_causal_graph must be True for counterfactuals")
        return self.causal.causal_query(h, slot_idx, value)
