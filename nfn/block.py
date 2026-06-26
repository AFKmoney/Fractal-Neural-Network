"""
LEAC Block — Unifie les Trois Piliers de l'Emergence

Architecture par bloc:
  1. Embedding Gematrique (5 systemes croises + biais d'attention)
  2. Attention Fractale Lineaire O(L*d^2) avec feature map fractale
  3. Soliton de Phase Kuramoto (coherence a longue portee)
  4. Phase-Routed MoE (routage von Mises, experts creux)
  5. [Optionnel] Graphe Causal (DAG + NOTEARS + do-calculus)
  6. [Optionnel] Self-Model (Global Workspace + Introspection)
  7. [Optionnel] Memoire de Travail (scratchpad differentiable)

Les 3 piliers de LEAC:
  Pilier 1 - COHERENCE: Dynamique de Phase Kuramoto
  Pilier 2 - RAISONNEMENT: SCM Causal (DAG + interventions + contre-factuels)
  Pilier 3 - INTROSPECTION: Self-Model (workspace global + auto-representation)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FNNConfig
from .moe import FractalLinearAttention, PhaseSoliton, PhaseRoutedMoE
from .causal import CausalGraphLayer
from .self_model import SelfModel
from .working_memory import FractalWorkingMemory


class FNNBlock(nn.Module):
    """
    Bloc LEAC unifiant les 3 piliers de la conscience emergente.

    Chaque bloc est: Gematria → FractalLinearAttn → Soliton → MoE
    +
    [Pilier 1] Kuramoto Phase Forcing (coherence dirigee par but)
    [Pilier 2] CausalGraphLayer (raisonnement causal structure)
    [Pilier 3] SelfModel (introspection + workspace global)
    """

    def __init__(self, cfg: FNNConfig, block_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.block_idx = block_idx
        d = cfg.d_model

        self.attn = FractalLinearAttention(
            d_model=d,
            n_heads=cfg.n_heads,
            n_levels=cfg.n_levels,
            dropout=cfg.dropout,
            causal=True,
        )

        self.soliton = PhaseSoliton(d, n_phases=cfg.nfmc_n_phases)

        self.moe = PhaseRoutedMoE(
            d_model=d,
            n_experts=cfg.moe_n_experts,
            d_ff_per_expert=cfg.moe_d_ff_per_expert,
            n_phases=cfg.nfmc_n_phases,
            top_k=cfg.moe_top_k,
            kappa=cfg.moe_kappa,
            dropout=cfg.dropout,
        )

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        # ── Pilier 2: Raisonnement Causal ──────────────────────────────────
        self.causal: Optional[CausalGraphLayer] = None
        if cfg.use_causal_graph:
            self.causal = CausalGraphLayer(
                d_model=d,
                n_slots=cfg.causal_n_slots,
                hidden=cfg.causal_hidden,
                sparsity=cfg.causal_sparsity,
                use_nonlinear=cfg.use_nonlinear_causal,
            )

        # ── Pilier 3: Self-Model (Introspection) ────────────────────────────
        self.self_model: Optional[SelfModel] = None
        if cfg.use_self_model:
            self.self_model = SelfModel(
                d_model=d,
                n_slots=cfg.self_model_n_slots,
                n_signals=cfg.self_model_n_signals,
            )

        # ── Memoire de Travail Differentiable ──────────────────────────────
        self.working_mem: Optional[FractalWorkingMemory] = None
        if cfg.use_working_memory:
            self.working_mem = FractalWorkingMemory(
                d_model=d,
                n_slots=cfg.wm_n_slots,
                n_heads=cfg.wm_n_heads,
                sharpness=cfg.wm_sharpness,
            )

        # ── Fusion des flux ─────────────────────────────────────────────────
        n_streams = 1
        if cfg.use_working_memory:
            n_streams += 1
        if n_streams > 1:
            self.fusion = nn.Linear(d * n_streams, d, bias=False)
            nn.init.zeros_(self.fusion.weight)
        else:
            self.fusion = None

    def forward(
        self,
        h: torch.Tensor,
        write_memory: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        h: [B, L, d]
        Returns: (h_out [B, L, d], losses dict)
        """
        losses: Dict[str, torch.Tensor] = {}

        # ── 1. Fractal Linear Attention + Soliton + MoE ──────────────────────
        h_attn = self.attn(self.norm1(h), level=self.block_idx)
        h = h + h_attn
        h = self.soliton(h)
        h_moe = self.moe(self.norm2(h))
        streams = [h_moe]

        # ── 2. Memoire de Travail ─────────────────────────────────────────────
        if self.working_mem is not None:
            h_wm = self.working_mem(h_moe, write=write_memory)
            streams.append(h_wm)

        # ── 3. Fusion ────────────────────────────────────────────────────────
        if self.fusion is not None and len(streams) > 1:
            h_fused = self.fusion(torch.cat(streams, dim=-1))
        else:
            h_fused = streams[0]

        # ── 4. Pilier 2: Graphe Causal ───────────────────────────────────────
        if self.causal is not None:
            h_fused, loss_dag = self.causal(h_fused)
            losses["causal"] = loss_dag * self.cfg.lambda_causal

            if self.training and self.cfg.lambda_counterfactual > 0:
                cf_loss = self.causal.counterfactual_loss(
                    h_fused, n_samples=1, lambda_cf=self.cfg.lambda_counterfactual
                )
                losses["counterfactual"] = cf_loss

        # ── 5. Pilier 3: Self-Model ─────────────────────────────────────────
        if self.self_model is not None:
            h_fused, self_state = self.self_model(h_fused, losses, write=write_memory)
            losses["self_model_coherence"] = self_state.var(dim=-1).mean() * 0.001

        return h_fused, losses