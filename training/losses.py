"""
Multi-objective loss for the NFN — Back-Propagation Through Phase (BPTP).

    L = L_task
      + λ_phase    · L_phase     (phase smoothness / continuity)
      + λ_freq     · L_freq      (sinusoidal amplitude sparsity)
      + λ_spectral · L_spectral  (Jacobian Frobenius — spectral stability)

All regularisation terms are differentiable and computed in a single pass.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nfn.config import NFNConfig
from nfn.connections import SinusoidalAggregator, SinusoidalGate


class NFNLoss:
    """Stateless loss helper — call .regularization() each forward pass."""

    def __init__(self, cfg: NFNConfig):
        self.cfg = cfg

    # ── Phase smoothness ─────────────────────────────────────────────────────

    @staticmethod
    def phase_loss(phases: List[torch.Tensor]) -> torch.Tensor:
        """
        Penalises abrupt phase jumps between consecutive nodes.
        L_phase = mean_level mean_batch Σ_i (1 - cos(θ_i - θ_{i-1}))
        """
        if not phases:
            return torch.tensor(0.0)
        total = torch.tensor(0.0, device=phases[0].device)
        for ph in phases:
            # ph: [B, N]  — differences along the N dimension
            if ph.shape[-1] > 1:
                diff = ph[:, 1:] - ph[:, :-1]    # [B, N-1]
                total = total + (1.0 - torch.cos(diff)).mean()
        return total / max(len(phases), 1)

    # ── Frequency amplitude sparsity ─────────────────────────────────────────

    @staticmethod
    def freq_loss(model: nn.Module) -> torch.Tensor:
        """L_freq = Σ_{gates} ||A||_1 — encourages sparse amplitude usage."""
        parts = [m.A.abs().mean() for m in model.modules() if isinstance(m, SinusoidalGate)]
        if not parts:
            return torch.tensor(0.0)
        return torch.stack(parts).mean()

    # ── Spectral stability (Jacobian Frobenius) ───────────────────────────────

    @staticmethod
    def spectral_loss(model: nn.Module) -> torch.Tensor:
        """Approximates ||J||_F via squared Frobenius norm of linear weights."""
        parts = [(m.weight ** 2).mean() for m in model.modules() if isinstance(m, nn.Linear)]
        if not parts:
            return torch.tensor(0.0)
        return torch.stack(parts).mean()

    # ── Combined regularisation ───────────────────────────────────────────────

    def regularization(
        self,
        phases: List[torch.Tensor],
        model: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        device = phases[0].device if phases else torch.device("cpu")
        cfg = self.cfg

        lp = self.phase_loss(phases).to(device)
        lf = self.freq_loss(model).to(device)
        ls = self.spectral_loss(model).to(device)

        total = (cfg.lambda_phase * lp
                 + cfg.lambda_freq * lf
                 + cfg.lambda_spectral * ls)

        return {
            "phase": lp,
            "freq": lf,
            "spectral": ls,
            "total_reg": total,
        }

    # ── Task loss ─────────────────────────────────────────────────────────────

    @staticmethod
    def cross_entropy(
        logits: torch.Tensor,   # [B, L, V]
        targets: torch.Tensor,  # [B, L]
        ignore_index: int = 0,
    ) -> torch.Tensor:
        B, L, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
            ignore_index=ignore_index,
        )
