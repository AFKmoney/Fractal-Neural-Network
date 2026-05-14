"""
Multi-objective loss for NFN AGI v5.0 — Back-Propagation Through Phase (BPTP).

    L = L_task
      + λ_phase      · L_phase       (phase smoothness / continuity)
      + λ_freq       · L_freq        (sinusoidal amplitude sparsity)
      + λ_spectral   · L_spectral    (Jacobian Frobenius — spectral stability)

AGI v4.0:
      + λ_causal     · L_causal      (DAG sparsity across blocks)
      + λ_goal       · L_goal        (phase-goal alignment)
      + λ_coherence  · L_coherence   (cross-modal phase coherence)
      + λ_ponder     · L_ponder      (ACT halting efficiency)
      + λ_pred       · L_pred        (predictive coding error)
      + λ_fe         · L_fe          (free energy / ELBO)

AGI v5.0 additions:
      + λ_value      · L_value       (value function TD error)
      + λ_intrinsic  · L_intrinsic   (forward model curiosity)
      + λ_tom        · L_tom         (theory of mind prediction)
      + λ_notears    · L_notears     (NOTEARS acyclicity penalty)
      + λ_cf         · L_cf          (counterfactual consistency)

All regularisation terms are differentiable and computed in a single pass.
Gradient tracking: per-signal gradient norms logged for diagnostic purposes.
"""

from typing import Dict, List, Optional, Tuple

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


class AGILoss:
    """
    Full AGI v5.0 loss aggregator.

    Combines all loss signals from AGINFNModel.forward() into a single
    weighted scalar with per-component tracking for logging.

    v5.0 additions:
      - value loss (TD error for value function)
      - intrinsic loss (forward model curiosity)
      - theory of mind loss
      - NOTEARS acyclicity + counterfactual consistency losses
      - per-signal gradient norm tracking

    Usage:
        criterion = AGILoss(cfg)
        loss, breakdown = criterion(model_losses)
        loss.backward()
    """

    def __init__(self, cfg: NFNConfig):
        self.cfg = cfg
        # Running EMA of per-signal losses for adaptive weighting
        self._ema: Dict[str, float] = {}
        self._ema_alpha = 0.95

    def _update_ema(self, key: str, val: float) -> float:
        if key not in self._ema:
            self._ema[key] = val
        else:
            self._ema[key] = self._ema_alpha * self._ema[key] + (1 - self._ema_alpha) * val
        return self._ema[key]

    def __call__(
        self,
        model_losses: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns (total_loss, breakdown_dict)."""
        device = next(iter(model_losses.values())).device
        breakdown: Dict[str, torch.Tensor] = {}

        # v4.0 signals
        v4_keys = ("lm", "causal", "goal", "coherence", "ponder", "pred", "free_energy", "consistency")
        for k in v4_keys:
            breakdown[k] = model_losses.get(k, torch.tensor(0.0, device=device))

        # v5.0 signals (with configurable weights)
        cfg = self.cfg
        v5_weighted = {
            "value":           (model_losses.get("value",    torch.tensor(0.0, device=device)),
                                getattr(cfg, "lambda_value",       0.01)),
            "intrinsic":       (model_losses.get("intrinsic", torch.tensor(0.0, device=device)),
                                getattr(cfg, "lambda_intrinsic",   0.05)),
            "tom":             (model_losses.get("tom",       torch.tensor(0.0, device=device)),
                                getattr(cfg, "lambda_tom",         0.05)),
            "notears":         (model_losses.get("notears",   torch.tensor(0.0, device=device)),
                                getattr(cfg, "lambda_notears",     0.001)),
            "counterfactual":  (model_losses.get("counterfactual", torch.tensor(0.0, device=device)),
                                getattr(cfg, "lambda_counterfactual", 0.01)),
        }
        for k, (loss, weight) in v5_weighted.items():
            breakdown[k] = loss * weight

        total = sum(breakdown.values())
        breakdown["total"] = total

        # Update EMAs for monitoring
        for k, v in breakdown.items():
            if k != "total":
                self._update_ema(k, v.item())

        return total, breakdown

    def get_ema_breakdown(self) -> Dict[str, float]:
        """Returns EMA-smoothed per-signal losses for monitoring."""
        return dict(self._ema)

    @staticmethod
    def log_breakdown(breakdown: Dict[str, torch.Tensor], step: int, prefix: str = "train") -> str:
        parts = [f"step={step}"]
        for k, v in breakdown.items():
            if isinstance(v, torch.Tensor) and abs(v.item()) > 1e-8:
                parts.append(f"{prefix}/{k}={v.item():.4f}")
        return "  ".join(parts)

    @staticmethod
    def per_signal_grad_norms(
        model: nn.Module,
        breakdown: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute gradient norm attributable to each loss signal.

        This is an approximation: we check which parameters have gradients
        and report the total gradient norm. For true per-signal attribution,
        one would need separate backward() calls (expensive).

        Used for diagnosing which signals dominate training.
        """
        total_norm = 0.0
        n_params   = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item() ** 2
                n_params   += 1
        total_norm = total_norm ** 0.5

        # Approximate attribution by loss magnitude (proportional)
        total_loss = sum(v.item() for v in breakdown.values() if isinstance(v, torch.Tensor) and v.requires_grad)
        if total_loss < 1e-8:
            return {}

        return {
            f"grad_norm_{k}": total_norm * v.item() / max(total_loss, 1e-8)
            for k, v in breakdown.items()
            if isinstance(v, torch.Tensor) and k != "total" and abs(v.item()) > 1e-8
        }
