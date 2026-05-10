"""
OnlineLearner — Test-Time Adaptation via LoRA Fast Weights

The main model weights are NEVER modified. Instead, a thin overlay of LoRA
adapters sits on top of the attention projections. These adapters update via a
few gradient steps whenever the model encounters text it finds surprising
(high perplexity). Between sessions they decay exponentially toward zero.

This gives the model a form of "working memory at the weight level":
  - Fast: 3–5 gradient steps on O(rank × d) params, not millions
  - Safe: main weights are frozen — no catastrophic forgetting
  - Forgetful by design: decay factor pulls adapters back toward the base model
  - Confident: gated by perplexity — model only updates on things it doesn't know

Typical usage:
    learner = OnlineLearner(agi_model, tokenizer, adapter_rank=8)
    # … at inference time, after reading new text:
    stats = learner.adapt_from_text("new context the model hasn't seen …")
    output = learner.generate("question about new context")
    # … at session end:
    learner.save_adapters("session.pt")
    # … next session:
    learner.load_adapters("session.pt")
    learner.decay(steps=100)   # simulate time passing
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nfn.agi_model import AGINFNModel
from nfn.tokenizer import NFNTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# LoRA wrapper
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with low-rank adapter matrices.
    Output = original(x) + (x @ A.T @ B.T) * (alpha / rank)

    The original weight is frozen. Only A and B are trainable.
    A is initialised with small noise; B is zero → adapter starts silent.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float = 8.0):
        super().__init__()
        self.linear  = linear
        self.rank    = rank
        self.scaling = alpha / rank

        d_in  = linear.in_features
        d_out = linear.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

        # Freeze original
        for p in self.linear.parameters():
            p.requires_grad_(False)

    # Proxy weight/bias so nn.MultiheadAttention can still read them directly
    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        delta = (x @ self.lora_A.T) @ self.lora_B.T
        return base + delta * self.scaling

    def decay_(self, factor: float) -> None:
        with torch.no_grad():
            self.lora_A.mul_(factor)
            self.lora_B.mul_(factor)

    def reset_(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.lora_A, std=0.02)
            self.lora_B.zero_()

    @property
    def adapter_norm(self) -> float:
        return (self.lora_A.norm() + self.lora_B.norm()).item() / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Online Learner
# ─────────────────────────────────────────────────────────────────────────────

class OnlineLearner:
    """
    Test-time adaptation wrapper for AGINFNModel.

    Injects LoRA adapters into all attention and projection layers, then
    exposes adapt() for in-context weight updates.

    Parameters
    ----------
    model            : frozen AGINFNModel
    tokenizer        : NFNTokenizer
    adapter_rank     : LoRA rank (higher = more capacity, more compute)
    alpha            : LoRA scaling = alpha / rank
    online_lr        : learning rate for adapter updates
    n_steps          : gradient steps per adapt() call
    decay_factor     : multiplicative decay applied after each adapt() call
                       (1.0 = no decay, 0.95 = forget ~5% per call)
    ppl_gate         : perplexity threshold — skip update if ppl < this value
                       (model already knows this content)
    max_adapt_tokens : maximum tokens per adapt() call
    """

    # Layer name suffixes that are candidates for adaptation
    _TARGET_SUFFIXES = (
        "qkv", "out", "q_proj", "k_proj", "v_proj", "out_proj",
        "to_q", "to_k", "to_v", "to_out",
        "w1", "w2", "fc1", "fc2",
    )

    def __init__(
        self,
        model:            AGINFNModel,
        tokenizer:        NFNTokenizer,
        adapter_rank:     int   = 8,
        alpha:            float = 16.0,
        online_lr:        float = 2e-4,
        n_steps:          int   = 4,
        decay_factor:     float = 0.97,
        ppl_gate:         float = 30.0,
        max_adapt_tokens: int   = 256,
    ):
        self.model            = model
        self.tokenizer        = tokenizer
        self.n_steps          = n_steps
        self.decay_factor     = decay_factor
        self.ppl_gate         = ppl_gate
        self.max_adapt_tokens = max_adapt_tokens

        self.adapters: List[LoRALinear] = []
        self._inject(adapter_rank, alpha)

        adapter_params = [p for a in self.adapters for p in a.parameters()
                          if p.requires_grad]
        self.optimizer = torch.optim.Adam(adapter_params, lr=online_lr,
                                          betas=(0.9, 0.95), eps=1e-8)

        # Statistics
        self.n_adapt_calls    = 0
        self.n_skipped        = 0
        self.total_loss       = 0.0

    # ── Adapter injection ─────────────────────────────────────────────────────

    def _inject(self, rank: int, alpha: float) -> None:
        device = next(self.model.parameters()).device

        def _should_adapt(name: str) -> bool:
            return any(name == s or name.endswith("." + s)
                       for s in self._TARGET_SUFFIXES)

        for module in self.model.modules():
            for attr_name, child in list(module.named_children()):
                if isinstance(child, nn.Linear) and _should_adapt(attr_name):
                    lora = LoRALinear(child, rank, alpha).to(device)
                    setattr(module, attr_name, lora)
                    self.adapters.append(lora)

    # ── Perplexity (no grad) ──────────────────────────────────────────────────

    @torch.no_grad()
    def _perplexity(self, ids: torch.Tensor) -> float:
        """Quick single-forward perplexity for confidence gating."""
        if ids.numel() < 2:
            return float("inf")
        x = ids[:-1].unsqueeze(0)
        y = ids[1:].unsqueeze(0)
        self.model.eval()
        logits, _ = self.model(x, write_memory=False)
        nll = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=self.model.cfg.pad_token_id,
        )
        return math.exp(min(nll.item(), 20.0))   # cap at e^20 to avoid inf

    # ── Core adaptation step ──────────────────────────────────────────────────

    def adapt(self, ids: torch.Tensor) -> Dict[str, object]:
        """
        Run n_steps gradient updates on adapter params for the given token ids.

        Parameters
        ----------
        ids : torch.Tensor
            Shape [L] or [1, L]. Will be clipped to max_adapt_tokens.

        Returns
        -------
        dict with keys: ppl, skipped, steps, loss, adapter_norm
        """
        self.n_adapt_calls += 1

        if ids.dim() == 2:
            ids = ids[0]
        ids = ids[:self.max_adapt_tokens + 1].to(
            next(self.model.parameters()).device
        )

        ppl = self._perplexity(ids)

        if ppl < self.ppl_gate:
            self.n_skipped += 1
            return {"ppl": ppl, "skipped": True, "steps": 0, "loss": 0.0,
                    "adapter_norm": self._mean_adapter_norm()}

        x = ids[:-1].unsqueeze(0)
        y = ids[1:].unsqueeze(0)

        self.model.train()
        final_loss = 0.0
        for _ in range(self.n_steps):
            self.optimizer.zero_grad(set_to_none=True)
            logits, _ = self.model(x, write_memory=False)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=self.model.cfg.pad_token_id,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for a in self.adapters for p in a.parameters()], 1.0
            )
            self.optimizer.step()
            final_loss = loss.item()

        self.model.eval()
        self.total_loss += final_loss

        self.decay(steps=1)

        return {
            "ppl":          ppl,
            "skipped":      False,
            "steps":        self.n_steps,
            "loss":         final_loss,
            "adapter_norm": self._mean_adapter_norm(),
        }

    def adapt_from_text(self, text: str) -> Dict[str, object]:
        """Tokenise text and call adapt()."""
        ids = torch.tensor(
            self.tokenizer.encode(text, add_bos=True),
            dtype=torch.long,
        )
        return self.adapt(ids)

    # ── Decay / reset ─────────────────────────────────────────────────────────

    def decay(self, steps: int = 1) -> None:
        """Apply exponential decay to all adapters (simulate forgetting over time)."""
        factor = self.decay_factor ** steps
        for a in self.adapters:
            a.decay_(factor)

    def reset(self) -> None:
        """Zero all adapters — full forgetting."""
        for a in self.adapters:
            a.reset_()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_adapters(self, path: str) -> None:
        """Save only the adapter weights (not the base model)."""
        state = {
            f"{i}": {"A": a.lora_A.data, "B": a.lora_B.data}
            for i, a in enumerate(self.adapters)
        }
        torch.save(state, path)

    def load_adapters(self, path: str) -> None:
        """Load adapter weights saved by save_adapters()."""
        state = torch.load(path, map_location=next(self.model.parameters()).device,
                           weights_only=True)
        for i, a in enumerate(self.adapters):
            key = str(i)
            if key in state:
                a.lora_A.data.copy_(state[key]["A"])
                a.lora_B.data.copy_(state[key]["B"])

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _mean_adapter_norm(self) -> float:
        if not self.adapters:
            return 0.0
        return sum(a.adapter_norm for a in self.adapters) / len(self.adapters)

    def stats(self) -> Dict[str, object]:
        n_adapter_params = sum(
            a.lora_A.numel() + a.lora_B.numel() for a in self.adapters
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "n_adapters":      len(self.adapters),
            "adapter_params":  n_adapter_params,
            "adapter_ratio":   f"{100 * n_adapter_params / max(total_params, 1):.2f}%",
            "adapt_calls":     self.n_adapt_calls,
            "skipped":         self.n_skipped,
            "mean_adapter_norm": self._mean_adapter_norm(),
            "avg_loss":        self.total_loss / max(self.n_adapt_calls - self.n_skipped, 1),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"OnlineLearner("
                f"adapters={s['n_adapters']}, "
                f"params={s['adapter_params']:,} ({s['adapter_ratio']}), "
                f"calls={s['adapt_calls']}, skipped={s['skipped']})")
