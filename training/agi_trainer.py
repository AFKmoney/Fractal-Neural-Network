"""
NFN AGI Trainer v4.0

Trains AGINFNModel end-to-end with all AGI loss signals.

Key differences from NFNTrainer:
  - Uses AGILoss (lm + causal + goal + ponder + free_energy + consistency + ...)
  - Curriculum learning: LM-only first, then AGI losses phase in after warmup
  - Goal annealing: set_goal() on prompt prefixes periodically
  - Memory consolidation every N steps (episodic → semantic)
  - Gradient clipping per-module (phase params need lower LR)
  - Streaming-aware: handles sequences > cfg.max_seq_len via ChunkedForward
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel, build_agi_model
from nfn.tokenizer import NFNTokenizer
from training.losses import AGILoss


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AGITextDataset:
    """
    Tokenises raw text and yields (input_ids, targets) batches.
    Supports sequences longer than seq_len via a sliding window.
    """

    def __init__(
        self,
        text: str,
        tokenizer: NFNTokenizer,
        seq_len: int,
        batch_size: int,
        stride: Optional[int] = None,   # None → seq_len (no overlap)
    ):
        ids = tokenizer.encode(text, add_bos=True, add_eos=False)
        self.data      = torch.tensor(ids, dtype=torch.long)
        self.seq_len   = seq_len
        self.batch_size = batch_size
        self.stride    = stride or seq_len

    def __len__(self) -> int:
        n = max(0, (len(self.data) - self.seq_len) // self.stride)
        return max(1, n // self.batch_size)

    def iter_batches(self, device: torch.device) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yields (input_ids, targets) pairs, each [B, seq_len]."""
        L = len(self.data)
        starts = list(range(0, max(1, L - self.seq_len), self.stride))
        if not starts:
            starts = [0]
        # Shuffle
        idx = torch.randperm(len(starts)).tolist()
        buf = []
        for i in idx:
            s = starts[i]
            chunk = self.data[s: s + self.seq_len + 1]
            if len(chunk) < 2:
                continue
            if len(chunk) < self.seq_len + 1:
                chunk = F.pad(chunk, (0, self.seq_len + 1 - len(chunk)))
            buf.append(chunk)
            if len(buf) == self.batch_size:
                batch = torch.stack(buf)          # [B, seq_len+1]
                x = batch[:, :-1].to(device)
                y = batch[:, 1:].to(device)
                yield x, y
                buf = []


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

def cosine_with_warmup(
    optimizer: AdamW,
    n_warmup: int,
    n_total: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < n_warmup:
            return step / max(n_warmup, 1)
        t = (step - n_warmup) / max(n_total - n_warmup, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# AGI Trainer
# ─────────────────────────────────────────────────────────────────────────────

class AGITrainer:
    """
    Full AGINFNModel trainer with curriculum learning.

    Curriculum schedule (controlled by agi_loss_start_step):
      Steps 0 → agi_loss_start_step:    LM loss only (stable base)
      Steps > agi_loss_start_step:      All AGI losses enabled, ramping up
        over agi_loss_ramp_steps

    Sinusoidal parameters (omega, phi, A) get 30% of the main LR and no
    weight decay — they are phase parameters, not weights.
    """

    def __init__(
        self,
        model: AGINFNModel,
        tokenizer: NFNTokenizer,
        cfg: Optional[NFNConfig] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        max_grad_norm: float = 1.0,
        dtype: torch.dtype = torch.float32,
        output_dir: str = "checkpoints",
        step_callback: Optional[Callable[[Dict], None]] = None,
        grad_accumulation_steps: int = 1,
        compile_model: bool = False,
        # AGI curriculum
        agi_loss_start_step: int = 500,
        agi_loss_ramp_steps: int = 200,
        # Goal setting: every N steps, set goal from current batch prefix
        goal_set_every: int = 50,
        goal_prefix_len: int = 8,
        # Consolidation
        consolidate_every: int = 100,
    ):
        self.model    = model
        self.tokenizer = tokenizer
        self.cfg      = cfg or model.cfg
        self.max_grad_norm = max_grad_norm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_callback = step_callback
        self.dtype    = dtype
        self._stop    = False
        self.grad_accumulation_steps = max(1, grad_accumulation_steps)

        self.agi_loss_start_step = agi_loss_start_step
        self.agi_loss_ramp_steps = max(1, agi_loss_ramp_steps)
        self.goal_set_every  = goal_set_every
        self.goal_prefix_len = goal_prefix_len
        self.consolidate_every = consolidate_every

        self.device = next(model.parameters()).device

        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                print("torch.compile enabled")
            except Exception as e:
                print(f"torch.compile skipped: {e}")

        # Phase params get separate lower LR
        phase_names = ("A", "omega", "phi", "log_gamma", "phase", "theta")
        phase_params, main_params = [], []
        for name, p in model.named_parameters():
            if any(k in name for k in phase_names):
                phase_params.append(p)
            else:
                main_params.append(p)

        self.optimizer = AdamW(
            [
                {"params": main_params,  "lr": lr,       "weight_decay": weight_decay},
                {"params": phase_params, "lr": lr * 0.3, "weight_decay": 0.0},
            ],
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        self.agi_loss = AGILoss(self.cfg)
        self.step  = 0
        self.history: List[Dict] = []

        use_amp = (dtype == torch.float16) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── AGI loss curriculum ramp ──────────────────────────────────────────────

    def _agi_weight(self) -> float:
        """Returns 0.0 before agi_loss_start_step, ramps to 1.0 over ramp_steps."""
        if self.step < self.agi_loss_start_step:
            return 0.0
        ramp = (self.step - self.agi_loss_start_step) / self.agi_loss_ramp_steps
        return min(1.0, ramp)

    # ── One optimizer step ────────────────────────────────────────────────────

    def _step(
        self,
        batches: List[Tuple[torch.Tensor, torch.Tensor]],
        set_goal: bool = False,
    ) -> Dict[str, float]:
        self.model.train()
        device_type = self.device.type
        accum = len(batches)
        agi_w = self._agi_weight()

        accum_metrics: Dict[str, float] = {}

        for i, (x, y) in enumerate(batches):
            # Set goal from first batch prefix (once per goal_set_every)
            if set_goal and i == 0:
                with torch.no_grad():
                    self.model.set_goal(x[:, :self.goal_prefix_len])

            with torch.autocast(device_type=device_type, dtype=self.dtype,
                                enabled=(self.dtype != torch.float32)):
                logits, losses = self.model(x, targets=y)

                # Pure LM loss
                lm_loss = losses.get("lm", torch.tensor(0.0, device=self.device))

                if agi_w > 0.0:
                    # All AGI losses scaled by curriculum weight
                    total_loss, breakdown = self.agi_loss(losses)
                    # Blend: use pure LM when agi_w=0, full AGI when agi_w=1
                    total_loss = (1 - agi_w) * lm_loss + agi_w * total_loss
                else:
                    total_loss = lm_loss
                    breakdown  = {"lm": lm_loss}

                total_loss = total_loss / accum

            self.scaler.scale(total_loss).backward()

            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    accum_metrics[k] = accum_metrics.get(k, 0.0) + v.item() / accum

        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        accum_metrics["grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        accum_metrics["agi_weight"] = agi_w
        accum_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        return accum_metrics

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save(self, tag: str = "latest") -> str:
        path = self.output_dir / f"agi_nfn_{tag}.pt"
        torch.save(
            {
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "step":            self.step,
                "cfg":             self.cfg.to_dict(),
            },
            path,
        )
        return str(path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "AGITrainer":
        device = device or torch.device("cpu")
        ckpt   = torch.load(path, map_location=device, weights_only=False)
        cfg    = NFNConfig.from_dict(ckpt["cfg"])
        model  = build_agi_model(
            vocab_size  = cfg.vocab_size,
            d_model     = cfg.d_model,
            n_blocks    = cfg.n_blocks,
            use_mod     = cfg.use_mixture_of_depths,
            use_mtp     = cfg.use_multi_token_pred,
            use_hyper   = cfg.use_hyper_net,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        tokenizer = NFNTokenizer()
        trainer = cls(model, tokenizer, cfg)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        trainer.step = ckpt.get("step", 0)
        return trainer

    # ── Main training loop ────────────────────────────────────────────────────

    def train(
        self,
        text: str,
        n_epochs: int = 1,
        seq_len: Optional[int] = None,
        batch_size: int = 4,
        n_warmup_steps: int = 100,
        save_every: int = 500,
        log_every: int = 10,
        eval_text: Optional[str] = None,
    ) -> List[Dict]:
        """
        Train on raw text with full AGI loss curriculum.

        Curriculum:
          - Steps 0 → agi_loss_start_step: LM only (stable language model base)
          - Steps > agi_loss_start_step: All AGI losses enabled with ramp
        """
        if seq_len is None:
            seq_len = min(self.cfg.max_seq_len, 512)

        dataset = AGITextDataset(text, self.tokenizer, seq_len, batch_size)
        n_steps_per_epoch = len(dataset)
        n_total = n_steps_per_epoch * n_epochs

        scheduler = cosine_with_warmup(self.optimizer, n_warmup_steps, n_total)

        t0 = time.time()
        self._stop = False
        accum_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for epoch in range(n_epochs):
            if self._stop:
                break
            for x, y in dataset.iter_batches(self.device):
                if self._stop:
                    break

                accum_buf.append((x, y))
                if len(accum_buf) < self.grad_accumulation_steps:
                    continue

                do_goal = (self.step % self.goal_set_every == 0)
                metrics = self._step(accum_buf, set_goal=do_goal)
                accum_buf = []

                # Reset goal every N steps (prevent goal lock-in)
                if do_goal and self.step > 0:
                    self.model.reset_goal()

                # Trigger episodic→semantic consolidation
                if self.step > 0 and self.step % self.consolidate_every == 0:
                    self._consolidate()

                scheduler.step()
                self.step += 1
                metrics["step"]   = self.step
                metrics["epoch"]  = epoch
                metrics["elapsed"] = time.time() - t0
                self.history.append(metrics)

                if self.step_callback:
                    self.step_callback(metrics)

                if self.step % log_every == 0:
                    lm   = metrics.get("lm",    metrics.get("total", 0.0))
                    causal = metrics.get("causal", 0.0)
                    goal   = metrics.get("goal",   0.0)
                    w      = metrics.get("agi_weight", 0.0)
                    gn     = metrics.get("grad_norm", 0.0)
                    lr     = metrics.get("lr", 0.0)
                    print(
                        f"step {self.step:5d} | lm {lm:.4f} | causal {causal:.5f} "
                        f"| goal {goal:.5f} | agi_w {w:.2f} | gn {gn:.2f} | lr {lr:.2e} "
                        f"| {metrics['elapsed']:.1f}s"
                    )

                if self.step % save_every == 0:
                    ckpt = self.save(f"step{self.step}")
                    print(f"  [saved {ckpt}]")

        self.save("final")
        return self.history

    def _consolidate(self):
        """Trigger episodic→semantic memory consolidation across all blocks."""
        self.model.eval()
        with torch.no_grad():
            for block in self.model.blocks:
                if hasattr(block, "memory") and block.memory is not None:
                    block.memory.maybe_consolidate()
        self.model.train()

    def stop(self):
        self._stop = True

    # ── Quick eval ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def eval_perplexity(self, text: str, seq_len: int = 512) -> float:
        """Compute perplexity on a held-out text string."""
        self.model.eval()
        ids = self.tokenizer.encode(text)[:seq_len + 1]
        if len(ids) < 2:
            return float("inf")
        x = torch.tensor(ids[:-1], device=self.device).unsqueeze(0)
        y = torch.tensor(ids[1:],  device=self.device).unsqueeze(0)
        logits, _ = self.model(x, targets=y)
        nll = F.cross_entropy(
            logits.view(-1, self.cfg.vocab_size),
            y.view(-1),
            ignore_index=self.cfg.pad_token_id,
        )
        self.model.train()
        return math.exp(nll.item())
