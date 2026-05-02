"""
NFN Trainer — Back-Propagation Through Phase (BPTP).

Features:
  - AdamW with cosine LR + warmup
  - Gradient clipping (especially important for sinusoidal param gradients)
  - Mixed-precision (fp16/bf16) support
  - Per-step metrics streamed via callback
  - Checkpoint save/load
  - Training on raw text (auto-chunked) or pre-tokenised tensors
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from nfn.config import NFNConfig
from nfn.network import NFNLanguageModel
from nfn.tokenizer import NFNTokenizer
from training.losses import NFNLoss


# ── Dataset helpers ───────────────────────────────────────────────────────────

class TextDataset:
    """Tokenises a raw string and yields [B, L+1] chunks for next-token prediction."""

    def __init__(
        self,
        text: str,
        tokenizer: NFNTokenizer,
        seq_len: int,
        batch_size: int,
    ):
        ids = tokenizer.encode(text, add_bos=True, add_eos=False)
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
        self.batch_size = batch_size

    def __len__(self) -> int:
        n = (len(self.data) - 1) // self.seq_len
        return n // self.batch_size

    def iter_batches(self, device: torch.device) -> Iterator[Dict[str, torch.Tensor]]:
        chunk_size = self.seq_len + 1
        n_chunks = (len(self.data) - 1) // chunk_size
        if n_chunks == 0:
            return
        # Shuffle chunk order each epoch
        perm = torch.randperm(n_chunks)
        for i in range(0, n_chunks - self.batch_size + 1, self.batch_size):
            batch_idx = perm[i: i + self.batch_size]
            batch = torch.stack([
                self.data[j * chunk_size: j * chunk_size + chunk_size]
                for j in batch_idx
            ])
            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)
            yield {"input_ids": x, "targets": y}


# ── Scheduler ─────────────────────────────────────────────────────────────────

def cosine_with_warmup(
    optimizer: AdamW,
    n_warmup: int,
    n_total: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < n_warmup:
            return step / max(n_warmup, 1)
        progress = (step - n_warmup) / max(n_total - n_warmup, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


# ── Trainer ───────────────────────────────────────────────────────────────────

class NFNTrainer:
    """
    Manages the training loop for an NFNLanguageModel.

    Parameters
    ----------
    model        : NFNLanguageModel
    tokenizer    : NFNTokenizer
    cfg          : NFNConfig
    lr           : learning rate (default 3e-4)
    weight_decay : AdamW weight decay
    max_grad_norm: gradient clipping
    dtype        : torch.float32 | torch.bfloat16 | torch.float16
    output_dir   : where to save checkpoints
    step_callback: called every step with metrics dict (for UI streaming)
    """

    def __init__(
        self,
        model: NFNLanguageModel,
        tokenizer: NFNTokenizer,
        cfg: Optional[NFNConfig] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        max_grad_norm: float = 1.0,
        dtype: torch.dtype = torch.float32,
        output_dir: str = "checkpoints",
        step_callback: Optional[Callable[[Dict], None]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg or model.cfg
        self.max_grad_norm = max_grad_norm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_callback = step_callback
        self.dtype = dtype
        self._stop = False

        self.device = next(model.parameters()).device

        # Separate sinusoidal params for potentially different LR
        sin_params, other_params = [], []
        for name, p in model.named_parameters():
            if any(k in name for k in ("A", "omega", "phi", "log_gamma")):
                sin_params.append(p)
            else:
                other_params.append(p)

        self.optimizer = AdamW(
            [
                {"params": other_params, "lr": lr, "weight_decay": weight_decay},
                {"params": sin_params, "lr": lr * 0.3, "weight_decay": 0.0},
            ],
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        self.loss_fn = NFNLoss(self.cfg)
        self.step = 0
        self.history: List[Dict] = []

        # AMP scaler
        use_amp = (dtype == torch.float16) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save(self, tag: str = "latest"):
        path = self.output_dir / f"nfn_{tag}.pt"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": self.step,
                "cfg": self.cfg.to_dict(),
            },
            path,
        )
        return str(path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "NFNTrainer":
        if device is None:
            device = torch.device("cpu")
        ckpt = torch.load(path, map_location=device)
        from nfn.config import NFNConfig
        from nfn.network import NFNLanguageModel
        from nfn.tokenizer import NFNTokenizer
        cfg = NFNConfig.from_dict(ckpt["cfg"])
        model = NFNLanguageModel(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        tokenizer = NFNTokenizer()
        trainer = cls(model, tokenizer, cfg)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        trainer.step = ckpt["step"]
        return trainer

    # ── Training step ─────────────────────────────────────────────────────────

    def _step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        with torch.autocast(
            device_type=self.device.type if hasattr(self.device, "type") else "cpu",
            dtype=self.dtype,
            enabled=(self.dtype != torch.float32),
        ):
            logits, aux = self.model(batch["input_ids"], targets=batch["targets"])
            losses = aux["loss_aux"]
            loss = losses["total"]

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": losses["task"].item(),
            "loss_total": loss.item(),
            "loss_phase": losses["phase"].item(),
            "loss_freq": losses["freq"].item(),
            "loss_spectral": losses["spectral"].item(),
            "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    # ── Main training loop ────────────────────────────────────────────────────

    def train(
        self,
        text: str,
        n_epochs: int = 1,
        seq_len: Optional[int] = None,
        batch_size: int = 4,
        n_warmup_steps: int = 100,
        eval_text: Optional[str] = None,
        save_every: int = 500,
        log_every: int = 10,
    ):
        """
        Train on raw text.

        Parameters
        ----------
        text       : raw training corpus
        n_epochs   : number of epochs
        seq_len    : context length (defaults to cfg.max_seq_len)
        batch_size : micro-batch size
        """
        if seq_len is None:
            seq_len = self.cfg.max_seq_len

        dataset = TextDataset(text, self.tokenizer, seq_len, batch_size)
        n_steps_per_epoch = len(dataset)
        n_total_steps = n_steps_per_epoch * n_epochs

        scheduler = cosine_with_warmup(self.optimizer, n_warmup_steps, n_total_steps)

        t0 = time.time()
        self._stop = False

        for epoch in range(n_epochs):
            if self._stop:
                break
            for batch in dataset.iter_batches(self.device):
                if self._stop:
                    break

                metrics = self._step(batch)
                scheduler.step()
                self.step += 1
                metrics["step"] = self.step
                metrics["epoch"] = epoch
                metrics["tokens_seen"] = self.step * batch_size * seq_len
                metrics["elapsed"] = time.time() - t0

                self.history.append(metrics)

                if self.step_callback:
                    self.step_callback(metrics)

                if self.step % log_every == 0:
                    print(
                        f"step {self.step:5d} | loss {metrics['loss']:.4f} "
                        f"| phase {metrics['loss_phase']:.5f} "
                        f"| lr {metrics['lr']:.2e} "
                        f"| {metrics['elapsed']:.1f}s"
                    )

                if self.step % save_every == 0:
                    self.save(f"step{self.step}")

        self.save("final")
        return self.history

    def stop(self):
        self._stop = True
