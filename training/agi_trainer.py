"""
NFN AGI Trainer v5.0

Training system for AGINFNModel that goes beyond standard next-token prediction.

Core features:
  1. Multi-objective loss with curriculum ramp (LM → all AGI losses)
  2. Self-play improvement loop with DPO-lite preference learning
  3. Constitutional self-critique (generate → critique → revise → train on revision)
  4. WAKE/SLEEP memory cycle (WAKE: write memory every step; SLEEP: consolidate)
  5. Curiosity-driven loss weighting (upweight high-entropy / surprising tokens)
  6. Intrinsic memory-use reward (reward memory retrievals that actually help)

These are genuine algorithmic changes to what the model trains on — not just
comments describing what they would do.
"""

import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel, build_agi_model
from nfn.tokenizer import NFNTokenizer
from nfn.online_learner import OnlineLearner
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
        text:       str,
        tokenizer:  NFNTokenizer,
        seq_len:    int,
        batch_size: int,
        stride:     Optional[int] = None,
    ):
        ids = tokenizer.encode(text, add_bos=True, add_eos=False)
        self.data       = torch.tensor(ids, dtype=torch.long)
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.stride     = stride or seq_len

    def __len__(self) -> int:
        n = max(0, (len(self.data) - self.seq_len) // self.stride)
        return max(1, n // self.batch_size)

    def iter_batches(self, device: torch.device) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yields (input_ids, targets) pairs, each [B, seq_len]."""
        L      = len(self.data)
        starts = list(range(0, max(1, L - self.seq_len), self.stride))
        if not starts:
            starts = [0]
        idx = torch.randperm(len(starts)).tolist()
        buf: List[torch.Tensor] = []
        for i in idx:
            s     = starts[i]
            chunk = self.data[s: s + self.seq_len + 1]
            if len(chunk) < 2:
                continue
            if len(chunk) < self.seq_len + 1:
                chunk = F.pad(chunk, (0, self.seq_len + 1 - len(chunk)))
            buf.append(chunk)
            if len(buf) == self.batch_size:
                batch = torch.stack(buf)      # [B, seq_len+1]
                x = batch[:, :-1].to(device)
                y = batch[:, 1:].to(device)
                yield x, y
                buf = []


# ─────────────────────────────────────────────────────────────────────────────
# Self-play buffer
# ─────────────────────────────────────────────────────────────────────────────

class SelfPlayBuffer:
    """
    Circular buffer of (prompt_ids, winner_ids, loser_ids, score_delta) tuples.

    Populated by the self-play step; can be consumed for offline DPO replay.
    score_delta = winner_score - loser_score (always >= 0 by construction).
    """

    def __init__(self, capacity: int = 256):
        self._buf: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        prompt_ids:  torch.Tensor,
        winner_ids:  torch.Tensor,
        loser_ids:   torch.Tensor,
        score_delta: float,
    ):
        # Store on CPU to avoid occupying VRAM between steps
        self._buf.append((
            prompt_ids.cpu(),
            winner_ids.cpu(),
            loser_ids.cpu(),
            score_delta,
        ))

    def sample(
        self, n: int, device: torch.device
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]]:
        if len(self._buf) < n:
            return None
        idx      = torch.randperm(len(self._buf))[:n].tolist()
        buf_list = list(self._buf)
        return [
            (buf_list[i][0].to(device),
             buf_list[i][1].to(device),
             buf_list[i][2].to(device),
             buf_list[i][3])
            for i in idx
        ]

    def __len__(self) -> int:
        return len(self._buf)


# ─────────────────────────────────────────────────────────────────────────────
# Curiosity weighter
# ─────────────────────────────────────────────────────────────────────────────

class CuriosityWeighter:
    """
    Computes per-example curiosity weights from output logit entropy.

    High entropy  → model is uncertain about that example → up-weight it.
    Low entropy   → model is confident about that example → down-weight it.

    Weights are normalised across the batch (sum = 1), so they replace the
    standard mean reduction without changing the loss scale when curiosity is
    uniform across the batch.
    """

    def __init__(self, tau: float = 1.0):
        self.tau = max(tau, 1e-5)

    def weights(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, L, V]
        Returns:
            w: [B] — per-example weight, sums to 1 across batch
        """
        with torch.no_grad():
            probs   = F.softmax(logits.detach(), dim=-1)           # [B, L, V]
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # [B, L]
            mean_h  = entropy.mean(dim=-1)                          # [B]
            w       = F.softmax(mean_h / self.tau, dim=0)          # [B]
        return w

    def weighted_loss(
        self,
        logits:  torch.Tensor,   # [B, L, V]
        targets: torch.Tensor,   # [B, L]
        pad_id:  int = 0,
    ) -> torch.Tensor:
        """
        Curiosity-weighted cross-entropy. Returns scalar.
        """
        w       = self.weights(logits)   # [B]
        B, L, V = logits.shape

        per_tok = F.cross_entropy(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
            ignore_index=pad_id,
            reduction="none",
        ).reshape(B, L)                                              # [B, L]

        valid  = (targets != pad_id).float()
        denom  = valid.sum(dim=-1).clamp(min=1)
        per_ex = (per_tok * valid).sum(dim=-1) / denom              # [B]

        return (w * per_ex).sum()


# ─────────────────────────────────────────────────────────────────────────────
# DPO-lite helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sequence_log_prob(
    model:  AGINFNModel,
    ids:    torch.Tensor,   # [1, L]
    pad_id: int = 0,
) -> torch.Tensor:
    """
    Sum of log-probs of `ids[1:]` under the model given `ids[:-1]` as context.
    Returns a scalar tensor (with grad when called inside autocast).
    """
    assert ids.dim() == 2 and ids.shape[0] == 1
    if ids.shape[1] < 2:
        return torch.tensor(0.0, device=ids.device)
    targets = ids[:, 1:]
    logits, _ = model(ids, targets=targets, write_memory=False)
    lp        = F.log_softmax(logits[:, :-1, :], dim=-1)   # [1, L-1, V]
    tgt       = ids[:, 1:]                                   # [1, L-1]
    mask      = (tgt != pad_id).float()
    return (lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1) * mask).sum()


def dpo_loss(
    log_p_winner: torch.Tensor,
    log_p_loser:  torch.Tensor,
    beta:         float = 0.1,
) -> torch.Tensor:
    """
    DPO-lite: -log sigmoid(beta * (log_p_winner - log_p_loser))

    No reference model required — implicit uniform reference.
    Pushes the model to assign higher likelihood to the winner sequence.
    """
    return -F.logsigmoid(beta * (log_p_winner - log_p_loser))


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate schedule
# ─────────────────────────────────────────────────────────────────────────────

def cosine_with_warmup(
    optimizer:    AdamW,
    n_warmup:     int,
    n_total:      int,
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
    Full AGINFNModel trainer with genuine AGI-specific training features.

    Training loop (per optimizer step):
      1. WAKE forward (write_memory=True) + curiosity-weighted AGI loss
      2. Every self_play_every steps:  self-play + DPO-lite update
      3. Every critique_every steps:   constitutional critique step
      4. Every sleep_every steps:      SLEEP cycle (consolidation + replay)
      5. Every goal_set_every steps:   set_goal from batch prefix

    Curriculum schedule:
      Steps 0 → agi_loss_start_step: LM loss only (stable language base)
      Steps agi_loss_start_step + agi_loss_ramp_steps+: all AGI losses at full weight
      (linear ramp in between)
    """

    def __init__(
        self,
        model:          AGINFNModel,
        tokenizer:      NFNTokenizer,
        cfg:            Optional[NFNConfig] = None,
        lr:             float = 3e-4,
        weight_decay:   float = 0.1,
        max_grad_norm:  float = 1.0,
        dtype:          torch.dtype = torch.float32,
        output_dir:     str = "checkpoints",
        step_callback:  Optional[Callable[[Dict], None]] = None,
        grad_accumulation_steps: int = 1,
        # Curriculum
        agi_loss_start_step: int = 200,
        agi_loss_ramp_steps: int = 100,
        # Self-play
        self_play_every:    int   = 50,
        n_candidates:       int   = 4,
        dpo_beta:           float = 0.1,
        dpo_loss_weight:    float = 0.3,
        # Constitutional
        critique_every:     int   = 100,
        critique_weight:    float = 2.0,
        # Memory cycle
        sleep_every:        int   = 200,
        sleep_replay_steps: int   = 10,
        # Curiosity
        curiosity_tau:      float = 1.0,
        curiosity_weight:   float = 0.3,
        # Goal
        goal_set_every:     int   = 25,
        goal_prefix_len:    int   = 8,
        # Feature flags
        use_self_play:      bool  = True,
        use_critique:       bool  = True,
        use_sleep:          bool  = True,
        use_curiosity:      bool  = True,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg or model.cfg
        self.max_grad_norm           = max_grad_norm
        self.output_dir              = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_callback           = step_callback
        self.dtype                   = dtype
        self._stop                   = False
        self.grad_accumulation_steps = max(1, grad_accumulation_steps)

        # Curriculum
        self.agi_loss_start_step = agi_loss_start_step
        self.agi_loss_ramp_steps = max(1, agi_loss_ramp_steps)

        # Self-play
        self.use_self_play   = use_self_play
        self.self_play_every = self_play_every
        self.n_candidates    = n_candidates
        self.dpo_beta        = dpo_beta
        self.dpo_loss_weight = dpo_loss_weight
        self.sp_buffer       = SelfPlayBuffer(capacity=512)

        # Constitutional critique
        self.use_critique    = use_critique
        self.critique_every  = critique_every
        self.critique_weight = critique_weight

        # SLEEP / memory consolidation
        self.use_sleep          = use_sleep
        self.sleep_every        = sleep_every
        self.sleep_replay_steps = sleep_replay_steps
        # Rolling buffer of recent (x, y) CPU tensors for SLEEP replay
        self._replay_buf: Deque[Tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=32)

        # Curiosity weighting
        self.use_curiosity    = use_curiosity
        self.curiosity_weight = curiosity_weight
        self.curiosity        = CuriosityWeighter(tau=curiosity_tau)

        # Goal
        self.goal_set_every  = goal_set_every
        self.goal_prefix_len = goal_prefix_len

        # Online learning (test-time adaptation)
        self.use_online_learning = False
        self._online_learner: Optional[OnlineLearner] = None

        self.device = next(model.parameters()).device

        # Phase parameters get a lower learning rate and no weight decay —
        # they represent oscillatory phase dynamics, not information weights.
        phase_kws    = ("A", "omega", "phi", "log_gamma", "phase", "theta")
        phase_params, main_params = [], []
        for name, p in model.named_parameters():
            if any(k in name for k in phase_kws):
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
        self.step     = 0
        self.history: List[Dict] = []

        use_amp = (dtype == torch.float16) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Curriculum weight ─────────────────────────────────────────────────────

    def _agi_weight(self) -> float:
        """0 before agi_loss_start_step; linearly ramps to 1 over ramp_steps."""
        if self.step < self.agi_loss_start_step:
            return 0.0
        ramp = (self.step - self.agi_loss_start_step) / self.agi_loss_ramp_steps
        return min(1.0, ramp)

    # ── Main forward pass (WAKE) ──────────────────────────────────────────────

    def _forward_step(
        self,
        x: torch.Tensor,   # [B, L]
        y: torch.Tensor,   # [B, L]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        One forward pass with memory writes (WAKE phase).
        Applies curiosity-weighted loss blended into the AGI objective.
        Returns (total_loss tensor, metrics dict).
        """
        agi_w       = self._agi_weight()
        device_type = self.device.type

        with torch.autocast(device_type=device_type, dtype=self.dtype,
                            enabled=(self.dtype != torch.float32)):
            # WAKE: write_memory=True so episodic buffer fills every step
            logits, losses = self.model(x, targets=y, write_memory=True)

            lm_loss = losses.get("lm", torch.tensor(0.0, device=self.device))

            if agi_w > 0.0:
                agi_total, _ = self.agi_loss(losses)
                # Blend LM-only phase with full AGI phase
                base_loss = (1.0 - agi_w) * lm_loss + agi_w * agi_total
            else:
                base_loss = lm_loss

            # Curiosity weighting: replace raw LM contribution with entropy-
            # weighted version to upweight surprising/hard examples.
            if self.use_curiosity and agi_w > 0.0:
                curious_lm = self.curiosity.weighted_loss(
                    logits, y, pad_id=self.cfg.pad_token_id
                )
                # Add the curiosity correction (positive when model is confused)
                base_loss = base_loss + self.curiosity_weight * (curious_lm - lm_loss.detach())

        metrics: Dict[str, float] = {
            k: v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)
        }
        metrics["agi_weight"] = agi_w
        return base_loss, metrics

    # ── Self-play step ────────────────────────────────────────────────────────

    def _self_play_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Self-play improvement loop:
          1. For each batch item, generate n_candidates completions with temperature
          2. Score each via -LM_loss (higher = model prefers it = better)
          3. Select best (winner) and worst (loser)
          4. DPO-lite: push log P(winner) > log P(loser)
          5. Distillation: cross-entropy toward winner tokens
          6. Store pair in self-play buffer for potential offline replay

        This is a genuine training signal: it forces the model to differentiate
        among its own outputs and prefer the ones it deems most likely.
        """
        if not self.use_self_play:
            return {}

        self.model.eval()
        B            = x.shape[0]
        dpo_losses:     List[torch.Tensor] = []
        distill_losses: List[torch.Tensor] = []

        for b in range(B):
            prompt = x[b: b + 1]   # [1, L]

            candidates: List[torch.Tensor] = []
            scores:     List[float]        = []

            # Generate candidates (no grad)
            with torch.no_grad():
                for _ in range(self.n_candidates):
                    cand = self.model.generate(
                        prompt,
                        max_new_tokens=min(32, x.shape[1] // 2),
                        temperature=1.2,
                        top_k=40,
                        top_p=0.95,
                    )
                    # Score = -lm_loss on the full sequence (prompt + completion)
                    if cand.shape[1] > 1:
                        tgt_c   = cand[:, 1:]
                        _, lss  = self.model(cand, targets=tgt_c, write_memory=False)
                        score   = -lss["lm"].item()
                    else:
                        score = -1e9
                    candidates.append(cand)
                    scores.append(score)

            best_idx  = int(max(range(len(scores)), key=lambda i: scores[i]))
            worst_idx = int(min(range(len(scores)), key=lambda i: scores[i]))

            if scores[best_idx] <= scores[worst_idx]:
                # Degenerate: all candidates identical — no useful signal
                continue

            winner = candidates[best_idx]
            loser  = candidates[worst_idx]
            self.sp_buffer.push(prompt, winner, loser,
                                scores[best_idx] - scores[worst_idx])

            # Gradient computations — model must be in train mode
            self.model.train()
            with torch.autocast(device_type=self.device.type, dtype=self.dtype,
                                enabled=(self.dtype != torch.float32)):
                log_p_w = _sequence_log_prob(self.model, winner, self.cfg.pad_token_id)
                log_p_l = _sequence_log_prob(self.model, loser,  self.cfg.pad_token_id)
                dpo     = dpo_loss(log_p_w, log_p_l, beta=self.dpo_beta)
                dpo_losses.append(dpo)

                # Soft distillation toward winner: train model to predict winner tokens
                win_tgt    = winner[:, 1:]
                win_logits, _ = self.model(winner, targets=win_tgt, write_memory=False)
                dist_loss  = F.cross_entropy(
                    win_logits[:, :-1].reshape(-1, self.cfg.vocab_size),
                    win_tgt.reshape(-1),
                    ignore_index=self.cfg.pad_token_id,
                )
                distill_losses.append(dist_loss)

            self.model.eval()

        self.model.train()

        sp_metrics: Dict[str, float] = {}
        if dpo_losses:
            dpo_total     = torch.stack(dpo_losses).mean()
            distill_total = torch.stack(distill_losses).mean()
            total_sp      = dpo_total + 0.5 * distill_total
            sp_metrics["sp_dpo"]     = dpo_total.item()
            sp_metrics["sp_distill"] = distill_total.item()
            sp_metrics["sp_total"]   = total_sp.item()
            self.scaler.scale(total_sp * self.dpo_loss_weight).backward()

        return sp_metrics

    # ── Constitutional critique step ──────────────────────────────────────────

    def _critique_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Constitutional self-critique loop:
          1. Generate a completion from the prompt (temperature=0.9)
          2. Append "[CRITIQUE]" marker and generate a critique
          3. Append "[REVISION]" marker and generate a revised completion
          4. Train the model on the revision with elevated weight

        This changes what the model trains on: instead of whatever came next
        in the corpus, it trains on a model-critiqued, model-revised version.
        Over many steps, this steers the model toward outputs that it itself
        judges to be improvable — a genuine self-improvement signal.
        """
        if not self.use_critique:
            return {}

        self.model.eval()
        # Process a single example per critique step for compute efficiency
        prompt = x[0:1]   # [1, L]

        crit_ids = torch.tensor(
            self.tokenizer.encode("\n[CRITIQUE]"),
            dtype=torch.long, device=self.device,
        ).unsqueeze(0)
        rev_ids  = torch.tensor(
            self.tokenizer.encode("\n[REVISION]"),
            dtype=torch.long, device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            # Step 1: generate a completion
            completion = self.model.generate(
                prompt,
                max_new_tokens=min(48, x.shape[1] // 2),
                temperature=0.9,
                top_k=50,
            )

            # Step 2: generate a critique of the completion
            critique_ctx = torch.cat([completion, crit_ids], dim=1)
            critique_ctx = critique_ctx[:, -x.shape[1]:]     # clamp to seq_len
            critique     = self.model.generate(
                critique_ctx,
                max_new_tokens=min(32, x.shape[1] // 4),
                temperature=0.8,
                top_k=50,
            )

            # Step 3: generate a revision based on the critique
            revision_ctx = torch.cat([critique, rev_ids], dim=1)
            revision_ctx = revision_ctx[:, -x.shape[1]:]
            revision     = self.model.generate(
                revision_ctx,
                max_new_tokens=min(48, x.shape[1] // 2),
                temperature=0.7,
                top_k=40,
            )

        if revision.shape[1] < 2:
            self.model.train()
            return {}

        # Train on the revision (elevated weight signals it is the preferred target)
        self.model.train()
        rev_targets = revision[:, 1:]

        with torch.autocast(device_type=self.device.type, dtype=self.dtype,
                            enabled=(self.dtype != torch.float32)):
            rev_logits, _ = self.model(revision, targets=rev_targets, write_memory=False)
            # Trim logits to align with targets
            L_tgt          = rev_targets.shape[1]
            rev_logits_trim = rev_logits[:, :L_tgt, :]
            crit_loss = F.cross_entropy(
                rev_logits_trim.reshape(-1, self.cfg.vocab_size),
                rev_targets.reshape(-1),
                ignore_index=self.cfg.pad_token_id,
            )

        self.scaler.scale(crit_loss * self.critique_weight).backward()
        return {"critique_loss": crit_loss.item()}

    # ── SLEEP cycle ───────────────────────────────────────────────────────────

    def _sleep_cycle(self):
        """
        SLEEP phase — two-stage memory consolidation:

        Stage 1 (consolidation): call maybe_consolidate() on every AGIBlock's
          TwoTierMemory. This triggers the episodic → semantic transfer when
          patterns have been seen often enough (consolidation_freq threshold).

        Stage 2 (replay): re-run recent training batches through the model with
          write_memory=True but no gradient. This lets the memory modules
          re-process their recent inputs and strengthen the semantic condensate —
          analogous to hippocampal replay during biological sleep.

        Neither stage computes gradients, so this is purely memory management.
        """
        if not self.use_sleep:
            return

        self.model.eval()
        with torch.no_grad():
            # Stage 1: episodic → semantic consolidation
            for block in self.model._agi_blocks:
                if hasattr(block, "memory") and block.memory is not None:
                    block.memory.maybe_consolidate()

            # Stage 2: replay recent batches
            replay_items = list(self._replay_buf)
            if replay_items:
                n_replay = min(self.sleep_replay_steps, len(replay_items))
                indices  = torch.randperm(len(replay_items))[:n_replay].tolist()
                for i in indices:
                    rx, ry = replay_items[i]
                    self.model(rx.to(self.device), targets=ry.to(self.device),
                               write_memory=True)

        self.model.train()

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save(self, tag: str = "latest") -> str:
        path = self.output_dir / f"agi_nfn_{tag}.pt"
        torch.save(
            {
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "step":            self.step,
                "cfg":             self.cfg.to_dict(),
                "history_tail":    self.history[-100:],
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
            vocab_size            = cfg.vocab_size,
            d_model               = cfg.d_model,
            n_blocks              = cfg.n_blocks,
            use_memory            = cfg.use_episodic_memory,
            use_working_memory    = cfg.use_working_memory,
            use_causal            = cfg.use_causal_graph,
            use_goal              = cfg.use_goal_predictor,
            use_reasoning         = cfg.use_recursive_reasoning,
            use_predictive_coding = cfg.use_predictive_coding,
            use_free_energy       = cfg.use_free_energy,
            use_self_consistency  = cfg.use_self_consistency,
            use_plan_executor     = cfg.use_plan_executor,
            use_bayesian          = cfg.use_bayesian_decoder,
            use_mod               = cfg.use_mixture_of_depths,
            use_mtp               = cfg.use_multi_token_pred,
            use_hyper             = cfg.use_hyper_net,
            use_ssm               = cfg.use_ssm,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        tokenizer         = NFNTokenizer()
        trainer           = cls(model, tokenizer, cfg)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        trainer.step      = ckpt.get("step", 0)
        trainer.history   = ckpt.get("history_tail", [])
        return trainer

    # ── Eval ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def eval_perplexity(self, text: str, seq_len: int = 512) -> float:
        """Compute perplexity on a held-out text string."""
        self.model.eval()
        ids = self.tokenizer.encode(text)[:seq_len + 1]
        if len(ids) < 2:
            self.model.train()
            return float("inf")
        x = torch.tensor(ids[:-1], device=self.device).unsqueeze(0)
        y = torch.tensor(ids[1:],  device=self.device).unsqueeze(0)
        logits, _ = self.model(x, targets=y, write_memory=False)
        nll = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            y.reshape(-1),
            ignore_index=self.cfg.pad_token_id,
        )
        self.model.train()
        return math.exp(nll.item())

    def stop(self):
        self._stop = True

    # ── Main training loop ────────────────────────────────────────────────────

    def train(
        self,
        text:           str,
        n_epochs:       int = 1,
        seq_len:        Optional[int] = None,
        batch_size:     int = 4,
        n_warmup_steps: int = 100,
        save_every:     int = 500,
        log_every:      int = 10,
        eval_text:      Optional[str] = None,
    ) -> List[Dict]:
        """
        Train on raw text with all AGI training features active.

        Per optimizer step:
          - WAKE forward (write_memory=True): episodic buffer fills
          - AGI multi-objective loss (curriculum gated)
          - Curiosity weighting (entropy-based upweighting of hard examples)
          - Every self_play_every: self-play + DPO-lite (same optimizer step)
          - Every critique_every:  constitutional critique (same optimizer step)
          - Every sleep_every:     SLEEP consolidation + replay (no grad)
          - Every goal_set_every:  set_goal from prompt prefix
        """
        if seq_len is None:
            seq_len = min(self.cfg.max_seq_len, 512)

        dataset        = AGITextDataset(text, self.tokenizer, seq_len, batch_size)
        n_steps_per_ep = len(dataset)
        n_total        = n_steps_per_ep * n_epochs

        scheduler  = cosine_with_warmup(self.optimizer, n_warmup_steps, n_total)
        t0         = time.time()
        self._stop = False
        accum_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for epoch in range(n_epochs):
            if self._stop:
                break

            for x, y in dataset.iter_batches(self.device):
                if self._stop:
                    break

                # Keep rolling buffer for SLEEP replay (stored on CPU)
                self._replay_buf.append((x.cpu(), y.cpu()))

                # ── Goal setting ──────────────────────────────────────────
                do_goal = (self.step % self.goal_set_every == 0)
                if do_goal:
                    with torch.no_grad():
                        pl = min(self.goal_prefix_len, x.shape[1])
                        self.model.set_goal(x[:, :pl])

                # ── Accumulate ────────────────────────────────────────────
                accum_buf.append((x, y))
                if len(accum_buf) < self.grad_accumulation_steps:
                    continue

                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                # ── 1. Standard WAKE forward + AGI loss ───────────────────
                step_metrics: Dict[str, float] = {}
                for acc_x, acc_y in accum_buf:
                    loss, fm = self._forward_step(acc_x, acc_y)
                    scaled   = loss / len(accum_buf)
                    self.scaler.scale(scaled).backward()
                    for k, v in fm.items():
                        step_metrics[k] = step_metrics.get(k, 0.0) + v / len(accum_buf)

                # ── 2. Self-play DPO update ───────────────────────────────
                if self.use_self_play and self.step % self.self_play_every == 0:
                    sp_m = self._self_play_step(x, y)
                    step_metrics.update(sp_m)

                # ── 3. Constitutional critique ────────────────────────────
                if self.use_critique and self.step % self.critique_every == 0:
                    cr_m = self._critique_step(x, y)
                    step_metrics.update(cr_m)

                # ── Gradient step (covers all .backward() calls above) ────
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                accum_buf = []

                # ── Goal reset ────────────────────────────────────────────
                if do_goal and self.step > 0:
                    self.model.reset_goal()

                # ── 4. SLEEP cycle (no grad) ──────────────────────────────
                if self.use_sleep and self.step > 0 and self.step % self.sleep_every == 0:
                    self._sleep_cycle()

                scheduler.step()
                self.step += 1

                # ── Bookkeeping ───────────────────────────────────────────
                step_metrics["step"]      = self.step
                step_metrics["epoch"]     = epoch
                step_metrics["elapsed"]   = time.time() - t0
                step_metrics["grad_norm"] = (
                    grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
                )
                step_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                self.history.append(step_metrics)

                if self.step_callback:
                    self.step_callback(step_metrics)

                if self.step % log_every == 0:
                    self._log(step_metrics)

                if eval_text and self.step % max(log_every * 5, 50) == 0:
                    ppl = self.eval_perplexity(eval_text, seq_len=min(seq_len, 512))
                    print(f"  [eval ppl: {ppl:.2f}]")

                if self.step % save_every == 0:
                    ckpt = self.save(f"step{self.step}")
                    print(f"  [saved {ckpt}]")

        self.save("final")
        return self.history

    # ── Online Learning Integration ────────────────────────────────────────────

    def enable_online_learning(
        self,
        adapter_rank: int = 8,
        online_lr: float = 2e-4,
        n_steps: int = 4,
        ppl_gate: float = 30.0,
    ):
        """Enable test-time LoRA adaptation via OnlineLearner."""
        self.use_online_learning = True
        self._online_learner = OnlineLearner(
            self.model,
            self.tokenizer,
            adapter_rank=adapter_rank,
            online_lr=online_lr,
            n_steps=n_steps,
            ppl_gate=ppl_gate,
        )
        return self._online_learner

    def online_adapt(self, text: str) -> Dict[str, object]:
        """Adapt model on new text at test time (LoRA fast weights)."""
        if self._online_learner is None:
            raise RuntimeError("Call enable_online_learning() first")
        return self._online_learner.adapt_from_text(text)

    def online_stats(self) -> Dict[str, object]:
        """Return OnlineLearner statistics."""
        if self._online_learner is None:
            return {"enabled": False}
        return {**self._online_learner.stats(), "enabled": True}

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, m: Dict[str, float]):
        lm     = m.get("lm",          m.get("total", 0.0))
        causal = m.get("causal",       0.0)
        goal   = m.get("goal",         0.0)
        fe     = m.get("free_energy",  0.0)
        sc     = m.get("consistency",  0.0)
        agi_w  = m.get("agi_weight",   0.0)
        gn     = m.get("grad_norm",    0.0)
        lr     = m.get("lr",           0.0)
        elapsed= m.get("elapsed",      0.0)

        extras = []
        if m.get("sp_total", 0.0) != 0.0:
            extras.append(f"sp_dpo={m.get('sp_dpo', 0.0):.4f}"
                          f" sp_dist={m.get('sp_distill', 0.0):.4f}")
        if m.get("critique_loss", 0.0) != 0.0:
            extras.append(f"crit={m['critique_loss']:.4f}")
        extras_str = "  " + "  ".join(extras) if extras else ""

        print(
            f"step {m['step']:5d} | ep {m.get('epoch', 0)} "
            f"| lm {lm:.4f} | causal {causal:.5f} | goal {goal:.5f} "
            f"| fe {fe:.5f} | sc {sc:.5f} "
            f"| agi_w {agi_w:.2f} | gn {gn:.2f} | lr {lr:.2e}"
            f"{extras_str} | {elapsed:.1f}s"
        )
