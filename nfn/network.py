"""
Neural Fractal Network – main language model.

Architecture per NFNBlock:
  1. Bottom-up (for each motif): leaf→root via SinusoidalAggregator
  2. Inter-motif coupling at each level via InterMotifCoupler
  3. Causal self-attention at top (compressed) level
  4. Top-down: root→leaf via SinusoidalBroadcast
  5. Motif mix → residual at leaf level

NFNLanguageModel:
  TokenEmbedding + PositionalEncoding
  → NFNBlock × n_blocks
  → LayerNorm → LMHead
"""

import math
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NFNConfig
from .connections import SinusoidalAggregator, SinusoidalBroadcast, InterMotifCoupler
from .topology import get_padded_length


# ─────────────────────────────────────────────────────────────────────────────
# Causal multi-head self-attention (top-level, over compressed sequence)
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask (upper-triangular = True means masked)
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in qkv]

        attn = (q @ k.transpose(-2, -1)) * self.scale           # [B, H, T, T]
        attn = attn.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, d)
        return self.out(out)


# ─────────────────────────────────────────────────────────────────────────────
# Single fractal motif branch (one set of K aggregators + K broadcasters)
# ─────────────────────────────────────────────────────────────────────────────

class MotifBranch(nn.Module):
    """
    Processes one fractal motif (e.g. binary_tree or cantor).
    Stores K SinusoidalAggregators (bottom-up) and K SinusoidalBroadcasters (top-down).
    """

    def __init__(self, cfg: NFNConfig, motif: str):
        super().__init__()
        self.motif = motif
        self.K = cfg.n_levels
        self.b = cfg.branching if motif != "cantor" else 3

        # Ω_k = ω_0 / λ^k  (deeper levels → lower frequency)
        omegas = [cfg.omega_base / (cfg.lambda_scale ** k) for k in range(cfg.n_levels)]

        self.aggregators = nn.ModuleList([
            SinusoidalAggregator(
                d_model=cfg.d_model,
                branching=self.b,
                rank=cfg.rank,
                omega_level=omegas[k],
                damping=cfg.damping,
                gamma_init=cfg.gamma_init,
                dropout=cfg.dropout,
            )
            for k in range(cfg.n_levels)
        ])

        self.broadcasters = nn.ModuleList([
            SinusoidalBroadcast(
                d_model=cfg.d_model,
                branching=self.b,
                rank=cfg.rank,
                damping=cfg.damping,
                dropout=cfg.dropout,
            )
            for k in range(cfg.n_levels)
        ])

    def bottom_up(
        self,
        x: torch.Tensor,   # [B, L, d]  (may not yet be padded for this motif)
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """
        Pads x to a multiple of b^K, then runs K aggregation steps.

        Returns
        -------
        levels   : list of length K+1 — levels[0] = padded x, levels[K] = root
        phases   : list of length K   — one phase tensor per level transition
        pad_len  : amount of padding added (to trim output in top_down)
        """
        B, L, d = x.shape
        stride = self.b ** self.K
        pad_needed = (stride - L % stride) % stride
        if pad_needed:
            pad = torch.zeros(B, pad_needed, d, device=device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        levels = [x]
        all_phases: List[torch.Tensor] = []

        current = x
        for k, agg in enumerate(self.aggregators):
            B, N_full, d = current.shape
            n_parents = N_full // self.b
            positions = torch.arange(n_parents, device=device)
            parent_h, phase = agg(current, positions)
            levels.append(parent_h)
            all_phases.append(phase)
            current = parent_h

        return levels, all_phases, pad_needed

    def top_down(
        self,
        levels: List[torch.Tensor],
        device: torch.device,
        orig_len: int,
    ) -> torch.Tensor:
        """
        Broadcasts context from root back to leaves.
        Returns updated leaf representation trimmed to orig_len: [B, orig_len, d].
        """
        current = levels[-1]   # [B, top_len, d]

        for k in reversed(range(self.K)):
            children = levels[k]               # [B, N*b, d]
            B, N_c, d = children.shape
            N = N_c // self.b
            positions = torch.arange(N, device=device)
            children_updated = self.broadcasters[k](children, current, positions)
            current = children_updated

        return current[:, :orig_len, :]   # trim padding


# ─────────────────────────────────────────────────────────────────────────────
# NFNBlock: one complete fractal processing block
# ─────────────────────────────────────────────────────────────────────────────

class NFNBlock(nn.Module):
    def __init__(self, cfg: NFNConfig):
        super().__init__()
        self.cfg = cfg
        self.b = cfg.branching

        # One branch per motif
        self.branches = nn.ModuleList([
            MotifBranch(cfg, motif) for motif in cfg.motifs
        ])

        # Inter-motif couplers at each level (only if >1 motif)
        self.couplers: Optional[nn.ModuleList] = None
        if cfg.n_motifs > 1:
            self.couplers = nn.ModuleList([
                InterMotifCoupler(cfg.d_model, cfg.rank, cfg.damping, cfg.dropout)
                for _ in range(cfg.n_levels + 1)
            ])

        # Causal self-attention at top level
        # max_len covers the worst-case (smallest b → largest top-len)
        top_len = max(1, cfg.max_seq_len // (2 ** cfg.n_levels))
        self.top_attn = CausalSelfAttention(
            cfg.d_model, cfg.n_heads, max_len=top_len, dropout=cfg.dropout
        )
        self.top_norm = nn.LayerNorm(cfg.d_model)

        # Mix multiple motif outputs at leaf level
        if cfg.n_motifs > 1:
            self.motif_mix = nn.Linear(cfg.d_model * cfg.n_motifs, cfg.d_model)
        else:
            self.motif_mix = nn.Identity()

        self.norm = nn.LayerNorm(cfg.d_model)

        # Temporal integration (P steps of state refinement)
        self.P = cfg.n_time_steps
        self.alpha = cfg.alpha
        self.temporal_ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pad sequence to multiple of b^K."""
        B, L, d = x.shape
        stride = self.b ** self.cfg.n_levels
        pad_len = (stride - L % stride) % stride
        if pad_len:
            pad = torch.zeros(B, pad_len, d, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        return x, L

    def _collect_phases(self, all_branch_phases: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Gather phases from all branches for loss computation."""
        flat = []
        for branch_phases in all_branch_phases:
            flat.extend(branch_phases)
        return flat

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,   # [B, L, d]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns (updated x [B, L, d], list of phase tensors for loss).
        """
        B, L, d = x.shape
        device = x.device
        residual = x
        orig_len = L

        # ── Bottom-up for all motifs (each branch pads to its own stride) ─────
        all_levels: List[List[torch.Tensor]] = []
        all_phases: List[List[torch.Tensor]] = []

        for branch in self.branches:
            lvls, phases, _ = branch.bottom_up(x, device)
            all_levels.append(lvls)
            all_phases.append(phases)

        # ── Inter-motif coupling — only at levels with matching sizes ──────────
        # Rebuild level lists to avoid in-place ops (which break autograd)
        if self.couplers is not None and len(self.branches) >= 2:
            n_shared = min(len(all_levels[0]), len(all_levels[1]))
            for lev_idx in range(n_shared):
                ha = all_levels[0][lev_idx]
                hb = all_levels[1][lev_idx]
                min_n = min(ha.shape[1], hb.shape[1])
                positions = torch.arange(min_n, device=device)
                ha_new, hb_new = self.couplers[lev_idx](
                    ha[:, :min_n], hb[:, :min_n], positions
                )
                # Concatenate back the un-coupled tail (if any), no in-place writes
                if ha.shape[1] > min_n:
                    ha_new = torch.cat([ha_new, ha[:, min_n:]], dim=1)
                if hb.shape[1] > min_n:
                    hb_new = torch.cat([hb_new, hb[:, min_n:]], dim=1)
                # Replace (list assignment, not tensor in-place)
                all_levels[0] = all_levels[0][:lev_idx] + [ha_new] + all_levels[0][lev_idx+1:]
                all_levels[1] = all_levels[1][:lev_idx] + [hb_new] + all_levels[1][lev_idx+1:]

        # ── Causal self-attention at top level ────────────────────────────────
        min_top = min(lvls[-1].shape[1] for lvls in all_levels)
        top = torch.stack([lvls[-1][:, :min_top] for lvls in all_levels], dim=0).mean(0)
        top = top + self.top_attn(self.top_norm(top))

        # Replace top level in each branch's level list (list assignment, not in-place)
        for m in range(len(self.branches)):
            branch_top = all_levels[m][-1]
            branch_top_len = branch_top.shape[1]
            if branch_top_len <= min_top:
                new_top = top[:, :branch_top_len]
            else:
                # Pad top with original tail
                new_top = torch.cat([top, branch_top[:, min_top:]], dim=1)
            all_levels[m] = all_levels[m][:-1] + [new_top]

        # ── Top-down for all motifs ───────────────────────────────────────────
        outputs = []
        for m, branch in enumerate(self.branches):
            out = branch.top_down(all_levels[m], device, orig_len)
            outputs.append(out)

        # ── Mix motif outputs ─────────────────────────────────────────────────
        if len(outputs) > 1:
            mixed = self.motif_mix(torch.cat(outputs, dim=-1))
        else:
            mixed = outputs[0]

        mixed = self.norm(mixed)

        # ── Temporal refinement (P mini-steps) ────────────────────────────────
        h = residual
        for _ in range(self.P):
            h = self.alpha * h + (1 - self.alpha) * self.temporal_ff(mixed + h)

        return h, self._collect_phases(all_phases)


# ─────────────────────────────────────────────────────────────────────────────
# Full NFN Language Model
# ─────────────────────────────────────────────────────────────────────────────

class NFNLanguageModel(nn.Module):
    """
    Causal language model powered by stacked NFN blocks.

    Forward signature:
        logits, aux = model(input_ids, targets=None)
        logits : [B, L, vocab_size]
        aux    : dict with 'phases', 'loss_aux' (None when targets is None)
    """

    def __init__(self, cfg: NFNConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([NFNBlock(cfg) for _ in range(cfg.n_blocks)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,         # [B, L]
        targets: Optional[torch.Tensor] = None,  # [B, L]
    ) -> Tuple[torch.Tensor, Dict]:
        B, L = input_ids.shape
        assert L <= self.cfg.max_seq_len, f"Sequence too long: {L} > {self.cfg.max_seq_len}"

        positions = torch.arange(L, device=input_ids.device)
        x = self.embed(input_ids) + self.pos_embed(positions).unsqueeze(0)

        all_phases: List[torch.Tensor] = []
        for block in self.blocks:
            x, phases = block(x)
            all_phases.extend(phases)

        x = self.final_norm(x)
        logits = self.lm_head(x)   # [B, L, vocab_size]

        aux: Dict = {"phases": all_phases, "loss_aux": None}

        if targets is not None:
            from training.losses import NFNLoss
            loss_fn = NFNLoss(self.cfg)
            task_loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=self.cfg.pad_token_id,
            )
            reg_losses = loss_fn.regularization(all_phases, self)
            total_loss = task_loss + reg_losses["total_reg"]
            aux["loss_aux"] = {
                "task": task_loss,
                "total": total_loss,
                **reg_losses,
            }

        return logits, aux

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,   # [B, prompt_len]
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            ctx = generated[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                topk_vals = torch.topk(logits, top_k).values[:, -1, None]
                logits[logits < topk_vals] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_summary(self) -> str:
        n = self.n_params()
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        if n >= 1e6:
            return f"{n/1e6:.2f}M"
        return f"{n/1e3:.1f}K"
