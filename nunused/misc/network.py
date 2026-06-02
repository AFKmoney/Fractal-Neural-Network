"""
Neural Fractal Network — Language Model (v2.0 / v3.0 hybrid)

Full architecture per NFNBlock:
  1. Working memory READ     (FractalMemoryBank / WorkingMemory)
  2. Bottom-up fractal       (SinusoidalAggregator × K, per motif)
  3. Kuramoto ODE phases     (KuramotoPhaseLayer at each level)
  4. Inter-motif coupling    (InterMotifCoupler)
  5. Flash self-attention    (CausalSelfAttention w/ RoPE + KV-cache)  ← top level
  5b. NFMC kernel layer      (NFMCKernelLayer — optional, cfg.use_nfmc)  ← v3.0
  6. Working memory WRITE    (update memory with current context)
  7. Top-down fractal        (SinusoidalBroadcast × K)
  8. Motif mix + residual
  9. Temporal refinement     (P steps: αh + (1-α)·FF(h+mixed))

NFNLanguageModel:
  TokenEmbedding (no positional embed — RoPE handles positions)
  → NFNBlock × n_blocks
  → LayerNorm → LMHead (weight-tied with embedding)

Standalone v3.0 model: see nfn/nfmc.py → NFMCLanguageModel
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NFNConfig
from .connections import SinusoidalAggregator, SinusoidalBroadcast, InterMotifCoupler
from .rope import RoPECache
from .kv_cache import AttentionKVCache, NFNKVCache
from .phase_ode import KuramotoPhaseLayer
from .memory import WorkingMemory, FractalMemoryBank
from .condensate import NFMCKernelLayer


# ─────────────────────────────────────────────────────────────────────────────
# Flash-attention multi-head self-attention with RoPE + KV-cache
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with:
      - Flash Attention via torch.nn.functional.scaled_dot_product_attention
      - Rotary Position Embeddings (RoPE) with NTK long-context scaling
      - Optional KV-cache for autoregressive generation
    """

    def __init__(self, cfg: NFNConfig, rope: RoPECache):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.H = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d = cfg.d_model
        self.use_rope = cfg.use_rope
        self.use_flash = cfg.use_flash_attn
        self.rope = rope   # shared RoPE across all blocks

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        # Fallback causal mask (used when flash_attn is unavailable)
        max_t = max(1, cfg.max_seq_len // (cfg.branching ** cfg.n_levels))
        mask = torch.triu(torch.ones(max_t, max_t, dtype=torch.bool), diagonal=1)
        self.register_buffer("_causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,                             # [B, T, d]
        kv_cache: Optional[AttentionKVCache] = None, # for generation
        kv_offset: int = 0,                          # position of first token
    ) -> torch.Tensor:
        B, T, d = x.shape

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d, dim=-1)

        # Multi-head reshape: [B, H, T, d_head]
        def mh(t):
            return t.view(B, T, self.H, self.d_head).transpose(1, 2)
        q, k, v = mh(q), mh(k), mh(v)

        # Apply RoPE
        if self.use_rope:
            q, k = self.rope(q, k, offset=kv_offset)

        # KV-cache update (autoregressive generation)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
            T_full = k.shape[2]
        else:
            T_full = T

        # Flash attention (PyTorch 2.0+ with SDPA)
        if self.use_flash and hasattr(F, "scaled_dot_product_attention"):
            # SDPA handles causal masking natively when is_causal=True
            # For generation with KV cache, we can't use is_causal (T_q ≠ T_kv)
            is_causal = (kv_cache is None)
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.drop.p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # Manual attention with causal mask
            scale = self.d_head ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale          # [B, H, T, T_full]
            if kv_cache is None and T > 1:
                mask = self._causal_mask[:T, :T_full]
                attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.out(out)


# ─────────────────────────────────────────────────────────────────────────────
# MotifBranch: one fractal motif with Kuramoto phases
# ─────────────────────────────────────────────────────────────────────────────

class MotifBranch(nn.Module):
    def __init__(self, cfg: NFNConfig, motif: str):
        super().__init__()
        self.motif = motif
        self.K = cfg.n_levels
        self.b = cfg.branching if motif != "cantor" else 3

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

        # Kuramoto phase layer at each level (replaces simple MLP phase)
        if cfg.use_kuramoto:
            self.kuramoto_layers = nn.ModuleList([
                KuramotoPhaseLayer(
                    d_model=cfg.d_model,
                    n_max_nodes=cfg.kuramoto_n_max,
                    rank=cfg.kuramoto_rank,
                    n_ode_steps=cfg.kuramoto_steps,
                    omega_init=omegas[k],
                )
                for k in range(cfg.n_levels)
            ])
        else:
            self.kuramoto_layers = None

    def bottom_up(
        self,
        x: torch.Tensor,
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        B, L, d = x.shape
        stride = self.b ** self.K
        pad_needed = (stride - L % stride) % stride
        if pad_needed:
            x = torch.cat([x, torch.zeros(B, pad_needed, d, device=device, dtype=x.dtype)], dim=1)

        levels = [x]
        all_phases: List[torch.Tensor] = []
        current = x

        for k, agg in enumerate(self.aggregators):
            B2, N_full, d2 = current.shape
            n_parents = N_full // self.b
            positions = torch.arange(n_parents, device=device)
            parent_h, phase = agg(current, positions)

            # Replace simple phase with Kuramoto ODE
            if self.kuramoto_layers is not None:
                parent_h, phase = self.kuramoto_layers[k](parent_h, phase)

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
        current = levels[-1]
        for k in reversed(range(self.K)):
            children = levels[k]
            B, N_c, d = children.shape
            N = N_c // self.b
            positions = torch.arange(N, device=device)
            current = self.broadcasters[k](children, current, positions)
        return current[:, :orig_len, :]


# ─────────────────────────────────────────────────────────────────────────────
# NFNBlock
# ─────────────────────────────────────────────────────────────────────────────

class NFNBlock(nn.Module):
    def __init__(self, cfg: NFNConfig, rope: RoPECache):
        super().__init__()
        self.cfg = cfg
        self.b = cfg.branching

        self.branches = nn.ModuleList([
            MotifBranch(cfg, motif) for motif in cfg.motifs
        ])

        # Inter-motif couplers
        self.couplers: Optional[nn.ModuleList] = None
        if cfg.n_motifs > 1:
            self.couplers = nn.ModuleList([
                InterMotifCoupler(cfg.d_model, cfg.rank, cfg.damping, cfg.dropout)
                for _ in range(cfg.n_levels + 1)
            ])

        # Flash attention with shared RoPE
        self.top_attn = CausalSelfAttention(cfg, rope)
        self.top_norm = nn.LayerNorm(cfg.d_model)

        # Working memory (optional)
        self.memory: Optional[WorkingMemory] = None
        if cfg.use_memory and not cfg.memory_per_level:
            self.memory = WorkingMemory(
                cfg.d_model, cfg.memory_slots, cfg.memory_heads, cfg.dropout
            )

        # Fractal memory bank (one bank per level)
        self.frac_memory: Optional[FractalMemoryBank] = None
        if cfg.use_memory and cfg.memory_per_level:
            self.frac_memory = FractalMemoryBank(
                cfg.d_model, cfg.n_levels,
                n_slots_per_level=cfg.memory_slots,
                n_heads=cfg.memory_heads,
                dropout=cfg.dropout,
            )

        # NFMC kernel layer (v3.0 — optional fractal kernel enrichment)
        self.nfmc: Optional[NFMCKernelLayer] = None
        if cfg.use_nfmc:
            self.nfmc = NFMCKernelLayer(
                d_model    = cfg.d_model,
                n_rff      = cfg.nfmc_n_rff,
                n_scales   = cfg.nfmc_n_scales,
                rank       = cfg.nfmc_rank,
                n_phases   = cfg.nfmc_n_phases,
                n_iter     = cfg.nfmc_lock_iter,
                eta        = cfg.nfmc_eta,
            )

        # Motif mix
        if cfg.n_motifs > 1:
            self.motif_mix = nn.Linear(cfg.d_model * cfg.n_motifs, cfg.d_model)
        else:
            self.motif_mix = nn.Identity()

        self.norm = nn.LayerNorm(cfg.d_model)

        # Temporal refinement
        self.P = cfg.n_time_steps
        self.alpha = cfg.alpha
        self.temporal_ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def _collect_phases(self, branch_phases):
        return [p for phases in branch_phases for p in phases]

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[AttentionKVCache] = None,
        kv_offset: int = 0,
        update_memory: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, L, d = x.shape
        device = x.device
        residual = x
        orig_len = L

        # ── Working memory READ (before fractal processing) ───────────────────
        if self.memory is not None:
            x, _ = self.memory(x, update_memory=False)   # read only here

        # ── Bottom-up ─────────────────────────────────────────────────────────
        all_levels: List[List[torch.Tensor]] = []
        all_phases: List[List[torch.Tensor]] = []

        for branch in self.branches:
            lvls, phases, _ = branch.bottom_up(x, device)
            all_levels.append(lvls)
            all_phases.append(phases)

        # ── Fractal memory enrichment (per level) ─────────────────────────────
        if self.frac_memory is not None:
            for m_idx in range(len(self.branches)):
                all_levels[m_idx] = self.frac_memory(
                    all_levels[m_idx], update=update_memory
                )

        # ── Inter-motif coupling ──────────────────────────────────────────────
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
                if ha.shape[1] > min_n:
                    ha_new = torch.cat([ha_new, ha[:, min_n:]], dim=1)
                if hb.shape[1] > min_n:
                    hb_new = torch.cat([hb_new, hb[:, min_n:]], dim=1)
                all_levels[0] = all_levels[0][:lev_idx] + [ha_new] + all_levels[0][lev_idx+1:]
                all_levels[1] = all_levels[1][:lev_idx] + [hb_new] + all_levels[1][lev_idx+1:]

        # ── Flash self-attention at top level ─────────────────────────────────
        min_top = min(lvls[-1].shape[1] for lvls in all_levels)
        top = torch.stack([lvls[-1][:, :min_top] for lvls in all_levels], dim=0).mean(0)
        top = top + self.top_attn(self.top_norm(top), kv_cache=kv_cache, kv_offset=kv_offset)

        # ── NFMC kernel enrichment (v3.0 — optional) ──────────────────────────
        if self.nfmc is not None:
            top = self.nfmc(top)

        # Inject updated top back (no in-place)
        for m_idx in range(len(self.branches)):
            branch_top = all_levels[m_idx][-1]
            branch_top_len = branch_top.shape[1]
            if branch_top_len <= min_top:
                new_top = top[:, :branch_top_len]
            else:
                new_top = torch.cat([top, branch_top[:, min_top:]], dim=1)
            all_levels[m_idx] = all_levels[m_idx][:-1] + [new_top]

        # ── Working memory WRITE (update with top-level context) ──────────────
        if self.memory is not None and update_memory:
            _, _ = self.memory(top, update_memory=True)

        # ── Top-down ──────────────────────────────────────────────────────────
        outputs = []
        for m_idx, branch in enumerate(self.branches):
            outputs.append(branch.top_down(all_levels[m_idx], device, orig_len))

        # ── Mix motifs ────────────────────────────────────────────────────────
        if len(outputs) > 1:
            mixed = self.motif_mix(torch.cat(outputs, dim=-1))
        else:
            mixed = outputs[0]
        mixed = self.norm(mixed)

        # ── Temporal refinement ───────────────────────────────────────────────
        h = residual
        for _ in range(self.P):
            h = self.alpha * h + (1 - self.alpha) * self.temporal_ff(mixed + h)

        return h, self._collect_phases(all_phases)


# ─────────────────────────────────────────────────────────────────────────────
# NFN Language Model
# ─────────────────────────────────────────────────────────────────────────────

class NFNLanguageModel(nn.Module):
    """
    Causal language model with the full NFN stack:
      - Flash Attention + RoPE (long context up to 128K via NTK)
      - Kuramoto ODE phase dynamics
      - Persistent fractal working memory
      - Multi-motif fractal topology
      - KV-cache for O(1) per-token generation
    """

    def __init__(self, cfg: NFNConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding only (no learned positional embedding — RoPE handles it)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)

        # Shared RoPE cache (one instance, used by all blocks)
        top_len = max(1, cfg.max_seq_len // (cfg.branching ** cfg.n_levels))
        self.rope = RoPECache(
            dim=cfg.d_model // cfg.n_heads,
            max_seq_len=top_len,
            base=cfg.rope_base,
            scale_factor=cfg.rope_scale_factor,
        )

        self.blocks = nn.ModuleList([NFNBlock(cfg, self.rope) for _ in range(cfg.n_blocks)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Memory management ─────────────────────────────────────────────────────

    def reset_memory(self, batch_size: int = 1):
        """Clear all working memory banks — call between conversations."""
        for block in self.blocks:
            if block.memory is not None:
                block.memory.reset(batch_size)
            if block.frac_memory is not None:
                block.frac_memory.reset_all(batch_size)

    def save_memory(self) -> List:
        """Serialise all memory states for persistence."""
        states = []
        for block in self.blocks:
            if block.memory is not None:
                states.append(("flat", block.memory.save_state()))
            elif block.frac_memory is not None:
                states.append(("fractal", block.frac_memory.save_states()))
            else:
                states.append(None)
        return states

    def load_memory(self, states: List):
        """Restore memory states."""
        for block, state in zip(self.blocks, states):
            if state is None:
                continue
            kind, data = state
            if kind == "flat" and block.memory is not None:
                block.memory.set_state(data)
            elif kind == "fractal" and block.frac_memory is not None:
                block.frac_memory.load_states(data)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,                       # [B, L]
        targets: Optional[torch.Tensor] = None,         # [B, L]
        kv_cache: Optional[NFNKVCache] = None,
        update_memory: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        B, L = input_ids.shape
        assert L <= self.cfg.max_seq_len, \
            f"Sequence length {L} exceeds max_seq_len {self.cfg.max_seq_len}"

        x = self.embed(input_ids)   # [B, L, d]  — no positional encoding, RoPE handles it

        kv_offset = kv_cache.step if kv_cache is not None else 0
        all_phases: List[torch.Tensor] = []

        for i, block in enumerate(self.blocks):
            attn_cache = kv_cache.attn_caches[i] if kv_cache is not None else None
            x, phases = block(
                x,
                kv_cache=attn_cache,
                kv_offset=kv_offset,
                update_memory=update_memory,
            )
            all_phases.extend(phases)

        if kv_cache is not None:
            kv_cache.step += L

        x = self.final_norm(x)
        logits = self.lm_head(x)   # [B, L, V]

        aux: Dict = {"phases": all_phases, "loss_aux": None}

        if targets is not None:
            from training.losses import NFNLoss
            loss_fn = NFNLoss(self.cfg)
            task_loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=self.cfg.pad_token_id,
            )
            reg = loss_fn.regularization(all_phases, self)
            total = task_loss + reg["total_reg"]
            aux["loss_aux"] = {
                "task": task_loss,
                "total": total,
                **reg,
            }

        return logits, aux

    # ── Autoregressive generation ─────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        self.eval()

        kv_cache = NFNKVCache(
            n_blocks=self.cfg.n_blocks,
            n_levels=self.cfg.n_levels,
            branching=self.cfg.branching,
            d_model=self.cfg.d_model,
        ) if use_cache else None

        # Prefill
        generated = input_ids.clone()
        logits, _ = self(generated, kv_cache=kv_cache, update_memory=True)

        for _ in range(max_new_tokens):
            # In cached mode, only feed the last token
            if use_cache:
                next_input = generated[:, -1:]
            else:
                next_input = generated[:, -self.cfg.max_seq_len:]

            logits, _ = self(next_input, kv_cache=kv_cache, update_memory=False)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                topk_val = torch.topk(next_logits, top_k).values[:, -1, None]
                next_logits = next_logits.masked_fill(next_logits < topk_val, float("-inf"))

            if top_p is not None and top_p < 1.0:
                sorted_l, sorted_i = torch.sort(next_logits, descending=True)
                cum_p = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                remove = cum_p - F.softmax(sorted_l, dim=-1) > top_p
                sorted_l[remove] = float("-inf")
                next_logits = torch.scatter(next_logits, 1, sorted_i, sorted_l)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    # ── Utilities ─────────────────────────────────────────────────────────────

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_summary(self) -> str:
        n = self.n_params()
        if n >= 1e9: return f"{n/1e9:.2f}B"
        if n >= 1e6: return f"{n/1e6:.2f}M"
        return f"{n/1e3:.1f}K"
