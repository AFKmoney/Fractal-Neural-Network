"""
NFN v4.0 — AGINFNModel

Full AGI language model stack integrating all v4.0 components:

  Token → AnalyticEmbedding
       → AGIBlock × n_blocks  (memory + causal + goal + phase)
       → [MultimodalFractalRFF]  (optional cross-modal fusion)
       → BayesianZipfianDecoder / ZipfianDecoder
       → Logits [B, L, V]

Losses returned per forward pass:
  "lm"        : primary language model cross-entropy
  "causal"    : DAG sparsity (sum over blocks)
  "goal"      : phase-goal alignment (sum over blocks)
  "coherence" : cross-modal phase coherence (if multimodal)
  "total"     : weighted sum of all above

Generation:
  model.generate(input_ids, max_new_tokens, ...)
  Supports greedy, top-k, top-p, and temperature sampling.
  Goal-directed generation: call model.set_goal(prompt_ids) first.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NFNConfig
from .agi_block import AGIBlock
from .hopfield import BayesianZipfianDecoder, ZipfianDecoder
from .analytic_embed import AnalyticTokenEmbedding


# ─────────────────────────────────────────────────────────────────────────────
# Optional multimodal import  (only if use_multimodal=True)
# ─────────────────────────────────────────────────────────────────────────────

def _try_import_multimodal():
    try:
        from .multimodal import MultimodalFractalRFF
        return MultimodalFractalRFF
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# AGINFNModel
# ─────────────────────────────────────────────────────────────────────────────

class AGINFNModel(nn.Module):
    """
    Full NFN v4.0 AGI language model.

    Differences from EfficientNFNLM (v3.2):
      • Each block is an AGIBlock (adds memory, causal DAG, goal forcing)
      • BayesianZipfianDecoder with condensate-derived uncertainty
      • Optional MultimodalFractalRFF for image/audio fusion
      • Goal-directed generation API (set_goal / reset_goal)
      • Richer loss dict from forward()
    """

    def __init__(self, cfg: NFNConfig):
        super().__init__()
        self.cfg = cfg

        # ── Embedding ──────────────────────────────────────────────────────
        self.embed = AnalyticTokenEmbedding(cfg.vocab_size, cfg.d_model)

        # ── AGI Blocks ─────────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            AGIBlock(cfg, block_idx=i) for i in range(cfg.n_blocks)
        ])

        # ── Final LayerNorm ────────────────────────────────────────────────
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # ── Decoder head ───────────────────────────────────────────────────
        if cfg.use_bayesian_decoder:
            self.lm_head = BayesianZipfianDecoder(
                in_dim              = cfg.d_model,
                vocab_size          = cfg.vocab_size,
                alpha               = cfg.nfmc_zipf_alpha,
                uncertainty_beta    = cfg.bayesian_uncertainty_beta,
            )
        else:
            self.lm_head = ZipfianDecoder(
                in_dim     = cfg.d_model,
                vocab_size = cfg.vocab_size,
                alpha      = cfg.nfmc_zipf_alpha,
            )

        # ── Multimodal fusion (optional) ───────────────────────────────────
        self.multimodal: Optional[nn.Module] = None
        if cfg.use_multimodal:
            MultimodalFractalRFF = _try_import_multimodal()
            if MultimodalFractalRFF is not None:
                self.multimodal = MultimodalFractalRFF(
                    d_shared    = cfg.d_model,
                    d_text      = cfg.d_model,
                    d_image     = cfg.multimodal_d_image,
                    d_audio     = cfg.multimodal_d_audio,
                    patch_hw    = (cfg.multimodal_patch_h, cfg.multimodal_patch_w),
                    n_rff       = cfg.nfmc_n_rff,
                    rank        = cfg.nfmc_rank,
                    n_phases    = cfg.nfmc_n_phases,
                )

    # ─── goal management ──────────────────────────────────────────────────────

    def set_goal(self, prompt_ids: torch.Tensor):
        """
        Encode the prompt into a goal phase and store it in all blocks.
        Call once before starting a constrained generation episode.

        prompt_ids: [B, L_prompt]
        """
        with torch.no_grad():
            h = self.embed(prompt_ids)
            for block in self.blocks:
                block.set_goal(h)
                h, _ = block(h, write_memory=False)

    def reset_goal(self):
        """Clear goal state in all blocks between generation episodes."""
        for block in self.blocks:
            block.reset_goal()

    def reset_memory(self):
        """Reset episodic memory across all blocks."""
        for block in self.blocks:
            if block.memory is not None:
                block.memory.reset()

    # ─── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:    torch.Tensor,                    # [B, L]
        targets:      Optional[torch.Tensor]  = None,  # [B, L]
        image:        Optional[torch.Tensor]  = None,  # [B, n_patches, d_image]
        audio:        Optional[torch.Tensor]  = None,  # [B, n_frames, d_audio]
        mask:         Optional[torch.Tensor]  = None,  # [B, L] attention mask
        write_memory: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
          logits : [B, L, vocab_size]
          losses : dict with keys "lm", "causal", "goal", "coherence", "total"
        """
        h = self.embed(input_ids)   # [B, L, d_model]

        losses: Dict[str, torch.Tensor] = {
            "causal":    torch.tensor(0.0, device=h.device),
            "goal":      torch.tensor(0.0, device=h.device),
            "coherence": torch.tensor(0.0, device=h.device),
        }

        # ── Multimodal fusion (before blocks) ─────────────────────────────
        if self.multimodal is not None and (image is not None or audio is not None):
            modal_out = self.multimodal(text=h, image=image, audio=audio)
            h = modal_out["text"]
            losses["coherence"] = (
                self.multimodal.loss_coherence() * self.cfg.lambda_coherence
            )

        # ── AGI Blocks ─────────────────────────────────────────────────────
        for block in self.blocks:
            h, block_losses = block(h, mask=mask, write_memory=write_memory)
            for k, v in block_losses.items():
                if k in losses:
                    losses[k] = losses[k] + v

        h = self.ln_f(h)

        # ── Decoder ────────────────────────────────────────────────────────
        if isinstance(self.lm_head, BayesianZipfianDecoder):
            logits = self.lm_head(
                h,
                sample_noise = self.cfg.bayesian_thompson_sampling and self.training,
            )
        else:
            logits = self.lm_head(h)   # [B, L, V]

        # ── Language model loss ────────────────────────────────────────────
        if targets is not None:
            if isinstance(self.lm_head, BayesianZipfianDecoder):
                losses["lm"] = self.lm_head.loss_calibration(logits, targets)
            else:
                losses["lm"] = F.cross_entropy(
                    logits.reshape(-1, self.cfg.vocab_size),
                    targets.reshape(-1),
                    ignore_index=-1,
                )
        else:
            losses["lm"] = torch.tensor(0.0, device=h.device)

        losses["total"] = sum(losses.values())

        return logits, losses

    # ─── generation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids:      torch.Tensor,           # [B, L]
        max_new_tokens: int = 128,
        temperature:    float = 1.0,
        top_k:          int = 0,
        top_p:          float = 1.0,
        eos_token_id:   Optional[int] = None,
        image:          Optional[torch.Tensor] = None,
        audio:          Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        Returns [B, L + max_new_tokens].
        """
        ids = input_ids
        eos = eos_token_id if eos_token_id is not None else self.cfg.eos_token_id

        for _ in range(max_new_tokens):
            # Only pass image/audio on first step (they're context, not per-token)
            logits, _ = self.forward(
                ids,
                image  = image if ids.shape[1] == input_ids.shape[1] else None,
                audio  = audio if ids.shape[1] == input_ids.shape[1] else None,
                write_memory = True,
            )
            next_logits = logits[:, -1, :]   # [B, V]

            if temperature != 1.0:
                next_logits = next_logits / max(temperature, 1e-5)

            # Top-k filtering
            if top_k > 0:
                topk_vals = torch.topk(next_logits, top_k).values[:, -1:]
                next_logits = next_logits.masked_fill(next_logits < topk_vals, -1e9)

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove_mask] = -1e9
                next_logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)   # [B, 1]

            ids = torch.cat([ids, next_id], dim=1)

            if eos is not None and (next_id == eos).all():
                break

        return ids

    # ─── parameter count ──────────────────────────────────────────────────────

    def param_count(self) -> Dict[str, int]:
        """Returns parameter counts by component."""
        def count(m):
            return sum(p.numel() for p in m.parameters())

        result = {
            "embed":    count(self.embed),
            "blocks":   count(self.blocks),
            "ln_f":     count(self.ln_f),
            "lm_head":  count(self.lm_head),
            "total":    count(self),
        }
        if self.multimodal is not None:
            result["multimodal"] = count(self.multimodal)
        return result

    def __repr__(self) -> str:
        pc = self.param_count()
        return (
            f"AGINFNModel(\n"
            f"  vocab={self.cfg.vocab_size}, d={self.cfg.d_model}, "
            f"blocks={self.cfg.n_blocks}\n"
            f"  memory={self.cfg.use_episodic_memory}, "
            f"causal={self.cfg.use_causal_graph}, "
            f"goal={self.cfg.use_goal_predictor}, "
            f"multimodal={self.cfg.use_multimodal}\n"
            f"  params={pc['total']:,}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience builder
# ─────────────────────────────────────────────────────────────────────────────

def build_agi_model(
    vocab_size: int,
    d_model: int = 256,
    n_blocks: int = 4,
    use_memory: bool = True,
    use_causal: bool = True,
    use_goal: bool = True,
    use_bayesian: bool = True,
    **kwargs,
) -> AGINFNModel:
    """
    Quick builder for AGINFNModel with sensible defaults.

    Example:
        model = build_agi_model(vocab_size=32000, d_model=512, n_blocks=8)
    """
    cfg = NFNConfig(
        vocab_size           = vocab_size,
        d_model              = d_model,
        n_blocks             = n_blocks,
        use_episodic_memory  = use_memory,
        use_causal_graph     = use_causal,
        use_goal_predictor   = use_goal,
        use_bayesian_decoder = use_bayesian,
        **kwargs,
    )
    return AGINFNModel(cfg)
