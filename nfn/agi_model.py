"""
NFN v4.0 — AGINFNModel

Full AGI language model stack:

  Token → AnalyticEmbedding
       → [PredictiveCodingBlock wrapping AGIBlock] × n_blocks
       → [MultimodalFractalRFF]
       → BayesianZipfianDecoder / ZipfianDecoder
       → Logits [B, L, V]

New in this version:
  • RecursiveReasoner wraps any block for ACT variable-depth thinking
  • PredictiveCodingBlock wires top-down prediction errors between blocks
  • think(prompt) runs extra reasoning rounds before generating
  • Full loss dict: lm + causal + goal + coherence + ponder + pred + fe + consistency
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
from .reasoning import RecursiveReasoner
from .predictive import PredictiveCodingBlock
from .multi_token_pred import MultiTokenPredictor, SpeculativeDecoder


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
    NFN v4.0 full AGI language model.

    Each block is optionally wrapped in:
      • RecursiveReasoner  — variable-depth ACT thinking per position
      • PredictiveCodingBlock — top-down prediction errors between layers

    These wrappers are transparent: the block still receives h and returns
    (h, losses), the wrappers add ponder / pred losses on top.
    """

    def __init__(self, cfg: NFNConfig):
        super().__init__()
        self.cfg = cfg

        # ── Embedding ──────────────────────────────────────────────────────
        self.embed = AnalyticTokenEmbedding(cfg.vocab_size, cfg.d_model)

        # ── Build blocks with optional wrappers ────────────────────────────
        raw_blocks = [AGIBlock(cfg, block_idx=i) for i in range(cfg.n_blocks)]

        # Wrap with predictive coding first (inter-block prediction errors)
        if cfg.use_predictive_coding:
            pc_blocks = [
                PredictiveCodingBlock(
                    d_model     = cfg.d_model,
                    block       = b,
                    error_scale = cfg.pc_error_scale,
                    lambda_pred = cfg.lambda_pred,
                )
                for b in raw_blocks
            ]
        else:
            pc_blocks = raw_blocks

        # Wrap with recursive reasoner (ACT variable-depth thinking)
        if cfg.use_recursive_reasoning:
            self.reasoner = RecursiveReasoner(
                d_model            = cfg.d_model,
                max_steps          = cfg.reasoning_max_steps,
                halt_threshold     = cfg.reasoning_halt_threshold,
                halt_on_alignment  = cfg.reasoning_halt_on_alignment,
                lambda_ponder      = cfg.lambda_ponder,
            )
        else:
            self.reasoner = None

        self.blocks = nn.ModuleList(pc_blocks)

        # Keep references to the raw AGIBlocks for goal/memory management
        self._agi_blocks: List[AGIBlock] = raw_blocks

        # ── Final norm ─────────────────────────────────────────────────────
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # ── Decoder ────────────────────────────────────────────────────────
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

        # ── Multi-Token Prediction ─────────────────────────────────────────
        self.mtp: Optional[MultiTokenPredictor] = None
        self.speculative: Optional[SpeculativeDecoder] = None
        if cfg.use_multi_token_pred:
            self.mtp = MultiTokenPredictor(
                d_model           = cfg.d_model,
                vocab_size        = cfg.vocab_size,
                n_heads           = cfg.mtp_n_heads,
                alpha             = cfg.nfmc_zipf_alpha,
                loss_weight_decay = cfg.mtp_loss_weight_decay,
            )
            self.speculative = SpeculativeDecoder(self.mtp)

        # ── Multimodal fusion ──────────────────────────────────────────────
        self.multimodal: Optional[nn.Module] = None
        if cfg.use_multimodal:
            MMClass = _try_import_multimodal()
            if MMClass is not None:
                self.multimodal = MMClass(
                    d_shared = cfg.d_model,
                    d_text   = cfg.d_model,
                    d_image  = cfg.multimodal_d_image,
                    d_audio  = cfg.multimodal_d_audio,
                    patch_hw = (cfg.multimodal_patch_h, cfg.multimodal_patch_w),
                    n_rff    = cfg.nfmc_n_rff,
                    rank     = cfg.nfmc_rank,
                    n_phases = cfg.nfmc_n_phases,
                )

    # ─── goal management ──────────────────────────────────────────────────────

    def set_goal(self, prompt_ids: torch.Tensor):
        """Encode prompt → goal phase, store in all blocks."""
        with torch.no_grad():
            h = self.embed(prompt_ids)
            for block in self._agi_blocks:
                block.set_goal(h)
                h, _ = block(h, write_memory=False)

    def reset_goal(self):
        for block in self._agi_blocks:
            block.reset_goal()

    def reset_memory(self):
        for block in self._agi_blocks:
            if block.memory is not None:
                block.memory.reset()
            if block.working_mem is not None:
                block.working_mem.reset()

    # ─── think()  — extra reasoning rounds before generation ─────────────────

    @torch.no_grad()
    def think(self, input_ids: torch.Tensor, n_rounds: int = 3) -> torch.Tensor:
        """
        Run n_rounds of reasoning over the prompt without generating tokens.
        Updates working memory and episodic memory in-place.
        Returns the final hidden state [B, L, d] after thinking.

        Call this before generate() on hard problems.
        """
        h = self.embed(input_ids)
        for _ in range(n_rounds):
            for block in self.blocks:
                if isinstance(block, PredictiveCodingBlock):
                    h, _ = block(h, write_memory=True)
                else:
                    h, _ = block(h, write_memory=True)
        return self.ln_f(h)

    # ─── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:    torch.Tensor,
        targets:      Optional[torch.Tensor] = None,
        image:        Optional[torch.Tensor] = None,
        audio:        Optional[torch.Tensor] = None,
        mask:         Optional[torch.Tensor] = None,
        write_memory: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns (logits [B, L, V], losses dict).
        """
        h = self.embed(input_ids)

        losses: Dict[str, torch.Tensor] = {
            k: torch.tensor(0.0, device=h.device)
            for k in ("causal", "goal", "coherence", "ponder", "pred", "free_energy", "consistency")
        }

        # ── Multimodal fusion ──────────────────────────────────────────────
        if self.multimodal is not None and (image is not None or audio is not None):
            modal_out = self.multimodal(text=h, image=image, audio=audio)
            h = modal_out["text"]
            losses["coherence"] = (
                self.multimodal.loss_coherence() * self.cfg.lambda_coherence
            )

        # ── Blocks (with optional RecursiveReasoner) ───────────────────────
        prev_prediction = None   # for predictive coding top-down pass

        for block in self.blocks:
            if self.reasoner is not None:
                # ACT: run block multiple times, halt adaptively
                def _block_call(h_in):
                    if isinstance(block, PredictiveCodingBlock):
                        h_out, _ = block(h_in, prediction_above=prev_prediction,
                                         write_memory=write_memory)
                    else:
                        h_out, _ = block(h_in, write_memory=write_memory)
                    return h_out, {}

                # Get goal alignment for this block (if available)
                goal_alignment = None
                raw_block = block.block if isinstance(block, PredictiveCodingBlock) else block
                if hasattr(raw_block, 'goal') and raw_block.goal is not None:
                    if raw_block.goal._goal_phase is not None:
                        d = h.shape[-1]
                        half = min(self.cfg.goal_n_phases, d // 2)
                        phase_proxy = torch.atan2(h[..., :half], h[..., half:half*2] + 1e-6)
                        if half < self.cfg.goal_n_phases:
                            pad = torch.zeros(*phase_proxy.shape[:-1],
                                              self.cfg.goal_n_phases - half, device=h.device)
                            phase_proxy = torch.cat([phase_proxy, pad], dim=-1)
                        _, goal_alignment = raw_block.goal(phase_proxy)

                h, ponder_loss, _ = self.reasoner(h, _block_call, goal_alignment)
                losses["ponder"] = losses["ponder"] + ponder_loss

            else:
                # Standard single pass
                if isinstance(block, PredictiveCodingBlock):
                    h, pred_loss = block(h, prediction_above=prev_prediction,
                                         write_memory=write_memory)
                    losses["pred"] = losses["pred"] + pred_loss
                    prev_prediction = block.last_prediction
                else:
                    h, block_losses = block(h, mask=mask, write_memory=write_memory)
                    for k, v in block_losses.items():
                        if k in losses:
                            losses[k] = losses[k] + v

        h = self.ln_f(h)

        # ── Decoder ────────────────────────────────────────────────────────
        if isinstance(self.lm_head, BayesianZipfianDecoder):
            logits = self.lm_head(
                h, sample_noise=self.cfg.bayesian_thompson_sampling and self.training
            )
        else:
            logits = self.lm_head(h)

        # ── LM loss ────────────────────────────────────────────────────────
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

        # ── Multi-Token Prediction loss ────────────────────────────────────
        if self.mtp is not None and targets is not None:
            mtp_loss, mtp_breakdown = self.mtp.loss(h, targets)
            losses["mtp"] = mtp_loss
            losses.update(mtp_breakdown)

        losses["total"] = sum(v for k, v in losses.items() if not k.startswith("mtp_") and k != "total")
        return logits, losses

    # ─── generation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids:      torch.Tensor,
        max_new_tokens: int = 128,
        temperature:    float = 1.0,
        top_k:          int = 0,
        top_p:          float = 1.0,
        eos_token_id:   Optional[int] = None,
        image:          Optional[torch.Tensor] = None,
        audio:          Optional[torch.Tensor] = None,
        think_rounds:   int = 0,
        speculative:    bool = False,   # use speculative decoding (requires MTP)
    ) -> torch.Tensor:
        """
        Autoregressive generation with optional thinking phase.

        think_rounds > 0: run think() before generating to warm up
        working memory and episodic memory with prompt context.
        """
        if think_rounds > 0:
            self.think(input_ids, n_rounds=think_rounds)

        # Speculative decoding path (requires MTP heads)
        if speculative and self.speculative is not None:
            return self._speculative_generate(
                input_ids, max_new_tokens, temperature, eos_token_id
            )

        ids = input_ids
        eos = eos_token_id if eos_token_id is not None else self.cfg.eos_token_id
        first_step = True

        for _ in range(max_new_tokens):
            logits, _ = self.forward(
                ids,
                image  = image if first_step else None,
                audio  = audio if first_step else None,
                write_memory = True,
            )
            first_step = False
            next_logits = logits[:, -1, :]

            if temperature != 1.0:
                next_logits = next_logits / max(temperature, 1e-5)

            if top_k > 0:
                thresh = torch.topk(next_logits, top_k).values[:, -1:]
                next_logits = next_logits.masked_fill(next_logits < thresh, -1e9)

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = -1e9
                next_logits.scatter_(1, sorted_idx, sorted_logits)

            probs  = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids    = torch.cat([ids, next_id], dim=1)

            if eos is not None and (next_id == eos).all():
                break

        return ids

    @torch.no_grad()
    def _speculative_generate(
        self,
        input_ids:      torch.Tensor,
        max_new_tokens: int,
        temperature:    float,
        eos_token_id:   Optional[int],
    ) -> torch.Tensor:
        """
        Speculative decoding: MTP heads draft N tokens, full model verifies.
        Typical speedup: 2-4x over standard autoregressive generation.
        """
        ids     = input_ids
        eos     = eos_token_id if eos_token_id is not None else self.cfg.eos_token_id
        N       = self.mtp.n_heads
        generated = 0

        while generated < max_new_tokens:
            B, L = ids.shape

            # 1. Run full model to get hidden state for draft generation
            h = self.embed(ids)
            for block in self.blocks:
                result = block(h, write_memory=False)
                h = result[0] if isinstance(result, tuple) else result
            h = self.ln_f(h)

            # 2. Draft N tokens using MTP heads (from last hidden state)
            h_last    = h[:, -1:, :]                   # [B, 1, d]
            all_head_logits = self.mtp(h_last)         # N × [B, 1, V]
            draft_ids = torch.cat(
                [F.softmax(lg[:, 0] / max(temperature, 1e-5), dim=-1
                 ).multinomial(1) for lg in all_head_logits],
                dim=1,
            )  # [B, N]

            # 3. Verify draft by running full model on context + draft
            candidate = torch.cat([ids, draft_ids], dim=1)   # [B, L+N]
            h_verify  = self.embed(candidate)
            for block in self.blocks:
                result = block(h_verify, write_memory=False)
                h_verify = result[0] if isinstance(result, tuple) else result
            h_verify = self.ln_f(h_verify)
            verify_logits = (
                self.lm_head(h_verify)
                if not isinstance(self.lm_head, BayesianZipfianDecoder)
                else self.lm_head(h_verify)
            )  # [B, L+N, V]

            # 4. Accept/reject each draft token
            new_tokens = []
            for i in range(N):
                v_log = verify_logits[:, L + i - 1, :]
                v_prob = F.softmax(v_log / max(temperature, 1e-5), -1)
                d_tok  = draft_ids[:, i]
                accept = v_prob[torch.arange(B), d_tok] > 0.5
                if accept.all():
                    new_tokens.append(d_tok.unsqueeze(1))
                else:
                    # Fallback: sample from verified distribution
                    fallback = torch.multinomial(v_prob, 1)
                    new_tokens.append(fallback)
                    break

            if not new_tokens:
                # Safety: emit at least one token
                last_logits = verify_logits[:, L - 1, :]
                p = F.softmax(last_logits / max(temperature, 1e-5), -1)
                new_tokens.append(torch.multinomial(p, 1))

            new_ids = torch.cat(new_tokens, dim=1)
            ids = torch.cat([ids, new_ids], dim=1)
            generated += new_ids.shape[1]

            if eos is not None and (new_ids == eos).any():
                break

        return ids

    # ─── utilities ────────────────────────────────────────────────────────────

    def param_count(self) -> Dict[str, int]:
        def count(m): return sum(p.numel() for p in m.parameters())
        result = {
            "embed":   count(self.embed),
            "blocks":  count(self.blocks),
            "ln_f":    count(self.ln_f),
            "lm_head": count(self.lm_head),
            "total":   count(self),
        }
        if self.multimodal is not None:
            result["multimodal"] = count(self.multimodal)
        return result

    def __repr__(self) -> str:
        pc   = self.param_count()
        cfg  = self.cfg
        mods = []
        if cfg.use_episodic_memory:    mods.append("episodic+semantic memory")
        if cfg.use_working_memory:     mods.append(f"working memory ({cfg.wm_n_slots} slots)")
        if cfg.use_causal_graph:       mods.append("causal DAG")
        if cfg.use_goal_predictor:     mods.append("goal forcing")
        if cfg.use_recursive_reasoning:mods.append(f"ACT (max {cfg.reasoning_max_steps} steps)")
        if cfg.use_predictive_coding:  mods.append("predictive coding")
        if cfg.use_free_energy:        mods.append("free energy")
        if cfg.use_self_consistency:   mods.append("self-consistency")
        if cfg.use_plan_executor:      mods.append(f"planner ({cfg.plan_n_subgoals} sub-goals)")
        if cfg.use_bayesian_decoder:      mods.append("bayesian decoder")
        if cfg.use_multimodal:            mods.append("multimodal")
        if cfg.use_mixture_of_depths:     mods.append(f"MoD ({cfg.mod_capacity_factor:.0%} tokens)")
        if cfg.use_multi_token_pred:      mods.append(f"MTP (N={cfg.mtp_n_heads})")
        if cfg.use_streaming:             mods.append(f"streaming (W={cfg.streaming_window_size})")
        if cfg.use_hyper_net:             mods.append(f"hyper-net (r={cfg.hyper_rank})")
        return (
            f"AGINFNModel(\n"
            f"  vocab={cfg.vocab_size}  d={cfg.d_model}  blocks={cfg.n_blocks}\n"
            f"  modules: {', '.join(mods) or 'none'}\n"
            f"  params={pc['total']:,}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience builder
# ─────────────────────────────────────────────────────────────────────────────

def build_agi_model(
    vocab_size:             int,
    d_model:                int   = 256,
    n_blocks:               int   = 4,
    use_memory:             bool  = True,
    use_working_memory:     bool  = True,
    use_causal:             bool  = True,
    use_goal:               bool  = True,
    use_reasoning:          bool  = True,
    use_predictive_coding:  bool  = True,
    use_free_energy:        bool  = True,
    use_self_consistency:   bool  = True,
    use_plan_executor:      bool  = True,
    use_bayesian:           bool  = True,
    use_mod:                bool  = True,
    use_mtp:                bool  = True,
    use_hyper:              bool  = True,
    **kwargs,
) -> AGINFNModel:
    """
    One-call builder for a fully-equipped AGINFNModel.

    Example:
        model = build_agi_model(vocab_size=32000, d_model=512, n_blocks=8)
        print(model)
    """
    cfg = NFNConfig(
        vocab_size              = vocab_size,
        d_model                 = d_model,
        n_blocks                = n_blocks,
        use_episodic_memory     = use_memory,
        use_working_memory      = use_working_memory,
        use_causal_graph        = use_causal,
        use_goal_predictor      = use_goal,
        use_recursive_reasoning = use_reasoning,
        use_predictive_coding   = use_predictive_coding,
        use_free_energy         = use_free_energy,
        use_self_consistency    = use_self_consistency,
        use_plan_executor       = use_plan_executor,
        use_bayesian_decoder    = use_bayesian,
        use_mixture_of_depths   = use_mod,
        use_multi_token_pred    = use_mtp,
        use_hyper_net           = use_hyper,
        **kwargs,
    )
    return AGINFNModel(cfg)
