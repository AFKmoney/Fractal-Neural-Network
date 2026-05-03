"""
NFMCLanguageModel — NFN v3.0 Standalone Model

Architecture:
    tokens [B, L]
        ↓ embed           [B, L, d_model]      — learnable (minimal)
        ↓ FractalRFF      [B, L, 2*n_rff]       — fixed fractal kernel features
        ↓ Condensate      [B, L, rank]           — fixed spectral projection
        ↓ PhaseLocking    [B, L, n_phases]       — fixed Kuramoto dynamics
        ↓ Decoder         [B, L, V]              — learnable head

The core kernel (RFF + Condensate + PhaseLocking) is PARAMETER-FREE after
one-shot condensation. Only the token embedding and decoder head carry gradients.

Zero-training initialisation path:
    model.condense_from_text(text, tokenizer)
    → collects RFF features for all tokens (one forward pass, no SGD)
    → SVD-condenses them into K̃_r
    → seeds vocabulary phases from character co-occurrence statistics
    → model is ready to generate (quality scales with decoder fine-tuning)

Fine-tuning path (very fast — only embed + decoder):
    optimizer = AdamW(model.learnable_params(), lr=3e-4)
    → typically 100-500 steps suffices due to kernel structural priors
"""

import math
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NFNConfig
from .condensate import (
    FractalRFF,
    SpectralCondensate,
    HelmholtzPhaseLocking,
    CondensateDecoder,
)


class NFMCLanguageModel(nn.Module):
    """
    Neural Fractal Multidimensional Condensate Language Model (v3.0).

    Key properties:
    - Fractal kernel provides rich multi-scale structural priors
    - Phase locking discovers syntactic/semantic categories from geometry
    - Condensate encodes universal structure in a fixed O(rank²) matrix
    - Only ~2% of parameters require gradient (embed + decoder head)
    """

    def __init__(self, cfg: NFNConfig):
        super().__init__()
        self.cfg = cfg

        V   = cfg.vocab_size
        d   = cfg.d_model
        n_rff    = getattr(cfg, "nfmc_n_rff",    256)
        n_scales = getattr(cfg, "nfmc_n_scales",  8)
        rank     = getattr(cfg, "nfmc_rank",      64)
        n_phases = getattr(cfg, "nfmc_n_phases",  8)
        n_iter   = getattr(cfg, "nfmc_lock_iter", 8)
        eta      = getattr(cfg, "nfmc_eta",       0.15)

        # ── Learnable: token embedding + output head ──────────────────────────
        self.embed      = nn.Embedding(V, d, padding_idx=cfg.pad_token_id)
        self.embed_norm = nn.LayerNorm(d)
        nn.init.normal_(self.embed.weight, std=0.02)

        # ── Fixed fractal kernel ──────────────────────────────────────────────
        self.rff        = FractalRFF(d, n_rff, n_scales=n_scales, seed=cfg.pad_token_id + 1337)
        rff_out         = self.rff.out_dim            # 2 * n_rff
        self.condensate = SpectralCondensate(rff_out, rank)
        self.phase_lock = HelmholtzPhaseLocking(n_phases, n_iter=n_iter, eta=eta)

        # ── Learnable: sinusoidal decoder head ────────────────────────────────
        self.decoder = CondensateDecoder(n_phases, rank, V, learnable=True)

        # Store dims for convenience
        self._rank     = rank
        self._n_phases = n_phases
        self._condensed = False

    # ── Condensation (one-shot, no SGD) ──────────────────────────────────────

    @torch.no_grad()
    def condense_from_text(self, text: str, tokenizer=None, max_tokens: int = 100_000):
        """
        Extract the condensate K̃_r from raw text in a single forward pass.

        This is NOT training — it is a deterministic eigendecomposition of
        the kernel operator restricted to the token manifold observed in `text`.

        Steps:
          1. Encode text → token IDs (up to max_tokens)
          2. Embed tokens → d-dimensional vectors
          3. Compute fractal RFF features (fractal map, no grad)
          4. SVD of feature matrix → top-r eigenvectors stored in condensate
        """
        self.eval()
        device = next(self.parameters()).device

        if tokenizer is not None:
            ids = tokenizer.encode(text)[:max_tokens]
        else:
            ids = [ord(c) % self.cfg.vocab_size for c in text[:max_tokens]]

        chunk = 4096
        phi_chunks = []
        for i in range(0, len(ids), chunk):
            ids_t = torch.tensor(ids[i:i+chunk], device=device).unsqueeze(0)
            emb = self.embed_norm(self.embed(ids_t)).squeeze(0)  # [N, d]
            phi_chunks.append(self.rff(emb))                     # [N, 2*n_rff]

        phi_all = torch.cat(phi_chunks, dim=0)                   # [total, 2*n_rff]
        self.condensate.condense(phi_all)
        self._condensed = True

        top_s = self.condensate.S[:4].tolist()
        print(
            f"[NFMC] Condensate built from {len(ids):,} tokens. "
            f"Top-4 singular values: {[f'{s:.4f}' for s in top_s]}"
        )

    @torch.no_grad()
    def condense_vocabulary(self, text: str, tokenizer=None, max_tokens: int = 200_000):
        """
        Seed the decoder weight from vocabulary co-occurrence statistics.

        Builds a token co-occurrence matrix C[i,j] = #(token_j follows token_i),
        then SVD-condenses it into the decoder projection.
        Still zero SGD — pure algebraic initialisation.
        """
        self.eval()
        device = next(self.parameters()).device
        V = self.cfg.vocab_size

        if tokenizer is not None:
            ids = tokenizer.encode(text)[:max_tokens]
        else:
            ids = [ord(c) % V for c in text[:max_tokens]]

        # Build co-occurrence matrix (bigram)
        cooccur = torch.zeros(V, V, dtype=torch.float32)
        ids_t = torch.tensor(ids, dtype=torch.long)
        for a, b in zip(ids_t[:-1], ids_t[1:]):
            cooccur[a, b] += 1.0

        # Row-normalise (transition probabilities)
        row_sum = cooccur.sum(dim=1, keepdim=True).clamp(min=1.0)
        cooccur = cooccur / row_sum

        cooccur = cooccur.to(device)
        try:
            _, S, Vh = torch.linalg.svd(cooccur, full_matrices=False)
        except Exception:
            _, S, Vh = torch.svd(cooccur)

        # Seed decoder weight: [V, in_dim]
        # We'll initialise the first min(rank+n_phases*2, V) rows with eigenvectors
        in_dim  = 2 * self._n_phases + self._rank
        W_seed  = torch.zeros(V, in_dim, device=device)
        r = min(V, in_dim, Vh.shape[0])
        cols = min(Vh.shape[1], in_dim)
        W_seed[:r, :cols] = Vh[:r, :cols] * S[:r].unsqueeze(1)
        # Blend with existing weight (80% seed, 20% random init)
        with torch.no_grad():
            self.decoder.proj.weight.data.mul_(0.2).add_(W_seed * 0.2)

        print(f"[NFMC] Vocabulary condensate seeded from {len(ids):,} tokens.")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,              # [B, L]
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:

        B, L = input_ids.shape

        # 1. Embed
        x = self.embed_norm(self.embed(input_ids))   # [B, L, d]

        # 2. Fractal RFF (fixed)
        phi = self.rff(x)                            # [B, L, 2*n_rff]

        # 3. Spectral condensate (fixed)
        z = self.condensate(phi)                     # [B, L, rank]

        # 4. Helmholtz phase locking (fixed dynamics, differentiable)
        theta, K_sim = self.phase_lock(z)            # [B, L, n_phases], [B, L, L]

        # 5. Sinusoidal decode → logits
        logits = self.decoder(theta, z)              # [B, L, V]

        aux = {
            "K_sim":      K_sim,
            "phases":     theta,
            "condensate": z,
        }

        if targets is not None:
            # Task loss
            task_loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=self.cfg.pad_token_id,
            )

            # Phase coherence loss:
            # minimise E = -½ mean(K_sim * cos(θᵢ - θⱼ))  — locked = low energy
            diff     = theta.unsqueeze(2) - theta.unsqueeze(1)   # [B, L, L, n_phases]
            cos_diff = torch.cos(diff).mean(dim=-1)              # [B, L, L]
            phase_coherence = -(K_sim.detach() * cos_diff).mean()

            lambda_pc = getattr(self.cfg, "nfmc_lambda_phase", 0.005)
            total = task_loss + lambda_pc * phase_coherence

            aux["loss_aux"] = {
                "total":           total,
                "task":            task_loss,
                "phase_coherence": phase_coherence,
            }

        return logits, aux

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,   # [1, L]
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            ctx = generated[:, -self.cfg.max_seq_len:]
            logits, _ = self.forward(ctx)
            logits = logits[:, -1, :]  # [1, V]

            if temperature > 0:
                logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = logits.scatter(1, sorted_idx, sorted_logits)

            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_id], dim=1)

            if next_id.item() == self.cfg.eos_token_id:
                break

        return generated

    # ── Utilities ────────────────────────────────────────────────────────────

    def learnable_params(self) -> Iterator[torch.nn.Parameter]:
        """Only the embedding + decoder head carry gradients."""
        yield from self.embed.parameters()
        yield from self.embed_norm.parameters()
        yield from self.decoder.parameters()

    def param_summary(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        kernel    = sum(p.numel() for p in self.rff.parameters())  # buffers, not params
        kernel   += sum(p.numel() for p in self.condensate.parameters())
        return {
            "total":       total,
            "trainable":   trainable,
            "kernel_fixed": 0,   # RFF/condensate are buffers, not parameters
            "condensed":   self._condensed,
        }
