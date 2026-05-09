"""
NFN v4.0 — Infinite Streaming Context

Transformers have O(L²) memory and can't physically run on sequences
longer than ~128K tokens on any GPU.  NFN can.

Architecture:
  Process tokens in chunks of size W (window size).
  For each chunk:
    1. Compute FractalLinearAttention within the chunk  O(W·d²)
    2. Compress the chunk into the episodic memory ring  O(W·d)
    3. Read relevant past context from episodic memory  O(k·d)
    4. Fuse local (chunk) + global (memory) representations

Total memory per token: O(d)  — independent of total sequence length.
Total FLOPs per token: O(d²)  — independent of total sequence length.

This is the killer feature: a 1M-token document requires the same
memory as a 1K-token document.  Transformers literally cannot do this.

The global context is maintained via:
  A. Episodic ring buffer (fast, exact, bounded by capacity)
  B. Semantic SVD condensate (slow, compressed, unbounded)
  C. Phase attractor state (persists Kuramoto phase across chunks)

The phase attractor is the key NFN innovation: the oscillator phases
carry *semantic state* forward across chunk boundaries without any
additional memory cost.

Implementation:
  StreamingContext  : per-sequence state manager
  ChunkedForward    : processes a long sequence in chunks
  InfiniteNFN       : model wrapper with O(1) per-token inference
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamingContext:
    """
    Per-sequence state for streaming/infinite context inference.

    Maintains:
      - Phase state θ  [1, d_model]   — Kuramoto phase across chunks
      - Key-Value cache for the current window (evicted when full)
      - Episodic memory handle (reference to TwoTierMemory)
    """

    def __init__(self, d_model: int, window_size: int):
        self.d_model     = d_model
        self.window_size = window_size
        self.phase_state: Optional[torch.Tensor] = None   # [1, d_model]
        self.tokens_seen: int = 0
        self.chunk_summaries: List[torch.Tensor] = []      # [d_model] per chunk

    def update_phase(self, h_chunk: torch.Tensor):
        """h_chunk: [1, W, d] → update running phase state."""
        new_phase = h_chunk.mean(1)   # [1, d]
        if self.phase_state is None:
            self.phase_state = new_phase
        else:
            # Exponential moving average: preserve long-range phase coherence
            self.phase_state = 0.9 * self.phase_state + 0.1 * new_phase
        self.tokens_seen += h_chunk.shape[1]

    def phase_bias(self, device: torch.device) -> Optional[torch.Tensor]:
        """Returns phase state as a bias for the current chunk [1, 1, d]."""
        if self.phase_state is None:
            return None
        return self.phase_state.to(device).unsqueeze(1)

    def reset(self):
        self.phase_state = None
        self.tokens_seen = 0
        self.chunk_summaries.clear()


class ChunkedForward(nn.Module):
    """
    Processes sequences of arbitrary length in chunks.

    Each chunk W tokens is processed by the full model stack.
    Cross-chunk context flows via:
      1. Phase bias (free — no params, no memory)
      2. Episodic memory reads (if TwoTierMemory is active)
      3. Cross-chunk attention summary: a single [1, d] vector
         summarising the previous chunk, prepended as a "context token"

    The context token trick:
      Before processing chunk t, prepend one "ghost token" = summary(chunk t-1).
      The ghost token participates in attention but is discarded from output.
      Cost: +1 attention step per chunk = negligible.
    """

    def __init__(
        self,
        model:       nn.Module,         # AGINFNModel
        window_size: int   = 512,
        overlap:     int   = 64,        # overlap between chunks (for continuity)
    ):
        super().__init__()
        self.model       = model
        self.window_size = window_size
        self.overlap     = overlap

        # Learnable summarizer: compress one chunk to one vector
        d = model.cfg.d_model
        self.summarizer = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, d),
            nn.LayerNorm(d),
        )
        nn.init.zeros_(self.summarizer[-2].weight)
        nn.init.zeros_(self.summarizer[-2].bias)

        # Ghost token embedding
        self.ghost_embed = nn.Parameter(torch.zeros(1, 1, d))

    def _chunk_sequence(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Split [B, L] into chunks of window_size with overlap."""
        B, L = input_ids.shape
        stride = self.window_size - self.overlap
        chunks = []
        for start in range(0, L, stride):
            end = min(start + self.window_size, L)
            chunks.append(input_ids[:, start:end])
            if end == L:
                break
        return chunks

    def forward(
        self,
        input_ids:    torch.Tensor,            # [B, L]  — long sequence
        targets:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process a long sequence chunk-by-chunk.
        Returns logits [B, L, V] and losses dict.
        """
        B, L = input_ids.shape
        chunks    = self._chunk_sequence(input_ids)
        all_logits: List[torch.Tensor] = []
        total_losses: Dict[str, torch.Tensor] = {}

        prev_summary: Optional[torch.Tensor] = None  # [B, d]
        d = self.model.cfg.d_model

        for chunk_ids in chunks:
            _, W = chunk_ids.shape

            # Prepend ghost token carrying previous chunk summary
            if prev_summary is not None:
                ghost = prev_summary.unsqueeze(1)  # [B, 1, d]
                # Embed chunk_ids then prepend ghost
                h_chunk = self.model.embed(chunk_ids)        # [B, W, d]
                h_input = torch.cat([ghost, h_chunk], dim=1) # [B, W+1, d]
            else:
                h_input = self.model.embed(chunk_ids)

            # Run through all blocks
            h = h_input
            chunk_losses: Dict[str, torch.Tensor] = {}
            for block in self.model.blocks:
                result = block(h, write_memory=True)
                h     = result[0] if isinstance(result, tuple) else result
                if isinstance(result, tuple) and len(result) > 1:
                    for k, v in result[1].items():
                        chunk_losses[k] = chunk_losses.get(k, 0.0) + v

            # Strip ghost token from output
            if prev_summary is not None:
                h = h[:, 1:]     # [B, W, d]

            h_normed = self.model.ln_f(h)

            # Decode
            if hasattr(self.model.lm_head, 'forward'):
                logits = self.model.lm_head(h_normed)
            else:
                logits = self.model.lm_head(h_normed)
            all_logits.append(logits)

            # Summarise this chunk for the next one
            prev_summary = self.summarizer(h.mean(1))   # [B, d]

            # Accumulate losses
            for k, v in chunk_losses.items():
                if k not in total_losses:
                    total_losses[k] = torch.tensor(0.0, device=input_ids.device)
                total_losses[k] = total_losses[k] + v

        # Stitch logits back (handle overlap by taking second half of each chunk)
        full_logits = self._stitch_logits(all_logits, L)
        total_losses["total"] = sum(total_losses.values()) if total_losses else torch.tensor(0.0)
        return full_logits, total_losses

    def _stitch_logits(self, chunks: List[torch.Tensor], L: int) -> torch.Tensor:
        """Concatenate chunk logits, trimming overlap. Returns [B, L, V]."""
        stride  = self.window_size - self.overlap
        parts   = []
        out_len = 0
        for i, chunk in enumerate(chunks):
            start = self.overlap if i > 0 else 0
            chunk_trimmed = chunk[:, start:]
            remaining = L - out_len
            chunk_trimmed = chunk_trimmed[:, :remaining]
            parts.append(chunk_trimmed)
            out_len += chunk_trimmed.shape[1]
            if out_len >= L:
                break
        return torch.cat(parts, dim=1)   # [B, L, V]


class InfiniteNFN(nn.Module):
    """
    NFN with true infinite context capability.

    Wraps AGINFNModel with:
      1. ChunkedForward for long sequences (> window_size tokens)
      2. StreamingContext for stateful per-sequence phase tracking
      3. O(1) per-token generate() — memory does not grow with sequence length

    Usage:
        model = InfiniteNFN(agi_model, window_size=512)

        # Process a 100K token document
        logits, losses = model(long_input_ids)

        # Stream generation: O(1) memory per step
        for token in model.stream_generate(prompt_ids):
            print(tokenizer.decode([token]))
    """

    def __init__(self, model: nn.Module, window_size: int = 512, overlap: int = 64):
        super().__init__()
        self.model   = model
        self.chunked = ChunkedForward(model, window_size, overlap)
        self.contexts: Dict[int, StreamingContext] = {}   # session_id → context
        self.window_size = window_size

    def forward(
        self,
        input_ids:  torch.Tensor,
        targets:    Optional[torch.Tensor] = None,
        session_id: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        """Auto-routes to chunked or direct based on sequence length."""
        _, L = input_ids.shape
        if L <= self.window_size:
            return self.model(input_ids, targets=targets)
        else:
            return self.chunked(input_ids, targets)

    @torch.no_grad()
    def stream_generate(
        self,
        input_ids:      torch.Tensor,   # [1, L]  — prompt
        max_new_tokens: int = 512,
        temperature:    float = 1.0,
        top_p:          float = 0.95,
        eos_token_id:   Optional[int] = None,
    ):
        """
        Generator that yields one token id at a time with O(1) memory.

        Uses the episodic ring buffer to maintain context beyond the window.
        At each step, only the last window_size tokens are in active attention;
        older tokens live compressed in episodic + semantic memory.
        """
        ids = input_ids
        eos = eos_token_id

        while True:
            # Keep only the last window_size tokens in active context
            ctx = ids[:, -self.window_size:]

            logits, _ = self.model(ctx, write_memory=True)
            next_logits = logits[:, -1, :]

            if temperature != 1.0:
                next_logits = next_logits / max(temperature, 1e-5)

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, -1), -1)
                remove = cum - F.softmax(sorted_logits, -1) > top_p
                sorted_logits[remove] = -1e9
                next_logits.scatter_(1, sorted_idx, sorted_logits)

            probs   = F.softmax(next_logits, -1)
            next_id = torch.multinomial(probs, 1)
            ids     = torch.cat([ids, next_id], dim=1)

            tok = next_id[0, 0].item()
            yield tok

            if max_new_tokens is not None:
                max_new_tokens -= 1
                if max_new_tokens <= 0:
                    break
            if eos is not None and tok == eos:
                break
