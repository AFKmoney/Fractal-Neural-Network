"""
Fractal KV-Cache for autoregressive generation.

Design insight: in the NFN fractal hierarchy, higher levels are recomputed
exponentially less often:
  - Level 0: every token (stride = 1)
  - Level 1: every b tokens (stride = b)
  - Level k: every b^k tokens (stride = b^k)

The cache stores:
  1. FractalStateCache : completed level-k representations [B, n_complete, d]
  2. LeafBuffer        : buffered but not-yet-aggregated level-0 states
  3. AttentionKVCache  : K, V tensors for each top-level attention layer

During generation step t:
  - Embed token t → append to LeafBuffer
  - For each level k, if len(LeafBuffer) % b^k == 0:
      recompute level-k aggregation from cache and buffer
      append new parent state to FractalStateCache[k]
  - Run top-down pass using all cached states + buffer
  - Return output at current position
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class AttentionKVCache:
    """
    Standard KV cache for the top-level causal self-attention.
    One instance per NFNBlock layer.
    """

    def __init__(self):
        self.k: Optional[torch.Tensor] = None   # [B, H, T_cached, d_head]
        self.v: Optional[torch.Tensor] = None   # [B, H, T_cached, d_head]
        self.length: int = 0

    def update(
        self,
        new_k: torch.Tensor,   # [B, H, T_new, d_head]
        new_v: torch.Tensor,   # [B, H, T_new, d_head]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K,V and return full cached K,V."""
        if self.k is None:
            self.k, self.v = new_k, new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=2)
            self.v = torch.cat([self.v, new_v], dim=2)
        self.length = self.k.shape[2]
        return self.k, self.v

    def reset(self):
        self.k = self.v = None
        self.length = 0


class FractalStateCache:
    """
    Stores completed fractal representations at every level.

    levels[k] : [B, n_complete_k, d]  — completed parent states
    buffer     : [B, n_buf, d]         — level-0 states not yet aggregated
    """

    def __init__(self, n_levels: int, branching: int, d_model: int):
        self.K = n_levels
        self.b = branching
        self.d = d_model
        self.levels: List[Optional[torch.Tensor]] = [None] * (n_levels + 1)
        self.buffer: Optional[torch.Tensor] = None   # pending level-0 states
        self.n_generated: int = 0                     # total tokens generated

    def reset(self):
        self.levels = [None] * (self.K + 1)
        self.buffer = None
        self.n_generated = 0

    def push_leaf(self, h: torch.Tensor):
        """Add one level-0 state [B, 1, d] to the buffer."""
        self.buffer = h if self.buffer is None else torch.cat([self.buffer, h], dim=1)
        self.n_generated += 1

    def get_all_level0(self) -> Optional[torch.Tensor]:
        """Return all level-0 states seen so far: cached + buffer."""
        if self.levels[0] is None:
            return self.buffer
        if self.buffer is None:
            return self.levels[0]
        return torch.cat([self.levels[0], self.buffer], dim=1)

    def flush_buffer(self, aggregators: nn.ModuleList, device: torch.device):
        """
        Process any complete chunks in the buffer through the aggregation hierarchy.
        Called after each push_leaf when buffer reaches b^k size at some level.
        """
        if self.buffer is None:
            return

        n_buf = self.buffer.shape[1]

        # Level 0: commit completed b-chunks
        if n_buf >= self.b and n_buf % self.b == 0:
            # Push completed level-0 chunks into levels[0]
            n_complete = (n_buf // self.b) * self.b
            self.levels[0] = (
                self.buffer[:, :n_complete]
                if self.levels[0] is None
                else torch.cat([self.levels[0], self.buffer[:, :n_complete]], dim=1)
            )
            # Remaining in buffer (partial chunk)
            remainder = n_buf % self.b
            self.buffer = self.buffer[:, n_complete:] if remainder else None

        # Propagate through aggregation levels
        for k in range(self.K):
            agg = aggregators[k]
            current = self.levels[k]
            if current is None:
                break
            n = current.shape[1]
            n_parents = n // self.b
            if n_parents == 0:
                break
            # Aggregate the complete chunks
            n_used = n_parents * self.b
            chunk = current[:, :n_used]   # [B, n_used, d]
            positions = torch.arange(n_parents, device=device)
            with torch.no_grad():
                parent_h, _ = agg(chunk, positions)   # [B, n_parents, d]

            self.levels[k + 1] = (
                parent_h if self.levels[k + 1] is None
                else torch.cat([self.levels[k + 1], parent_h], dim=1)
            )
            # Keep the tail that wasn't aggregated
            tail = current[:, n_used:]
            self.levels[k] = tail if tail.shape[1] > 0 else None


class NFNKVCache:
    """
    Aggregates all caches for one FNNModel during generation.

    Usage:
        cache = NFNKVCache(cfg)
        # first call (prefill):
        logits, aux = model(prompt_ids, kv_cache=cache)
        # subsequent calls (decode, one token at a time):
        for _ in range(max_new_tokens):
            logits, aux = model(next_id.unsqueeze(1), kv_cache=cache)
    """

    def __init__(self, n_blocks: int, n_levels: int, branching: int, d_model: int):
        # One attention KV cache per block
        self.attn_caches: List[AttentionKVCache] = [
            AttentionKVCache() for _ in range(n_blocks)
        ]
        # One fractal state cache per motif per block
        # (simplified: share across motifs)
        self.fractal_caches: List[FractalStateCache] = [
            FractalStateCache(n_levels, branching, d_model)
            for _ in range(n_blocks)
        ]
        self.step: int = 0   # total tokens processed

    def reset(self):
        for c in self.attn_caches:
            c.reset()
        for c in self.fractal_caches:
            c.reset()
        self.step = 0
