"""
NFN v4.0 — Two-Tier Memory: Episodic + Semantic

Inspired by the hippocampus → neocortex consolidation pathway in mammalian
memory:

  Episodic  (hippocampus) : exact event traces, indexed by context hash,
                            fast write, fast indexed read, bounded capacity.
  Semantic  (neocortex)   : slow condensate updated only when episodic
                            patterns repeat above a frequency threshold —
                            no catastrophic forgetting.

Architecture:
  ┌──────────────────────────────────────────────────────┐
  │  Input context  h ∈ [B, L, d]                        │
  │       │                                               │
  │  ┌────▼──────────────────────────┐                        │
  │  │  EpisodicStore            │  fast write O(1)/token │
  │  │  key   = FractalRFF(h)    │  fixed-size ring buf   │
  │  │  value = h                │  nearest-neighbour read│
  │  └────────────┼──────────────┘                        │
  │               │  consolidation (async, freq-triggered) │
  │  ┌────────────▼──────────────┐                        │
  │  │  SemanticConsolidator     │  rank-1 SVD update     │
  │  │  condensate U, S, V       │  no SGD                │
  │  └────────────┼──────────────┘                        │
  │               │                                        │
  │  ┌────────────▼──────────────┐                        │
  │  │  MemoryGate               │  α·episodic + β·semantic│
  │  └───────────────────────────┘                        │
  └──────────────────────────────────────────────────────┘
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Episodic Store  (hippocampus — fast, exact, bounded)
# ─────────────────────────────────────────────────────────────────────────────

class EpisodicStore(nn.Module):
    """
    Ring-buffer of (key, value) pairs.

    key   = L2-normalised FractalRFF projection of the context summary
    value = context summary vector (mean-pooled hidden state)

    Read  : k-NN in key space → weighted sum of values (soft attention)
    Write : append to ring, evict oldest when full
    """

    def __init__(
        self,
        d_model: int,
        capacity: int = 2048,   # max stored episodes
        key_dim: int = 64,      # RFF key dimension
        n_read: int = 8,        # top-k neighbours to read
        n_scales: int = 4,
    ):
        super().__init__()
        self.d_model  = d_model
        self.capacity = capacity
        self.key_dim  = key_dim
        self.n_read   = n_read

        # Fixed random projection for keys (no gradient)
        W = torch.randn(d_model, key_dim // 2)
        self.register_buffer("W_key", W)
        scales = [2.0 ** k for k in range(n_scales)]
        freqs  = torch.tensor(scales).repeat_interleave(key_dim // (2 * n_scales) + 1)[:key_dim // 2]
        self.register_buffer("freq_scales", freqs)

        # Ring buffer — stored as buffers so they survive save/load
        self.register_buffer("keys",   torch.zeros(capacity, key_dim))
        self.register_buffer("values", torch.zeros(capacity, d_model))
        self.register_buffer("ptr",    torch.tensor(0, dtype=torch.long))
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long))

        # Frequency counter for consolidation triggers
        self.register_buffer("freq_count", torch.zeros(capacity))

        # Learnable gate
        self.read_gate = nn.Linear(d_model, 1)

    def _encode_key(self, h: torch.Tensor) -> torch.Tensor:
        """h: [d_model] → key: [key_dim]"""
        proj = h @ self.W_key * self.freq_scales.unsqueeze(0)
        return F.normalize(torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1), dim=-1)

    @torch.no_grad()
    def write(self, h: torch.Tensor):
        """
        h: [B, d_model] — one summary per batch item.
        Writes each to the ring buffer.
        """
        B = h.shape[0]
        keys = self._encode_key(h)                 # [B, key_dim]
        for b in range(B):
            idx = self.ptr.item() % self.capacity
            self.keys[idx]   = keys[b]
            self.values[idx] = h[b].detach()
            self.freq_count[idx] = 0
            self.ptr.add_(1)
            if self.filled < self.capacity:
                self.filled.add_(1)

    def read(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query: [B, d_model]
        Returns:
          retrieved: [B, d_model]  — weighted sum of top-k values
          indices:   [B, n_read]   — indices of matched episodes
        """
        n = self.filled.item()
        if n == 0:
            return torch.zeros_like(query), torch.zeros(query.shape[0], self.n_read, dtype=torch.long)

        q_key = self._encode_key(query)               # [B, key_dim]
        sims  = q_key @ self.keys[:n].T               # [B, n]
        k     = min(self.n_read, n)
        top_v, top_i = torch.topk(sims, k, dim=-1)   # [B, k]

        weights    = F.softmax(top_v * math.sqrt(self.key_dim), dim=-1)  # [B, k]
        top_vals   = self.values[:n][top_i.reshape(-1)].view(*top_i.shape, self.d_model)  # [B, k, d]
        retrieved  = (weights.unsqueeze(-1) * top_vals).sum(1)           # [B, d]

        # Increment frequency counts for matched episodes (consolidation signal)
        with torch.no_grad():
            flat_i = top_i.reshape(-1)
            self.freq_count[flat_i % n] += 1.0

        gate = torch.sigmoid(self.read_gate(query))    # [B, 1]
        return retrieved * gate, top_i

    def most_frequent(self, threshold: float = 3.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (keys, values) of episodes repeated above threshold."""
        n = self.filled.item()
        if n == 0:
            return torch.zeros(0, self.key_dim), torch.zeros(0, self.d_model)
        mask = self.freq_count[:n] >= threshold
        return self.keys[:n][mask], self.values[:n][mask]

    def reset(self):
        self.ptr.zero_()
        self.filled.zero_()
        self.freq_count.zero_()
        self.keys.zero_()
        self.values.zero_()


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Consolidator  (neocortex — slow, compressed, persistent)
# ─────────────────────────────────────────────────────────────────────────────

class SemanticConsolidator(nn.Module):
    """
    Maintains a low-rank condensate of frequently-repeated episodic patterns.

    Update rule (incremental SVD, rank-1):
      Given a new batch of frequent episodes X ∈ R^{N×d},
      extend current U, S, V with a rank-1 update — O(d·r) per step, no SGD.

    This is the mechanism by which "short-term memory becomes long-term knowledge"
    without catastrophic forgetting: the SVD basis rotates smoothly.
    """

    def __init__(self, d_model: int, rank: int = 64, consolidation_freq: float = 3.0):
        super().__init__()
        self.d_model = d_model
        self.rank    = rank
        self.consolidation_freq = consolidation_freq

        # Condensate buffers (initialised to identity-like)
        self.register_buffer("U", torch.zeros(d_model, rank))
        self.register_buffer("S", torch.zeros(rank))
        self.register_buffer("initialized", torch.tensor(False))

        # Projection to read from condensate
        self.out_proj = nn.Linear(rank, d_model, bias=False)

    @torch.no_grad()
    def consolidate(self, values: torch.Tensor):
        """
        values: [N, d_model] — frequent episode vectors.
        Performs a truncated SVD update, keeping rank components.
        """
        if values.shape[0] < 2:
            return

        values = values - values.mean(0)
        _, S_new, Vh = torch.linalg.svd(values, full_matrices=False)
        V_new = Vh[:self.rank].T           # [d_model, min(N,rank)]
        r     = min(self.rank, V_new.shape[1])

        if not self.initialized.item():
            self.U[:, :r] = V_new[:, :r]
            self.S[:r]    = S_new[:r] / (S_new[0] + 1e-8)
            self.initialized.fill_(True)
        else:
            # Merge: blend old U with new V_new via weighted average in SVD space
            alpha = 0.3   # new info weight
            merged = torch.cat([
                self.U * self.S.unsqueeze(0),           # [d, r]  old weighted
                V_new[:, :r] * (alpha * S_new[:r] / (S_new[0] + 1e-8)).unsqueeze(0),
            ], dim=1)                                   # [d, 2r]
            _, S_m, Vh_m = torch.linalg.svd(merged, full_matrices=False)
            r_new = min(self.rank, Vh_m.shape[0])
            self.U[:, :r_new] = Vh_m[:r_new].T
            self.S[:r_new]    = S_m[:r_new] / (S_m[0] + 1e-8)

    def read(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, d_model]
        Returns semantic enrichment [B, d_model] via condensate projection.
        """
        if not self.initialized.item():
            return torch.zeros_like(h)
        coords = h @ self.U              # [B, rank]
        coords = coords * self.S.unsqueeze(0)
        return self.out_proj(coords)     # [B, d_model]


# ─────────────────────────────────────────────────────────────────────────────
# TwoTierMemory  (full hippocampus→neocortex module)
# ─────────────────────────────────────────────────────────────────────────────

class TwoTierMemory(nn.Module):
    """
    Unified episodic + semantic memory with automatic consolidation.

    Usage per forward pass:
      1. read(query)         — blend episodic + semantic retrieval
      2. write(summary)      — store current context in episodic ring
      3. maybe_consolidate() — triggered when episode frequency exceeds threshold
    """

    def __init__(
        self,
        d_model:             int,
        episodic_capacity:   int   = 2048,
        episodic_key_dim:    int   = 64,
        episodic_n_read:     int   = 8,
        semantic_rank:       int   = 64,
        consolidation_freq:  float = 3.0,
        consolidation_every: int   = 50,   # steps between consolidation checks
    ):
        super().__init__()
        self.d_model   = d_model
        self.consol_every = consolidation_every
        self._step        = 0

        self.episodic  = EpisodicStore(
            d_model, episodic_capacity, episodic_key_dim, episodic_n_read,
        )
        self.semantic  = SemanticConsolidator(
            d_model, semantic_rank, consolidation_freq,
        )

        # Blend gate: learn α·episodic + β·semantic + γ·passthrough
        self.blend = nn.Linear(d_model * 3, d_model)
        nn.init.zeros_(self.blend.weight)
        nn.init.zeros_(self.blend.bias)

    def forward(
        self,
        h: torch.Tensor,            # [B, L, d]
        write: bool = True,
    ) -> torch.Tensor:
        """
        Reads episodic + semantic memories, fuses with h, optionally writes.
        Returns enriched h [B, L, d].
        """
        B, L, d = h.shape
        summary = h.mean(1)              # [B, d]

        ep_ret, _  = self.episodic.read(summary)   # [B, d]
        sem_ret    = self.semantic.read(summary)    # [B, d]

        # Expand to sequence length and blend
        ep_exp  = ep_ret.unsqueeze(1).expand(B, L, d)
        sem_exp = sem_ret.unsqueeze(1).expand(B, L, d)
        fused   = self.blend(torch.cat([h, ep_exp, sem_exp], dim=-1))  # [B, L, d]
        out     = h + fused * 0.1

        if write:
            self.episodic.write(summary)
            self._step += 1
            if self._step % self.consol_every == 0:
                self.maybe_consolidate()

        return out

    @torch.no_grad()
    def maybe_consolidate(self):
        """Trigger semantic consolidation from frequent episodic patterns."""
        _, vals = self.episodic.most_frequent(self.semantic.consolidation_freq)
        if vals.shape[0] >= 2:
            self.semantic.consolidate(vals)

    def reset(self):
        self.episodic.reset()
        self._step = 0

    def save_state(self) -> dict:
        return {
            "episodic_keys":   self.episodic.keys.clone(),
            "episodic_values": self.episodic.values.clone(),
            "episodic_ptr":    self.episodic.ptr.clone(),
            "episodic_filled": self.episodic.filled.clone(),
            "episodic_freq":   self.episodic.freq_count.clone(),
            "semantic_U":      self.semantic.U.clone(),
            "semantic_S":      self.semantic.S.clone(),
            "semantic_init":   self.semantic.initialized.clone(),
        }

    def load_state(self, state: dict):
        self.episodic.keys.copy_(state["episodic_keys"])
        self.episodic.values.copy_(state["episodic_values"])
        self.episodic.ptr.copy_(state["episodic_ptr"])
        self.episodic.filled.copy_(state["episodic_filled"])
        self.episodic.freq_count.copy_(state["episodic_freq"])
        self.semantic.U.copy_(state["semantic_U"])
        self.semantic.S.copy_(state["semantic_S"])
        self.semantic.initialized.copy_(state["semantic_init"])
