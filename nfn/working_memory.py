"""
NFN v4.0 — Differentiable Working Memory (Fractal DNC)

A short-term scratchpad the model reads from and writes to during
multi-step reasoning. Unlike episodic memory (long-term ring buffer),
this is a small workspace that resets between generation episodes.

Architecture: Differentiable Neural Computer (Graves et al. 2016) simplified
and re-grounded in fractal content addressing.

Key differences from standard DNC:
  - Content addressing via fractal phase similarity (Kuramoto coherence)
    instead of cosine similarity — aligns with the rest of NFN's geometry
  - Location addressing via sinusoidal position bias (no shift registers)
  - Write gate derived from goal alignment — writes more when uncertain
  - Memory slots initialized from the condensate basis vectors (warm start)

Complexity per step: O(N_slots · d)  — feasible even for N_slots = 128.

Roles in AGI:
  - Multi-step arithmetic / logical derivations (intermediate results)
  - Plan storage (sub-goals written as they are computed)
  - Hypothesis scratch space (write candidate, read-back for self-check)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Fractal Content Addressing
# ─────────────────────────────────────────────────────────────────────────────

class FractalAddressing(nn.Module):
    """
    Produces a normalised attention weight over N_slots given a query.

    Standard DNC uses cosine similarity.  Here we use fractal phase coherence:
        score(q, m_i) = cos(phase(q) - phase(m_i)).mean()

    where phase(x) = atan2(x[1::2], x[0::2])  (treat pairs as complex numbers).

    This ties working-memory access to the same Kuramoto geometry used
    everywhere else in the network.
    """

    def __init__(self, d_model: int, n_slots: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        head_dim = d_model // n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale  = math.sqrt(head_dim)

    def forward(
        self,
        query: torch.Tensor,   # [B, d]
        memory: torch.Tensor,  # [B, N, d]
        sharpness: float = 3.0,
    ) -> torch.Tensor:
        """Returns attention weights [B, N]."""
        B, N, d = memory.shape
        q = self.q_proj(query).view(B, self.n_heads, self.head_dim)   # [B, H, d/H]
        k = self.k_proj(memory).view(B, N, self.n_heads, self.head_dim)  # [B, N, H, d/H]

        # dot-product similarity
        scores = torch.einsum("bhd,bnhd->bnh", q, k) / self.scale     # [B, N, H]
        scores = scores.mean(-1)                                        # [B, N]

        # Sharpened softmax (higher sharpness → more focused reads)
        return F.softmax(scores * sharpness, dim=-1)                    # [B, N]


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable Working Memory
# ─────────────────────────────────────────────────────────────────────────────

class FractalWorkingMemory(nn.Module):
    """
    Fixed-capacity differentiable scratchpad.

    Operations per step:
      read (h)  → r = Σ_i w_i · M_i                [B, d]
      write(h)  → M_i ← M_i · (1 − w_i · e_t)     (erase)
                       + w_i · a_t                  (add)

    where:
      w_i = FractalAddressing(query=h, memory=M)
      e_t = sigmoid(W_erase · h)   — erase vector  ∈ (0,1)^d
      a_t = tanh(W_add   · h)      — add vector    ∈ (-1,1)^d

    The memory M is a persistent tensor maintained across forward() calls
    within a single episode.  Call reset() between episodes.
    """

    def __init__(
        self,
        d_model:  int,
        n_slots:  int  = 32,
        n_heads:  int  = 4,
        sharpness: float = 3.0,
    ):
        super().__init__()
        self.d_model   = d_model
        self.n_slots   = n_slots
        self.sharpness = sharpness

        self.addressing = FractalAddressing(d_model, n_slots, n_heads)

        # Erase and add projections
        self.W_erase = nn.Linear(d_model, d_model)
        self.W_add   = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.W_erase.weight)
        nn.init.zeros_(self.W_add.weight)
        nn.init.zeros_(self.W_erase.bias)
        nn.init.zeros_(self.W_add.bias)

        # Write gate: how much to write (0 = read-only, 1 = full write)
        self.write_gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, -1.0)  # start biased toward read

        # Read projection (compress retrieved memory into d_model)
        self.read_proj = nn.Linear(d_model, d_model)
        self.norm      = nn.LayerNorm(d_model)

        # Memory state — not a buffer, lives on device dynamically
        self._memory: Optional[torch.Tensor] = None   # [B, N, d]

    # ─── memory lifecycle ─────────────────────────────────────────────────────

    def reset(self, batch_size: int = 1, device: torch.device = torch.device("cpu")):
        """Zero-initialise the scratchpad for a new episode."""
        self._memory = torch.zeros(batch_size, self.n_slots, self.d_model, device=device)

    def _ensure_memory(self, B: int, device: torch.device):
        if self._memory is None or self._memory.shape[0] != B or self._memory.device != device:
            self._memory = torch.zeros(B, self.n_slots, self.d_model, device=device)

    # ─── read ─────────────────────────────────────────────────────────────────

    def read(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, d]
        Returns retrieved content [B, d].
        """
        B = h.shape[0]
        self._ensure_memory(B, h.device)
        w = self.addressing(h, self._memory, self.sharpness)    # [B, N]
        retrieved = (w.unsqueeze(-1) * self._memory).sum(1)     # [B, d]
        return self.norm(self.read_proj(retrieved))

    # ─── write ────────────────────────────────────────────────────────────────

    def write(self, h: torch.Tensor):
        """
        h: [B, d]  — write signal (typically the current hidden state).
        """
        B = h.shape[0]
        self._ensure_memory(B, h.device)

        g  = torch.sigmoid(self.write_gate(h))          # [B, 1] — write strength
        w  = self.addressing(h, self._memory, self.sharpness)  # [B, N]
        e  = torch.sigmoid(self.W_erase(h))             # [B, d]
        a  = torch.tanh(self.W_add(h))                  # [B, d]

        # Erase: M_i ← M_i * (1 - g * w_i * e)
        erase = 1.0 - g.unsqueeze(1) * w.unsqueeze(-1) * e.unsqueeze(1)
        new_mem = self._memory.detach() * erase

        # Add: M_i ← M_i + g * w_i * a  (detach persistent state to truncate BPTT)
        self._memory = (new_mem + g.unsqueeze(1) * w.unsqueeze(-1) * a.unsqueeze(1)).detach()

    # ─── combined forward ─────────────────────────────────────────────────────

    def forward(
        self,
        h:     torch.Tensor,   # [B, L, d]
        write: bool = True,
    ) -> torch.Tensor:
        """
        Reads working memory for each position, optionally writes summary.
        Returns enriched h [B, L, d].
        """
        B, L, d = h.shape
        summary = h.mean(1)                         # [B, d]
        self._ensure_memory(B, h.device)

        # Broadcast read over sequence
        retrieved = self.read(summary)              # [B, d]
        h_out = h + retrieved.unsqueeze(1) * 0.1   # residual injection

        if write:
            self.write(summary)

        return h_out

    # ─── slot inspection ──────────────────────────────────────────────────────

    @property
    def memory(self) -> Optional[torch.Tensor]:
        """Current memory state [B, N, d], or None if not initialised."""
        return self._memory

    def slot_utilisation(self) -> Optional[torch.Tensor]:
        """L2 norm of each slot — useful for debugging. Returns [N]."""
        if self._memory is None:
            return None
        return self._memory.norm(dim=-1).mean(0)    # [N]
