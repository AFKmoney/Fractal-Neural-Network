"""
Persistent inter-context working memory for the NFN.

Implements a differentiable memory bank of M slots that:
  1. READS  : the model queries memory via cross-attention at every forward pass
  2. WRITES : memory is updated via a gated write mechanism after each call
  3. PERSISTS: memory state is serializable (save/load between conversations)

Architecture:
    Memory  [M, d]  ← M learned slot vectors

    Read  : Q = LayerNorm(x) @ W_q
            K, V = Memory @ W_k, Memory @ W_v
            output = softmax(QKᵀ/√d) @ V

    Write : gate  = sigmoid(x_summary @ W_gate)   # [M] importance
            new_m = x_summary @ W_write            # [M, d] candidate
            Memory ← (1 - gate)·Memory + gate·new_m   (per-slot EMA)

The memory is a nn.Parameter when persistent=False (reset each episode),
or a registered buffer when persistent=True (preserved across calls).

Inspiration:
  - Graves et al. (2016) Neural Turing Machine
  - Bulatov et al. (2022) Recurrent Memory Transformer
  - Munkhdalai et al. (2019) Metalearned Neural Memory
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorkingMemory(nn.Module):
    """
    M-slot persistent working memory with cross-attention read and gated write.

    Parameters
    ----------
    d_model  : hidden dimension
    n_slots  : M, number of memory slots
    n_heads  : attention heads for memory read
    dropout  : attention dropout
    """

    def __init__(
        self,
        d_model: int,
        n_slots: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.M = n_slots
        self.d = d_model
        self.H = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Persistent memory slots (initialised to small random values)
        self.memory_init = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)

        # Read projections (cross-attention: Q from context, K/V from memory)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Write projections
        self.W_write = nn.Linear(d_model, d_model, bias=False)
        self.W_gate  = nn.Linear(d_model, n_slots, bias=True)

        # Layer norms
        self.norm_ctx = nn.LayerNorm(d_model)
        self.norm_mem = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

        self.attn_drop = nn.Dropout(dropout)

        # Runtime memory state (not a parameter — managed externally)
        # shape: [B, M, d]  or None (use memory_init)
        self._state: Optional[torch.Tensor] = None

    # ── State management ──────────────────────────────────────────────────────

    def reset(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """Reset memory to initial values."""
        dev = device or self.memory_init.device
        self._state = self.memory_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(dev)

    def get_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return current memory state [B, M, d], initialising if needed."""
        if self._state is None or self._state.shape[0] != batch_size:
            self.reset(batch_size, device)
        return self._state.to(device)

    def set_state(self, state: torch.Tensor):
        """Restore memory state (e.g. loaded from disk)."""
        self._state = state.detach()

    def save_state(self) -> Optional[torch.Tensor]:
        """Return a detached copy of the current state for serialisation."""
        return self._state.detach().clone() if self._state is not None else None

    # ── Read: cross-attention (context → memory) ──────────────────────────────

    def read(
        self,
        x: torch.Tensor,        # [B, L, d]  context
        memory: torch.Tensor,   # [B, M, d]
    ) -> torch.Tensor:
        """
        Read from memory via multi-head cross-attention.
        Returns memory-enriched context [B, L, d].
        """
        B, L, d = x.shape
        x_n = self.norm_ctx(x)
        m_n = self.norm_mem(memory)

        def split_heads(t):
            return t.view(t.shape[0], t.shape[1], self.H, self.d_head).transpose(1, 2)

        Q = split_heads(self.W_q(x_n))    # [B, H, L, d_head]
        K = split_heads(self.W_k(m_n))    # [B, H, M, d_head]
        V = split_heads(self.W_v(m_n))    # [B, H, M, d_head]

        # Scaled dot-product (no causal mask — all memory slots are visible)
        attn = (Q @ K.transpose(-2, -1)) * self.scale   # [B, H, L, M]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, L, d)  # [B, L, d]
        return self.norm_out(x + self.W_o(out))

    # ── Write: gated update ──────────────────────────────────────────────────

    def write(
        self,
        x: torch.Tensor,        # [B, L, d]  context
        memory: torch.Tensor,   # [B, M, d]  current memory
    ) -> torch.Tensor:
        """
        Update memory using a summary of the current context.
        Returns updated memory [B, M, d].
        """
        # Summarise context: mean pooling
        summary = x.mean(dim=1)   # [B, d]

        # Gate: how much to update each slot
        gate = torch.sigmoid(self.W_gate(summary))   # [B, M]
        gate = gate.unsqueeze(-1)                     # [B, M, 1]

        # Candidate new values
        candidate = self.W_write(summary)             # [B, d]
        candidate = candidate.unsqueeze(1).expand(-1, self.M, -1)  # [B, M, d]

        # EMA-style gated write
        new_memory = (1 - gate) * memory + gate * candidate
        return new_memory

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,           # [B, L, d]
        update_memory: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory, optionally write back.

        Returns
        -------
        x_enriched : [B, L, d]   context enriched with memory
        memory     : [B, M, d]   updated memory state
        """
        B, L, d = x.shape
        memory = self.get_state(B, x.device)

        # Read
        x_enriched = self.read(x, memory)

        # Write
        if update_memory:
            new_memory = self.write(x, memory)
            self._state = new_memory.detach()   # detach to avoid unbounded backprop
            memory = new_memory

        return x_enriched, memory


# ─────────────────────────────────────────────────────────────────────────────
# Fractal memory: memory organised at each fractal level
# (higher levels = more abstract, persistent memories)
# ─────────────────────────────────────────────────────────────────────────────

class FractalMemoryBank(nn.Module):
    """
    K+1 independent memory banks, one per fractal level.
    Higher levels store more abstract and longer-term memories.

    Level 0: episodic / token-level (fast refresh)
    Level K: semantic / document-level (slow refresh)
    """

    def __init__(self, d_model: int, n_levels: int, n_slots_per_level: int = 32,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Fewer slots at lower levels (token-level needs less long-term storage)
        slots = [max(8, n_slots_per_level // (2 ** k)) for k in range(n_levels + 1)]
        self.banks = nn.ModuleList([
            WorkingMemory(d_model, n_slots=s, n_heads=n_heads, dropout=dropout)
            for s in slots
        ])
        self.n_levels = n_levels

    def forward(
        self,
        level_repr: list,        # list of [B, N_k, d] tensors, one per level
        update: bool = True,
    ) -> list:
        """Enrich each level's representation with its dedicated memory bank."""
        enriched = []
        for k, (bank, h) in enumerate(zip(self.banks, level_repr)):
            h_enriched, _ = bank(h, update_memory=update)
            enriched.append(h_enriched)
        return enriched

    def reset_all(self, batch_size: int = 1, device: Optional[torch.device] = None):
        for bank in self.banks:
            bank.reset(batch_size, device)

    def save_states(self) -> list:
        return [b.save_state() for b in self.banks]

    def load_states(self, states: list):
        for bank, state in zip(self.banks, states):
            if state is not None:
                bank.set_state(state)
