"""
NFN v4.0 — Causal Graph Layer

A differentiable sparse DAG learned over the fractal memory slots.
Enables counterfactual / interventional reasoning (do-calculus style).

Theory
------
Given memory slots M ∈ R^{n_slots × d}, we learn a causal adjacency matrix
A ∈ [0,1]^{n_slots × n_slots} such that:

  • A is a DAG  (no directed cycles)
  • A[i,j] encodes the causal influence of slot i → slot j
  • Interventions: do(M_i = v) sets slot i to v and propagates downstream

The DAG constraint is enforced via the NOTEARS / acyclicity penalty:
    h(A) = tr(e^{A ⊙ A}) - n_slots = 0  iff A is acyclic

Rather than solving the full NOTEARS optimisation (which requires outer loop),
we use a DAG-GNN-style approach:
    A_dag = A ⊙ tril_mask  (lower-triangular = topological ordering by slot index)

This is a strong but practical approximation: slot ordering corresponds to the
fractal hierarchy (slot 0 = coarsest, slot n = finest), which already implies
a natural causal direction (coarse → fine).

Do-Calculus Intervention
------------------------
  do(slot i = v):
    M'_j = M_j + A[i,j] * (v - M_i)  for j downstream of i

This is a linear SCM (Structural Causal Model) intervention.
Non-linear SCMs can be obtained by replacing the linear propagation with
a GNN message-passing step (future work).

Architecture
------------
  CausalEdgeNet   : learns A from slot features
  CausalPropagator: propagates interventions through the DAG
  CausalGraphLayer: full module — read → infer A → propagate → enrich h
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flash_attn import FlashAttention


# ─────────────────────────────────────────────────────────────────────────────
# Causal Edge Network  (learns the adjacency matrix)
# ─────────────────────────────────────────────────────────────────────────────

class CausalEdgeNet(nn.Module):
    """
    Infers a sparse causal adjacency matrix A ∈ [0,1]^{n×n} from slot features.

    Architecture:
      For each pair (i,j), score(i,j) = MLP([m_i ‖ m_j])
      A = sigmoid(score) ⊙ tril_mask  — lower-triangular DAG

    Sparsity is encouraged by an L1 penalty on A (loss_sparse).
    """

    def __init__(self, d_model: int, n_slots: int, hidden: int = 64, sparsity: float = 0.01):
        super().__init__()
        self.n_slots   = n_slots
        self.sparsity  = sparsity

        self.edge_mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.edge_mlp[-1].weight, std=0.01)
        nn.init.constant_(self.edge_mlp[-1].bias, -2.0)  # start sparse

        # Fixed lower-triangular mask (slot 0 → slot n, no self-loops)
        tril = torch.tril(torch.ones(n_slots, n_slots), diagonal=-1)
        self.register_buffer("tril_mask", tril)

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        slots: [B, n_slots, d_model]
        Returns:
          A     : [B, n_slots, n_slots]  — causal adjacency (lower-triangular)
          loss_s: scalar                 — sparsity penalty
        """
        B, n, d = slots.shape
        si = slots.unsqueeze(2).expand(B, n, n, d)  # [B, n, n, d]
        sj = slots.unsqueeze(1).expand(B, n, n, d)  # [B, n, n, d]
        pair = torch.cat([si, sj], dim=-1)           # [B, n, n, 2d]

        scores = self.edge_mlp(pair).squeeze(-1)     # [B, n, n]
        A = torch.sigmoid(scores) * self.tril_mask   # [B, n, n]  — DAG

        loss_s = self.sparsity * A.mean()
        return A, loss_s


# ─────────────────────────────────────────────────────────────────────────────
# Causal Propagator  (linear SCM message passing)
# ─────────────────────────────────────────────────────────────────────────────

class CausalPropagator(nn.Module):
    """
    Propagates causal influence through the DAG.

    Forward pass (observational):
      M_out = M + A^T · M   (each slot receives weighted sum of its parents)

    Intervention do(M_i = v):
      M'_j = M_j + A[i,j] * (v - M_i)  for all j
    """

    def __init__(self, d_model: int, n_slots: int):
        super().__init__()
        self.msg_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.msg_proj.weight)

    def forward(self, slots: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        slots : [B, n_slots, d_model]
        A     : [B, n_slots, n_slots]  — A[b,i,j] = causal weight i→j
        Returns enriched slots [B, n_slots, d_model].
        """
        msg = self.msg_proj(slots)          # [B, n, d]
        # Aggregate: slot j receives Σ_i A[i,j] * msg_i
        received = torch.bmm(A.transpose(1, 2), msg)   # [B, n, d]
        return slots + received * 0.1

    @torch.no_grad()
    def intervene(
        self,
        slots:    torch.Tensor,    # [B, n_slots, d_model]
        A:        torch.Tensor,    # [B, n_slots, n_slots]
        slot_idx: int,             # which slot to intervene on
        value:    torch.Tensor,    # [B, d_model]  — intervention value
    ) -> torch.Tensor:
        """
        do(M_{slot_idx} = value) — hard intervention.
        Propagates the change downstream through A.
        Returns modified slot matrix.
        """
        slots = slots.clone()
        delta = value - slots[:, slot_idx, :]           # [B, d]
        # Downstream influence: A[slot_idx, j] * delta for all j > slot_idx
        downstream = A[:, slot_idx, :].unsqueeze(-1) * delta.unsqueeze(1)  # [B, n, d]
        slots = slots + downstream
        slots[:, slot_idx, :] = value
        return slots


# ─────────────────────────────────────────────────────────────────────────────
# Nonlinear Causal Propagator  (GNN-style message passing)
# ─────────────────────────────────────────────────────────────────────────────

class NonlinearCausalPropagator(nn.Module):
    """
    Nonlinear SCM message passing: replaces linear A^T · M with learned
    nonlinear messages. Each edge (i→j) computes:
    
      msg_ij = MLP([m_i ‖ m_j ‖ A[i,j]])  ∈ R^d
      gate_ij = σ(w_gate · [m_i ‖ m_j])    ∈ [0,1]
    
    Slot j receives:  Δm_j = Σ_{i→j} gate_ij · msg_ij
    
    This enables modelling nonlinear causal mechanisms (X→Y where Y = f(X, noise))
    rather than only linear ones (Y = a·X + noise).
    """

    def __init__(self, d_model: int, n_slots: int, msg_hidden: int = 128):
        super().__init__()
        self.n_slots = n_slots
        
        # Message MLP: takes [m_i, m_j, edge_weight] → message vector
        self.msg_mlp = nn.Sequential(
            nn.Linear(d_model * 2 + 1, msg_hidden),
            nn.SiLU(),
            nn.Linear(msg_hidden, d_model),
        )
        nn.init.zeros_(self.msg_mlp[-1].weight)
        nn.init.zeros_(self.msg_mlp[-1].bias)
        
        # Gating: controls how much each message passes through
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_net[0].bias, -2.0)
        
        self.tril_mask: Optional[torch.Tensor] = None

    def forward(self, slots: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        slots : [B, n_slots, d]
        A     : [B, n_slots, n_slots]  — A[b,i,j] = causal weight i→j
        Returns enriched slots [B, n_slots, d].
        """
        B, n, d = slots.shape
        
        if self.tril_mask is None or self.tril_mask.shape[0] != n:
            self.tril_mask = torch.tril(torch.ones(n, n, device=slots.device), diagonal=-1)
        
        si = slots.unsqueeze(2).expand(B, n, n, d)  # [B, n, n, d] — from
        sj = slots.unsqueeze(1).expand(B, n, n, d)  # [B, n, n, d] — to
        
        A_exp = A.unsqueeze(-1)  # [B, n, n, 1]
        
        # Message: MLP([m_i, m_j, A_ij])
        msg_input = torch.cat([si, sj, A_exp.expand(B, n, n, 1)], dim=-1)
        msgs = self.msg_mlp(msg_input)  # [B, n, n, d]
        
        # Gate: σ(w · [m_i, m_j])
        gates = self.gate_net(torch.cat([si, sj], dim=-1))  # [B, n, n, 1]
        
        # Mask to only pass messages along DAG edges
        mask = self.tril_mask.unsqueeze(0).unsqueeze(-1)  # [1, n, n, 1]
        gated_msgs = msgs * gates * mask  # [B, n, n, d]
        
        # Aggregate: slot j receives Σ_i gated_msgs[i,j]
        received = gated_msgs.sum(dim=1)  # [B, n, d]
        
        return slots + received * 0.1

    @torch.no_grad()
    def intervene(
        self,
        slots:    torch.Tensor,
        A:        torch.Tensor,
        slot_idx: int,
        value:    torch.Tensor,
    ) -> torch.Tensor:
        """
        do(M_{slot_idx} = value) — hard intervention with nonlinear propagation.
        Propagates the DELTA through nonlinear messages for downstream slots.
        """
        slots_new = slots.clone()
        old_val = slots_new[:, slot_idx, :]
        slots_new[:, slot_idx, :] = value
        delta = value - old_val  # [B, d]
        
        # Iterative propagation: recompute messages with updated slots
        for _ in range(3):  # fixed 3 iterations of nonlinear propagation
            B, n, d = slots_new.shape
            si = slots_new.unsqueeze(2).expand(B, n, n, d)
            sj = slots_new.unsqueeze(1).expand(B, n, n, d)
            A_exp = A.unsqueeze(-1)
            
            msg_input = torch.cat([si, sj, A_exp.expand(B, n, n, 1)], dim=-1)
            msgs = self.msg_mlp(msg_input)
            gates = self.gate_net(torch.cat([si, sj], dim=-1))
            
            if self.tril_mask is None or self.tril_mask.shape[0] != n:
                self.tril_mask = torch.tril(torch.ones(n, n, device=slots.device), diagonal=-1)
            mask = self.tril_mask.unsqueeze(0).unsqueeze(-1)
            gated_msgs = msgs * gates * mask
            
            received = gated_msgs.sum(dim=1)
            slots_new = slots_new + received * 0.1
            slots_new[:, slot_idx, :] = value  # re-fix intervened slot
        
        return slots_new


# ─────────────────────────────────────────────────────────────────────────────
# Causal Graph Layer  (full module)
# ─────────────────────────────────────────────────────────────────────────────

class CausalGraphLayer(nn.Module):
    """
    Full causal reasoning module for NFN.

    Takes hidden states h [B, L, d] and a set of memory slots [B, n_slots, d].
    Learns a causal DAG over the slots, propagates causal influence, and
    enriches h with the result.

    Used as a post-processor on the fractal memory bank:
      h_out = h + causal_layer(h, memory_slots)
    """

    def __init__(
        self,
        d_model:   int,
        n_slots:   int,
        hidden:    int   = 64,
        sparsity:       float = 0.01,
        use_nonlinear:  bool  = False,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.d_model = d_model

        # Compress h to slot-sized summaries
        self.slot_proj = nn.Linear(d_model, d_model)

        # Causal components
        self.edge_net   = CausalEdgeNet(d_model, n_slots, hidden, sparsity)
        self.propagator = NonlinearCausalPropagator(d_model, n_slots) if use_nonlinear \
                          else CausalPropagator(d_model, n_slots)

        # Read back into h
        self.read_attn = FlashAttention(d_model, n_heads=4)
        self.norm      = nn.LayerNorm(d_model)

    def forward(
        self,
        h:     torch.Tensor,             # [B, L, d]
        slots: Optional[torch.Tensor] = None,  # [B, n_slots, d]  or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          h_out    : [B, L, d]   — causally enriched hidden states
          loss_dag : scalar      — sparsity penalty on A
        """
        B, L, d = h.shape

        # If no slots provided, derive them from h via pooling
        if slots is None:
            if L < self.n_slots:
                # Sequence shorter than n_slots — pad then pool
                padded = F.pad(h, (0, 0, 0, self.n_slots - L))   # [B, n_slots, d]
                slots = padded
            else:
                chunk  = L // self.n_slots
                keep   = chunk * self.n_slots
                slots  = h[:, :keep].reshape(B, self.n_slots, chunk, d).mean(2)

        slots = self.slot_proj(slots)            # [B, n_slots, d]

        # Infer causal graph
        A, loss_dag = self.edge_net(slots)       # [B, n_slots, n_slots], scalar

        # Propagate causal influence through DAG
        slots_enriched = self.propagator(slots, A)   # [B, n_slots, d]

        # Read causally-enriched information back into h
        h_out = self.read_attn(h, slots_enriched, slots_enriched)
        h_out = self.norm(h + h_out)

        return h_out, loss_dag

    def causal_query(
        self,
        h:        torch.Tensor,       # [B, L, d]
        slot_idx: int,
        value:    torch.Tensor,       # [B, d]
    ) -> torch.Tensor:
        """
        Counterfactual query: what would h look like if slot `slot_idx` were
        set to `value`? (do-calculus intervention)
        """
        B, L, d = h.shape
        if L < self.n_slots:
            slots = F.pad(h, (0, 0, 0, self.n_slots - L))
        else:
            chunk = L // self.n_slots
            slots = h[:, :chunk * self.n_slots].reshape(B, self.n_slots, chunk, d).mean(2)
        slots = self.slot_proj(slots)

        A, _ = self.edge_net(slots)
        slots_intervened = self.propagator.intervene(slots, A, slot_idx, value)
        h_cf = self.read_attn(h, slots_intervened, slots_intervened)
        return self.norm(h + h_cf)
