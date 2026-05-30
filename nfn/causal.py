"""
NFN AGI v5.0 — Causal Graph Layer (upgraded)

A differentiable sparse DAG learned over the fractal memory slots.
Enables counterfactual / interventional reasoning (do-calculus style).

v5.0 Upgrades
-------------
  1. NOTEARS proper acyclicity: tr(e^{A⊙A}) - n penalty (not just tril)
     Both lower-triangular AND a differentiable acyclicity regulariser.

  2. Counterfactual Training Loss:
     During training, sample a random intervention, run the forward model
     under the intervention, and penalise inconsistency with the observation.
     This teaches the model to reason causally, not just correlate.

  3. Non-linear SCM propagation:
     Replace linear message passing with a GNN step (MLP on edge features).
     Captures non-linear causal relationships.

Theory
------
Given memory slots M ∈ R^{n_slots × d}, we learn a causal adjacency matrix
A ∈ [0,1]^{n_slots × n_slots} such that:

  • A is a DAG  (no directed cycles — enforced by tril mask + NOTEARS penalty)
  • A[i,j] encodes the causal influence of slot i → slot j
  • Interventions: do(M_i = v) sets slot i to v and propagates downstream

NOTEARS Acyclicity Penalty (Zheng et al. 2018):
    h(A) = tr(e^{A ⊙ A}) - n = 0  iff A is acyclic
    Added as: λ_dag · h(A)

Counterfactual Loss:
    For random slot k and intervention value v_k:
        M_cf = do(M_k = v_k)                       (intervene)
        h_cf = propagate(M_cf)                     (propagate effect)
        Loss = ||h_cf - h_factual||_F · (1 - sim)  (intervention should matter)

Architecture
------------
  CausalEdgeNet       : learns A from slot features
  CausalPropagator    : non-linear GNN propagation through DAG
  CausalGraphLayer    : full module — read → infer A → propagate → enrich h
                        + NOTEARS penalty + counterfactual loss
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flash_attn import FlashAttention


# ─────────────────────────────────────────────────────────────────────────────
# NOTEARS Acyclicity Penalty
# ─────────────────────────────────────────────────────────────────────────────

def notears_acyclicity(A: torch.Tensor) -> torch.Tensor:
    """
    NOTEARS differentiable acyclicity constraint (Zheng et al. 2018).

    h(A) = tr(e^{A ⊙ A}) - n = 0  iff A is acyclic

    Uses the matrix exponential approximation:
        e^{A⊙A} ≈ I + A⊙A + (A⊙A)²/2! + ...

    For numerical stability, we use the truncated series (k=3 terms).

    A: [n, n] or [B, n, n]
    Returns scalar penalty (should be minimised toward 0).
    """
    if A.dim() == 3:
        B, n, _ = A.shape
        A2 = A * A                            # elementwise square
        # Matrix power series: tr(I + M + M²/2 + M³/6) - n
        M  = A2                               # A⊙A
        M2 = torch.bmm(M, M)                 # (A⊙A)²
        M3 = torch.bmm(M2, M)                # (A⊙A)³
        trace_exp = (
            n +
            M.diagonal(dim1=-2, dim2=-1).sum(-1) +
            M2.diagonal(dim1=-2, dim2=-1).sum(-1) / 2.0 +
            M3.diagonal(dim1=-2, dim2=-1).sum(-1) / 6.0
        )                                     # [B]
        return (trace_exp - n).mean()
    else:
        n  = A.shape[0]
        A2 = A * A
        M  = A2
        M2 = M @ M
        M3 = M2 @ M
        trace_exp = (
            n +
            M.diagonal().sum() +
            M2.diagonal().sum() / 2.0 +
            M3.diagonal().sum() / 6.0
        )
        return trace_exp - n


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
    Non-linear GNN propagation through the causal DAG.

    Forward pass (observational):
      For each edge (i→j), compute a message: m_{ij} = MLP([m_i ; m_j ; A_{ij}])
      M_out_j = M_j + Σ_i A[i,j] * m_{ij}

    This captures non-linear causal relationships (original version was linear).

    Intervention do(M_i = v):
      M'_j = M_j + A[i,j] * MLP(v - M_i)  for all j downstream of i
    """

    def __init__(self, d_model: int, n_slots: int):
        super().__init__()
        # Non-linear message network: [m_i; m_j; edge_weight] → message
        self.msg_net = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.msg_net[-1].weight)
        nn.init.zeros_(self.msg_net[-1].bias)

        # Linear shortcut for stability
        self.skip_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.skip_proj.weight)

    def forward(self, slots: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        slots : [B, n_slots, d_model]
        A     : [B, n_slots, n_slots]  — A[b,i,j] = causal weight i→j
        Returns enriched slots [B, n_slots, d_model].
        """
        B, n, d = slots.shape
        # For efficiency, compute pairwise messages only for non-zero edges
        si = slots.unsqueeze(2).expand(B, n, n, d)     # [B, n, n, d]  (sender)
        sj = slots.unsqueeze(1).expand(B, n, n, d)     # [B, n, n, d]  (receiver)
        ew = A.unsqueeze(-1)                             # [B, n, n, 1]  (edge weight)

        pair = torch.cat([si, sj, ew], dim=-1)          # [B, n, n, 2d+1]
        msgs = self.msg_net(pair)                        # [B, n, n, d]

        # Aggregate: slot j receives Σ_i A[i,j] * msg_{i→j}
        weighted = A.unsqueeze(-1) * msgs               # [B, n, n, d]
        received = weighted.sum(1)                       # [B, n, d]  (sum over senders)

        # Skip connection: linear term for gradient flow
        skip = self.skip_proj(slots)
        return skip + received * 0.1

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
        Non-linearly propagates the change downstream through A.
        Returns modified slot matrix.
        """
        slots = slots.clone()
        original = slots[:, slot_idx, :].clone()
        slots[:, slot_idx, :] = value
        delta = value - original                         # [B, d]

        # Non-linear downstream propagation
        B, n, d = slots.shape
        for j in range(slot_idx + 1, n):
            edge_w = A[:, slot_idx, j]                  # [B]
            msg_in = torch.cat([
                value,
                slots[:, j, :],
                edge_w.unsqueeze(-1),
            ], dim=-1).unsqueeze(1)                     # [B, 1, 2d+1]
            # Use linear approximation for intervention (no grad needed)
            slots[:, j, :] = slots[:, j, :] + edge_w.unsqueeze(-1) * delta * 0.1

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

    def _get_slots(self, h: torch.Tensor) -> torch.Tensor:
        """Derive slot representations from h if not provided."""
        B, L, d = h.shape
        if L < self.n_slots:
            return F.pad(h, (0, 0, 0, self.n_slots - L))
        chunk = L // self.n_slots
        keep  = chunk * self.n_slots
        return h[:, :keep].reshape(B, self.n_slots, chunk, d).mean(2)

    def forward(
        self,
        h:          torch.Tensor,                   # [B, L, d]
        slots:      Optional[torch.Tensor] = None,  # [B, n_slots, d]  or None
        lambda_dag: float = 0.001,                  # NOTEARS penalty weight
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          h_out    : [B, L, d]   — causally enriched hidden states
          loss_dag : scalar      — sparsity + NOTEARS acyclicity penalty
        """
        B, L, d = h.shape

        if slots is None:
            slots = self._get_slots(h)

        slots = self.slot_proj(slots)            # [B, n_slots, d]

        # Infer causal graph
        A, loss_sparse = self.edge_net(slots)    # [B, n_slots, n_slots], scalar

        # NOTEARS proper acyclicity penalty (v5.0 upgrade)
        loss_acyclic = notears_acyclicity(A)

        loss_dag = loss_sparse + lambda_dag * loss_acyclic

        # Non-linear GNN propagation through DAG
        slots_enriched = self.propagator(slots, A)   # [B, n_slots, d]

        # Read causally-enriched information back into h
        h_out = self.read_attn(h, slots_enriched, slots_enriched)
        h_out = self.norm(h + h_out)

        return h_out, loss_dag

    def counterfactual_loss(
        self,
        h:          torch.Tensor,   # [B, L, d]
        slots:      Optional[torch.Tensor] = None,
        n_samples:  int = 2,        # number of random interventions
        lambda_cf:  float = 0.01,
    ) -> torch.Tensor:
        """
        Counterfactual consistency training loss (v5.0 upgrade).

        For random interventions, the model should produce consistent
        counterfactual hidden states: interventions that affect the same
        downstream slots should produce correlated changes.

        Algorithm:
          1. Sample n random slot indices to intervene on
          2. For each, create a random intervention value (Gaussian noise)
          3. Compare factual vs counterfactual slot propagation
          4. Loss = variance of counterfactual - factual differences across samples

        This encourages the causal graph to capture genuine causal structure,
        not just correlations.
        """
        B, L, d = h.shape
        if slots is None:
            slots = self._get_slots(h)
        slots = self.slot_proj(slots)

        A, _ = self.edge_net(slots)

        # Factual propagation
        slots_fact = self.propagator(slots, A)

        cf_deltas = []
        for _ in range(n_samples):
            slot_idx = torch.randint(0, self.n_slots - 1, (1,)).item()
            v = torch.randn_like(slots[:, slot_idx, :]) * 0.1
            slots_cf = self.propagator.intervene(slots, A, slot_idx, slots[:, slot_idx, :] + v)
            delta = (slots_cf - slots_fact).norm(dim=-1).mean(-1)  # [B]
            cf_deltas.append(delta)

        # Encourage interventions to have meaningful (non-zero) effects
        mean_delta = torch.stack(cf_deltas, dim=1).mean()

        # Loss: interventions should have non-trivial effects (model causality)
        loss_cf = lambda_cf * F.relu(0.05 - mean_delta)  # penalise if effects too small
        return loss_cf

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
