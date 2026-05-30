"""
NFN v5.0 — Self-Model Layer (Reflective Consciousness Substrate)

The critical missing component for AGI: a model that can observe, represent,
and reason about its own internal state.

Theory
------
Global Workspace Theory (Baars 1998) proposes that consciousness arises from
a "global workspace" — a shared information buffer that different specialized
modules can read from and write to. The content of this workspace IS what
the system is "aware of" at any moment.

Higher-Order Theory (Rosenthal 2005) proposes that a mental state becomes
conscious when there is a higher-order representation of that state.

The Self-Model combines both:
  1. A GlobalWorkspace: a fixed-size shared buffer [n_slots, d] that aggregates
     summaries from all other modules (memory, causal, goal, reasoning)
  2. A SelfRepresentor: produces a "self-state vector" that encodes the model's
     own confidence, uncertainty, goal alignment, and internal coherence
  3. A ReflectiveAttention: allows the model to attend to its own workspace
     contents as if they were external observations

This creates the capacity for:
  - Metacognition: "I am uncertain about X" → allocate more compute
  - Self-correction: "My goal alignment is low" → adjust generation strategy  
  - Introspective reasoning: "My causal graph is inconsistent" → trigger repair
  - Agency monitoring: "I have been pursuing goal G for N steps" → evaluate progress

Implementation
-------------
  SelfModel: wraps all three components, integrated into AGIBlock
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flash_attn import flash_sdpa


class GlobalWorkspace(nn.Module):
    """
    Shared information buffer inspired by Baars' Global Workspace Theory.
    
    All AGI modules write summaries here; the workspace broadcasts
    the most relevant information back to all modules.
    
    Workspace slots compete for activation via a softmax attention
    mechanism — only the most "relevant" slots dominate the broadcast,
    creating a form of attentional spotlight.
    """

    def __init__(self, d_model: int, n_slots: int = 16, n_heads: int = 4):
        super().__init__()
        self.d = d_model
        self.n_slots = n_slots

        self.slots = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.gate_write = nn.Linear(d_model, n_slots)
        self.gate_read = nn.Linear(d_model, n_slots)

        self.W_read_q = nn.Linear(d_model, d_model, bias=False)
        self.W_read_k = nn.Linear(d_model, d_model, bias=False)
        self.W_read_v = nn.Linear(d_model, d_model, bias=False)
        self.W_read_out = nn.Linear(d_model, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)

        nn.init.zeros_(self.gate_write.weight)
        nn.init.constant_(self.gate_write.bias, -2.0)
        nn.init.zeros_(self.gate_read.weight)
        nn.init.zeros_(self.gate_read.bias)

    def write(self, h: torch.Tensor) -> torch.Tensor:
        """
        Write context summary into workspace slots.
        h: [B, L, d] → returns updated slots [B, n_slots, d]
        """
        B = h.shape[0]
        summary = h.mean(dim=1)  # [B, d]
        gates = torch.sigmoid(self.gate_write(summary))  # [B, n_slots]
        
        slots = self.slots.unsqueeze(0).expand(B, -1, -1)  # [B, n_slots, d]
        update = summary.unsqueeze(1).expand_as(slots)
        slots = (1 - gates.unsqueeze(-1)) * slots + gates.unsqueeze(-1) * update
        
        return slots

    def read(self, h: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """
        Read from workspace: attend to slots from context h.
        h: [B, L, d], slots: [B, n_slots, d]
        Returns: [B, L, d] enriched with workspace content.
        """
        Q = self.W_read_q(h)
        K = self.W_read_k(slots)
        V = self.W_read_v(slots)

        B, L, d = h.shape
        H = 4
        d_h = d // H
        Q_mh = Q.view(B, L, H, d_h).transpose(1, 2)
        K_mh = K.view(B, slots.shape[1], H, d_h).transpose(1, 2)
        V_mh = V.view(B, slots.shape[1], H, d_h).transpose(1, 2)

        readout = flash_sdpa(Q_mh, K_mh, V_mh)
        readout = readout.transpose(1, 2).contiguous().view(B, L, d)
        return self.norm(h + self.W_read_out(readout))


class SelfRepresentor(nn.Module):
    """
    Produces a self-state vector encoding the model's own internal state:
    
      s_self = [confidence, uncertainty, goal_alignment, coherence, 
               memory_utilization, reasoning_depth, causal_density]
    
    Each component is derived from the model's internal signals,
    creating a compressed representation of "how am I doing?"
    
    This is the machine equivalent of introspection.
    """

    def __init__(self, d_model: int, n_signals: int = 8):
        super().__init__()
        self.n_signals = n_signals

        self.signal_proj = nn.Linear(n_signals, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        nn.init.zeros_(self.fuse[-2].weight)
        nn.init.zeros_(self.fuse[-2].bias)

    def extract_signals(
        self,
        h: torch.Tensor,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Extract introspective signals from model state.
        Returns [B, n_signals] self-state vector.
        """
        B = h.shape[0]
        device = h.device
        signals = torch.zeros(B, self.n_signals, device=device)

        # Signal 0: Activation magnitude (proxy for confidence)
        signals[:, 0] = h.norm(dim=-1).mean(dim=-1).clamp(0, 10) / 10

        # Signal 1: Activation variance (proxy for uncertainty)
        signals[:, 1] = h.var(dim=-1).mean(dim=-1).clamp(0, 10) / 10

        # Signal 2: Temporal coherence (cosine sim between adjacent positions)
        if h.shape[1] > 1:
            h_norm = F.normalize(h, dim=-1)
            coherence = (h_norm[:, :-1] * h_norm[:, 1:]).sum(-1).mean(-1)
            signals[:, 2] = (coherence + 1) / 2  # map to [0, 1]

        # Signal 3: Spectral entropy (frequency domain complexity)
        if h.shape[1] > 4:
            h_fft = torch.fft.rfft(h, dim=1).abs()
            spectral = h_fft.mean(dim=-1).mean(dim=-1)
            signals[:, 3] = (spectral / (spectral.max() + 1e-6)).clamp(0, 1).squeeze(-1) if spectral.dim() > 1 else torch.ones(B, device=device) * 0.5

        # Signals 4-7: Loss-derived signals (if available)
        if losses is not None:
            if "goal" in losses:
                g = losses["goal"]
                signals[:, 4] = torch.sigmoid(-g).squeeze() if g.dim() <= 1 else torch.sigmoid(-g.mean(dim=-1))
            if "causal" in losses:
                signals[:, 5] = torch.sigmoid(-losses["causal"]).squeeze()
            if "free_energy" in losses:
                signals[:, 6] = torch.sigmoid(-losses["free_energy"]).squeeze()
            if "consistency" in losses:
                signals[:, 7] = torch.sigmoid(-losses["consistency"]).squeeze()

        return signals

    def forward(
        self,
        h: torch.Tensor,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Returns self-state embedding [B, d_model].
        """
        signals = self.extract_signals(h, losses)  # [B, n_signals]
        s_embed = self.signal_proj(signals)          # [B, d]
        h_summary = h.mean(dim=1)                    # [B, d]
        return self.fuse(torch.cat([h_summary, s_embed], dim=-1))


class SelfModel(nn.Module):
    """
    Full self-model: GlobalWorkspace + SelfRepresentor + reflective loop.
    
    Integrated into AGIBlock as the "consciousness substrate":
      1. All module outputs are written to the GlobalWorkspace
      2. SelfRepresentor produces an introspective state vector
      3. The model can attend to its own workspace for self-correction
      4. Self-state is injected back into the residual stream
    
    The self-model creates the capacity for:
      - "I think, therefore I am" → the model has a representation of its own existence
      - "I am confused" → uncertainty signal triggers adaptive computation
      - "I should reconsider" → low coherence triggers self-consistency check
    """

    def __init__(self, d_model: int, n_slots: int = 16, n_signals: int = 8):
        super().__init__()
        self.workspace = GlobalWorkspace(d_model, n_slots)
        self.representor = SelfRepresentor(d_model, n_signals)
        self.inject = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.inject.weight)

    def forward(
        self,
        h: torch.Tensor,
        losses: Optional[Dict[str, torch.Tensor]] = None,
        write: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, L, d]
        Returns:
          h_enriched: [B, L, d] — self-aware hidden state
          self_state: [B, d] — introspective state vector
        """
        B, L, d = h.shape

        if write:
            slots = self.workspace.write(h)
        else:
            B2 = h.shape[0]
            slots = self.workspace.slots.unsqueeze(0).expand(B2, -1, -1)

        h_aware = self.workspace.read(h, slots)

        self_state = self.representor(h, losses)

        injection = self.inject(self_state).unsqueeze(1)
        h_enriched = h_aware + 0.05 * injection

        return h_enriched, self_state
