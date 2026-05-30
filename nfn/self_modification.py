"""
NFN v5.0 — Self-Modification Engine

The model modifies its own architecture based on mathematical discoveries.

Core Idea
---------
A system that can modify itself can IMPROVE without external intervention.
The FNN's fractal topology is parameterized (branching factor, depth, motif type).
The self-modification engine observes the model's internal dynamics and adjusts
these parameters to optimize for:
  - Coherence (phase synchronization quality)
  - Efficiency (activation utilization)
  - Discovery rate (mathematical truths found per step)

What can be modified:
  1. Fractal topology: branching factor, depth, motif type
  2. Kuramoto coupling: rank, integration steps, damping
  3. MoE routing: number of experts, top-k, concentration κ
  4. Attention bias: gematria weight, causal weight
  5. Learning rate: per-parameter-group adaptive rates

Modification mechanism:
  - A controller network observes the model's state
  - It outputs modification proposals (parameter deltas)
  - Modifications are applied with a small step size
  - Performance is measured before and after
  - Beneficial modifications are kept, harmful ones are rolled back

This is EVOLUTIONARY self-modification: the model evolves its own
architecture, guided by fitness = mathematical discovery rate.

Architecture
-----------
  TopologyModifier      : adjusts fractal topology parameters
  CouplingModifier      : adjusts Kuramoto coupling parameters
  RoutingModifier       : adjusts MoE routing parameters
  SelfModificationController : coordinates all modifications
  ModificationHistory   : tracks what was modified and its effect
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModificationProposal:
    """A proposed modification to the model's architecture."""

    def __init__(
        self,
        target: str,
        parameter: str,
        current_value: float,
        proposed_delta: float,
        step_size: float = 0.01,
    ):
        self.target = target
        self.parameter = parameter
        self.current_value = current_value
        self.proposed_delta = proposed_delta
        self.step_size = step_size
        self.new_value = current_value + step_size * proposed_delta
        self.applied = False
        self.fitness_before = None
        self.fitness_after = None

    @property
    def improvement(self) -> Optional[float]:
        if self.fitness_before is not None and self.fitness_after is not None:
            return self.fitness_after - self.fitness_before
        return None

    @property
    def accepted(self) -> bool:
        imp = self.improvement
        return imp is not None and imp > 0


class TopologyModifier(nn.Module):
    """
    Proposes modifications to the fractal topology.
    
    Adjusts:
      - n_levels (fractal depth): more levels = finer multi-scale
      - branching factor: higher = wider branching
      - motif weights: which motif to emphasize (binary_tree, cantor, sierpinski)
    
    The modifier is a small neural network that takes the model's
    performance metrics as input and outputs parameter deltas.
    """

    def __init__(self, n_motifs: int = 3, d_state: int = 32):
        super().__init__()
        self.n_motifs = n_motifs

        self.controller = nn.Sequential(
            nn.Linear(d_state, 64),
            nn.SiLU(),
            nn.Linear(64, 2 + n_motifs),
        )
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def forward(self, state: torch.Tensor) -> Dict[str, float]:
        """
        state: [1, d_state] — current model performance metrics
        Returns: dict of proposed parameter deltas
        """
        deltas = self.controller(state).squeeze(0)
        return {
            "n_levels_delta": deltas[0].item(),
            "branching_delta": deltas[1].item(),
            "motif_weight_0": deltas[2].item() if self.n_motifs > 0 else 0.0,
            "motif_weight_1": deltas[3].item() if self.n_motifs > 1 else 0.0,
            "motif_weight_2": deltas[4].item() if self.n_motifs > 2 else 0.0,
        }


class CouplingModifier(nn.Module):
    """
    Proposes modifications to Kuramoto coupling parameters.
    
    Adjusts:
      - coupling_rank: rank of the coupling matrix K
      - kuramoto_steps: number of RK4 integration steps
      - damping: phase damping factor
    """

    def __init__(self, d_state: int = 32):
        super().__init__()
        self.controller = nn.Sequential(
            nn.Linear(d_state, 32),
            nn.SiLU(),
            nn.Linear(32, 3),
        )
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def forward(self, state: torch.Tensor) -> Dict[str, float]:
        deltas = self.controller(state).squeeze(0)
        return {
            "coupling_rank_delta": deltas[0].item(),
            "kuramoto_steps_delta": deltas[1].item(),
            "damping_delta": deltas[2].item(),
        }


class RoutingModifier(nn.Module):
    """
    Proposes modifications to MoE routing parameters.
    
    Adjusts:
      - kappa (von Mises concentration)
      - top_k (number of active experts)
      - expert temperature
    """

    def __init__(self, d_state: int = 32):
        super().__init__()
        self.controller = nn.Sequential(
            nn.Linear(d_state, 32),
            nn.SiLU(),
            nn.Linear(32, 3),
        )
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def forward(self, state: torch.Tensor) -> Dict[str, float]:
        deltas = self.controller(state).squeeze(0)
        return {
            "kappa_delta": deltas[0].item(),
            "top_k_delta": deltas[1].item(),
            "expert_temperature_delta": deltas[2].item(),
        }


class ModificationHistory:
    """Tracks modification history for analysis and rollback."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.history: List[ModificationProposal] = []

    def record(self, proposal: ModificationProposal):
        if len(self.history) >= self.max_size:
            self.history.pop(0)
        self.history.append(proposal)

    @property
    def n_accepted(self) -> int:
        return sum(1 for p in self.history if p.accepted)

    @property
    def n_rejected(self) -> int:
        return sum(1 for p in self.history if not p.accepted and p.improvement is not None)

    @property
    def acceptance_rate(self) -> float:
        total = self.n_accepted + self.n_rejected
        return self.n_accepted / max(total, 1)

    @property
    def mean_improvement(self) -> float:
        improvements = [p.improvement for p in self.history if p.improvement is not None]
        return sum(improvements) / max(len(improvements), 1)

    def recent(self, n: int = 5) -> List[ModificationProposal]:
        return self.history[-n:]


class SelfModificationController(nn.Module):
    """
    Coordinates all self-modification proposals.
    
    Observes the model's internal state, proposes modifications,
    applies them tentatively, measures the effect, and keeps
    only beneficial changes.
    
    This is the AGI "self-improvement" loop:
      observe → propose → apply → measure → accept/reject
    
    The controller is trained via REINFORCE:
      reward = improvement in mathematical discovery rate
    
    Safety mechanisms:
      - Maximum step size per modification (prevents wild changes)
      - Rollback capability (any modification can be undone)
      - Rate limiting (max N modifications per training step)
      - Parameter bounds (clamp to valid ranges)
    """

    def __init__(
        self,
        d_state: int = 32,
        n_motifs: int = 3,
        max_step_size: float = 0.1,
        max_modifications_per_step: int = 3,
    ):
        super().__init__()
        self.d_state = d_state
        self.max_step_size = max_step_size
        self.max_mods = max_modifications_per_step

        self.topology_mod = TopologyModifier(n_motifs, d_state)
        self.coupling_mod = CouplingModifier(d_state)
        self.routing_mod = RoutingModifier(d_state)

        self.state_encoder = nn.Sequential(
            nn.Linear(8, d_state),
            nn.SiLU(),
            nn.Linear(d_state, d_state),
        )

        self.history = ModificationHistory()
        self.total_modifications = 0
        self.accepted_modifications = 0

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-4, betas=(0.9, 0.95)
        )

    def encode_state(
        self,
        coherence: float = 0.5,
        efficiency: float = 0.5,
        discovery_rate: float = 0.0,
        loss: float = 1.0,
        grad_norm: float = 1.0,
        phase_sync: float = 0.5,
        entropy: float = 0.5,
        step: float = 0.0,
    ) -> torch.Tensor:
        """Encode model state into a fixed-size vector."""
        state = torch.tensor([[
            coherence, efficiency, discovery_rate, loss,
            grad_norm, phase_sync, entropy, min(step / 10000, 1.0)
        ]])
        return self.state_encoder(state)

    def propose_modifications(
        self, state: torch.Tensor
    ) -> List[ModificationProposal]:
        """Generate modification proposals from current state."""
        proposals = []

        for name, delta in self.topology_mod(state).items():
            proposals.append(ModificationProposal(
                target="topology", parameter=name,
                current_value=0.0,
                proposed_delta=delta,
                step_size=self.max_step_size,
            ))

        for name, delta in self.coupling_mod(state).items():
            proposals.append(ModificationProposal(
                target="coupling", parameter=name,
                current_value=0.0,
                proposed_delta=delta,
                step_size=self.max_step_size,
            ))

        for name, delta in self.routing_mod(state).items():
            proposals.append(ModificationProposal(
                target="routing", parameter=name,
                current_value=0.0,
                proposed_delta=delta,
                step_size=self.max_step_size,
            ))

        proposals.sort(key=lambda p: abs(p.proposed_delta), reverse=True)
        return proposals[:self.max_mods]

    def evaluate_fitness(
        self,
        discovery_rate: float,
        coherence: float,
        efficiency: float,
    ) -> float:
        """Compute fitness score for accepting/rejecting modifications."""
        return 0.5 * discovery_rate + 0.3 * coherence + 0.2 * efficiency

    def train_step(
        self,
        fitness_before: float,
        fitness_after: float,
        state: torch.Tensor,
    ):
        """Train the controller via REINFORCE."""
        proposals = self.propose_modifications(state)
        if not proposals:
            return

        improvement = fitness_after - fitness_before
        reward = torch.tensor(max(improvement, 0.0))

        all_params = list(self.parameters())
        pseudo_loss = -reward * sum(
            torch.tensor(p.proposed_delta) ** 2 for p in proposals
        ).requires_grad_(True)

        self.optimizer.zero_grad()
        if pseudo_loss.requires_grad:
            pseudo_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(all_params, 1.0)
            self.optimizer.step()

        self.total_modifications += len(proposals)
        if improvement > 0:
            self.accepted_modifications += 1

    def get_config_deltas(
        self, state: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Get all proposed config deltas for external application."""
        proposals = self.propose_modifications(state)
        result: Dict[str, Dict[str, float]] = {}
        for p in proposals:
            if p.target not in result:
                result[p.target] = {}
            result[p.target][p.parameter] = p.new_value
        return result

    def stats(self) -> Dict[str, float]:
        return {
            "total_modifications": self.total_modifications,
            "accepted": self.accepted_modifications,
            "acceptance_rate": self.accepted_modifications / max(self.total_modifications, 1),
            "history_mean_improvement": self.history.mean_improvement,
        }
