"""
FNN — Fractal Neural Network (core)

Import surface after quarantine 2026-09-11.
Architecture unchanged: FractalLinearAttention + PhaseSoliton + PhaseRoutedMoE.

Graft modules (PRISM, interface, AGI trainers, multimodal, …)
live under `_quarantine/` on this branch's working copy notes.
They are not imported here.
"""

from .config import FNNConfig
from .model import FNNModel, build_fnn_model
from .block import FNNBlock
from .gematria import GematriaEmbedding, GematriaAttentionBias
from .phase_ode import KuramotoPhaseLayer, PhaseGoalForcing
from .fractal import FractalLinearAttention, PhaseSoliton, PhaseRoutedMoE
from .causal import CausalGraphLayer, notears_acyclicity
from .self_model import GlobalWorkspace, SelfRepresentor, SelfModel
from .lifecycle import FNNLifecycle

__all__ = [
    "FNNConfig",
    "FNNModel",
    "FNNBlock",
    "FNNLifecycle",
    "build_fnn_model",
    "GematriaEmbedding",
    "GematriaAttentionBias",
    "KuramotoPhaseLayer",
    "PhaseGoalForcing",
    "FractalLinearAttention",
    "PhaseSoliton",
    "PhaseRoutedMoE",
    "CausalGraphLayer",
    "notears_acyclicity",
    "GlobalWorkspace",
    "SelfRepresentor",
    "SelfModel",
]
