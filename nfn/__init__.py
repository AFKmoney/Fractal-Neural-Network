"""
FNN v6.0 — Fractal Neural Network

Architecture neurale fractale unifiee, fondee sur:
  - Dynamique de phase Kuramoto (coherence)
  - SCM causal (DAG + do-calculus)
  - Self-Model (espace de travail global)

Modules avances (optionnels via FNNConfig):
  - Gematria hyperbolique (topos semantique de Poincare)
  - Attention holographique AdS/CFT
  - Reseau de tenseurs MERA O(log L)
  - Boucle auto-referentielle de Godel
  - Flot de renormalisation (auto-organisation critique)

Auto-Genese Mathematique: carburant d'apprentissage auto-supervise infini.
"""

from .config import FNNConfig
from .model import FNNModel, build_fnn_model
from .block import FNNBlock
from .gematria import GematriaEmbedding, GematriaAttentionBias
from .phase_ode import KuramotoPhaseLayer, PhaseGoalForcing
from .fractal import FractalLinearAttention, PhaseSoliton, PhaseRoutedMoE
from .causal import CausalGraphLayer, notears_acyclicity
from .self_model import GlobalWorkspace, SelfRepresentor, SelfModel
from .auto_genesis import MathTruthEngine, ConjectureLoop, ProofLoop, SelfModificationController
from .lifecycle import FNNLifecycle

# ── Modules avances ───────────────────────────────────────────────────────────
from .hyperbolic_gematria import (
    PoincareBall, HyperbolicGematriaTable, HyperbolicGematriaAttention,
    SheafTheoryLayer, HyperbolicGematriaModule,
)
from .ads_cft import (
    AdSMetric, AdSBulkProjector, ER_EPR_Bridge, AdSCFTAttention,
)
from .tensor_network import (
    Disentangler, Isometry, MERALayer, MERAAttention, TensorNetworkEncoder,
)
from .godel_loop import (
    SelfReferenceOperator, IncompletenessDetector, GodelFixedPoint,
)
from .rg_flow import (
    ScaleDecomposition, RGFlowLayer, CriticalityOptimizer, RGFlowScheduler,
)

__all__ = [
    # Core
    "FNNConfig", "FNNModel", "FNNBlock", "FNNLifecycle", "build_fnn_model",
    # Gematria
    "GematriaEmbedding", "GematriaAttentionBias",
    # Kuramoto
    "KuramotoPhaseLayer", "PhaseGoalForcing",
    # Fractal
    "FractalLinearAttention", "PhaseSoliton", "PhaseRoutedMoE",
    # Causal
    "CausalGraphLayer", "notears_acyclicity",
    # Self-Model
    "GlobalWorkspace", "SelfRepresentor", "SelfModel",
    # Auto-Genesis
    "MathTruthEngine", "ConjectureLoop", "ProofLoop", "SelfModificationController",
    # Hyperbolic Gematria
    "PoincareBall", "HyperbolicGematriaTable", "HyperbolicGematriaAttention",
    "SheafTheoryLayer", "HyperbolicGematriaModule",
    # AdS/CFT
    "AdSMetric", "AdSBulkProjector", "ER_EPR_Bridge", "AdSCFTAttention",
    # MERA
    "Disentangler", "Isometry", "MERALayer", "MERAAttention", "TensorNetworkEncoder",
    # Godel
    "SelfReferenceOperator", "IncompletenessDetector", "GodelFixedPoint",
    # RG Flow
    "ScaleDecomposition", "RGFlowLayer", "CriticalityOptimizer", "RGFlowScheduler",
    # Compatibilite LEAC (deconseille, retirer dans une future version)
    "LEACConfig", "LEACModel", "LEACBlock", "LEACLifecycle",
]
__version__ = "6.0.0"

# ── Alias de compatibilite (LEAC → FNN) ───────────────────────────────────────
# Conserves temporairement pour eviter de casser le code qui importe encore
# l'ancien nommage. A retirer une fois toute la codebase migree vers FNN*.
LEACConfig = FNNConfig
LEACModel = FNNModel
LEACBlock = FNNBlock
LEACLifecycle = FNNLifecycle
