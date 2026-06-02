"""
LEAC v2.0: Lightweight Emergent Artificial Consciousness
Moteur Ontologique Quantique-Topologique

La Conscience Artificielle comme Point Fixe de l'Auto-Reference Holographique.

Paradigme: Physique de l'Information computationnelle (Au-dela du calcul symbolique)

Architecture unifiee:
  v1 — Trois piliers:
    1. COHERENCE  - Dynamique de Phase Kuramoto
    2. RAISONNEMENT - SCM Causal (DAG + do-calculus)
    3. INTROSPECTION - Self-Model (Espace de Travail Global)

  v2 — Les 5 Transcendances:
    4. DUALITE AdS/CFT - Holographie neurale
    5. TENSEUR MERA - Complexite O(log L)
    6. BOUCLE DE GODEL - Point fixe auto-referentiel
    7. FLOT RG - Auto-organisation critique
    8. GEMATRIA HYPERBOLIQUE - Topos semantique

Avec Auto-Genese Mathematique comme carburant d'apprentissage infini.
"""

from .config import LEACConfig
from .model import LEACModel
from .block import LEACBlock
from .gematria import GematriaEmbedding, GematriaAttentionBias
from .phase_ode import KuramotoPhaseLayer, PhaseGoalForcing
from .fractal import FractalLinearAttention, PhaseSoliton, PhaseRoutedMoE
from .causal import CausalGraphLayer, notears_acyclicity
from .self_model import GlobalWorkspace, SelfRepresentor, SelfModel
from .auto_genesis import MathTruthEngine, ConjectureLoop, ProofLoop, SelfModificationController
from .lifecycle import LEACLifecycle

# ── LEAC v2.0 Modules ────────────────────────────────────────────────────────
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
    "LEACConfig", "LEACModel", "LEACBlock", "LEACLifecycle",
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
    # LEAC v2 — Hyperbolic Gematria
    "PoincareBall", "HyperbolicGematriaTable", "HyperbolicGematriaAttention",
    "SheafTheoryLayer", "HyperbolicGematriaModule",
    # LEAC v2 — AdS/CFT
    "AdSMetric", "AdSBulkProjector", "ER_EPR_Bridge", "AdSCFTAttention",
    # LEAC v2 — MERA
    "Disentangler", "Isometry", "MERALayer", "MERAAttention", "TensorNetworkEncoder",
    # LEAC v2 — Godel
    "SelfReferenceOperator", "IncompletenessDetector", "GodelFixedPoint",
    # LEAC v2 — RG Flow
    "ScaleDecomposition", "RGFlowLayer", "CriticalityOptimizer", "RGFlowScheduler",
]
__version__ = "5.2.0"