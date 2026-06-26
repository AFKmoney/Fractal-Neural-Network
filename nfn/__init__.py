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

# ── Modules optionnels (recuperes de UNUSED/misc, Phase 2) ────────────────────
from .ssm import FractalSSM, SSMBlock, HybridNFNBlock
from .mixture_of_depths import MixtureOfDepths, MoDRouter
from .hyper import ContextHyperNet, HyperAdapter, HyperLinear
from .predictive import PredictiveCodingBlock, FreeEnergyMinimiser
from .multi_token_pred import MultiTokenPredictor, MTPHead
from .reasoning import RecursiveReasoner, HaltingUnit, SelfConsistencyCheck
from .program_synthesis import ProgramSynthesizer
from .multimodal import MultimodalFractalRFF, CrossModalSync
from .phase_ode import HierarchicalGoalDecomposer

# ── Infrastructure long-contexte (orphelins rebranches, Phase 3) ─────────────
from .rope import RoPECache, precompute_freqs_cis, apply_rotary_emb
from .kv_cache import AttentionKVCache, FractalStateCache, NFNKVCache
from .topology import (
    FractalLevel, build_binary_tree, build_cantor, build_sierpinski,
    build_motif, get_padded_length,
)
from .connections import (
    SinusoidalGate, SinusoidalAggregator, SinusoidalBroadcast, InterMotifCoupler,
)

# ── Sous-package PRISM (mecanismes integres depuis PRISM-KB) ──────────────────
# Import paresseux pour ne pas penaliser l'import de base. Acceder via:
#   from nfn.prism import Prism, PrismConfig, HoloTape, MultiRateBus, ...
# Mecanismes: backbone multi-taux (MRB), MoE polymorphique heterogene,
# memoire holographique VSA (zero entraînement), scaling progressif (PCS),
# CogLoop, curriculum, pretraining modulaire.

__all__ = [
    # Core
    "FNNConfig", "FNNModel", "FNNBlock", "FNNLifecycle", "build_fnn_model",
    # Gematria
    "GematriaEmbedding", "GematriaAttentionBias",
    # Kuramoto
    "KuramotoPhaseLayer", "PhaseGoalForcing", "HierarchicalGoalDecomposer",
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
    # Modules optionnels (Phase 2)
    "FractalSSM", "SSMBlock", "HybridNFNBlock",
    "MixtureOfDepths", "MoDRouter",
    "ContextHyperNet", "HyperAdapter", "HyperLinear",
    "PredictiveCodingBlock", "FreeEnergyMinimiser",
    "MultiTokenPredictor", "MTPHead",
    "RecursiveReasoner", "HaltingUnit", "SelfConsistencyCheck",
    "ProgramSynthesizer",
    "MultimodalFractalRFF", "CrossModalSync",
    # Long-contexte / fractal topology (Phase 3)
    "RoPECache", "precompute_freqs_cis", "apply_rotary_emb",
    "AttentionKVCache", "FractalStateCache", "NFNKVCache",
    "FractalLevel", "build_binary_tree", "build_cantor", "build_sierpinski",
    "build_motif", "get_padded_length",
    "SinusoidalGate", "SinusoidalAggregator", "SinusoidalBroadcast", "InterMotifCoupler",
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
