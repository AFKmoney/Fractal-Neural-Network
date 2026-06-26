"""PRISM — Polymorphic Recurrent Intelligence with Shared Memory.

A novel language model architecture that unifies four paradigms under one
abstraction, integrated into FNN as an alternative/research backbone:

* **Sub-quadratic continuous backbone** — the Multi-Rate Bus (MRB), a bank of
  recurrent filters at logarithmically-spaced decay rates with a learned
  per-token scale gate.
* **Polymorphic MoE** — a router selects which *kind* of computation a token
  undergoes: Neural (MLP), Memory (read/write head), or Symbolic (a library of
  differentiable primitives).
* **Shared differentiable memory bus** — a single memory tape flows through all
  layers and time steps, acting as the Global Workspace through which the
  heterogeneous experts communicate.
* **Differentiable symbolic reasoning** — typed primitives are soft-selected
  and composed end-to-end inside the MoE router.
* **Holographic memory (PRISM-Holo)** — an algebraic VSA tape that binds and
  retrieves facts with zero training, as a drop-in for the memory expert.
* **Progressive Capacity Stacking** — grow a model 350M → 700M → 1B with weight
  inheritance (~40-50% wall-clock savings).
* **CogLoop** — a PERCEIVE → REFLECT → RESPOND → CONSOLIDATE cognitive loop
  with persistent two-tier (working + long-term) memory.
"""

from nfn.prism.config import PrismConfig, MemoryConfig
from nfn.prism.model import Prism, PrismOutput
from nfn.prism.block import PrismBlock
from nfn.prism.norm import RMSNorm

# ── Mechanisms ────────────────────────────────────────────────────────────────
from nfn.prism.mrb import MultiRateBus, MRBOutput
from nfn.prism.memory import MemoryHead, MemoryState
from nfn.prism.symbolic import SymbolicLibrary
from nfn.prism.experts import (
    Expert, NeuralExpert, MemoryExpert, SymbolicExpert, ExpertStats, build_expert,
)
from nfn.prism.router import PolymorphicRouter

# ── Holographic memory (PRISM-Holo) ──────────────────────────────────────────
from nfn.prism.holo import HoloTape, HoloEncoder, HoloHead

# ── Scaling & training levers ────────────────────────────────────────────────
from nfn.prism.pcs import grow_model, resolve_schedule, StageSpec
from nfn.prism.curriculum import CurriculumSchedule, TokenRecycler
from nfn.prism.modular import modular_config, assemble_experts

# ── Cognitive loop ───────────────────────────────────────────────────────────
from nfn.prism.cogloop import CogLoop, CogAnswer
from nfn.prism.cogmemory import CogMemory, Episode, WorkingMemory, LongTermStore

__all__ = [
    # Core
    "PrismConfig", "MemoryConfig", "Prism", "PrismOutput", "PrismBlock", "RMSNorm",
    # Mechanisms
    "MultiRateBus", "MRBOutput",
    "MemoryHead", "MemoryState",
    "SymbolicLibrary",
    "Expert", "NeuralExpert", "MemoryExpert", "SymbolicExpert", "ExpertStats",
    "build_expert",
    "PolymorphicRouter",
    # Holographic
    "HoloTape", "HoloEncoder", "HoloHead",
    # Scaling
    "grow_model", "resolve_schedule", "StageSpec",
    "CurriculumSchedule", "TokenRecycler",
    "modular_config", "assemble_experts",
    # Cognitive loop
    "CogLoop", "CogAnswer", "CogMemory", "Episode", "WorkingMemory", "LongTermStore",
]
__version__ = "0.1.0"
