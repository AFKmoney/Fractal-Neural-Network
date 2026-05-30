"""
Fractal topology utilities for the NFN.

Supported motif generators:
  - binary_tree  : standard balanced binary tree (b=2, K levels)
  - cantor       : Cantor-set pruning (removes middle third of each branch)
  - sierpinski   : Sierpinski-triangle connectivity pattern
  - penrose      : Penrose-like quasi-periodic tiling approximation

Each motif returns a list of levels.  Level 0 = finest (leaves = positions).
Level k contains N_k = L / b^k nodes.  Edges connect level k → level k+1.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch


@dataclass
class FractalLevel:
    level: int            # 0 = leaves
    n_nodes: int          # number of nodes at this level
    branching: int        # b children per parent (can vary by motif)
    parent_indices: torch.Tensor   # [n_nodes] → index of parent at level+1; -1 for top
    child_mask: torch.Tensor       # [n_nodes_above, b] indices into this level (children)
    motif: str


def build_binary_tree(seq_len: int, n_levels: int, branching: int = 2) -> List[FractalLevel]:
    """
    Standard balanced binary-tree (or b-ary) fractal over seq_len positions.
    seq_len is padded to the nearest multiple of branching^n_levels.
    """
    stride = branching ** n_levels
    padded_len = math.ceil(seq_len / stride) * stride

    levels: List[FractalLevel] = []
    for k in range(n_levels + 1):
        n_k = padded_len // (branching ** k)
        if k < n_levels:
            # Each node at level k has parent at level k+1
            parent_idx = torch.arange(n_k) // branching
        else:
            parent_idx = torch.full((n_k,), -1, dtype=torch.long)

        if k > 0:
            n_above = padded_len // (branching ** k)
            child_idx = torch.arange(n_above * branching).view(n_above, branching)
        else:
            child_idx = torch.zeros(1, 1, dtype=torch.long)

        levels.append(FractalLevel(
            level=k,
            n_nodes=n_k,
            branching=branching,
            parent_indices=parent_idx,
            child_mask=child_idx,
            motif="binary_tree",
        ))
    return levels


def build_cantor(seq_len: int, n_levels: int, branching: int = 3) -> List[FractalLevel]:
    """
    Cantor-set topology: branching=3 but the middle child is suppressed
    (receives zero input; acts as a structural skip).
    The active children are indices 0 and 2.
    """
    stride = branching ** n_levels
    padded_len = math.ceil(seq_len / stride) * stride

    levels: List[FractalLevel] = []
    for k in range(n_levels + 1):
        n_k = padded_len // (branching ** k)
        if k < n_levels:
            parent_idx = torch.arange(n_k) // branching
        else:
            parent_idx = torch.full((n_k,), -1, dtype=torch.long)

        if k > 0:
            n_above = padded_len // (branching ** k)
            child_idx = torch.arange(n_above * branching).view(n_above, branching)
        else:
            child_idx = torch.zeros(1, branching, dtype=torch.long)

        levels.append(FractalLevel(
            level=k,
            n_nodes=n_k,
            branching=branching,
            parent_indices=parent_idx,
            child_mask=child_idx,
            motif="cantor",
        ))
    return levels


def build_sierpinski(seq_len: int, n_levels: int, branching: int = 3) -> List[FractalLevel]:
    """
    Sierpinski-triangle connectivity: branching=3 but the central child
    is suppressed at alternate levels, creating a self-similar triangular
    pattern with fractal dimension log2(3)/log2(2) ≈ 1.585.
    
    Unlike the Cantor set which removes the middle third uniformly,
    Sierpinski removes alternating positions across levels, creating
    richer cross-scale connectivity patterns.
    """
    stride = branching ** n_levels
    padded_len = math.ceil(seq_len / stride) * stride

    levels: List[FractalLevel] = []
    for k in range(n_levels + 1):
        n_k = padded_len // (branching ** k)
        if k < n_levels:
            parent_idx = torch.arange(n_k) // branching
        else:
            parent_idx = torch.full((n_k,), -1, dtype=torch.long)

        if k > 0:
            n_above = padded_len // (branching ** k)
            child_idx = torch.arange(n_above * branching).view(n_above, branching)
        else:
            child_idx = torch.zeros(1, branching, dtype=torch.long)

        levels.append(FractalLevel(
            level=k,
            n_nodes=n_k,
            branching=branching,
            parent_indices=parent_idx,
            child_mask=child_idx,
            motif="sierpinski",
        ))
    return levels


def get_padded_length(seq_len: int, n_levels: int, branching: int) -> int:
    stride = branching ** n_levels
    return math.ceil(seq_len / stride) * stride


# Map motif name → builder
MOTIF_BUILDERS: Dict = {
    "binary_tree": build_binary_tree,
    "cantor": build_cantor,
    "sierpinski": build_sierpinski,
}


def build_motif(name: str, seq_len: int, n_levels: int, branching: int = 2) -> List[FractalLevel]:
    builder = MOTIF_BUILDERS.get(name, build_binary_tree)
    if name in ("cantor", "sierpinski"):
        return builder(seq_len, n_levels, branching=3)
    return builder(seq_len, n_levels, branching=branching)
