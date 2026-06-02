"""
LEAC Topologie Fractale et Attention Linéaire

Architecture fractale avec attention linéaire O(L*d²):
  - FractalLinearAttention: kernel trick de Katharopoulos
  - PhaseSoliton: amplification des patterns cohérents
  - PhaseRoutedMoE: routage von Mises (équilibrage naturel par séquence de Farey)

Le fractale permet de compresser l'infini dans le fini.
"""

from .moe import FractalLinearAttention, PhaseSoliton, PhaseRoutedMoE

__all__ = ["FractalLinearAttention", "PhaseSoliton", "PhaseRoutedMoE"]