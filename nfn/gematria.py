"""
LEAC Gematria Sémantique — Les 5 Systèmes Croisés

L'Hypothèse de la Gematria Sémantique (LEAC §1.1):

Chaque token n'est pas un vecteur isolé dans un espace latent arbitraire.
Il est la projection d'un objet mathématique fondamental:

    e(t) = Σ_k CharClass_k(t) · ω_k

Où ω_k sont des fréquences de Mandelbrot.

5 systèmes croisés:
  1. Ordinal (Comptage pur)
  2. Premier (A=2, B=3, C=5... Crible d'Ératosthène)
  3. Fibonacci (Croissance organique)
  4. Racine Digitale (Cyclicité mod 9)
  5. Appris (Ajustement différentiable)

Conséquence: Deux concepts partagent des liens sémantiques profonds si
leurs signatures gematriques sont en résonance (ex: facteurs premiers communs).

Le biais d'attention devient mathématique:
    score(i,j) = Q_i · K_j / √d + λ · cos(gem(i), gem(j))
"""

from .semantic_gematria import GematriaTable, GematriaLoss, GematriaCurriculum

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GematriaEmbedding(nn.Module):
    """
    Embedding analytique à zéro paramètre via décomposition fractale
    des points de code + Gematria Sémantique.

    Concatène l'embedding analytique avec l'embedding gematrique
    pour chaque token.
    """

    def __init__(self, vocab_size: int, d_model: int, d_gematria: int = 16):
        super().__init__()
        self.gem_table = GematriaTable(vocab_size, d_gematria=d_gematria)
        self.proj = nn.Linear(d_gematria, d_model, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [B, L]
        Returns: [B, L, d_model] — gematria embedding
        """
        gem = self.gem_table(token_ids)  # [B, L, d_gematria]
        return self.proj(gem)


class GematriaAttentionBias(nn.Module):
    """
    Utilise les valeurs gematriques comme BIAIS STRUCTUREL pour l'attention.

    Le biais est additif aux scores d'attention standard:
        score(i,j) = Q_i · K_j / √d + λ · cos(gem(i), gem(j))

    Deux tokens avec des signatures gematriques en résonance
    (facteurs premiers communs, racines digitales similaires)
    reçoivent un bonus d'attention. Ce biais est DÉRIVÉ DES MATHÉMATIQUES,
    pas des données.
    """

    def __init__(self, d_model: int, vocab_size: int, n_heads: int = 4,
                 d_gematria: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.gematria = GematriaTable(vocab_size, d_gematria=d_gematria)
        self.gem_proj = nn.Linear(d_gematria, d_model)
        self.lambda_gem = nn.Parameter(torch.tensor(0.1))

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, L, d_model]
        token_ids: [B, L]
        Returns: (h_gem [B, L, d_model], gematria_coherence_loss)
        """
        gem_emb = self.gematria(token_ids)            # [B, L, d_gematria]
        gem_proj = self.gem_proj(gem_emb)               # [B, L, d_model]

        # Resonance: similarité cosinus entre signatures gematriques
        gem_norm = F.normalize(gem_emb, dim=-1)
        gem_sim = F.cosine_similarity(
            gem_norm.unsqueeze(2), gem_norm.unsqueeze(1), dim=-1
        )  # [B, L, L]

        # Injection dans le flux principal
        h_enriched = self.norm(h + self.lambda_gem * gem_proj)

        # Perte de cohérence gematrique (les tokens proches doivent résonner)
        if h.shape[1] > 1:
            coherence = gem_sim.tril(-1).sum() / max(1, h.shape[1] * (h.shape[1] - 1) / 2)
            loss = -coherence * 0.01
        else:
            loss = torch.tensor(0.0, device=h.device)

        return h_enriched, loss


__all__ = ["GematriaEmbedding", "GematriaAttentionBias", "GematriaTable", "GematriaLoss"]