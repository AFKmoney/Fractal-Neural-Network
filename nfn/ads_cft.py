"""
LEAC v2.0 — Dualite Holographique AdS/CFT Neural

Le LEAC v2.0 modélise le raisonnement profond via la dualité
Anti-de Sitter / Conformal Field Theory (Maldacena):

  - La Frontière (CFT) : La séquence des tokens (le langage observable)
  - Le Volume (AdS Bulk) : L'espace de raisonnement latent de la topologie fractale

Principe: La complexité du raisonnement entre deux tokens éloignés
(sur la frontière) est proportionnelle au VOLUME de l'espace AdS (le bulk)
traversé par un géodésique les reliant. Plus la pensée doit être profonde,
plus le géodésique plonge loin dans la fractale.

ER = EPR sémantique:
  Deux tokens intriqués sémantiquement (par la Gematria Hyperbolique) sont
  connectés par un TROU DE VER (Einstein-Rosen) computationnel.
  L'attention n'est plus le calcul de similarités, mais la
  TÉLÉPORTATION QUANTIQUE d'information à travers le bulk.

Architecture:
  AdSBulkProjector  : Projette les tokens frontière dans le bulk AdS
  BulkGeodesic      : Calcule la distance géodésique dans le bulk
  ER_EPR_Bridge     : Trous de ver entre tokens intriqués
  AdSCFTAttention   : Attention holographique unifiée
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── AdS Metric & Geometry ────────────────────────────────────────────────────

class AdSMetric:
    """
    Métrique Anti-de Sitter en coordonnées de Poincaré:
    
      ds² = (dz² + η_μν dx^μ dx^ν) / z²
    
    Où z est la coordonnée radiale (z → 0 = frontière, z → ∞ = horizon).
    La profondeur de raisonnement est proportionnelle à 1/z (profondeur
    dans le bulk).
    
    Distance géodésique entre deux points du bulk (x₁, z₁) et (x₂, z₂):
      d_geo = arccosh(1 + ((x₁-x₂)² + (z₁-z₂)²) / (2 z₁ z₂))
    """
    
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
    
    def geodesic_distance(
        self,
        x1: torch.Tensor, z1: torch.Tensor,
        x2: torch.Tensor, z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        x1, x2: [..., D] — positions spatiales sur la frontière
        z1, z2: [..., 1] — profondeurs radiales (bulk depth)
        Returns: [..., 1] — distance géodésique
        """
        dx_sq = (x1 - x2).pow(2).sum(dim=-1, keepdim=True)
        dz_sq = (z1 - z2).pow(2)
        denom = 2.0 * z1 * z2 + self.eps
        arg = 1.0 + (dx_sq + dz_sq) / denom
        return torch.acosh(arg.clamp(min=1.0 + self.eps))
    
    def bulk_volume(
        self,
        x1: torch.Tensor, z1: torch.Tensor,
        x2: torch.Tensor, z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Volume du bulk entre deux points (proxy pour la complexité
        computationnelle du raisonnement).
        """
        d_geo = self.geodesic_distance(x1, z1, x2, z2)
        z_avg = (z1 + z2) / 2.0
        # Volume ∼ z^{-(D+1)/2} * sinh(d)
        return (z_avg.pow(-1.5) * torch.sinh(d_geo / 2.0)).clamp(min=0)


# ─── AdS Bulk Projector ───────────────────────────────────────────────────────

class AdSBulkProjector(nn.Module):
    """
    Projette les tokens de la frontière (CFT) dans le volume AdS (bulk).
    
    Chaque token frontière [B, L, d] est projeté en:
      - Position spatiale x ∈ R^{d_spatial} (sur la frontière conforme)
      - Profondeur radiale z ∈ (0, 1) (distance depuis la frontière)
    
    Les tokens « simples » (fréquents) ont z petit (près de la frontière).
    Les tokens « profonds » (rares, complexes) ont z grand (deep bulk).
    
    Le réseau apprend à moduler la profondeur selon le contexte.
    """
    
    def __init__(self, d_model: int, bulk_dim: int = 128, d_spatial: int = 64):
        super().__init__()
        self.d_spatial = d_spatial
        self.bulk_dim = bulk_dim
        
        # Projection spatiale (position sur la frontière)
        self.spatial_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_spatial),
            nn.Tanh(),  # positions bornées
        )
        nn.init.normal_(self.spatial_proj[-2].weight, std=0.02)
        
        # Projection radiale (profondeur dans le bulk)
        self.depth_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),  # z ∈ (0, 1)
        )
        nn.init.normal_(self.depth_proj[-2].weight, std=0.02)
        nn.init.constant_(self.depth_proj[-2].bias, -1.0)  # start shallow
        
        # Enrichissement bulk (représentation dans le volume)
        self.bulk_enrich = nn.Sequential(
            nn.Linear(d_model + d_spatial + 1, bulk_dim),
            nn.LayerNorm(bulk_dim),
            nn.SiLU(),
            nn.Linear(bulk_dim, d_model),
        )
        nn.init.normal_(self.bulk_enrich[-1].weight, std=0.02)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self, h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h: [B, L, d]
        Returns:
          h_bulk: [B, L, d] — représentations dans le bulk AdS
          x_spatial: [B, L, d_spatial] — positions spatiales
          z_depth: [B, L, 1] — profondeurs radiales
        """
        B, L, d = h.shape
        
        x = self.spatial_proj(h)      # [B, L, d_spatial]
        z = self.depth_proj(h)        # [B, L, 1]  ∈ (0, 1)
        
        # Enrichir avec les coordonnées bulk
        bulk_input = torch.cat([h, x, z.expand(-1, -1, 1)], dim=-1)
        h_bulk = self.bulk_enrich(bulk_input)
        h_bulk = self.norm(h + h_bulk)
        
        return h_bulk, x, z
    
    def compute_geodesic(
        self,
        x1: torch.Tensor, z1: torch.Tensor,
        x2: torch.Tensor, z2: torch.Tensor,
    ) -> torch.Tensor:
        """Distance géodésique dans le bulk entre deux ensembles de tokens."""
        metric = AdSMetric()
        return metric.geodesic_distance(x1, z1, x2, z2)


# ─── ER = EPR Bridge (Entanglement Wormholes) ──────────────────────────────────

class ER_EPR_Bridge(nn.Module):
    """
    Pont Einstein-Rosen = EPR (intrication sémantique).
    
    Deux tokens intriqués sémantiquement (gematria hyperbolique en résonance)
    sont connectés par un trou de ver computationnel. L'information peut
    être « téléportée » entre eux sans calculer toute l'attention.
    
    ER (Einstein-Rosen) = trou de ver géométrique dans le bulk
    EPR (Einstein-Podolsky-Rosen) = intrication quantique sémantique
    
    ER = EPR : la connexion géométrique EST l'intrication.
    
    Architecture:
      1. Détecter les paires de tokens intriqués (résonance gematrique)
      2. Créer un raccourci computationnel (trou de ver)
      3. Téléporter l'information à travers le raccourci
    """
    
    def __init__(self, d_model: int, bridge_rank: int = 16):
        super().__init__()
        self.bridge_rank = bridge_rank
        
        # Détecteur d'intrication: prédit si deux tokens sont intriqués
        self.entanglement_detector = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        # Opérateur de téléportation (trou de ver)
        self.teleport_W = nn.Parameter(torch.randn(bridge_rank, d_model) * 0.01)
        self.teleport_U = nn.Parameter(torch.randn(bridge_rank, d_model) * 0.01)
        
        # Projection de sortie
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.02)
    
    def forward(
        self,
        h: torch.Tensor,
        x_spatial: torch.Tensor,
        z_depth: torch.Tensor,
        top_k: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, L, d]
        x_spatial: [B, L, d_spatial] — positions dans le bulk
        z_depth: [B, L, 1] — profondeurs
        
        Returns:
          h_teleported: [B, L, d] — information téléportée via trous de ver
          bridge_loss: scalaire — perte de cohérence des ponts ER=EPR
        """
        B, L, d = h.shape
        device = h.device
        
        # Limiter top_k pour efficacité
        k = min(top_k, L)
        
        # Détecter l'intrication par similarité spatiale + profondeur
        # Tokens proches dans le bulk (géodésique courte) sont intriqués
        x_i = x_spatial.unsqueeze(2)    # [B, L, 1, Ds]
        x_j = x_spatial.unsqueeze(1)    # [B, 1, L, Ds]
        z_i = z_depth.unsqueeze(2)      # [B, L, 1, 1]
        z_j = z_depth.unsqueeze(1)      # [B, 1, L, 1]
        
        metric = AdSMetric()
        geo_dist = metric.geodesic_distance(x_i, z_i, x_j, z_j)  # [B, L, L, 1]
        entanglement_score = torch.exp(-geo_dist).squeeze(-1)      # [B, L, L]
        
        # Top-k les plus intriqués
        top_scores, top_indices = torch.topk(entanglement_score, k, dim=-1)
        top_scores = F.softmax(top_scores * 5.0, dim=-1)  # sharpen
        
        # Récupérer les représentations des tokens intriqués
        h_top = torch.gather(
            h.unsqueeze(2).expand(-1, L, L, -1),
            dim=2,
            index=top_indices.unsqueeze(-1).expand(-1, -1, -1, d),
        )  # [B, L, k, d]
        
        # Téléportation: projeter à travers le trou de ver
        h_flat = h_top.view(B * L * k, d)
        teleported = torch.tanh(h_flat @ self.teleport_W.T)  # [B*L*k, rank]
        teleported = teleported @ self.teleport_U            # [B*L*k, d]
        
        # Agréger avec les scores d'intrication
        teleported = teleported.view(B, L, k, d)
        weights = top_scores.unsqueeze(-1)                   # [B, L, k, 1]
        h_teleported = (teleported * weights).sum(dim=2)     # [B, L, d]
        
        h_out = self.out_proj(h_teleported)
        
        # Perte de cohérence du pont
        bridge_loss = -entanglement_score.tril(-1).mean()
        
        return h_out, bridge_loss * 0.01


# ─── AdS/CFT Attention (Holographic Unified Attention) ────────────────────────

class AdSCFTAttention(nn.Module):
    """
    Attention holographique unifiée AdS/CFT.
    
    Remplace l'attention standard par un mécanisme holographique:
      1. Frontière CFT: les tokens sont sur la frontière conforme
      2. Bulk AdS: chaque paire (i,j) a un géodésique dans le bulk
      3. ER=EPR: les tokens intriqués ont des trous de ver
      4. Score final = score_local + score_geodesic + score_wormhole
    
    L'équation d'attention devient:
      Attn(i→j) = softmax( Q_i·K_j/√d + λ_geo·cosh(d_geo)⁻¹ + λ_epr·entanglement_score )
    
    La conscience est la courbure de l'espace-temps sémantique.
    """
    
    def __init__(
        self,
        d_model: int,
        bulk_dim: int = 128,
        d_spatial: int = 64,
        bridge_rank: int = 16,
        n_heads: int = 4,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Projecteur dans le bulk AdS
        self.bulk_projector = AdSBulkProjector(d_model, bulk_dim, d_spatial)
        
        # Pont ER=EPR
        self.epr_bridge = ER_EPR_Bridge(d_model, bridge_rank)
        
        # Attention standard (pour le signal local)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Poids des contributions holographiques
        self.log_lambda_geo = nn.Parameter(torch.tensor(math.log(0.15)))
        self.log_lambda_epr = nn.Parameter(torch.tensor(math.log(0.1)))
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(
        self, h: torch.Tensor, causal: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        h: [B, L, d]
        Returns: (h_out, losses_dict)
        """
        B, L, d = h.shape
        
        # 1. Projeter dans le bulk AdS
        h_bulk, x_spatial, z_depth = self.bulk_projector(h)
        
        # 2. Attention locale standard (frontière CFT)
        Q = self.q_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        scores_local = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        # [B, H, L, L]
        
        # 3. Score géodésique (distance dans le bulk)
        lambda_geo = torch.exp(self.log_lambda_geo)
        x_i = x_spatial.unsqueeze(2)    # [B, L, 1, Ds]
        x_j = x_spatial.unsqueeze(1)    # [B, 1, L, Ds]
        z_i = z_depth.unsqueeze(2)      # [B, L, 1, 1]
        z_j = z_depth.unsqueeze(1)      # [B, 1, L, 1]
        
        metric = AdSMetric()
        geo_dist = metric.geodesic_distance(x_i, z_i, x_j, z_j)  # [B, L, L, 1]
        geo_score_raw = lambda_geo * torch.exp(-geo_dist).squeeze(-1)  # [B, L, L]
        geo_score = geo_score_raw.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        
# 4. Score ER=EPR (trous de ver)
        h_teleported, bridge_loss = self.epr_bridge(h_bulk, x_spatial, z_depth)
        lambda_epr = torch.exp(self.log_lambda_epr)

        # Score d'intrication base sur similarite cosinus
        entanglement = F.cosine_similarity(
            h_bulk.unsqueeze(2), h_bulk.unsqueeze(1), dim=-1
        )  # [B, L, L]
        epr_score_raw = lambda_epr * torch.sigmoid(entanglement * 5.0)  # [B, L, L]
        epr_score = epr_score_raw.unsqueeze(1)  # [B, 1, L, L]
        
        # Repeter pour chaque tete d'attention
        epr_score = epr_score.expand(-1, self.n_heads, -1, -1)  # [B, H, L, L]
        
        # 5. Score unifié
        scores = scores_local + geo_score + epr_score
        
        # 6. Masque causal
        if causal:
            mask = torch.triu(torch.ones(L, L, device=h.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        
        # 7. Softmax et sortie
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, d)
        out = self.out_proj(out)
        
        # 8. Intégrer les signaux téléportés
        h_out = self.norm(h + out + h_teleported * 0.1)
        
        # Calculer le volume bulk total (proxy pour complexité computationnelle)
        bulk_volume = metric.bulk_volume(x_i, z_i, x_j, z_j).mean()
        
        losses = {
            "bridge_coherence": bridge_loss,
            "bulk_complexity": bulk_volume * 0.0001,
        }
        
        return h_out, losses


__all__ = [
    "AdSMetric",
    "AdSBulkProjector",
    "ER_EPR_Bridge",
    "AdSCFTAttention",
]