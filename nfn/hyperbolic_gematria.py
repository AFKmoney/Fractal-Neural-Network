"""
LEAC v2.0 — Gematria Hyperbolique (Poincaré Embeddings)

La limite de la Gematria classique est qu'elle repose sur une métrique
euclidienne plate. Le langage naturel n'est pas plat : les concepts
s'étirent et se contractent.

Innovations LEAC v2:
  1. Plongement dans l'espace hyperbolique de Poincaré (H^n)
  2. Les nombres premiers sont des points sur la frontière conforme
  3. Distance géodésique = dissimilarité sémantique fondamentale
  4. Théorie des Faisceaux (Sheaf Theory) : un mot n'a pas de sens fixe
     → c'est un Faisceau. Sa signification locale dépend du contexte.
     L'hallucination est un défaut de cohomologie.

Espace de Poincaré (modèle boule):
  d_H(z_i, z_j) = arccosh(1 + 2*‖z_i - z_j‖² / ((1-‖z_i‖²)(1-‖z_j‖²)))

Où z_i ∈ B^n (boule unité ouverte). La frontière ∂B^n = S^{n-1} contient
les « tokens idéaux » (nombres premiers). Le bulk contient les concepts
composés et le contexte.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Poincaré Ball Model ─────────────────────────────────────────────────────

class PoincareBall:
    """
    Modèle de la boule de Poincaré pour la géométrie hyperbolique.
    
    d_H(z_i, z_j) = arccosh(1 + 2*‖z_i - z_j‖² / ((1-‖z_i‖²)(1-‖z_j‖²)))
    
    L'espace est courbé négativement : les distances explosent près
    de la frontière. Les concepts génériques sont près du centre,
    les concepts très spécifiques près de la frontière.
    """
    
    def __init__(self, eps: float = 1e-12):
        self.eps = eps
    
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map: transports v from x along geodesic."""
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        lambda_x = 2.0 / (1.0 - x.norm(dim=-1, keepdim=True).pow(2).clamp(min=self.eps))
        denom = v_norm * lambda_x
        tanh_factor = torch.tanh(denom / 2.0).clamp(-0.9999, 0.9999)
        direction = v / v_norm
        return self.mobius_add(x, tanh_factor * direction)
    
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map: vector from x to y in tangent space."""
        diff = self.mobius_add(-x, y)
        diff_norm = diff.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        lambda_x = 2.0 / (1.0 - x.norm(dim=-1, keepdim=True).pow(2).clamp(min=self.eps))
        artanh = 0.5 * torch.log((1.0 + diff_norm) / (1.0 - diff_norm + self.eps))
        scale = 2.0 * artanh / (lambda_x * diff_norm + self.eps)
        return scale * diff
    
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Addition de Möbius dans la boule de Poincaré."""
        x_norm_sq = x.norm(dim=-1, keepdim=True).pow(2)
        y_norm_sq = y.norm(dim=-1, keepdim=True).pow(2)
        x_dot_y = (x * y).sum(dim=-1, keepdim=True)
        num = (1.0 + 2.0 * x_dot_y + y_norm_sq) * x + (1.0 - x_norm_sq) * y
        den = 1.0 + 2.0 * x_dot_y + x_norm_sq * y_norm_sq
        return num / (den.clamp(min=self.eps))
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Distance hyperbolique entre deux points de la boule."""
        numerator = (x - y).norm(dim=-1, keepdim=True).pow(2)
        denom = (1.0 - x.norm(dim=-1, keepdim=True).pow(2)).clamp(min=self.eps) * \
                (1.0 - y.norm(dim=-1, keepdim=True).pow(2)).clamp(min=self.eps)
        arg = 1.0 + 2.0 * numerator / denom.clamp(min=self.eps)
        return torch.acosh(arg.clamp(min=1.0 + self.eps))


# ─── Hyperbolic Gematria Table ───────────────────────────────────────────────

class HyperbolicGematriaTable(nn.Module):
    """
    Plonge les tokens dans l'espace hyperbolique de Poincaré.
    
    Chaque token est représenté comme un point z ∈ B^n (boule de Poincaré).
    Les tokens fondamentaux (petits nombres premiers) sont près de la
    frontière (‖z‖ → 1). Les tokens composés sont dans le bulk.
    
    5 systèmes gematriques sont plongés:
      - Ordinal → position radiale (1 - 1/n)
      - Premier → angle azimutal (θ = 2π·prime/next_prime)
      - Fibonacci → angle polaire (φ = π·fib/log(max_fib))
      - Racine Digitale → twist (mod 9 → 9 positions angulaires)
      - Appris → petit offset entraînable dans le tangent space
    """
    
    def __init__(self, vocab_size: int, hyperbolic_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = hyperbolic_dim
        
        # ── 1. Ordinal: position radiale ──────────────────────────────────────
        radial = torch.log1p(torch.arange(1, vocab_size + 1, dtype=torch.float))
        radial = radial / radial.max() * 0.999  # dans (0, 0.999)
        self.register_buffer("radial", radial)
        
        # ── 2. Premier: directions angulaires (ℂ boundary) ──────────────────
        primes = self._sieve(vocab_size * 10)
        angles = torch.zeros(vocab_size)
        for i in range(vocab_size):
            if i < len(primes):
                angles[i] = 2.0 * math.pi * (primes[max(0, i)] % 360) / 360.0
            else:
                angles[i] = 2.0 * math.pi * i / vocab_size
        self.register_buffer("prime_angles", angles)
        
        # ── 3. Fibonacci: composante angulaire supplémentaire ────────────────
        # Utiliser log(index+1) au lieu des valeurs Fibonacci (overflow float32)
        fib_log = torch.log1p(torch.arange(vocab_size, dtype=torch.float))
        self.register_buffer("fib_angles", 2.0 * math.pi * fib_log / fib_log.max())
        
        # ── 4. Racine Digitale: twist mod 9 ──────────────────────────────────
        digital_roots = torch.tensor([self._dr(i) for i in range(1, vocab_size + 1)], dtype=torch.float)
        self.register_buffer("digital_roots", digital_roots)
        
        # ── 5. Appris: offset entraînable ────────────────────────────────────
        self.learned_offset = nn.Parameter(torch.randn(vocab_size, hyperbolic_dim) * 0.01)
        
        # ── Recouvrement du cercle en directions ─────────────────────────────
        n_dir = hyperbolic_dim // 2  # paires (cos, sin) pour chaque direction
        self.n_directions = n_dir
        
        # Entraînable: les directions de l'espace hyperbolique
        self.direction_weights = nn.Parameter(torch.randn(n_dir, 3) * 0.1)  # weights for prime/fib/dr
        self.base_directions = nn.Parameter(torch.randn(n_dir, 2) * 0.1)     # base (cos, sin) per direction
        
    def _sieve(self, n: int) -> List[int]:
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(n + 1) if is_prime[i]]
    
    def _dr(self, n: int) -> int:
        while n > 9:
            n = sum(int(d) for d in str(n))
        return n
    
    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        token_ids: [B, L]
        Returns:
          z_hyperbolic: [B, L, hyperbolic_dim] — points dans la boule de Poincaré
          gem_boundary: [B, L, hyperbolic_dim] — projections sur la frontière
        """
        ids = token_ids.reshape(-1).clamp(0, self.vocab_size - 1)
        N = ids.shape[0]
        
        # Récupérer les valeurs pré-calculées
        r = self.radial[ids].unsqueeze(-1)                    # [N, 1]
        theta_p = self.prime_angles[ids].unsqueeze(-1)       # [N, 1]
        theta_f = self.fib_angles[ids].unsqueeze(-1)         # [N, 1]
        dr = self.digital_roots[ids].unsqueeze(-1)            # [N, 1]
        
        # Pour chaque direction de l'espace hyperbolique, créer une paire (cos, sin)
        directions = []
        for k in range(self.n_directions):
            w_p, w_f, w_d = self.direction_weights[k]  # poids par direction
            angle = (torch.tanh(w_p) * theta_p +
                     torch.tanh(w_f) * theta_f +
                     torch.tanh(w_d) * dr * 2.0 * math.pi / 9.0)
            cos_d = torch.cos(angle + self.base_directions[k, 0])
            sin_d = torch.sin(angle + self.base_directions[k, 1])
            directions.append(cos_d * r)
            directions.append(sin_d * r)
        
        z_base = torch.cat(directions, dim=-1)                # [N, hyperbolic_dim]
        
        # Offset appris (dans le tangent space, projeté via exp_map)
        learned = self.learned_offset[ids] * 0.01             # [N, hyperbolic_dim]
        z_base = z_base + learned
        
        # Garder dans la boule: ‖z‖ < 1
        norm = z_base.norm(dim=-1, keepdim=True)
        z = z_base / (norm.clamp(min=0.001) + 1e-6)
        # Appliquer le rayon radial
        z = z * r
        
        # Frontière: les tokens "idéaux" (norme = 0.999)
        boundary = F.normalize(z_base, dim=-1) * 0.999
        
        B = token_ids.shape[0]
        L = token_ids.shape[1]
        return z.view(B, L, -1), boundary.view(B, L, -1)


class HyperbolicGematriaAttention(nn.Module):
    """
    Attention basée sur la distance hyperbolique gematrique.
    
    score_hyp(i, j) = exp(-d_H(z_i, z_j) / τ)
    
    Où d_H est la distance de Poincaré. Les tokens gematriquement
    proches dans l'espace hyperbolique reçoivent un bonus d'attention
    exponentiellement grand.
    
    Le biais d'attention devient GEOMETRIQUE:
      score(i,j) = (Q_i · K_j) / √d + λ_hyp · exp(-d_H(gem_i, gem_j) / τ)
    """
    
    def __init__(self, hyperbolic_dim: int = 64, temperature: float = 1.0):
        super().__init__()
        self.poincare = PoincareBall()
        self.log_tau = nn.Parameter(torch.tensor(math.log(temperature)))
        self.lambda_hyp = nn.Parameter(torch.tensor(0.15))
    
    def forward(
        self, z_hyp: torch.Tensor, z_boundary: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z_hyp: [B, L, D] — points dans la boule
        z_boundary: [B, L, D] — projections frontière
        Returns:
          attn_bias: [B, L, L] — biais d'attention géométrique
          coherence_loss: scalaire — perte de cohérence cohomologique
        """
        B, L, _ = z_hyp.shape
        tau = torch.exp(self.log_tau)
        
        # Distance hyperbolique entre toutes les paires
        z_i = z_hyp.unsqueeze(2)   # [B, L, 1, D]
        z_j = z_hyp.unsqueeze(1)   # [B, 1, L, D]
        d_hyp = self.poincare.distance(z_i, z_j).squeeze(-1)  # [B, L, L]
        
        attn_bias = self.lambda_hyp * torch.exp(-d_hyp / tau)
        
        # Perte de cohérence: vérifier que les paires proches en distance
        # hyperbolique ont aussi des représentations contextuelles cohérentes
        # (principe de cohomologie des faisceaux)
        boundary_sim = F.cosine_similarity(
            z_boundary.unsqueeze(2), z_boundary.unsqueeze(1), dim=-1
        )
        
        # Pénaliser si distance hyperbolique ≠ similarité cosinus
        # Cohomologie: les faisceaux doivent être localement cohérents
        coherence = torch.exp(-d_hyp).tril(-1).sum() / max(1, L * (L - 1) / 2.0)
        boundary_coherence = boundary_sim.tril(-1).sum() / max(1, L * (L - 1) / 2.0)
        
        # Défaut de cohomologie: mismatch entre géométrie et sémantique
        cohomology_defect = (torch.abs(coherence - boundary_coherence)).mean()
        
        return attn_bias, -coherence * 0.01 + cohomology_defect * 0.005


class SheafTheoryLayer(nn.Module):
    """
    Théorie des Faisceaux pour la cohérence sémantique contextuelle.
    
    Un mot n'a pas de sens intrinsèque fixe. C'est un « Faisceau » (Sheaf).
    Son sens local dépend du voisinage (contexte), et la cohérence globale
    de la phrase est garantie par la cohomologie du faisceau.
    
    L'hallucination est un DÉFAUT DE COHOMOLOGIE.
    
    Architecture:
      1. Pour chaque token, calculer sa section locale (contexte gauche + droit)
      2. Vérifier les conditions de recollement entre sections adjacentes
      3. Propagation de cohérence via opérateurs de restriction
    """
    
    def __init__(self, d_model: int, n_stalks: int = 4):
        super().__init__()
        self.n_stalks = n_stalks
        self.d_model = d_model
        d_stalk = d_model // n_stalks
        
        # Opérateur de restriction (restreint le contexte à une stalk)
        self.restriction = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_stalk),
                nn.LayerNorm(d_stalk),
                nn.SiLU(),
            ) for _ in range(n_stalks)
        ])
        
        # Opérateur de recollement (vérifie la compatibilité entre stalks)
        # Input: 2 * d_stalk (paire de stalks adjacentes)
        self.gluing = nn.Sequential(
            nn.Linear(d_stalk * 2, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        
        # Opérateur de cohomologie
        # Input: n_stalks * d_stalk = d_model (les stalks reconstituent d_model)
        self.cohomology = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, L, d]
        Returns:
          h_sheaf: [B, L, d] — enrichi par cohomologie des faisceaux
          defect_loss: scalaire — défaut de cohomologie
        """
        B, L, d = h.shape
        
        # 1. Restreindre à chaque stalk
        stalks = [rest(h) for rest in self.restriction]  # n_stalks × [B, L, d_stalk]
        
        # 2. Vérifier les conditions de recollement entre tokens adjacents
        gluing_losses = []
        for k in range(self.n_stalks):
            s = stalks[k]
            if L > 1:
                # Comparer les restrictions de tokens adjacents
                s_left = s[:, :-1]   # [B, L-1, d_stalk]
                s_right = s[:, 1:]   # [B, L-1, d_stalk]
                # L'opérateur de recollement vérifie si les sections sont compatibles
                glue_input = torch.cat([s_left, s_right], dim=-1)
                glue_score = self.gluing(glue_input).squeeze(-1)  # [B, L-1]
                # Pénaliser les discontinuités (défauts de recollement)
                gluing_losses.append(F.relu(-glue_score).mean())
        
        defect_loss = torch.stack(gluing_losses).mean() if gluing_losses else \
                      torch.tensor(0.0, device=h.device)
        
        # 3. Cohomologie: vérifier la cohérence globale
        all_stalks = torch.cat(stalks, dim=-1)  # [B, L, d]
        cohomology_signal = self.cohomology(all_stalks)
        h_sheaf = self.norm(h + cohomology_signal * 0.1)
        
        return h_sheaf, defect_loss * 0.01


class HyperbolicGematriaModule(nn.Module):
    """
    Module complet de Gematria Hyperbolique pour LEAC v2.
    
    Pipeline:
      Token IDs → HyperbolicGematriaTable → PoincaréBall embeddings
      → HyperbolicGematriaAttention (biais géométrique)
      → SheafTheoryLayer (cohérence cohomologique)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        hyperbolic_dim: int = 64,
        n_stalks: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.gem_table = HyperbolicGematriaTable(vocab_size, hyperbolic_dim)
        self.geo_attn = HyperbolicGematriaAttention(hyperbolic_dim, temperature)
        self.sheaf = SheafTheoryLayer(d_model, n_stalks)
        
        # Projection hyperbolic_dim → d_model
        self.hyp_to_d = nn.Linear(hyperbolic_dim, d_model, bias=False)
        nn.init.normal_(self.hyp_to_d.weight, std=0.02)
    
    def forward(
        self, h: torch.Tensor, token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        h: [B, L, d_model]
        token_ids: [B, L]
        Returns: (h_enriched, losses_dict)
        """
        # 1. Plongement hyperbolique
        z_hyp, z_boundary = self.gem_table(token_ids)
        
        # 2. Biais d'attention géométrique
        attn_bias, gem_loss = self.geo_attn(z_hyp, z_boundary)
        
        # 3. Injecter l'information hyperbolique dans le flux principal
        hyp_signal = self.hyp_to_d(z_hyp)
        h_hyperbolic = h + hyp_signal * 0.1
        
        # 4. Vérifier la cohomologie des faisceaux
        h_sheaf, sheaf_loss = self.sheaf(h_hyperbolic)
        
        losses = {
            "hyperbolic_gematria": gem_loss,
            "sheaf_cohomology": sheaf_loss,
        }
        
        return h_sheaf, losses


__all__ = [
    "PoincareBall",
    "HyperbolicGematriaTable",
    "HyperbolicGematriaAttention",
    "SheafTheoryLayer",
    "HyperbolicGematriaModule",
]