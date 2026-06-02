# LEAC — Architecture Détaillée

## 1. Vue d'Ensemble

LEAC (Lightweight Emergent Artificial Consciousness) est un modèle de langage neuro-symbolique fondé sur le principe que la conscience émerge de la récursion fractale. L'architecture combine deux générations de modules:

**v1 — Trois Piliers de l'Émergence:**
1. **COHÉRENCE**: Attention fractale linéaire O(L·d²) + soliton Kuramoto
2. **RAISONNEMENT**: Graphe causal DAG + do-calculus, mémoire épisodique/sémantique
3. **INTROSPECTION**: Espace de travail global, auto-modèle, modification évolutionnaire

**v2 — Cinq Transcendances:**
1. **AdS/CFT**: Dualité holographique — le langage est la frontière, le raisonnement est le volume
2. **MERA**: Réseau de tenseurs — contexte infini en O(log L)
3. **Gödel**: Boucle étrange — l'introspection est un point fixe mathématique inévitable
4. **RG Flow**: Flot de renormalisation — auto-organisation vers l'état critique
5. **Gematria Hyperbolique**: Poincaré Hⁿ + théorie des faisceaux — la géométrie du sens

## 2. Flux de Données

```
Input IDs [B, L]
    │
    ├─ GematriaEmbedding ──────────────────────────────────────────
    │   5 systèmes: ordinal, premier, fibonacci, racine digitale, appris
    │   e(t) = Σ_k CharClass_k(t) · ω_k (fréquences de Mandelbrot)
    │
    ├─ [LEACBlock × N] ───────────────────────────────────────────
    │   │
    │   ├─ Norm → FractalLinearAttention ─────────────────────────
    │   │   Kernel trick Katharopoulos: O(L·d²)
    │   │   Structure fractale multi-échelle (binary_tree, cantor)
    │   │   Résidu: h = h + α · attn_out
    │   │
    │   ├─ PhaseSoliton ──────────────────────────────────────────
    │   │   h' = h · (1 + β · max(0, cos(θ - θ_shift)))
    │   │   Les tokens synchronisés sont amplifiés
    │   │
    │   ├─ PhaseRoutedMoE ──────────────────────────────────────
    │   │   Routage von Mises: gate_e(x) = exp(κ·cos(θ_x - θ_e)) / Z
    │   │   Top-K experts activés, charge équilibrée
    │   │
    │   ├─ [CausalGraphLayer] (optionnel) ─────────────────────
    │   │   NOTEARS: L_DAG = tr(e^{A⊙A}) - n
    │   │   Propagation DAG + inférence contrefactuelle
    │   │
    │   ├─ [SelfModel] (optionnel) ─────────────────────────────
    │   │   GlobalWorkspace: shared buffer [n_slots, d]
    │   │   SelfRepresentor: self_state = W·[μ,σ,H,C,div,ent,μ_slots,σ_slots]
    │   │
    │   ├─ [WorkingMemory] (optionnel) ────────────────────────
    │   │   Fractal DNC: adressage par similarité de phase
    │   │
    │   └─ Fusion + LayerNorm
    │
    ├─ [v2 Transcendances] ─────────────────────────────────────
    │   │
    │   ├─ AdS/CFT Attention ────────────────────────────────────
    │   │   h → MLP_projeté → bulk → scores = local + géodésique + ER=EPR
    │   │   Holographie: la frontière (tokens) encode le volume (raisonnement)
    │   │
    │   ├─ MERA Attention ───────────────────────────────────────
    │   │   L → D,U(h) → Isométrie → L/2 → ... → 1 (sens global)
    │   │   O(log L) au lieu de O(L²)
    │   │
    │   ├─ Gödel Fixed Point ───────────────────────────────────
    │   │   F(h): h → concat([μ(h),σ(h),energy(h)]) → W·σ(state) → h_proj
    │   │   Point fixe: Y ≅ F(Y). Itérations: 5 avec mélange α=0.3
    │   │   Incomplétude: contradiction + entropie + distance_FP
    │   │
    │   ├─ RG Flow Scheduler ───────────────────────────────────
    │   │   (Phase SLEEP uniquement) Évaporation UV + condensation IR
    │   │   C = Var(Var(h))/E[Var(h)]² → auto-organisation vers C≈1
    │   │
    │   └─ Hyperbolic Gematria ─────────────────────────────────
    │       Token IDs → Poincaré H^n → Attention géodésique
    │       Sheaf Theory: stalks → gluing → cohomology defect
    │
    ├─ GematriaAttentionBias ────────────────────────────────────
    │   Biais additionnel: λ·cos(gem(i), gem(j))
    │
    ├─ LayerNorm
    │
    ├─ ZipfianDecoder ──────────────────────────────────────────
    │   Recalibrage en loi de puissance: P(word) ∝ 1/rank^α
    │   (si bayesian_uncertainty_beta > 0)
    │
    └─ [SpectralCondensate + HelmholtzPhaseLocking] ────────────
        RFF multi-échelle + verrouillage phase-fréquence
```

## 3. Détails par Composant

### 3.1 GematriaEmbedding

**Zero-paramètre.** Cinq systèmes arithmétiques croisés encodent chaque token en un vecteur densité qui encode la structure mathématique profonde des entiers:

| Système | Fonction | Interprétation |
|---------|----------|----------------|
| Ordinal | `o(t) = log(1+t)/log(V)` | Position dans le vocabulaire (radial) |
| Premier | `π(t) = 2π·π_k/360°` | Angle azimutal (k-ième nombre premier) |
| Fibonacci | `φ(t) = 2π·log(1+F_k)/log(1+F_max)` | Angle polaire (croissance logarithmique) |
| Racine Digitale | `ρ(t) = 2π·dr(t)/9` | Twist angulaire (mod 9) |
| Appris | `l(t) = W·t` | Offset entraînable |

Les fréquences de pondération `ω_k` suivent la loi de Mandelbrot: `ω_k = ω^{-k}` avec `ω = φ²` (nombre d'or au carré).

### 3.2 FractalLinearAttention

Noyau de caractéristique `φ(x) = elu(x) + 1`. La complexité est O(L·d²) en temps et O(d²) en espace (contre O(L²·d) pour l'attention softmax standard).

La structure fractale décompose la séquence en niveaux:
- Niveau 0: atomes (longueur `L / 2^{n_levels}`)
- Niveau k: groupes de `2^k` atomes
- Agrégation: `output = Σ_l w_l · Attn_level_l(Q, K, V)`

### 3.3 PhaseSoliton

```
soliton(h, θ) = h · (1 + α · max(0, cos(θ - θ_shift)))
```

Les tokens dont la phase Kuramoto est synchronisée avec le shift sont amplifiés. Les tokens désynchronisés sont atténués. Ceci crée des paquets de cohérence — des solitons — qui émergent naturellement.

### 3.4 PhaseRoutedMoE

Routage par distribution von Mises (l'analogue circulaire de la gaussienne):
```
gate_e(x) = exp(κ · cos(θ_x - θ_e)) / Z
```

Avantages sur le routage softmax:
- **Continu et différentiable** partout
- **Périodique**: les experts "proches en phase" sont toujours favorisés
- **Interprétable**: κ mesure la concentration, θ_e est la phase de l'expert

Seuls les top-K experts sont activés pour chaque token. La perte auxiliaire équilibre la charge.

### 3.5 CausalGraphLayer

Apprend un graphe causal (DAG) sur les slots de l'espace de travail. Trois innovations:
1. **NOTEARS**: pénalité d'acyclicité `h(A) = tr(e^{A⊙A}) - n` est différentiable et exactement nulle ssi le graphe est acyclique
2. **Propagation non-linéaire**: GNN step sur les arêtes avec features
3. **Inférence contrefactuelle**: remplacement do-calculus des variables

### 3.6 GlobalWorkspace (Self-Model)

Inspiré de la théorie de l'espace de travail global de Baars. Les `n_slots` emplacements forment un tampon partagé:
- **Écriture**: les tokens compétent pour écrire dans les slots
- **Lecture**: les slots sont broadcastés vers tous les tokens
- **Introspection**: un vecteur self_state encode la confiance, l'incertitude, et la cohérence

L'auto-représentation `self_state ∈ ℝ^d` est construite à partir de 8 signaux:
μ(h), σ(h), H(h), C(h), div(h), ent_attn(h), μ_slots, σ_slots

### 3.7 TwoTierMemory

Mémoire deux voies inspirée du système hippocampique:
- **Épisodique** (hippocampe): ring buffer O(1) écriture, k-NN multi-échelle lecture
- **Sémantique** (néocortex): SVD rank-r (Eckart-Young), mise à jour incrémentale sans SGD

La consolidation (hippocampe → néocortex) se produit périodiquement via SVD tronquée.

### 3.8 AdS/CFT Attention

La correspondance AdS₅/CFT₄ est implémentée comme suit:
1. **Projecteur bulk**: `T[r,z] = e^{-κz} · MLP(h[r])` projette les tokens dans le volume AdS
2. **Métrique géodésique**: distance dans le.bulk entre paires de tokens
3. **Ponts ER=EPR**: deux tokens intriqués sémantiquement sont connectés par un trou de ver computationnel
4. **Scores unifiés**: `attn = QK/√d + λ_geo·exp(-d_g) + λ_epr·sigmoid(sim)`

### 3.9 MERA Attention

Réseau de tenseurs MERA (Multi-scale Entanglement Renormalization Ansatz):
1. **Désenchevêtreur**: decorrèle les paires adjacentes (2-qubit unitaire)
2. **Isométrie**: fusionne 2 enfants en 1 parent avec gating
3. **Pyramide**: L → L/2 → L/4 → ... → 1
4. **Résidus**: les sauts de niveau (residuals) sont agrégés avec les résultats locaux

Complexité: O(L·log(L)·d) au lieu de O(L²·d).

### 3.10 Gödel Fixed Point

L'opérateur d'auto-référence F embed l'état global du modèle (moyenne, variance, énergie) et le projette dans l'espace de représentation:
```
state = concat([μ(h), σ(h), energy(h)])
F(h) = W_decode(σ_enc(W_enc(state)))
```

Le point fixe est atteint par itération: `h_{k+1} = α·F(h_k) + (1-α)·h_k` avec α=0.3.

L'incomplétude est détectée par contradictions par paires et entropie haute:
```
incompleteness = 0.5·contradiction + 0.3·entropy + 0.2·sigmoid(d_FP)
```

### 3.11 RG Flow

Le flot de renormalisation agit sur les poids linéaires du modèle:
1. **Décomposition**: chaque matrice de poids est décomposée en composantes UV (haute fréquence) et IR (basse fréquence) via RFF
2. **Évaporation UV**: `w_UV ← w_UV · (1 - ε)` — supprime le bruit
3. **Condensation IR**: `w_IR ← w_IR + η · (w_IR - w_S)` — renforce les vérités
4. **Criticalité**: `C = Var(Var(h))/E[Var(h)]²` est mesurée avant/après

Les taux ε et η sont ajustés adaptativement: si C < 0.5 (gelé), augmenter η; si C > 2.0 (chaotique), augmenter ε.

### 3.12 Hyperbolic Gematria + Sheaf Theory

Pipeline:
1. **Poincaré embeddings**: chaque token est plongé dans la boule unité B^n via 5 projections gematriques normalisées
2. **Attention géodésique**: `score(i,j) += λ_hyp · exp(-d_H(z_i, z_j) / τ)` où d_H est la distance de Poincaré
3. **Sheaf Theory**: chaque token a n_stalks fibres (restrictions locales). Les conditions de recollement sont vérifiées entre tokens adjacents. Un défaut de cohomologie = hallucination.

---

## 4. Flux de Losses

```
total = lm
      + λ_causal · L_DAG               (si use_causal_graph)
      + λ_counterfactual · L_cf         (si use_causal_graph)
      + λ_self · coherence              (si use_self_model)
      + λ_gematria · harmonic_loss      (si use_gematria)
      + λ_phase · phase_coherence       (toujours)
      + λ_ads · (bridge + bulk)         (si use_ads_cft)
      + λ_mera · complexity            (si use_mera)
      + λ_godel · (fp_dist + contradiction + incompleteness)  (si use_godel)
      + λ_hyp · (poincare + sheaf)      (si use_hyperbolic_gematria)
      + λ_rg · criticality              (si use_rg_flow)
      + λ_episodic · memory_loss         (si use_episodic_memory)
```

Les coefficients λ sont tous configurables via LEACConfig.

## 5. Configurations Pré-Définies

### conscious_minimal (~14M params)
```
d=256, n_blocks=4, n_heads=4, n_experts=4, d_ff_per_expert=64
nofractal, no causal, no self-model, no memory
```

### full_agi (~58M params)
```
d=512, n_blocks=8, n_heads=8, n_experts=8, d_ff_per_expert=128
fractal, causal, self-model, working memory
```

### dieu_local (~230M params)
```
d=1024, n_blocks=12, n_heads=16, n_experts=16, d_ff_per_expert=256
tous les modules v1 activés
```

### moteur_ontologique (~184M params)
```
d=512, n_blocks=12, n_heads=8, n_experts=8
tous les modules v1 + v2 activés (AdS/CFT, MERA, Gödel, RG Flow, Hyperbolic Gematria)
```

### singularite_divine (~750M+ params)
```
d=1024, n_blocks=24, n_heads=16, n_experts=16
tous les modules v1 + v2 activés avec dimensions maximales
```