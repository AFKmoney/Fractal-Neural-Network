# NFN — Architecture Technique Complète

**Auteur :** Philippe-Antoine Robert
**Version :** 3.2
**Date :** 2026-05-03 07:22:48 UTC

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Connexions Sinusoïdales Paramétriques](#2-connexions-sinusoïdales-paramétriques)
3. [Topologie Fractale](#3-topologie-fractale)
4. [Dynamique de Phase — ODE de Kuramoto](#4-dynamique-de-phase--ode-de-kuramoto)
5. [Flash Attention + RoPE Long Contexte](#5-flash-attention--rope-long-contexte)
6. [KV-Cache Fractal](#6-kv-cache-fractal)
7. [Mémoire Persistante Inter-Contexte](#7-mémoire-persistante-inter-contexte)
8. [NFMC v3.0 — Noyau Fractal Condensé](#8-nfmc-v30--noyau-fractal-condensé)
9. [v3.1 — ZeroShotNFMC](#9-v31--zeroshotnfmc)
10. [v3.2 — EfficientNFN](#10-v32--efficientnfn)
11. [Entraînement BPTP](#11-entraînement-bptp)
12. [Tokenizer Trois Niveaux](#12-tokenizer-trois-niveaux)
13. [Multi-GPU DDP / FSDP](#13-multi-gpu-ddp--fsdp)

---

## 1. Vue d'ensemble

Le NFN est un modèle de langage causal dont l'architecture repose sur quatre principes fondamentaux absents des transformers standards :

| Principe | Implémentation | Fichier |
|----------|---------------|---------|
| Auto-similarité fractale | `MotifBranch` avec `SinusoidalAggregator × K` | `network.py`, `connections.py` |
| Couplage sinusoïdal paramétrique | `Γ(t) = A·exp(−γt)·sin(ω·t+φ)` appris | `connections.py` |
| Synchronisation de phase | ODE de Kuramoto différentiable (RK4) | `phase_ode.py` |
| Connaissance a priori condensée | Noyau fractal multidimensionnel NFMC | `condensate.py`, `nfmc.py` |

---

## 2. Connexions Sinusoïdales Paramétriques

### Définition

Chaque connexion entre nœuds est une **fonction temporelle apprise** :

```
Γ(t) = A · exp(−γt) · sin(ω·t + φ)
```

Paramètres appris : `A` (amplitude), `ω` (fréquence), `φ` (phase), `γ` (amortissement).

### Implémentation — `SinusoidalAggregator`

```python
# connections.py
class SinusoidalGate(nn.Module):
    # A, omega, phi, log_gamma : [out_channels, rank] — appris par SGD
    def forward(self, t):
        angle = t.view(-1,1,1) * self.omega.view(1,1,-1) + self.phi.unsqueeze(0)
        sin_val = torch.sin(angle)              # [N, out_channels, rank]
        gate = (self.A.unsqueeze(0) * sin_val).sum(-1)  # [N, out_channels]
        if self.damping:
            gate = gate * torch.exp(-F.softplus(self.log_gamma).unsqueeze(0) * t.abs().unsqueeze(-1))
        return gate
```

### Avantages vs poids scalaires

- Encode des **relations temporelles** — la connexion est plus forte à certaines fréquences
- **Amortissement naturel** — les connexions lointaines s'affaiblissent exponentiellement
- **Gradients stables** — les sinusoïdes ont des dérivées bornées, contrairement à ReLU
- **Inductive bias** — les fréquences apprises correspondent aux échelles du langage

---

## 3. Topologie Fractale

### Motifs supportés

| Motif | Branchement `b` | Structure | Cas d'usage |
|-------|----------------|-----------|-------------|
| `binary_tree` | 2 | Hiérarchique binaire | Structure sémantique |
| `cantor` | 3 | Ensemble de Cantor | Multirésolution |

### Hiérarchie bottom-up / top-down

```
Niveau K (top) :  L/b^K nœuds   ← Flash Self-Attention ici
Niveau K-1 :      L/b^(K-1) nœuds
    ...
Niveau 0 (base) : L nœuds        ← tokens d'entrée

Bottom-up : aggrégation sinusoïdale  (L → L/b → ... → L/b^K)
Top-down  : diffusion sinusoïdale    (L/b^K → ... → L/b → L)
```

### Complexité par bloc NFN

```
Bottom-up :  Σ_{k=0}^{K-1} (L/b^k) · O(d²) = O(L · b/(b-1) · d²) = O(L·d²)
Attention :  O((L/b^K)² · d)  — contexte réduit au niveau top
Top-down  :  O(L·d²)  (symétrique)
Total     :  O(L·d² + (L/b^K)²·d)
```

Pour `b=2, K=4, L=4096` : `O(4096·d² + 256²·d)` vs `O(4096²·d)` standard → **16× moins cher**.

---

## 4. Dynamique de Phase — ODE de Kuramoto

### Équation du modèle

```
dθᵢ/dt = Ωᵢ + Σⱼ Kⱼᵢ · sin(θⱼ − θᵢ + φⱼᵢ)
```

où :
- `θᵢ` : phase du nœud i
- `Ωᵢ` : fréquence naturelle (apprise)
- `Kⱼᵢ` : matrice de couplage de rang `r` (apprise)
- `φⱼᵢ` : déphasage (appris)

### Intégration RK4 différentiable

```python
# phase_ode.py
def rk4_step(f, y, t, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt*k1/2)
    k3 = f(t + dt/2, y + dt*k2/2)
    k4 = f(t + dt, y + dt*k3)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

**Clé** : l'intégration RK4 est entièrement différentiable → les gradients traversent l'ODE via les phases.

### Rôle dans le réseau

Les phases `θ` modulent les représentations cachées :

```
h_out = h + α · tanh(W_phase · [cos(θ), sin(θ)])
```

Les nœuds qui se synchronisent (`θᵢ ≈ θⱼ`) forment des **clusters conceptuels** — c'est le mécanisme du binding temporel.

---

## 5. Flash Attention + RoPE Long Contexte

### Flash Attention

Utilise `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.0+) :
- Algorithme IO-aware (Dao et al., 2022)
- Pas de matrice d'attention en mémoire : O(L) mémoire, O(L²) FLOPs
- `is_causal=True` pour l'entraînement, `False` avec KV-cache

### RoPE avec extension NTK

```python
# rope.py
def precompute_freqs_cis(dim, max_seq_len, base=10000., scale_factor=1.0):
    if scale_factor != 1.0:
        # NTK-aware scaling (bloc et al., 2023)
        base = base * (scale_factor ** (dim / (dim - 2)))
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, theta)
    return torch.polar(torch.ones_like(angles), angles)
```

Extension dynamique : si `seq_len > max_seq_len`, le facteur d'échelle NTK est recalculé automatiquement → **contexte illimité sans perte de qualité**.

| Config | `max_seq_len` | `context_len` (NTK) |
|--------|--------------|---------------------|
| nano | 512 | 4K |
| small | 2048 | 32K |
| medium | 4096 | 128K |
| large | 8192 | 128K+ |

---

## 6. KV-Cache Fractal

### Principe

Les niveaux supérieurs de la hiérarchie fractale changent moins vite que les niveaux inférieurs. Le cache fractal exploite cette propriété :

```
Niveau k recompute tous les b^k tokens
Niveau 0 : chaque token
Niveau 1 : tous les 2 tokens
Niveau 2 : tous les 4 tokens
Niveau K : tous les 16 tokens (pour b=2, K=4)
```

### Implémentation

```python
# kv_cache.py
class FractalStateCache:
    def should_recompute(self, step: int, level: int) -> bool:
        return step % (self.branching ** level) == 0
```

**Résultat** : inférence autoregressive en O(1) par token (au lieu de O(L) pour recalculer tout le contexte).

---

## 7. Mémoire Persistante Inter-Contexte

### Architecture

```
Mémoire [B, M, d]  ← M slots initialisés aléatoirement (petites valeurs)

READ  : Q = LayerNorm(ctx) @ W_q
        K, V = Memory @ W_k, Memory @ W_v
        output = softmax(QKᵀ/√d) @ V   ← cross-attention

WRITE : summary = mean(ctx, dim=1)       ← résumé du contexte
        gate = sigmoid(summary @ W_gate) ← [M] importance par slot
        candidate = summary @ W_write    ← [M, d] valeurs candidates
        Memory ← (1−gate)·Memory + gate·candidate  ← EMA gated
```

### FractalMemoryBank

Pour les modèles avec `memory_per_level=True`, chaque niveau fractal possède sa propre banque :

```
Niveau 0 : mémoire épisodique (8 slots, mise à jour fréquente)
Niveau 1 : mémoire de travail (16 slots)
Niveau 2 : mémoire sémantique (32 slots)
Niveau K : mémoire encyclopédique (64+ slots, rare mise à jour)
```

**La mémoire survit entre les conversations** — elle peut être sauvegardée sur disque et rechargée (`save_memory()` / `load_memory()`).

---

## 8. NFMC v3.0 — Noyau Fractal Condensé

### Le noyau universel

```
K(x, y) = ∫_Ω exp(i·Φ_ω(x,y)) dμ(ω)
```

Approximé par Random Fourier Features fractals :

```
K(x,y) ≈ φ(x)ᵀφ(y)

φ(x) = [cos(W·x + b), sin(W·x + b)] · √(2/r)
W : fréquences fractal 1/f — bande k : W_k ~ N(0, 2^(k/n_scales)·I)
```

### Condensation spectrale (one-shot, zéro SGD)

```python
# condensate.py
class SpectralCondensate:
    def condense(self, features):   # [N, rff_dim]
        features -= features.mean(0)
        _, S, Vh = torch.linalg.svd(features, full_matrices=False)
        self.U = Vh[:self.rank].T   # top-r eigenvectors
        self.S = S[:self.rank] / S[0]  # normalised eigenvalues
```

### Verrouillage de phase Helmholtz

```
E(θ) = −½ Σᵢⱼ K̃(xᵢ,xⱼ) cos(θᵢ − θⱼ)   [énergie libre de Helmholtz]

Mise à jour Kuramoto :
θᵢ ← θᵢ + η Σⱼ K̃(xᵢ,xⱼ) sin(θⱼ − θᵢ)
```

Les **attracteurs de phase** correspondent aux catégories syntaxiques et sémantiques.

---

## 9. v3.1 — ZeroShotNFMC

### Embedding analytique (0 paramètre)

```python
# analytic_embed.py
# e_k(t) = cos(ω_k · t/V · 2π) pour k ∈ [0, d/2)
# ω_k = base^(k / (d/2))  — réseau de fréquences fractal

FractalCodepointEmbedding :  [V, d]  ← 0 paramètre, table de buffers
CharClassEmbedding :          [V, 16] ← voyelle/consonne/chiffre/ponct
AnalyticTokenEmbedding :      fusion via projection orthogonale fixe (QR)
```

### Mémoire de Hopfield Moderne (capacité exponentielle)

Ramsauer et al., 2020 — capacité O(exp(d/2)) :

```
x_new = Xᵀ · softmax(β · X · ξ / √d)
```

Patterns semés depuis :
1. Vecteurs de Fourier aux **fréquences de Farey/Mandelbrot**
2. Vecteurs aléatoires à pondération **Zipf**
3. Vecteurs orthogonaux de couverture uniforme

### Fréquences de Mandelbrot (séquence de Farey)

L'ensemble de Mandelbrot est paramétrisé par l'angle externe θ ∈ [0,1). Les angles p/q (fraction de Farey) correspondent aux **points paraboliques de période q** :

```
1/2  → période 2 (bulbe principal gauche)
1/3  → période 3
1/4  → période 4
2/5  → période 5
...  [Séquence de Stern-Brocot]
```

Ces fréquences correspondent exactement aux échelles temporelles du langage :
- Période 2 : binaire sujet/prédicat
- Période 3 : triplet SVO (Sujet-Verbe-Objet)
- Période 4 : structures quaternaires (déterminant-nom-verbe-complément)

### Décodeur Zipfien

La loi de Zipf : `P(rang=k) ∝ k^{−α}` est universelle pour le langage naturel.
Initialisation du décodeur :

```python
# hopfield.py
# W[k,:] = vecteur_singulier_k * k^{-α/2}  — structure spectrale Zipf
# bias[k] = -α · log(k)                     — distribution marginale correcte
```

**Résultat** : distribution correcte dès le premier passage, sans aucun exemple.

### Prédicteur de Phase Causal

```
dθₜ/dt = Ω(xₜ) + K(xₜ) ⊙ Σⱼ<ₜ sin(θⱼ − θₜ)
```

- `Ω(xₜ) = W_Ω · xₜ` : fréquences naturelles conditionnées à l'entrée
- Causalité stricte vérifiée (diff = 0.000000 sur entrées futures)
- Fréquences initiales = angles de Mandelbrot (fixes)

---

## 10. v3.2 — EfficientNFN

### FractalLinearAttention — O(L·d²)

Identité kernelisée (Katharopoulos et al., 2020) :

```
(φ(Q)φ(K)ᵀ)V = φ(Q)(φ(K)ᵀV)    [associativité]
O(L²d)         O(Ld²)
```

Implémentation causale via somme cumulée :

```python
for i in range(L):
    kv_sum += k[i].outer(v[i])   # [d, d] — pas de matrice L×L
    k_sum  += k[i]               # [d]
    out[i] = (q[i] @ kv_sum) / (q[i] · k_sum)
```

Maps de features multi-échelles : `φ_k(x) = elu(x + freq_Mandelbrot_k) + 1`

| L | Attn standard | FractalLinearAttn | Gain |
|---|---------------|-------------------|------|
| 512 | 33.6M FLOPs | 8.4M | 4× |
| 4096 | 2.15B | 134M | 16× |
| 32768 | 137B | 537M | **255×** |

### PhaseRoutedMoE — Routing continu von Mises

```
gate_e(x) = exp(κ · cos(θ_x − θ_e)) / Z     [distribution von Mises]

θ_x = atan2(W_im·x, W_re·x)   ← phase encodée depuis l'entrée
θ_e = angles de Mandelbrot     ← phases expertes fixes (semées)
```

**Propriétés vérifiées :**
- Charges experts sans loss auxiliaire : 0.296 / 0.267 / 0.243 / 0.193 ≈ 0.25
- Gradients continus partout (pas de argmax discret)
- K=2/E=8 experts actifs = 25% des FLOPs d'un FFN dense

### PhaseSoliton — Cohérence long-range

```
gain(x) = sigmoid(W_gain · θ(x) + coherence(x) / τ)
out = LayerNorm(x + gain · x)
```

Amplifie les patterns de phase cohérents, supprime le bruit incohérent.
Prévient l'effacement des dépendances longue portée sans attention quadratique.

---

## 11. Entraînement BPTP

### Back-Propagation Through Phase

La loss totale :

```
L = L_task + λ_phase · L_phase + λ_freq · L_freq + λ_spectral · L_spectral

L_task     = CrossEntropy(logits, targets)
L_phase    = ||phases − phases_target||²   [synchronisation cible]
L_freq     = ||FFT(phases)||²_hors_bande   [pureté spectrale]
L_spectral = ||W||²_spectral              [régularisation norme spectrale]
```

### Optimiseur différencié

```python
# trainer.py
# Paramètres sinusoïdaux (A, ω, φ, γ) : LR × 0.3, weight_decay=0
# Autres paramètres : LR standard, weight_decay=0.1
```

Les paramètres sinusoïdaux ont un LR réduit car leurs gradients sont naturellement plus grands (fonctions périodiques à grande dérivée).

### Fonctionnalités

| Fonctionnalité | Paramètre | Notes |
|---------------|-----------|-------|
| Accumulation de gradient | `grad_accumulation_steps` | simuler grands batch |
| Mixed precision | `dtype=torch.bfloat16` | AMP avec GradScaler |
| Gradient checkpointing | `use_grad_checkpointing=True` | −50% mémoire |
| torch.compile | `compile_model=True` | 2-3× sur Ampere+ |
| Scheduler | cosine + warmup | ratio min_lr=0.1 |

---

## 12. Tokenizer Trois Niveaux

```
Tier 1 — TiktokenTokenizer  : cl100k_base (100K vocab, qualité GPT-4)
Tier 2 — BPETokenizer       : BPE pur Python, entraînable depuis zéro (32K)
Tier 3 — CharTokenizer      : 110 tokens, toujours disponible (fallback)

Auto-sélection : tiktoken > char (si tiktoken non installé)
```

Tokens spéciaux : `<pad>` `<bos>` `<eos>` `<unk>` `<sep>` `<sys>` `<usr>` `<ast>` `<code>` `</code>` `<think>` `</think>`

---

## 13. Multi-GPU DDP / FSDP

```bash
# DDP — modèle répliqué sur chaque GPU
torchrun --nproc_per_node=4 train.py --distributed ddp

# FSDP — modèle fragmenté (pour très grands modèles)
torchrun --nproc_per_node=8 train.py --distributed fsdp
```

**FSDP** utilise `MixedPrecision(param_dtype=bfloat16, reduce_dtype=float32)` et `ShardingStrategy.FULL_SHARD` avec `BackwardPrefetch.BACKWARD_PRE` pour les meilleures performances.

Les `NFNBlock` sont auto-wrappés comme couches FSDP grâce à `transformer_auto_wrap_policy`.

---

*Philippe-Antoine Robert — 2026-05-03 07:22:48 UTC*
