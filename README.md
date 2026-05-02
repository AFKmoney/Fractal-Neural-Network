# Neural Fractal Network (NFN)

> **Réseau neuronal à topologie fractale multiple et couplage par représentation sinusoïdale paramétrique**
> *Vers une architecture neuro-inspirée pour la proto-intelligence générale*
>
> Auteur : Philippe-Antoine Robert — Version 2.0, 2026-05-02

---

## Vue d'ensemble

Le **Neural Fractal Network (NFN)** est une architecture de langage entièrement nouvelle qui combine :

1. **Topologie fractale récursive** — chaque sous-réseau est une copie contractée de l'ensemble, encodant explicitement l'auto-similarité et l'invariance d'échelle.
2. **Connexions sinusoïdales paramétriques (NSRN)** — les poids ne sont pas des scalaires mais des fonctions du temps : `Γ(t) = A·exp(-γt)·sin(ω·t + φ)`, avec A, ω, φ, γ tous appris par rétropropagation.
3. **Dynamique de phase** — chaque nœud possède une phase θ évoluant selon un oscillateur de Kuramoto étendu, permettant la synchronisation inter-nœuds et le **binding temporel**.
4. **Multi-motifs** — superposition de motifs fractals hétérogènes (arbre binaire, Cantor) couplés via des interactions cross-fréquence.

### Innovation clé : connexion sinusoïdale

```
Γ_{j→i}(t) = A_{ji} · exp(-γ_{ji}·t) · sin(ω_{ji}·t + φ_{ji})
```

Les paramètres `(A, ω, φ, γ)` sont appris, permettant un **multiplexage fréquentiel** : plusieurs signaux coexistent dans un même lien physique.

---

## Architecture

```
NFNLanguageModel
│
├── TokenEmbedding + PositionalEncoding
│
└── NFNBlock × n_blocks
    ├── Bottom-up (pour chaque motif fractal)
    │   Level 0 [B, L, d]     ← tokens
    │   Level 1 [B, L/b, d]   ← SinusoidalAggregator
    │   Level 2 [B, L/b², d]  ← SinusoidalAggregator
    │   ...
    │   Level K [B, L/bᴷ, d]  ← racine
    │
    ├── Couplage inter-motifs (InterMotifCoupler à chaque niveau)
    ├── Attention causale au niveau supérieur (CausalSelfAttention)
    ├── Top-down (SinusoidalBroadcast × K niveaux)
    └── Raffinement temporel (P mini-pas de récurrence)
│
└── LayerNorm → LMHead
```

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Démarrage rapide

```bash
# Test rapide (< 1 min sur CPU)
python examples/quickstart.py

# Entraînement sur Shakespeare
python examples/train_shakespeare.py

# Lancer l'interface web
python run.py

# Entraînement CLI
python train.py --text data/corpus.txt --config small --epochs 10
```

---

## Interface web

```bash
python run.py --port 8000
# → http://localhost:8000
```

Onglets disponibles :
- **Chat** — conversation multi-tours avec streaming WebSocket
- **Code** — complétion, explication, refactorisation, génération
- **Agent** — raisonnement ReAct multi-étapes avec outils (calculate, search, think)
- **Entraînement** — BPTP en direct avec courbe de perte en temps réel
- **Modèle** — visualisation de la topologie fractale et des oscillations de phase

---

## Configurations

| Config | d_model | n_levels | n_blocks | Params (approx.) |
|--------|---------|----------|----------|-----------------|
| `nano`   | 128    | 3        | 2        | ~2M             |
| `small`  | 256    | 4        | 4        | ~15M            |
| `medium` | 512    | 4        | 8        | ~85M            |

```bash
python run.py --config small
python train.py --config medium --epochs 20 --text data/corpus.txt
```

---

## Entraînement

```python
from nfn.config import NFNConfig
from nfn.network import NFNLanguageModel
from nfn.tokenizer import NFNTokenizer
from training.trainer import NFNTrainer

cfg = NFNConfig(d_model=256, n_levels=4, n_blocks=4, motifs=["binary_tree","cantor"])
tok = NFNTokenizer()
cfg.vocab_size = tok.vocab_size
model = NFNLanguageModel(cfg)

trainer = NFNTrainer(model, tok, cfg, lr=3e-4)
trainer.train(open("data/corpus.txt").read(), n_epochs=5, batch_size=4)
```

### Perte multi-objectif (BPTP)

```
L = L_tâche
  + λ_phase    · L_phase     (continuité des phases)
  + λ_freq     · L_freq      (parcimonie des amplitudes sinusoïdales)
  + λ_spectral · L_spectral  (stabilité spectrale / Jacobienne)
```

---

## Inférence

```python
from inference.engine import NFNInferenceEngine

engine = NFNInferenceEngine(model, tokenizer)

# Génération simple
text = engine.generate("Le NFN est", max_new_tokens=200, temperature=0.8)

# Streaming token par token
for tok in engine.stream("Bonjour", max_new_tokens=100):
    print(tok, end="", flush=True)

# Chat multi-tours
reply = engine.chat([{"role":"user","content":"Explique le NFN"}])

# Perplexité
ppl = engine.perplexity("texte de test")
```

Stratégies de décodage : `top_p` (nucleus), `top_k`, `greedy`, `beam`, `mirostat_v2`.

---

## Structure du projet

```
flow/
├── nfn/
│   ├── config.py          # NFNConfig dataclass
│   ├── connections.py     # SinusoidalGate, SinusoidalAggregator, InterMotifCoupler
│   ├── topology.py        # Constructeurs de graphes fractals
│   ├── network.py         # NFNLanguageModel, NFNBlock, MotifBranch
│   └── tokenizer.py       # CharTokenizer + BPETokenizer
├── training/
│   ├── losses.py          # NFNLoss (phase, freq, spectral)
│   └── trainer.py         # NFNTrainer (BPTP, AdamW, scheduler)
├── inference/
│   └── engine.py          # NFNInferenceEngine (stream, beam, mirostat)
├── interface/
│   ├── app.py             # FastAPI + WebSocket
│   ├── agents.py          # ChatAgent, CodeAgent, ReasoningAgent
│   └── static/            # index.html, style.css, app.js
├── examples/
│   ├── quickstart.py
│   └── train_shakespeare.py
├── configs/
│   ├── nano.json
│   ├── small.json
│   └── medium.json
├── train.py               # CLI entraînement
└── run.py                 # Lance l'interface web
```

---

## Fondements théoriques

- **Topologie fractale** : Mandelbrot (1982), FractalNet (Larsson et al., 2016)
- **Connexions sinusoïdales** : SIREN (Sitzmann et al., 2020)
- **Dynamique d'oscillateurs** : Kuramoto (1984), Hoppensteadt & Izhikevich (1999)
- **Liage temporel** : Singer (1999), von der Malsburg (1994)
- **Global Workspace** : Baars (1988)
- **Neural ODE** : Chen et al. (2018)

---

*NFN v1.0 — Laboratoire indépendant NeuroFractal — Philippe-Antoine Robert*
