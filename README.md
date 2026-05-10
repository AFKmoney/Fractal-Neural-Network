# Neural Fractal Network (NFN)

> **Architecture de modèle de langage — géométrie fractale, dynamique de phase, apprentissage autonome**

**Auteur :** Philippe-Antoine Robert  
**Version :** 5.0  
**Licence :** Propriétaire — Philippe-Antoine Robert, tous droits réservés

---

## Description honnête

NFN est une **architecture de recherche** pour les modèles de langage. Ce n'est pas un modèle pré-entraîné prêt à l'emploi — c'est l'architecture et le système d'entraînement. Pour obtenir un modèle capable de converser, il faut l'entraîner sur un corpus réel (gigaoctets de texte) avec du calcul GPU (heures à jours selon la taille).

Ce que l'architecture apporte par rapport à un transformer standard :

| Composant | Ce qu'il remplace | Avantage |
|-----------|-------------------|----------|
| Fractal linear attention | Attention softmax O(L²) | Complexité O(L·d²), 255× plus rapide à L=32 768 |
| Dynamique de phase Kuramoto | Embeddings de position | Synchronie d'oscillateurs comme mesure de similarité |
| Embedding analytique | Table d'embedding apprise | 0 paramètre — pas de cold-start |
| Décodeur Zipf-initialisé | Init aléatoire | Correspond à la distribution naturelle du langage dès le départ |
| Condensat spectral (SVD) | Rien (transformer sans mémoire) | Connaissance persistante sans oubli catastrophique |

**v5.0 ajoute un système d'entraînement complet** avec 5 signaux simultanés qui vont au-delà de la prédiction du prochain token.

---

## Installation

### Prérequis

- **Python 3.10+** — [télécharger](https://www.python.org/downloads/)
- **pip** (inclus avec Python)
- **Git** — [télécharger](https://git-scm.com/)
- GPU NVIDIA recommandé pour l'entraînement (CPU fonctionne pour les tests)

### 1. Cloner le dépôt

```bash
git clone https://github.com/AFKmoney/FNN.git
cd FNN
```

### 2. Installer les dépendances

```bash
pip install -e .
```

Cela installe automatiquement : `torch`, `fastapi`, `uvicorn`, `numpy`, `tqdm`, etc.

### 3. Dépendances optionnelles

```bash
# Tokenizer plus rapide (recommandé)
pip install tiktoken

# Interface native Windows (fenêtre desktop au lieu du navigateur)
pip install pywebview

# GPU CUDA 12.1 (si tu as une carte NVIDIA)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# GPU CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon (Mac M1/M2/M3)
pip install torch  # la version MPS est incluse depuis torch 2.0
```

### 4. Vérifier l'installation

```bash
python -m pytest tests/ -q
# → 67 tests passent
```

---

## Lancer l'application

### Windows — double-clic

Double-clique sur **`start.bat`** dans le dossier du projet.

Une fenêtre console s'ouvre, le serveur démarre, le navigateur s'ouvre automatiquement sur `http://127.0.0.1:8000`.

### Ligne de commande (tous systèmes)

```bash
# Démarrage rapide (modèle nano, port 8000)
python run.py

# Modèle plus grand
python run.py --config small

# Charger un checkpoint entraîné
python run.py --model checkpoints/agi_nfn_final.pt

# Activer l'adaptation test-time (TTL) au démarrage
python run.py --ttl --adapter-rank 8

# Port différent, sans ouvrir le navigateur automatiquement
python run.py --port 8080 --no-open

# Exposer sur le réseau local (pour accéder depuis un autre PC)
python run.py --host 0.0.0.0 --port 8000
```

### Linux / Mac

```bash
chmod +x start.sh
./start.sh                    # équivalent de python run.py
./start.sh --config small
```

---

## Interface web

L'interface s'ouvre à `http://127.0.0.1:8000` et comprend 6 onglets :

| Onglet | Fonction |
|--------|----------|
| **Chat** | Conversation avec le modèle, streaming token par token |
| **Code** | Complétion, explication, refactorisation de code |
| **Agent** | Raisonnement multi-étapes avec outils (calculer, chercher, analyser) |
| **Entraînement** | Lancer un entraînement depuis l'interface, voir les métriques en direct |
| **Explorer le Web** | Donner une URL → le modèle lit la page et s'adapte. Mode auto : navigation autonome par mots-clés |
| **Adaptation TTL** | Contrôles des adaptateurs LoRA — activer/désactiver, voir les stats, reset |

---

## Entraîner le modèle

### Mode AGI (recommandé)

5 signaux d'entraînement simultanés :

```bash
# Entraînement minimal pour tester (corpus intégré, pas de données externes)
python train_agi.py --config nano --epochs 3

# Avec tes propres données
python train_agi.py --text data/mon_corpus.txt --config nano --epochs 10

# Modèle moyen, avec adaptation test-time pendant l'entraînement
python train_agi.py \
    --text data/mon_corpus.txt \
    --config medium \
    --epochs 10 \
    --batch 4 \
    --lr 3e-4 \
    --ttl --adapter-rank 8 \
    --sample-every 200

# Sans self-play (plus rapide)
python train_agi.py --text data/mon_corpus.txt --no-self-play --no-critique

# Reprendre depuis un checkpoint
python train_agi.py --resume checkpoints/agi_nfn_step500.pt --epochs 5
```

### Tous les flags d'entraînement

```
--text PATH          Fichier texte d'entraînement (UTF-8)
--config NAME        nano | small | medium | large
--epochs N           Nombre d'époques (défaut: 3)
--batch N            Batch size (défaut: 4)
--lr FLOAT           Learning rate (défaut: 3e-4)
--seq-len N          Longueur de contexte (défaut: depuis config)
--output DIR         Dossier des checkpoints (défaut: checkpoints/)
--save-every N       Sauvegarder tous les N steps (défaut: 500)
--log-every N        Logger tous les N steps (défaut: 10)
--device auto|cpu|cuda|mps
--fp16               Précision mixte fp16 (CUDA seulement)
--grad-accum N       Accumulation de gradient (défaut: 1)

Signaux AGI :
--no-self-play       Désactiver self-play DPO-lite
--no-critique        Désactiver critique constitutionnelle
--no-sleep           Désactiver cycle WAKE/SLEEP
--no-curiosity       Désactiver pondération par curiosité
--agi-start N        Step où les pertes AGI commencent (défaut: 200)
--agi-ramp N         Steps de montée en charge (défaut: 100)

Adaptation test-time :
--ttl                Activer les adaptateurs LoRA pendant l'entraînement
--adapter-rank N     Rang LoRA (défaut: 8)
--online-lr FLOAT    LR des adaptateurs (défaut: 2e-4)
--online-steps N     Steps de gradient par adaptation (défaut: 4)
--ppl-gate FLOAT     Seuil ppl — skip si modèle connaît déjà (défaut: 30)
--save-adapters PATH Sauvegarder les adaptateurs en fin d'entraînement
--load-adapters PATH Charger des adaptateurs au démarrage

Évaluation :
--eval-text PATH     Texte de validation pour la perplexité
--eval-every N       Évaluer tous les N steps
--sample-every N     Générer un exemple tous les N steps
--sample-prompt STR  Prompt pour les exemples
```

### Mode LM standard

```bash
python train.py --text data/mon_corpus.txt --config nano --epochs 5
```

---

## Système d'entraînement AGI (v5.0)

Un LLM standard = minimiser la cross-entropie sur le prochain token. NFN v5.0 entraîne avec 5 signaux en parallèle :

| Signal | Mécanisme |
|--------|-----------|
| **LM + curiosité** | Cross-entropie pondérée par l'entropie par token — les exemples surprenants reçoivent un signal de gradient plus fort |
| **Perte AGI multi-objectif** | DAG causal + alignement de but + cohérence de phase + ACT halting + codage prédictif + énergie libre + self-consistency |
| **Self-play DPO-lite** | Génère N candidats → trie par −perte LM → gradient DPO préférence + distillation vers le meilleur |
| **Critique constitutionnelle** | Génère → ajoute `[CRITIQUE]` → génère critique → ajoute `[REVISION]` → entraîne sur la révision à 2× poids |
| **Cycle WAKE/SLEEP** | Chaque step : écriture en mémoire épisodique. Tous les N steps : consolidation épisodique→sémantique + replay |

Curriculum : steps 0→200 LM seulement (base stable), puis montée progressive vers les pertes AGI complètes sur 100 steps.

---

## Adaptation test-time (apprendre sans ré-entraîner)

Le flag `--ttl` active des **adaptateurs LoRA** — une couche légère sur les projections d'attention qui se met à jour à l'inférence :

```
Poids de base  →  GELÉS  (résultat de l'entraînement, jamais modifiés)
Adaptateurs LoRA → MISE À JOUR  (O(rank × d) params, ~0.1% du modèle)
```

Fonctionnement :
1. Le modèle lit un nouveau texte → calcule la perplexité
2. Si ppl > seuil (texte surprenant) : N steps de gradient sur les adaptateurs seulement
3. Décroissance exponentielle des adaptateurs après chaque mise à jour (oubli contrôlé)
4. Les adaptateurs se sauvegardent et se rechargent entre sessions

```bash
# Activer TTL dans l'interface web
python run.py --ttl

# Sauvegarder/charger les adaptateurs
python train_agi.py --save-adapters checkpoints/adapters.pt
python run.py --model checkpoints/agi_nfn_final.pt --ttl
# puis charger via l'onglet "Adaptation TTL" de l'interface
```

---

## Exploration internet

Le modèle peut lire des pages web et s'adapter en temps réel, sans installer de dépendances supplémentaires (utilise uniquement la stdlib Python) :

**Via l'interface web :** Onglet "Explorer le Web" → coller une URL → le modèle lit la page, affiche la perplexité avant/après.

**Mode autonome :** Donner une URL de départ + des mots-clés → le modèle navigue de page en page automatiquement.

**Via Python :**

```python
from nfn.web_explorer import WebExplorer
from nfn.online_learner import OnlineLearner

explorer = WebExplorer()
learner  = OnlineLearner(model, tokenizer, adapter_rank=8)

# Lire une page
page = explorer.fetch("https://fr.wikipedia.org/wiki/Intelligence_artificielle")
print(f"Titre: {page['title']} — {page['n_chars']} caractères")

# Adapter le modèle depuis cette page
stats = learner.adapt_from_text(page['text'])
print(f"ppl: {stats['ppl']:.1f} → loss: {stats['loss']:.4f}")

# Navigation autonome
for page in explorer.explore("https://fr.wikipedia.org/wiki/Réseau_de_neurones",
                              n_pages=10,
                              keywords=["apprentissage", "architecture"]):
    stats = learner.adapt_from_text(page['text'])
    print(f"  {page['title']}: ppl {stats['ppl']:.0f}")
```

---

## API Python

```python
from nfn.agi_model import build_agi_model
from nfn.tokenizer import NFNTokenizer
from nfn.online_learner import OnlineLearner

tok   = NFNTokenizer()
model = build_agi_model(vocab_size=tok.vocab_size, d_model=512, n_blocks=8)

# Génération dirigée par un but
model.set_goal(tok.encode("Expliquer étape par étape"))
ids = tok.encode("Le concept principal est", add_bos=True)
out = model.generate(ids, max_new_tokens=200, temperature=0.8)
print(tok.decode(out[0].tolist()))

# Réinitialiser le but
model.reset_goal()

# Adaptation test-time
learner = OnlineLearner(model, tok, adapter_rank=8, ppl_gate=30.0)
learner.adapt_from_text("Nouveau texte que le modèle n'a pas vu...")
print(learner.stats())
# → {'n_adapters': 12, 'adapter_params': 13824, 'adapter_ratio': '0.43%', ...}

# Sauvegarder les adaptateurs
learner.save_adapters("session_adapters.pt")
learner.load_adapters("session_adapters.pt")
learner.reset()   # oubli complet
learner.decay(steps=10)  # décroissance manuelle
```

---

## API HTTP

Le serveur expose une API REST + WebSocket à `http://127.0.0.1:8000` :

```
POST /api/chat              → Chat multi-tour (JSON, bloquant)
WS   /ws/chat               → Chat streaming (WebSocket)
POST /api/generate          → Génération brute
POST /api/think             → Raisonnement + réponse (N rounds)
POST /api/agent/run         → Agent avec outils
POST /api/learn             → Apprendre depuis texte (mémoire épisodique)
POST /api/explore/url       → Lire une URL et s'adapter
POST /api/explore/text      → Adapter depuis texte brut
WS   /ws/explore            → Exploration autonome en streaming
GET  /api/ttl/stats         → Stats des adaptateurs LoRA
POST /api/ttl/enable        → Activer TTL
POST /api/ttl/disable       → Désactiver TTL
POST /api/ttl/reset         → Remettre adaptateurs à zéro
POST /api/train/start       → Démarrer entraînement en arrière-plan
POST /api/train/stop        → Arrêter
GET  /api/train/status      → Métriques d'entraînement en direct
WS   /ws/train              → Métriques live (WebSocket)
GET  /api/status            → Info modèle, modules actifs
```

---

## Architecture

```
NFN v5.0
┌──────────────────────────────────────────────────────────┐
│  Token Input [B, L]                                      │
│       │                                                  │
│  AnalyticTokenEmbedding  [0 paramètres]                  │
│  Fourier fractal + géométrie de classe de caractère      │
│       │                                                  │
│  AGIBlock × n_blocks                                     │
│  ├─ EfficientNFNBlock                                    │
│  │   ├─ FractalLinearAttention  O(L·d²)                  │
│  │   ├─ PhaseSoliton                                     │
│  │   └─ PhaseRoutedMoE                                   │
│  ├─ TwoTierMemory  (anneau épisodique + SVD sémantique)  │
│  ├─ CausalGraphLayer  (DAG + do-calculus)                │
│  ├─ PhaseGoalPredictor  (forçage λ·sin(θ*−θ))           │
│  ├─ RecursiveReasoner  (halting ACT)                     │
│  └─ PredictiveCodingBlock                                │
│       │                                                  │
│  BayesianZipfianDecoder                                  │
│  Logits [B, L, V]                                        │
└──────────────────────────────────────────────────────────┘
  Adaptateurs LoRA optionnels sur toutes les projections
  (TTL — poids de base gelés)
```

---

## Tailles de modèle

| Config | Paramètres | RAM CPU | VRAM GPU | Usage |
|--------|-----------|---------|----------|-------|
| `nano` | ~3M | ~100 MB | ~200 MB | Tests rapides, CI |
| `small` | ~15M | ~500 MB | ~800 MB | Expériences |
| `medium` | ~85M | ~2 GB | ~3 GB | Entraînement sérieux |
| `large` | ~350M | ~8 GB | ~12 GB | Production |

---

## Efficacité

| Métrique | Transformer dense | NFN v5.0 |
|----------|------------------|----------|
| FLOPs attention (L=512) | 33.6M | 8.4M **(4×)** |
| FLOPs attention (L=4 096) | 2.15B | 134M **(16×)** |
| FLOPs attention (L=32 768) | 137B | 537M **(255×)** |
| Paramètres embedding | standard | **0** (analytique) |
| Mémoire cross-session | aucune | anneau épisodique + SVD |
| Adaptation inférence | aucune | LoRA fast weights (~0.1%) |

---

## Structure des fichiers

```
FNN/
├── nfn/                        Architecture principale
│   ├── config.py               NFNConfig — tous les hyperparamètres
│   ├── analytic_embed.py       Embedding analytique 0-paramètre
│   ├── condensate.py           FractalRFF + SpectralCondensate
│   ├── moe.py                  PhaseRoutedMoE + FractalLinearAttention
│   ├── efficient_block.py      EfficientNFNBlock
│   ├── episodic_memory.py      TwoTierMemory (épisodique + sémantique)
│   ├── causal.py               CausalGraphLayer — DAG + do-calculus
│   ├── goal.py                 PhaseGoalPredictor — forçage Kuramoto
│   ├── reasoning.py            RecursiveReasoner (ACT halting)
│   ├── predictive.py           PredictiveCodingBlock
│   ├── hyper.py                ContextHyperNet
│   ├── ssm.py                  FractalSSM (style Mamba, optionnel)
│   ├── online_learner.py       OnlineLearner — adaptation LoRA TTL
│   ├── web_explorer.py         WebExplorer — navigation internet stdlib
│   ├── agi_block.py            AGIBlock — bloc complet v5.0
│   ├── agi_model.py            AGINFNModel — stack complet
│   ├── network.py              NFNLanguageModel (LM standard)
│   └── tokenizer.py            Tokenizer 3 niveaux
│
├── training/
│   ├── agi_trainer.py          AGITrainer — 5 signaux d'entraînement
│   ├── trainer.py              NFNTrainer — LM standard
│   └── losses.py               NFNLoss + AGILoss
│
├── inference/
│   └── engine.py               Streaming, beam, décodage spéculatif
│
├── interface/
│   ├── app.py                  Serveur FastAPI + WebSocket
│   ├── agents.py               Agents Chat / Code / Raisonnement
│   └── static/
│       ├── index.html          Interface web (6 onglets)
│       ├── style.css           Thème sombre
│       └── app.js              Logique frontend
│
├── configs/
│   ├── nano.json               ~3M paramètres — tests rapides
│   ├── small.json              ~15M paramètres
│   ├── medium.json             ~85M paramètres
│   └── large.json              ~350M paramètres
│
├── tests/                      67 tests unitaires
├── docs/                       Documentation technique
│   ├── ARCHITECTURE.md         Formalisme mathématique
│   ├── CHANGELOG.md            Historique des versions
│   ├── API.md                  Référence API
│   └── THEORY.md               Théorie : fractales, Kuramoto, NFMC
│
├── train_agi.py                Point d'entrée entraînement AGI (v5.0)
├── train.py                    Point d'entrée entraînement LM standard
├── run.py                      Lanceur de l'interface web
├── start.bat                   Lanceur Windows (double-clic)
├── start.sh                    Lanceur Linux / Mac
└── requirements.txt            Dépendances
```

---

## Tests

```bash
python -m pytest tests/ -q      # 67 tests, ~80 secondes sur CPU
python -m pytest tests/ -v      # détail de chaque test
python -m pytest tests/test_agi.py -v    # tests AGI seulement
```

---

## Dépannage

**`ModuleNotFoundError: No module named 'nfn'`**
```bash
pip install -e .   # installer en mode développement depuis la racine du projet
```

**`CUDA out of memory`**
```bash
python train_agi.py --config nano --batch 1 --seq-len 64  # réduire la taille
```

**L'interface ne s'ouvre pas**
```bash
python run.py --no-open   # désactiver l'ouverture auto
# puis aller manuellement sur http://127.0.0.1:8000
```

**Erreur à l'entraînement sur Mac (MPS)**
```bash
python train_agi.py --device cpu   # MPS a des limitations avec certaines ops
```

**Le modèle génère du bruit**  
Normal pour un modèle non entraîné. Il faut l'entraîner sur un corpus réel en premier.

---

## Ce que c'est / ce que ce n'est pas

**C'est :** Une architecture de recherche bien structurée avec un système d'entraînement avancé. Tous les composants sont différentiables et testés. Les signaux d'entraînement (DPO-lite, critique constitutionnelle, WAKE/SLEEP) s'appuient sur des publications de recherche.

**Ce n'est pas :** Un modèle pré-entraîné. On ne peut pas avoir une vraie conversation avec lui sans l'entraîner d'abord sur de vraies données. Les innovations architecturales donnent des avantages structurels mais ne remplacent pas les données et le calcul.

**Sur l'étiquette "AGI" :** Elle décrit l'*objectif* d'entraînement (multi-signal, auto-correctif, dirigé par un but) — pas une affirmation d'intelligence générale dans les poids non entraînés.

---

*Philippe-Antoine Robert*  
*« L'intelligence n'est pas une question de taille. C'est une question de structure. »*
