# FNN — Fractal Neural Network

Bloc de language model : **attention linéaire fractale**, **soliton de phase**, **MoE routé par von Mises**.

Ce n’est **pas** un Transformer. Softmax L×L n’est pas le bloc.
Ce n’est **pas** Fractus CTE (projet séparé).
Ce n’est **pas** un checkpoint avec lequel chatter.

Branche de travail : `clean/quarantine`.
`main` n’a pas été écrasé.

```bash
git clone -b clean/quarantine https://github.com/AFKmoney/Fractal-Neural-Network.git
cd Fractal-Neural-Network
pip install torch
```

---

## Pourquoi cette branche existe

Le repo May 2026 a été gonflé par des sessions Claude/ChatGPT : UI, PRISM, scripts « AGI », fichiers morts, le bloc FNN poussé vers un transformer.

`clean/quarantine` garde le **bloc vivant** et liste le reste dans [`_quarantine/MANIFEST.md`](_quarantine/MANIFEST.md). On n’efface pas l’historique git.

Cœur vivant :

| Fichier | Rôle |
|---|---|
| `nfn/block.py` | 1 bloc = attn → soliton → MoE |
| `nfn/moe.py` | `FractalLinearAttention`, `PhaseRoutedMoE`, `PhaseSoliton` |
| `nfn/model.py` | pile de blocs + tête |
| `nfn/config.py` | `FNNConfig` + flags |
| `nfn/analytic_embed.py` | embed 0 params |
| `nfn/phase_ode.py` | phase / Kuramoto |
| `nfn/tokenizer.py` | char + BPE maison |

Flags greffe (`use_ads_cft`, `use_godel_loop`, `use_causal_graph`, …) : **OFF** sur les runs ci-dessous.

---

## Ce que fait un bloc

```
x
 → FractalLinearAttention   φ(q)·Σ φ(k)ᵀv   (Katharopoulos)
 → PhaseSoliton             gain sur les motifs de phase cohérents
 → PhaseRoutedMoE           gate von Mises, top-k experts
 → x + …
```

Routing : gate_e(x) = softmax_e( κ · mean cos(θ_x − θ_e) ).
θ_e seedé Mandelbrot. Top-k **après** le softmax.

### Patch vitesse (2026-09-12)

Avant, le MoE calculait les **E** experts puis masquait. L’attn causal posait `[B,H,L,d,d]`.

Maintenant (`nfn/moe.py`) :

1. GEMM seulement sur les tokens routés vers l’expert e (vrai top-k).
2. Attn par **chunks** + carry (S, z). Pic `[B,H,C,d,d]`, C=32, plus `[L,d,d]`.

Mêmes noms de poids. Un `.pt` d’avant charge encore.
Détail : [`docs/SPEED.md`](docs/SPEED.md).

À E=4 et L=128 CPU, le gros GEMM unique peut rester aussi vite. Le gain est **E grand** ou **L ≥ 4k**.

`generate()` recalcule encore tout le préfixe. `nfn/kv_cache.py` n’est pas branché. Decode ≠ train.

---

## Entraînement perpétuel (règle Fractus)

Le FNN n’est pas un train-and-ship. Le cerveau reste, le body se patche, l’offset est dans un manifest.

| Fichier | Rôle |
|---|---|
| `examples/live_train.py` | une slice, puis stop ; relancer = continuer |
| `RESUME_d256.json` | `tokens_seen`, `step`, `loss` |
| `fnn_d256.pt` | cerveau d=256 (local, pas dans git) |
| `fnn_talk.pt` | cerveau d=128 archivé — **on n’y écrit plus** |

Interdits :

- recréer / wipe le `.pt`
- `tokens_seen = 0` si le manifest a du progrès
- écraser `fnn_talk.pt` avec un d=256 — largeur nouvelle = **fichier neuf**
- rejouer une « phase 1 » par-dessus un offset > 0

```bash
python examples/live_train.py
# kill → relancer. Il rouvre le cœur.
```

[`docs/LIVE.md`](docs/LIVE.md)

---

## Chinchilla — on compte les **actifs**

Hoffmann : ~20 tokens / paramètre **qui voit le token**.
MoE 4→2 : la moitié des poids experts est froide.

Overtrain petit modèle (bande Phi/Gemma) : ~100× actifs.

| Cerveau | Actifs | 20× | 100× |
|---|---|---|---|
| 80k mini | 77–80 k | 1.5 M | 8 M |
| d=128 talk | 403 k | 8.1 M | 40 M |
| d=256 live | 1.07 M | 21 M | 107 M |
| 1 B dense | 1 B | 20 B | 100 B |
| 1 B total / ~119 M actifs | 119 M | 2.4 B | 12 B |
| 1 B actifs | 1 B | 20 B | 100 B |

FLOP dense : C ≈ 6ND. 1 B × 20 B tok ≈ 1.2×10²⁰ FLOP.

---

## Runs mesurés (CPU, 2026-09-11 → 12)

Tous : **2 blocs**, flags greffe OFF.

### A — feuille de 6 phrases, char, 20×

77 056 actifs. 1.54 M tok. Loss 4.68 → **0.032**. Récite la feuille. Pas un LM.

### B — BPE-400, Tiny Shakespeare, 100×

79 522 actifs. 7.95 M tok. Loss 5.94 → **2.60** (plateau dès ~3–4 M). Pastiche, pas de phrase.

### C — d=128, BPE-1024, ~10×

402 962 actifs. ~4.2 M tok. Loss 6.95 → **2.96**. Costume de pièce, boucle the/to. Cerveau `fnn_talk.pt` archivé.

### D — d=256 live (en cours)

1 067 026 actifs. Slice 1 : step 800, 409 600 tok, loss 6.91 → **3.82**. Encore 0.4× Chinchilla. `RESUME_d256.json`.

Verdict : le bloc apprend. « Parler » = plus d’actifs **et** des milliards de tokens, pas un epoch de plus sur 80k.

[`docs/CHINCHILLA.md`](docs/CHINCHILLA.md)

---

## Plus rapide qu’un Transformer ?

| Régime | Plus vite ? |
|---|---|
| Train L ≤ 512, GPU, FlashAttention | souvent **non** |
| Train L ≥ 8k | attn FNN **oui** |
| MoE avant le patch | non (les E experts passaient) |
| MoE après, E grand | oui |
| Decode actuel | **non** (pas de KV cache sur `generate`) |

Le FNN n’est pas « un transformer accéléré ». C’est un autre bloc.

---

## Quick start

```python
import torch
from nfn.config import FNNConfig
from nfn.model import FNNModel
from nfn.tokenizer import NFNTokenizer

tok = NFNTokenizer()
cfg = FNNConfig(
    vocab_size=tok.vocab_size,
    d_model=64, n_blocks=2, n_heads=2,
    use_causal_graph=False,
    use_self_model=False,
    use_working_memory=False,
    use_ads_cft=False,
    use_mera=False,
    use_godel_loop=False,
    use_rg_flow=False,
)
model = FNNModel(cfg)
x = torch.randint(0, cfg.vocab_size, (1, 32))
logits, losses = model(x, targets=x)
print(logits.shape, losses["lm"].item())
```

Smoke : `python examples/mini_train.py`
Live d=256 : `python examples/live_train.py`

---

## Docs

| Doc | Contenu |
|---|---|
| [`docs/CHINCHILLA.md`](docs/CHINCHILLA.md) | runs A–D |
| [`docs/LIVE.md`](docs/LIVE.md) | open-heart, interdits |
| [`docs/SPEED.md`](docs/SPEED.md) | top-k + chunk attn |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | texte long antérieur |
| [`_quarantine/MANIFEST.md`](_quarantine/MANIFEST.md) | greffe mise de côté |

Les PDF / `PAPER.md` datent d’avant le ménage. Conflit → **ce README + CHINCHILLA + LIVE + SPEED**.

## License

MIT
