# Chinchilla adapté — mini FNN

2 blocs, d=64, MoE 4→2, embed analytique. Pas plus de blocs.

## Params

| Run | Vocab | Total | Actifs | Cible |
|---|---|---|---|---|
| Char sheet 20× | 110 | 110 976 | 77 056 | 1.54 M tok |
| BPE Shakespeare 100× | 400 | 97 058 | **79 522** | **7.95 M tok** |

`moe_d_ff_per_expert` 64→32 pour garder ~80k actifs malgré le `lm_head` BPE.

## Run A — 20× char, 6 phrases

1.54 M tok, 213 s CPU, loss 4.68 → **0.032**.
Récite la feuille. Pas un LM.

## Run B — 100× BPE-400, Tiny Shakespeare

| | |
|---|---|
| Corpus | 1 115 394 chars → 832 807 tok BPE (≈1.34 char/tok) |
| Tokens vus | **7 951 360** (100 % du 100×) |
| Steps | 7765 × batch 8 × seq 128 |
| Loss | 5.94 → **2.60** (plateau ~2.52–2.70 dès ~3 M) |

Génération T=0.7 top-k=20 :

- `First Citizen:` → label + anglais cassé (`how go I … That pon actime`)
- `To be or not` → continue en pastiche, pas le soliloque
- `ROMEO:` → blank verse fake + faux speaker `BE prienteregood`
- `The king` → `HAN OHARG OK II` + mots shakespeariens brouillés
- `Hello world` → bascule quand même dans le registre théâtre

Verdict : le bloc **fit du vrai texte**. À 80k actifs + BPE 400, 100× ne suffit pas à articuler hors script. Le plafond n'est plus le train (loss plateaux). C'est la capacité / le vocab.

Prochain levier si on reste à 2 blocs : BPE 2k–4k (casse le budget 80k via `lm_head`) **ou** plus de data diverse sans plus de poids — peu probable de décoller sous ~2.5 CE char-ish.
