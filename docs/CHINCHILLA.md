# Mini FNN runs

Toujours 2 blocs. Flags greffe OFF.

## A — 80k actifs, BPE-400, 100× Shakespeare

7.95 M tok, loss → 2.60 plateau. Pastiche, pas de phrase.

## B — talk attempt, d=128, BPE-1024

| | |
|---|---|
| Total / actifs | 536 338 / **402 962** |
| Vocab | 1024 (756 merges) |
| Tokens vus | ~4.2 M (~10× actifs) |
| Loss | 6.95 → **2.96** |

Génération (T=0.35–0.4) :

- `ROMEO:` → `She the you to me I and ... I have you`
- `JULIET:` → `What you to the ... wretrown`
- `To be or not to be` → `of the the the to to find` + faux speaker `BRIZABETH`
- `The king` → faux `PRIARENCE` / `BRIXENCE`

Il a le **costume** (labels, retours ligne, thou/have). Pas la phrase.
Plus de tokens sur ce 400k aide encore un peu (3.09→2.96) mais ça part en boucle `the/to`.
Pour parler pour vrai : plus d'actifs (d=256+) ou un GPU. Pas un 3e bloc, pas un 100× de plus sur 400k.
