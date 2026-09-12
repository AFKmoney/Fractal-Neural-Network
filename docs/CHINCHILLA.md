# Chinchilla adapté — mini FNN (2026-09-11)

Mesure sur le bloc vivant, flags greffe OFF.
`d=64`, `n_blocks=2`, MoE 4 experts top-2, embed analytique (0 params).

## Params

| | |
|---|---|
| Total | 110 976 |
| MoE (experts) | 67 840 (~61 %) |
| Actifs estimés | 77 056 (top-2 / 4) |
| `lm_head` | 7 150 |
| `blocks` | 103 698 |

Chinchilla dense classique = 20 tokens / paramètre **total**.
Ici on compte les **actifs**, comme pour un MoE : le reste des experts ne voit pas le token.

| Règle | Tokens |
|---|---|
| 20 × total | 2.22 M |
| 20 × actifs | **1.54 M** |
| Overtrain petit modèle (≈100 × actifs) | 7.71 M |

## Ce run

| | |
|---|---|
| Tokens vus (200 + 4000 steps × 96) | ~0.40 M |
| Fraction du 20 × actifs | **~26 %** |
| Wall CPU | ~60 s |
| Loss | 4.67 → 1.63 → **0.12** |

À 26 % Chinchilla le modèle a surtout **mémorisé** les 6 phrases du corpus. Pas un LM.

Génération (T=0.4, top-k=8) :

- `Je suis` → `Je suis un FNN. Le bloc est est atention lineainea`
- `Le bloc` → `Le bloc ests attention lineaire. Le soliton synchronise la phase.`
- `Ce n'est` → traces de `attention lineaire` / `soliton`

## Pour qu'il parle pour vrai

Il manque encore ~1.1 M tokens au plan 20× actifs, et plutôt 7 M si on veut un petit modèle qui généralise.
Tokenizer char (110) = plafond. BPE ensuite, pas plus de blocs.
