# Chinchilla adapté — mini FNN

Bloc vivant seulement. Flags greffe OFF.
`d=64`, 2 blocs, MoE 4→2, embed analytique.

## Params

| | |
|---|---|
| Total | 110 976 |
| Actifs (top-2/4) | 77 056 |
| 20 × actifs | **1 541 120 tokens** |
| 100 × actifs (overtrain) | 7.71 M |

## Run 20× (2026-09-12, CPU)

| | |
|---|---|
| Tokens vus | **1 541 088** (100 % du 20×) |
| Steps | 16 053 × seq 96 |
| Wall | 213.5 s |
| Loss | 4.68 → **0.032** |

Génération T=0.3 top-k=6 :

- `Je suis` → `Je suis un FNN. Le bloc est attention lineaire. Le soliton synchroni`
- `Le bloc` → `... von Mises. Ce n'est pas un transformer.`
- `Ce n'est` → `Ce n'est un FNN. Le bloc est attention lineaire. Le soliton synchronise la phase`
- `Le transformer est` (hors ordre du corpus) → `Le transformer est. Ce n'est un transformer. Bonjour. Je suis un FNN.`

Verdict : le bloc apprend. À 20× sur 6 phrases il **récite la feuille**. Pas un LM. Prochaine étape data + BPE, pas plus de poids.
