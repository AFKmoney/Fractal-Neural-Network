# Speed patches (2026-09-12)

Body only. Mêmes noms de paramètres. Les `.pt` existants chargent.

## 1. MoE top-k réel

Avant : `x @ W1.view(E*d_ff, d)` = E experts, puis masque.
Après : boucle sur E, GEMM sur les tokens routés vers e seulement.

Sortie identique à ~1e-7.

## 2. Attn chunked + carry (S, z)

Plus de `[B,H,L,d,d]`. Pic `[B,H,C,d,d]`, C=32, plus l’état S, z.

Chunk vs C=L : maxdiff ~1e-7.

## Où ça paie

E=4, L=128, CPU : wall-clock plat.
Ça paie quand E monte ou L ≥ 4k.
