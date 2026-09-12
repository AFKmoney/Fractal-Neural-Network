# Speed patches (2026-09-12)

Body only. Same parameter names. `fnn_d256.pt` / `fnn_talk.pt` still load.

## 1. MoE — experts top-k seulement

Avant : `x @ W1.view(E*d_ff, d)` = les **E** experts, puis masque.
Après : boucle sur E, GEMM seulement sur les tokens routés vers e.

Même sortie à 1e-7 près (les experts masqués étaient ×0).

À E=4, L=128, CPU, le gros GEMM unique peut rester aussi vite — le gain est E grand / d_ff grand.

## 2. Attn — plus de `[B,H,L,d,d]`

Carry `(S, z)` par chunk (`chunk_size=32`).
Pic mémoire : `[B,H,C,d,d]` pas `[B,H,L,d,d]`.
Chunk vs C=L : maxdiff 1e-7.

À L=128 le wall-clock est plat (4 petits chunks). À L≥4k le pic RAM / alloc disparaît.
