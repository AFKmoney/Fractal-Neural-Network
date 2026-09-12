# Runs mini FNN (2026-09-11 → 12)

Toujours 2 blocs. Flags greffe OFF. Actifs = total − (1 − K/E) × poids MoE.

## A — char, 6 phrases, 20×

77 056 actifs. 1.54 M tok. Loss 4.68 → 0.032. Récite la feuille.

## B — BPE-400, Shakespeare, 100×

79 522 actifs. 7.95 M tok. Loss 5.94 → 2.60, plateau dès 3–4 M. Pastiche.

## C — d=128, BPE-1024, ~10×

402 962 actifs. ~4.2 M tok. Loss 6.95 → 2.96. Costume de pièce. Cerveau `fnn_talk.pt` — ne plus écrire dessus.

## D — d=256 live

1 067 026 actifs. Slice 1 : 409 600 tok, step 800, loss 6.91 → 3.82.
`fnn_d256.pt` + `RESUME_d256.json`. Relancer = continuer.

## Règle 1B

20× sur les **actifs**. 1 B dense → 20 B tok. 1 B total / 119 M actifs → ~2.4 B tok.
