# Mini FNN runs

Toujours 2 blocs.

## A — 80k BPE-400 100×

Loss → 2.60 plateau. Pastiche.

## B — d=128 BPE-1024 ~4.2M tok

403k actifs, loss 6.95 → 2.96. Costume de pièce, pas de phrase.
Cerveau : `fnn_talk.pt` — **on n’y touche plus**.

## C — d=256 live (perpétuel)

Nouveau fichier `fnn_d256.pt`. Resume via `RESUME_d256.json`.
Voir `docs/LIVE.md`.
