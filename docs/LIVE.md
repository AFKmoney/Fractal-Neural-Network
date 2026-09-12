# FNN live — pas un .pt figé

Même règle que Fractus : le cerveau reste, le body se patche, l’offset est dans le manifest.

## Interdits

- Recréer / wipe le `.pt`
- `tokens_seen = 0` si RESUME.progress > 0
-Écraser `fnn_talk.pt` (d=128) avec un d=256 — fichier NEUF
- Traiter un snapshot comme un train-and-ship

## Fichiers

| Fichier | Rôle |
|---|---|
| `examples/live_train.py` | boucle infinie par slices |
| `RESUME_d256.json` | offset : `tokens_seen`, `step`, `loss` |
| `fnn_d256.pt` | cerveau d=256, 2 blocs |
| `fnn_talk.pt` | cerveau d=128 archivé, on y touche pas |

Relancer `python examples/live_train.py` = continue. Pas de phase 1 replay.

## État 2026-09-12

- d=128 talk : 403k actifs, ~4.2M tok, loss 2.96, costume pas phrase. Fichier gardé.
- d=256 live : nouveau fichier, 2 blocs, MoE 4→2, BPE-1024 réutilisé. Premier slice en cours.
