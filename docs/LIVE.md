# FNN live — pas un .pt figé

Même règle que Fractus : le cerveau reste, le body se patche, l’offset est dans le manifest.

## Fichiers

| Fichier | Rôle |
|---|---|
| `examples/live_train.py` | une slice puis exit ; relancer continue |
| `RESUME_d256.json` | `tokens_seen`, `step`, `loss` |
| `fnn_d256.pt` | cerveau d=256, 2 blocs |
| `fnn_talk.pt` | d=128 archivé |

## Interdits

- Recréer / wipe le `.pt`
- `tokens_seen = 0` si RESUME.progress > 0
- Écraser `fnn_talk.pt` avec un d=256
- Rejouer une phase 1 par-dessus un offset > 0

`live_train.py` refuse d’écrire `tokens_seen=0` par-dessus un cerveau existant.

## État

d=256 slice 1 : step 800, 409 600 tok, loss 3.82. Prochain lancement reprend là.
