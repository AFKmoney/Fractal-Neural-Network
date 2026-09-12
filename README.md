# FNN — Fractal Neural Network

Language-model block built from **fractal linear attention**, **Kuramoto phase**, and **phase-routed MoE**.
Not a Transformer encoder stack. Softmax self-attention is not the block.

This branch (`clean/quarantine`) is a cleanup of the May 2026 “saved” tree.
Claude/ChatGPT sessions piled UI, a second architecture (PRISM), AGI trainers, and unused modules on top of the FNN block. Those files are listed in [`_quarantine/MANIFEST.md`](_quarantine/MANIFEST.md). `main` is untouched.

## What the live block does

`nfn/block.py` forward:

1. `FractalLinearAttention` — Katharopoulos kernel, O(L·d²), no L×L matrix
2. `PhaseSoliton` — amplify coherent phase patterns
3. `PhaseRoutedMoE` — von Mises / phase routing, top-k experts

Optional (flags on `FNNConfig`, off on the mini run): causal DAG, self-model, working memory.

Core files: `nfn/block.py`, `nfn/moe.py`, `nfn/phase_ode.py`, `nfn/model.py`, `nfn/config.py`, `nfn/analytic_embed.py`, `nfn/gematria.py`.

## Mini run (2026-09-11, CPU)

| | |
|---|---|
| Params | 110 976 total / ~77 056 actifs (MoE 4→2) |
| Chinchilla 20× actifs | 1.54 M tokens |
| Tokens vus | ~0.40 M (**26 %** du 20×) |
| Loss | 4.67 → **0.12** |
| Generate | commence à recracher les phrases du corpus |

Détails : [`docs/CHINCHILLA.md`](docs/CHINCHILLA.md). Script : `examples/mini_train.py`.

```bash
git clone -b clean/quarantine https://github.com/AFKmoney/Fractal-Neural-Network.git
cd Fractal-Neural-Network
pip install torch
python examples/mini_train.py
```

## What this is not

- Not a trained checkpoint you can chat with
- Not Fractus CTE (separate project)
- Not PRISM (quarantined)
- Not “AGI”

## License

MIT
