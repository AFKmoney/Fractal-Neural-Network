# FNN — Fractal Neural Network

Language-model block built from **fractal linear attention**, **Kuramoto phase**, and **phase-routed MoE**.
Not a Transformer encoder stack. Softmax self-attention is not the block.

This branch (`clean/quarantine`) is a cleanup of the May 2026 “saved” tree.
Claude/ChatGPT sessions piled UI, a second architecture (PRISM), AGI trainers, and unused modules on top of the FNN block. Those files are listed in [`_quarantine/MANIFEST.md`](_quarantine/MANIFEST.md). They are not deleted from git history (`main` is untouched).

## What the live block does

`nfn/block.py` forward:

1. `FractalLinearAttention` — Katharopoulos kernel, O(L·d²), no L×L matrix
2. `PhaseSoliton` — amplify coherent phase patterns
3. `PhaseRoutedMoE` — von Mises / phase routing, top-k experts

Optional (flags on `FNNConfig`, off on the mini run): causal DAG, self-model, working memory.

Core files: `nfn/block.py`, `nfn/moe.py`, `nfn/phase_ode.py`, `nfn/model.py`, `nfn/config.py`, `nfn/analytic_embed.py`, `nfn/gematria.py`.

`nfn/fractal.py` is a re-export of the three classes in `moe.py`.

## Quick start

```bash
pip install torch
git clone -b clean/quarantine https://github.com/AFKmoney/Fractal-Neural-Network.git
cd Fractal-Neural-Network
```

```python
import torch
from nfn.config import FNNConfig
from nfn.model import FNNModel
from nfn.tokenizer import NFNTokenizer

tok = NFNTokenizer()
cfg = FNNConfig(
    vocab_size=tok.vocab_size,
    d_model=64, n_blocks=2, n_heads=2,
    use_causal_graph=False,
    use_self_model=False,
    use_working_memory=False,
    use_ads_cft=False,
    use_mera=False,
    use_godel_loop=False,
    use_rg_flow=False,
)
model = FNNModel(cfg)
x = torch.randint(0, cfg.vocab_size, (1, 32))
logits, losses = model(x, targets=x)
print(logits.shape, losses["lm"].item())
```

Mini train (2026-09-11, CPU): 2 blocks, d=64, ~111k params, 40 AdamW steps on a short French corpus.
`loss_before=4.66` → `loss_after=2.75`. Smoke test, not a language model. Script: `examples/mini_train.py`.

## What this is not

- Not a trained checkpoint you can chat with
- Not Fractus CTE (separate project)
- Not PRISM (quarantined)
- Not “AGI”

## License

MIT
