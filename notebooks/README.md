# NFN — Free GPU Training Notebooks

Train your own NFN model for **$0** using free cloud GPU tiers.

| Platform | GPU | VRAM | Free quota | Notebook | Best for |
|----------|-----|------|-----------|----------|----------|
| **Kaggle** | T4 / P100 | 16 GB | 30h/week | [`kaggle_train.ipynb`](kaggle_train.ipynb) | Longest free runs |
| **Google Colab** | T4 | 15 GB | ~12h/session | [`colab_train.ipynb`](colab_train.ipynb) | Easiest setup |
| **Lightning.ai** | T4 | 16 GB | 22h/month | [`lightning_train.ipynb`](lightning_train.ipynb) | Persistent disk |

---

## Quick start — Kaggle (recommended, most free GPU time)

1. Go to [kaggle.com](https://www.kaggle.com) → Sign in
2. `+ New Notebook` → upload `kaggle_train.ipynb`
3. `Settings → Accelerator → GPU T4 x1`
4. `Run All`

Checkpoints download automatically when session ends.

---

## Quick start — Google Colab

Click the badge to open directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AFKmoney/FNN/blob/main/notebooks/colab_train.ipynb)

Or:
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. `File → Open notebook → GitHub` → paste `AFKmoney/FNN`
3. Select `notebooks/colab_train.ipynb`
4. `Runtime → Change runtime type → T4 GPU`
5. `Runtime → Run all`

Checkpoints are saved to your Google Drive automatically.

---

## Quick start — Lightning.ai

1. Go to [lightning.ai](https://lightning.ai) → Sign up free
2. `New Studio → Jupyter Notebook`
3. Upload `lightning_train.ipynb`
4. Enable GPU (top-right selector)
5. Run all cells

**Best option if you want persistence** — your files and checkpoints survive between sessions.

---

## What gets trained

| Notebook | Config | Params | Dataset | Time (T4) |
|----------|--------|--------|---------|-----------|
| Kaggle | medium | ~85M | Simple Wikipedia 50M chars | ~12h |
| Colab | small | ~15M | Simple Wikipedia 30M chars | ~2h |
| Lightning | medium | ~85M | Simple Wikipedia 80M chars | ~12h |

All presets use `--fp16` for memory efficiency and save checkpoints every 500 steps.

---

## Resuming after disconnect

All notebooks auto-detect existing checkpoints:

```python
# Just re-run the training cell — it resumes automatically
if os.path.exists('checkpoints/agi_nfn_latest.pt'):
    # resumes from checkpoint
```

For Colab: checkpoints are on Google Drive — they survive session resets.
For Kaggle: download checkpoints from the output tab before session ends.
For Lightning: checkpoints persist on `/teamspace/studios/` disk automatically.

---

## After training

Once training is done, test the model:

```python
from nfn.agi_model import build_agi_model
from nfn.tokenizer import NFNTokenizer
import torch

tok   = NFNTokenizer()
ckpt  = torch.load('checkpoints/agi_nfn_final.pt', map_location='cpu')
model = build_agi_model(vocab_size=tok.vocab_size, d_model=512, n_blocks=8)
model.load_state_dict(ckpt['model_state'])
model.eval()

ids = tok.encode("The future of intelligence is", add_bos=True)
out = model.generate(ids, max_new_tokens=150, temperature=0.8)
print(tok.decode(out[0].tolist()))
```
