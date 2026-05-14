# Cloud GPU Training Guide — NFN v5.0

Train your own AGI model on a cloud GPU. This guide covers everything from
renting a GPU to running your first training session.

---

## Quick Start (3 steps)

**Step 1 — Rent a cloud GPU**

| Provider | Cheapest GPU | Price | Good for |
|----------|-------------|-------|----------|
| [Lambda Labs](https://lambdalabs.com) | A10 (24 GB) | ~$0.60/h | `medium_wikipedia` |
| [RunPod](https://runpod.io) | RTX 4090 (24 GB) | ~$0.50/h | `medium_wikipedia` |
| [Vast.ai](https://vast.ai) | RTX 3090 (24 GB) | ~$0.20/h | `small_wikipedia` |
| [Paperspace](https://paperspace.com) | A100 (40 GB) | ~$1.10/h | `large_pile` |
| [Google Colab Pro](https://colab.research.google.com) | A100 (40 GB) | ~$10/month | All presets |

Choose Ubuntu 22.04 LTS or 20.04 LTS as the OS image.

**Step 2 — SSH in and run the setup script**

```bash
ssh user@your-instance-ip

# One-command setup + training (auto-detects GPU, picks best preset):
curl -fsSL https://raw.githubusercontent.com/AFKmoney/FNN/main/setup_cloud.sh | bash

# Or if you already cloned the repo:
bash setup_cloud.sh
```

**Step 3 — Watch training in real time**

```
══ Training ══
  Tip: run in tmux/screen to keep training after SSH disconnect

  step     1 | lm 6.8921 | ppl   987.3 | agi_w 0.00 [warmup] | gn 1.23 | lr 3.0e-04
  step    10 | lm 5.4231 | ppl   226.5 | agi_w 0.00 [warmup] | gn 0.87 | lr 3.0e-04
  step   100 | lm 4.2156 | ppl    67.8 | agi_w 0.12 [ramp]   | gn 0.65 | lr 2.8e-04
  💾 Checkpoint saved → checkpoints/agi_nfn_latest.pt
```

That's it.

---

## All Presets

```bash
python cloud_train.py --list-presets
```

| Preset | Dataset | Config | VRAM | Est. time |
|--------|---------|--------|------|-----------|
| `nano_shakespeare` | Shakespeare | nano | < 1 GB | ~5 min |
| `small_gutenberg` | Gutenberg top 100 | small | ~4 GB | ~1 h |
| `small_wikipedia` | Simple Wikipedia | small | ~4 GB | ~2 h |
| `medium_wikipedia` | Simple Wikipedia | medium | ~12 GB | ~12 h |
| `medium_openwebtext` | OpenWebText 10% | medium | ~12 GB | ~18 h |
| `medium_ccnews` | CC-News | medium | ~12 GB | ~10 h |
| `large_pile` | The Pile 10% | large | ~40 GB | ~48 h |

---

## Surviving SSH Disconnects

Cloud connections drop. Use **tmux** to keep training running:

```bash
# Install tmux (usually pre-installed)
sudo apt-get install -y tmux

# Start a detachable session
tmux new -s nfn

# Run training inside tmux
python cloud_train.py --preset small_wikipedia

# Detach from session (training keeps running in background)
# Press: Ctrl+B, then D

# === SSH disconnect, come back later ===

# Reconnect
ssh user@your-instance-ip

# Re-attach to training session
tmux attach -t nfn

# Or resume from checkpoint if session was lost
python cloud_train.py --resume checkpoints/agi_nfn_latest.pt
```

The script saves a `checkpoints/agi_nfn_latest.pt` checkpoint every 500 steps
automatically. If your session crashes, just resume:

```bash
python cloud_train.py --resume checkpoints/agi_nfn_latest.pt
```

---

## Datasets

All datasets download automatically on first run and are cached locally.

```bash
# List all available datasets
python cloud_train.py --list-datasets

# Use a specific dataset
python cloud_train.py --dataset wikipedia-en-simple --config small

# Use a local file
python cloud_train.py --data-file my_corpus.txt --config small

# Download only (no training)
python -m datasets.downloader wikipedia-en-simple
```

| Dataset | Description | Size | Command |
|---------|-------------|------|---------|
| `tiny-shakespeare` | Shakespeare's complete works | 1 MB | Quick tests |
| `gutenberg-top100` | Top 100 Gutenberg books | 20 MB | Literary style |
| `wikipedia-en-simple` | Simple English Wikipedia | 120 MB | General knowledge |
| `openwebtext-10pct` | 10% of OpenWebText | 2 GB | Web-style language |
| `cc-news` | CC-News sample | 1 GB | News articles |
| `wikipedia-en` | Full English Wikipedia | 20 GB | Large models |
| `pile-10pct` | The Pile 10% | 8 GB | Diverse high-quality |

For HuggingFace datasets (`openwebtext-10pct`, `cc-news`, `wikipedia-en`, `pile-10pct`),
the `datasets` library is required:

```bash
pip install datasets
```

---

## Custom Datasets

```bash
# Use any .txt file
python cloud_train.py --data-file path/to/corpus.txt --config small

# Mix datasets: download and concatenate manually
python -c "
from datasets.downloader import DatasetDownloader
dl = DatasetDownloader('data')
wiki = dl.get('wikipedia-en-simple')
guten = dl.get('gutenberg-top100')
combined = wiki + '\n\n' + guten
open('data/combined.txt', 'w').write(combined)
"
python cloud_train.py --data-file data/combined.txt --config medium
```

---

## Recommended Configs per GPU

| GPU | VRAM | Recommended preset |
|-----|------|--------------------|
| Tesla T4 | 16 GB | `small_wikipedia` with `--batch 8` |
| RTX 3090 / 4090 | 24 GB | `medium_wikipedia` |
| A10 | 24 GB | `medium_wikipedia` |
| A100 40 GB | 40 GB | `large_pile` or `medium_openwebtext` |
| A100 80 GB | 80 GB | `large_pile` with `--batch 16` |
| H100 | 80 GB | `large_pile` with `--batch 32` |

For multi-GPU, use `--grad-accum N` to simulate a larger batch:

```bash
# Simulate batch=64 with 8 GPUs × batch=8
python cloud_train.py --preset medium_wikipedia --batch 8 --grad-accum 8
```

---

## Advanced Options

```bash
python cloud_train.py \
    --preset medium_wikipedia \
    --epochs 10 \                      # override epochs
    --batch 16 \                       # override batch size
    --lr 1e-4 \                        # override learning rate
    --seq-len 2048 \                   # longer context
    --grad-accum 4 \                   # gradient accumulation
    --save-every 200 \                 # save more often
    --sample-every 1000 \              # periodic text generation
    --ttl \                            # enable test-time learning
    --eval-split 0.01                  # 1% validation set
```

### Disable specific AGI signals (faster, more stable early on)

```bash
# Just LM + curiosity (like a regular language model but with AGI architecture)
python cloud_train.py --preset small_wikipedia --no-self-play --no-critique

# All signals except sleep (saves memory, ~20% faster)
python cloud_train.py --preset small_wikipedia --no-sleep
```

---

## Monitoring

### Local monitoring

Training metrics are logged to `train_YYYYMMDD_HHMMSS.log` by default.

Metric legend:

```
step   10 | lm 4.21 | ppl  67.8 | agi_w 0.12 [ramp] | gn 0.65 | lr 2.8e-04  DPO=0.234
│          │          │           │                    │          │             │
│          │          │           │                    │          │             └ DPO-lite loss (self-play)
│          │          │           │                    │          └── Learning rate
│          │          │           │                    └── Gradient norm (watch for explosion > 5)
│          │          │           └── Curriculum phase: warmup → ramp → adaptive
│          │          └── AGI loss weight [0=LM only, 1=full AGI]
│          └── Per-token perplexity (exp(lm_loss))
└── Step number
```

**Healthy training signs:**
- LM loss decreasing over time
- Gradient norm stable (0.5–3.0)
- AGI weight reaching 1.0 after ~300 steps
- Perplexity dropping below 50 for real datasets

**Warning signs:**
- Gradient norm > 10 → reduce `--lr`
- Loss increasing after 1000 steps → reduce `--lr` or try `--no-self-play`
- OOM error → reduce `--batch` or `--seq-len`

### Remote monitoring via SSH tunnel

```bash
# Forward port from cloud server to your laptop
ssh -L 8000:localhost:8000 user@your-instance-ip

# Start the web interface on the server
python run.py --model checkpoints/agi_nfn_latest.pt

# Open in your laptop's browser
open http://localhost:8000
```

---

## Checkpoints

Checkpoints are saved to `checkpoints/` by default:

```
checkpoints/
├── agi_nfn_step500.pt     ← periodic checkpoint
├── agi_nfn_step1000.pt
├── agi_nfn_latest.pt      ← always the most recent (overwritten each save)
└── agi_nfn_final.pt       ← saved when training completes
```

To use a checkpoint:

```bash
# Web interface
python run.py --model checkpoints/agi_nfn_final.pt

# Continue training from checkpoint
python cloud_train.py --resume checkpoints/agi_nfn_final.pt

# Python API
import torch
from nfn.agi_model import AGINFNModel
ckpt  = torch.load("checkpoints/agi_nfn_final.pt")
model = AGINFNModel(...)
model.load_state_dict(ckpt["model_state"])
```

---

## Cost Estimates

| Preset | GPU | Hours | Approx. cost |
|--------|-----|-------|--------------|
| `nano_shakespeare` | Any | 0.1 h | $0.06 |
| `small_wikipedia` | T4 | 6 h | $0.50 |
| `small_wikipedia` | A100 | 2 h | $2.20 |
| `medium_wikipedia` | A100 | 12 h | $13.20 |
| `large_pile` | A100 80GB | 48 h | $52.80 |

Tips to reduce cost:
- Use spot instances (Vast.ai, RunPod) for non-critical runs (~50% cheaper)
- Start with `nano` to verify your setup before running large jobs
- Use `--max-chars` to limit dataset size for budget testing:
  ```bash
  python cloud_train.py --preset medium_wikipedia --max-chars 10000000  # 10 MB only
  ```

---

## Troubleshooting

**CUDA out of memory**
```bash
# Reduce batch and/or sequence length
python cloud_train.py --preset medium_wikipedia --batch 4 --seq-len 512
```

**Training loss not decreasing**
```bash
# Reduce learning rate
python cloud_train.py --preset small_wikipedia --lr 1e-4

# Or disable AGI signals and just do LM first
python cloud_train.py --preset small_wikipedia --no-self-play --no-critique --no-sleep
```

**SSH disconnected — training was in progress**
```bash
# Resume from latest checkpoint
python cloud_train.py --resume checkpoints/agi_nfn_latest.pt

# Or use tmux next time to avoid this
bash setup_cloud.sh --tmux
```

**Dataset download fails**
```bash
# Try downloading manually and using local file
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python cloud_train.py --data-file input.txt --config small
```

**`ModuleNotFoundError: No module named 'nfn'`**
```bash
pip install -e .    # from inside the FNN directory
```
