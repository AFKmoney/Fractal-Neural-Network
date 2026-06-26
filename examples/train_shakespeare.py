#!/usr/bin/env python3
"""
Example: train a nano NFN on the Shakespeare mini-corpus.

Download corpus with:
    curl -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Then run:
    python examples/train_shakespeare.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from nfn.config import FNNConfig
from nfn.model import FNNModel
from nfn.tokenizer import NFNTokenizer
from inference.engine import NFNInferenceEngine
from training.trainer import NFNTrainer

# ── Config ────────────────────────────────────────────────────────────────────
cfg = FNNConfig(
    d_model=128,
    d_ff=512,
    n_levels=3,
    branching=2,
    motifs=["binary_tree", "cantor"],
    n_blocks=2,
    n_heads=4,
    max_seq_len=256,
    dropout=0.1,
)

# ── Data ──────────────────────────────────────────────────────────────────────
data_path = ROOT / "data" / "shakespeare.txt"
if data_path.exists():
    text = data_path.read_text(encoding="utf-8")
    print(f"Loaded {len(text):,} chars from {data_path}")
else:
    # Minimal built-in excerpt for demonstration
    text = (
        "ROMEO: But, soft! What light through yonder window breaks?\n"
        "It is the east, and Juliet is the sun.\n"
        "Arise, fair sun, and kill the envious moon,\n"
        "Who is already sick and pale with grief,\n"
        "That thou, her maid, art far more fair than she.\n"
        "JULIET: O Romeo, Romeo! wherefore art thou Romeo?\n"
        "Deny thy father and refuse thy name.\n"
        "Or, if thou wilt not, be but sworn my love,\n"
        "And I'll no longer be a Capulet.\n"
    ) * 200
    print(f"Using built-in excerpt ({len(text):,} chars). Download shakespeare.txt for better results.")

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = NFNTokenizer()
cfg.vocab_size = tokenizer.vocab_size
cfg.pad_token_id = tokenizer.pad_token_id
cfg.bos_token_id = tokenizer.bos_token_id
cfg.eos_token_id = tokenizer.eos_token_id

model = FNNModel(cfg).to(device)
print(f"NFN | {model.param_summary()} params | device={device}")

# ── Train ─────────────────────────────────────────────────────────────────────
engine = NFNInferenceEngine(model, tokenizer, device)

def sample_callback(metrics):
    if metrics["step"] % 100 == 0:
        sample = engine.generate("ROMEO:", max_new_tokens=60, temperature=0.8)
        print(f"\n── step {metrics['step']} | loss={metrics['loss']:.4f} ──")
        print("ROMEO:" + sample)
        print()

trainer = NFNTrainer(
    model, tokenizer, cfg,
    lr=3e-4,
    output_dir=str(ROOT / "checkpoints"),
    step_callback=sample_callback,
)

trainer.train(text, n_epochs=5, seq_len=128, batch_size=4, log_every=20)

# ── Final sample ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL SAMPLES")
print("="*60)
for prompt in ["ROMEO:", "JULIET:", "To be or not to be"]:
    print(f"\n[{prompt}]")
    print(prompt + engine.generate(prompt, max_new_tokens=100, temperature=0.7))
