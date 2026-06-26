#!/usr/bin/env python3
"""
NFN Quickstart — crée un modèle nano, l'entraîne 50 steps et génère du texte.
Tourne en moins d'une minute sur CPU.

    python examples/quickstart.py
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

CORPUS = (
    "Le Neural Fractal Network (NFN) est une architecture neuro-inspirée.\n"
    "Il utilise une topologie fractale récursive et des connexions sinusoïdales.\n"
    "Chaque connexion : Γ(t) = A·exp(-γt)·sin(ω·t + φ) avec A,ω,φ appris.\n"
    "Les nœuds oscillent et se synchronisent pour lier les représentations.\n"
    "L'entraînement BPTP minimise L_tâche + L_phase + L_fréquence.\n"
    "Le réseau supporte plusieurs motifs : arbre binaire, Cantor, Sierpinski.\n"
    "La topologie auto-similaire encode l'invariance d'échelle.\n"
    "Les attracteurs de phase implémentent une mémoire associative.\n"
    "NFN est la première architecture unisant fractalité et oscillations apprenables.\n"
) * 30

# Tiny model for quick demo
cfg = FNNConfig(
    d_model=64,
    d_ff=256,
    n_levels=2,
    branching=2,
    motifs=["binary_tree"],
    n_blocks=2,
    n_heads=2,
    max_seq_len=128,
    dropout=0.0,
)

device = torch.device("cpu")
tokenizer = NFNTokenizer()
cfg.vocab_size = tokenizer.vocab_size
cfg.pad_token_id = tokenizer.pad_token_id
cfg.bos_token_id = tokenizer.bos_token_id
cfg.eos_token_id = tokenizer.eos_token_id

model = FNNModel(cfg).to(device)
engine = NFNInferenceEngine(model, tokenizer, device=device)

print(f"NFN nano | {sum(p.numel() for p in model.parameters()):,} params")
print(f"Vocab: {cfg.vocab_size} tokens (char-level)")
print()

# Sample before training
print("── Avant entraînement ──")
before = engine.generate("Le NFN", max_new_tokens=50, temperature=1.0)
print("Le NFN" + before)
print()

# Train
trainer = NFNTrainer(model, tokenizer, cfg, lr=5e-3, output_dir="/tmp/nfn_qs")
trainer.train(CORPUS, n_epochs=2, seq_len=64, batch_size=4, log_every=25)

# Sample after training
print()
print("── Après entraînement ──")
for prompt in ["Le NFN", "Les nœuds", "L'entraînement"]:
    out = engine.generate(prompt, max_new_tokens=60, temperature=0.7, top_k=20)
    print(f"[{prompt}] {prompt}{out}")
    print()

print("✅ Quickstart terminé!")
