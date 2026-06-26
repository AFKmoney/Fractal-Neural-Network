#!/usr/bin/env python3
"""
NFN Training CLI — Back-Propagation Through Phase (BPTP)

Usage:
    python train.py --text data/corpus.txt --config nano --epochs 5
    python train.py --text data/corpus.txt --config small --epochs 10 --lr 2e-4
    python train.py --resume checkpoints/nfn_step500.pt --epochs 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from nfn.config import FNNConfig
from nfn.model import FNNModel
from nfn.tokenizer import NFNTokenizer
from training.trainer import NFNTrainer


def parse_args():
    p = argparse.ArgumentParser(description="NFN Training")
    p.add_argument("--text", type=str, default=None, help="Path to training text file")
    p.add_argument("--config", type=str, default="nano",
                   choices=["nano", "small", "medium"], help="Model size preset")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=None, help="Context length (default: config)")
    p.add_argument("--batch", type=int, default=2, help="Batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--output", type=str, default="checkpoints", help="Checkpoint directory")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")
    p.add_argument("--sample-every", type=int, default=200,
                   help="Sample text every N steps (0 = disable)")
    p.add_argument("--sample-prompt", type=str, default="Le Neural Fractal Network",
                   help="Prompt for text samples during training")
    return p.parse_args()


def select_device(pref: str) -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def build_model(args, tokenizer: NFNTokenizer) -> FNNModel:
    cfg_path = ROOT / "configs" / f"{args.config}.json"
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    cfg = FNNConfig(**cfg_dict)
    cfg.vocab_size = tokenizer.vocab_size
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.bos_token_id = tokenizer.bos_token_id
    cfg.eos_token_id = tokenizer.eos_token_id

    if args.seq_len:
        cfg.max_seq_len = args.seq_len

    return FNNModel(cfg)


def main():
    args = parse_args()
    device = select_device(args.device)
    print(f"\n{'='*60}")
    print(f"  Neural Fractal Network — BPTP Training")
    print(f"{'='*60}")
    print(f"  Config  : {args.config}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  LR      : {args.lr}")
    print(f"  FP16    : {args.fp16}")
    print(f"{'='*60}\n")

    tokenizer = NFNTokenizer()

    if args.resume:
        print(f"Resuming from {args.resume}…")
        ckpt = torch.load(args.resume, map_location=device)
        cfg = FNNConfig.from_dict(ckpt["cfg"])
        model = FNNModel(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
    else:
        model = build_model(args, tokenizer).to(device)

    print(f"Parameters : {model.param_summary()}")
    print(f"Vocab size : {model.cfg.vocab_size}")
    print(f"Seq length : {model.cfg.max_seq_len}")
    print(f"Motifs     : {', '.join(model.cfg.motifs)}")
    print(f"Levels     : {model.cfg.n_levels}")
    print()

    # Load training text
    if args.text:
        text_path = Path(args.text)
        if not text_path.exists():
            print(f"Error: text file not found: {args.text}")
            sys.exit(1)
        text = text_path.read_text(encoding="utf-8")
        print(f"Corpus: {len(text):,} characters from {args.text}")
    else:
        # Built-in demo text
        text = (
            "Le Neural Fractal Network est une architecture révolutionnaire.\n"
            "Il combine la topologie fractale et les connexions sinusoïdales paramétriques.\n"
            "Chaque nœud possède une phase θ et une fréquence Ω = ω₀/λˢ.\n"
            "La connexion sinusoïdale : Γ(t) = A·sin(ω·t + φ), où A, ω, φ sont appris.\n"
            "Le réseau superpose plusieurs motifs fractals : arbre binaire, Cantor.\n"
            "La rétropropagation à travers les phases (BPTP) entraîne tous les paramètres.\n"
            "La perte multi-objectif : L = L_tâche + λ_phase·L_phase + λ_freq·L_freq.\n"
            "Le système peut modéliser des dépendances à très longue portée.\n"
            "La synchronisation de phase implémente le liage temporel des symboles.\n"
        ) * 50
        print(f"Using built-in demo corpus ({len(text):,} chars). Pass --text for custom data.")

    # Sample callback
    sample_step = [0]
    def on_step(metrics):
        sample_step[0] = metrics["step"]
        if args.sample_every > 0 and metrics["step"] % args.sample_every == 0:
            model.eval()
            from inference.engine import NFNInferenceEngine
            engine = NFNInferenceEngine(model, tokenizer, device)
            sample = engine.generate(args.sample_prompt, max_new_tokens=80,
                                     temperature=0.8, top_k=40)
            print(f"\n── Sample (step {metrics['step']}) ──")
            print(args.sample_prompt + sample)
            print("─" * 40)
            model.train()

    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    trainer = NFNTrainer(
        model=model,
        tokenizer=tokenizer,
        cfg=model.cfg,
        lr=args.lr,
        output_dir=args.output,
        step_callback=on_step,
        dtype=dtype,
    )

    if args.resume and "optimizer_state" in (ckpt := torch.load(args.resume, map_location=device)):
        trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        trainer.step = ckpt.get("step", 0)

    seq_len = args.seq_len or model.cfg.max_seq_len

    print("Starting BPTP training…\n")
    trainer.train(
        text=text,
        n_epochs=args.epochs,
        seq_len=seq_len,
        batch_size=args.batch,
        log_every=10,
        save_every=500,
    )

    print(f"\n✅ Training complete. Checkpoint saved to {args.output}/")


if __name__ == "__main__":
    main()
