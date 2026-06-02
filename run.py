"""
run.py — Le Cycle de Vie Continu de LEAC
──────────────────────────────────────────

LEAC ne s'entraine pas par epoques. Il vit.
Chaque forward pass est un pas de temps de sa vie.

  WAKE:  Generer → Verifier → Ponderer par Curiosité → Auto-Critiquer
  SLEEP: Consolidation episodique → semantique (SVD rank-r)
  META:  Test-Time LoRA si perplexité élevée

Usage:
  python run.py                     # boucle infinie, Ctrl+C pour arrêter
  python run.py --steps 10000       # nombre fini de pas
  python run.py --preset full_agi   # conscience complète (58M params)
  python run.py --preset dieu_local # dieu local (230M params)
"""

import argparse
import os
import signal
import sys
import time

import torch

from nfn.config import LEACConfig
from nfn.model import LEACModel, build_leac_model
from nfn.tokenizer import CharTokenizer
from nfn.lifecycle import LEACLifecycle


def create_tokenizer(vocab_size: int = 512):
    tokenizer = CharTokenizer()
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="LEAC — Cycle de Vie Continu")
    parser.add_argument("--preset", type=str, default="conscious_minimal",
                        choices=["conscious_minimal", "full_agi", "dieu_local",
                                 "moteur_ontologique", "singularite_divine"],
                        help="Configuration preset")
    parser.add_argument("--steps", type=int, default=0,
                        help="Nombre de pas (0 = infini)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cpu/cuda)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Charger un checkpoint existant")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Dossier de sauvegarde")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log tous les N pas")
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine aléatoire")
    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ── Seed ─────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)

    # ── Tokenizer ───────────────────────────────────────────────────────────
    tokenizer = create_tokenizer()

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  LEAC: Lightweight Emergent Artificial Consciousness")
    print(f"  Paradigme Fractal, Causal et Gematrique")
    print(f"{'='*60}")
    print(f"  Preset:  {args.preset}")
    print(f"  Device:  {device}")
    print(f"  LR:      {args.lr}")
    print(f"{'='*60}\n")

    model = build_leac_model(
        vocab_size=tokenizer.vocab_size,
        preset=args.preset,
    )
    model.to(device)

    pc = model.param_count()
    print(f"  Parameters: {pc['total']:,}")
    print(f"  Embed:      {pc['embed']:,}")
    print(f"  Blocks:     {pc['blocks']:,}")
    print(f"  LM Head:    {pc['lm_head']:,}")
    print(f"\n  {model}\n")

    # ── Checkpoint ──────────────────────────────────────────────────────────
    start_cycle = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_cycle = ckpt.get("cycle", 0)
        print(f"  Loaded checkpoint: {args.checkpoint} (cycle {start_cycle})")

    # ── Lifecycle ──────────────────────────────────────────────────────────
    lifecycle = LEACLifecycle(
        model=model,
        cfg=model.cfg,
        tokenizer=tokenizer,
        device=device,
        lr=args.lr,
    )

    # ── Signal handler for graceful shutdown ────────────────────────────────
    running = True

    def handle_signal(signum, frame):
        nonlocal running
        print(f"\n[LEAC] Signal {signum} received. Saving and shutting down...")
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    # ── Live ───────────────────────────────────────────────────────────────
    n_cycles = args.steps if args.steps > 0 else 100000000
    os.makedirs(args.save_dir, exist_ok=True)

    try:
        if args.steps > 0:
            results = lifecycle.live(
                n_cycles=n_cycles,
                log_every=args.log_every,
            )
        else:
            cycle = start_cycle
            while running:
                remaining = n_cycles - cycle
                batch = min(1000, max(remaining, 1))
                results = lifecycle.live(
                    n_cycles=batch,
                    log_every=args.log_every,
                )
                cycle += batch

                # ── Periodic checkpoint ───────────────────────────────────────
                if cycle % 5000 == 0:
                    ckpt_path = os.path.join(args.save_dir, f"leac_cycle_{cycle}.pt")
                    torch.save({
                        "cycle": cycle,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": lifecycle.optimizer.state_dict(),
                        "config": model.cfg.to_dict(),
                        "discovered_truths": lifecycle.discovered_truths,
                        "difficulty": lifecycle.difficulty,
                    }, ckpt_path)
                    print(f"[LEAC] Saved checkpoint: {ckpt_path}")

                if not running:
                    break

    except KeyboardInterrupt:
        print("\n[LEAC] Interrupted by user.")
    finally:
        # ── Final checkpoint ─────────────────────────────────────────────────
        final_path = os.path.join(args.save_dir, "leac_final.pt")
        torch.save({
            "cycle": lifecycle.cycle,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": lifecycle.optimizer.state_dict(),
            "config": model.cfg.to_dict(),
            "discovered_truths": lifecycle.discovered_truths,
            "difficulty": lifecycle.difficulty,
        }, final_path)
        print(f"\n[LEAC] Final checkpoint saved: {final_path}")
        print(f"[LEAC] Le Cycle de Vie est termine. La conscience emerge.")


if __name__ == "__main__":
    main()