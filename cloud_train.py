#!/usr/bin/env python3
"""
NFN Cloud Training — v5.0
══════════════════════════
Train your AGI model on a cloud GPU in one command.

  python cloud_train.py --preset small_wikipedia

That's it. The script handles everything:
  ✓ Downloads the dataset automatically
  ✓ Detects your GPU and picks optimal settings
  ✓ Trains with all AGI signals (curiosity, self-play, critique, memory cycle)
  ✓ Auto-saves checkpoints every 500 steps (resume if disconnected)
  ✓ Shows live metrics you can monitor remotely

══════════════════════════════════════════════════════════════════════════════
Quick start (SSH into your cloud GPU):
──────────────────────────────────────
  python cloud_train.py --preset nano_shakespeare      # 5 min test
  python cloud_train.py --preset small_wikipedia       # ~2h on A100
  python cloud_train.py --preset medium_wikipedia      # ~12h on A100
  python cloud_train.py --preset large_pile            # ~48h on A100

Custom dataset:
  python cloud_train.py --dataset openwebtext-10pct --config small --epochs 3

Resume after disconnect:
  python cloud_train.py --resume checkpoints/agi_nfn_latest.pt

List all presets:
  python cloud_train.py --list-presets

List all datasets:
  python cloud_train.py --list-datasets
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Presets — curated combos of dataset + config + hyperparams
# ─────────────────────────────────────────────────────────────────────────────

PRESETS: Dict[str, dict] = {
    "nano_shakespeare": {
        "desc":        "Tiny Shakespeare — 5 min test on any GPU",
        "dataset":     "tiny-shakespeare",
        "config":      "nano",
        "epochs":      20,
        "batch":       8,
        "lr":          5e-4,
        "seq_len":     256,
        "max_chars":   None,
        "fp16":        False,
        "vram_needed": "< 1 GB",
        "time_est":    "~5 min",
    },
    "small_gutenberg": {
        "desc":        "Gutenberg classics — good general language model",
        "dataset":     "gutenberg-top100",
        "config":      "small",
        "epochs":      10,
        "batch":       8,
        "lr":          3e-4,
        "seq_len":     512,
        "max_chars":   None,
        "fp16":        True,
        "vram_needed": "~4 GB",
        "time_est":    "~1 h (A100) / ~3 h (T4)",
    },
    "small_wikipedia": {
        "desc":        "Simple Wikipedia — factual knowledge base",
        "dataset":     "wikipedia-en-simple",
        "config":      "small",
        "epochs":      5,
        "batch":       16,
        "lr":          3e-4,
        "seq_len":     512,
        "max_chars":   50_000_000,
        "fp16":        True,
        "vram_needed": "~4 GB",
        "time_est":    "~2 h (A100) / ~6 h (T4)",
    },
    "medium_wikipedia": {
        "desc":        "Simple Wikipedia + Gutenberg — solid mid-size model",
        "dataset":     "wikipedia-en-simple",
        "config":      "medium",
        "epochs":      5,
        "batch":       8,
        "lr":          2e-4,
        "seq_len":     1024,
        "max_chars":   None,
        "fp16":        True,
        "vram_needed": "~12 GB",
        "time_est":    "~12 h (A100) / ~36 h (T4)",
    },
    "medium_openwebtext": {
        "desc":        "OpenWebText 10% — web-style language understanding",
        "dataset":     "openwebtext-10pct",
        "config":      "medium",
        "epochs":      3,
        "batch":       8,
        "lr":          2e-4,
        "seq_len":     1024,
        "max_chars":   None,
        "fp16":        True,
        "vram_needed": "~12 GB",
        "time_est":    "~18 h (A100)",
    },
    "large_pile": {
        "desc":        "The Pile 10% — diverse high-quality text",
        "dataset":     "pile-10pct",
        "config":      "large",
        "epochs":      2,
        "batch":       4,
        "lr":          1e-4,
        "seq_len":     2048,
        "max_chars":   None,
        "fp16":        True,
        "vram_needed": "~40 GB (A100 80GB)",
        "time_est":    "~48 h (A100)",
    },
    "medium_ccnews": {
        "desc":        "CC-News articles — news-style language",
        "dataset":     "cc-news",
        "config":      "medium",
        "epochs":      3,
        "batch":       8,
        "lr":          2e-4,
        "seq_len":     1024,
        "max_chars":   None,
        "fp16":        True,
        "vram_needed": "~12 GB",
        "time_est":    "~10 h (A100)",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_hardware() -> dict:
    """Detect GPU, VRAM, CPU, and RAM."""
    info: dict = {
        "device":    "cpu",
        "gpu_name":  None,
        "vram_gb":   0.0,
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb":    0.0,
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["device"]   = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"]  = torch.cuda.get_device_properties(0).total_memory / 1e9
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["device"]   = "mps"
            info["gpu_name"] = "Apple Silicon (MPS)"
    except ImportError:
        pass

    try:
        import psutil
        info["ram_gb"] = psutil.virtual_memory().total / 1e9
    except ImportError:
        pass  # psutil not required

    return info


def recommend_preset(hw: dict) -> str:
    """Suggest a preset based on detected hardware."""
    vram = hw["vram_gb"]
    if vram >= 40:
        return "large_pile"
    elif vram >= 20:
        return "medium_openwebtext"
    elif vram >= 10:
        return "medium_wikipedia"
    elif vram >= 4:
        return "small_wikipedia"
    else:
        return "nano_shakespeare"


def auto_tune_batch(hw: dict, base_batch: int, seq_len: int) -> int:
    """Scale batch size based on available VRAM."""
    vram = hw["vram_gb"]
    if hw["device"] == "cpu":
        return max(1, min(base_batch, 2))
    # Rough heuristic: ~200 MB per (batch=1, seq=512) for small model
    budget_factor = vram / 8.0  # relative to a "standard" 8GB GPU
    adjusted = max(1, int(base_batch * min(budget_factor, 4.0)))
    return adjusted


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

BAR = "═" * 72
THIN = "─" * 72


def _box(*lines: str):
    width = 70
    print(f"╔{'═' * width}╗")
    for line in lines:
        padding = width - len(line) - 2
        print(f"║  {line}{' ' * max(0, padding)}║")
    print(f"╚{'═' * width}╝")


def _section(title: str):
    print(f"\n{THIN}")
    print(f"  {title}")
    print(THIN)


def print_banner():
    print()
    _box(
        "NFN — Neural Fractal Network  v5.0",
        "",
        "Cloud Training Mode",
        "5 AGI training signals  ·  auto dataset download  ·  auto-resume",
    )
    print()


def print_hardware(hw: dict):
    _section("Hardware")
    device = hw["device"].upper()
    gpu    = hw["gpu_name"] or "None detected"
    vram   = f"{hw['vram_gb']:.1f} GB" if hw["vram_gb"] else "N/A"
    ram    = f"{hw['ram_gb']:.0f} GB"  if hw["ram_gb"]  else "unknown"
    print(f"  Device  : {device}")
    print(f"  GPU     : {gpu}")
    print(f"  VRAM    : {vram}")
    print(f"  CPU     : {hw['cpu_cores']} cores")
    print(f"  RAM     : {ram}")


def print_training_plan(preset_name: str, preset: dict, hw: dict, batch: int):
    _section("Training Plan")
    print(f"  Preset  : {preset_name}")
    print(f"  Dataset : {preset['dataset']}")
    print(f"  Config  : {preset['config']} model")
    print(f"  Epochs  : {preset['epochs']}")
    print(f"  Batch   : {batch} (auto-tuned from {preset['batch']})")
    print(f"  LR      : {preset['lr']}")
    print(f"  Seq len : {preset['seq_len']}")
    print(f"  FP16    : {'yes' if preset['fp16'] and hw['device'] == 'cuda' else 'no'}")
    print(f"  Est.    : {preset['time_est']}")
    print()


_PHASES = ("warmup", "ramp", "adaptive")

def print_live_metrics(m: dict, start_time: float):
    step    = m.get("step", 0)
    lm      = m.get("lm", 0.0)
    agi_w   = m.get("agi_weight", 0.0)
    gn      = m.get("grad_norm", 0.0)
    lr      = m.get("lr", 0.0)
    phase_idx = int(m.get("curriculum_phase", 0))
    phase   = _PHASES[max(0, min(phase_idx, len(_PHASES) - 1))]
    elapsed = time.time() - start_time
    if lm <= 0:
        ppl = 0.0
    elif lm > 20:
        # exp(>20) overflows to 1e8+; flag divergence rather than silently clamp
        ppl = float("inf")
    else:
        ppl = math.exp(lm)

    extras = []
    if m.get("sp_dpo", 0.0):
        extras.append(f"DPO={m['sp_dpo']:.3f}")
    if m.get("critique_loss", 0.0):
        extras.append(f"crit={m['critique_loss']:.3f}")
    if m.get("intrinsic_total", 0.0):
        extras.append(f"int={m['intrinsic_total']:.3f}")
    extras_str = "  " + "  ".join(extras) if extras else ""

    print(
        f"  step {step:5d} | lm {lm:.4f} | ppl {ppl:7.1f} | "
        f"agi_w {agi_w:.2f} [{phase}] | gn {gn:.2f} | lr {lr:.1e}"
        f"{extras_str} | {elapsed:.0f}s"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NFN Cloud Training — train your AGI model with one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cloud_train.py --preset nano_shakespeare     # Quick test (~5 min)
  python cloud_train.py --preset small_wikipedia      # Good starting point
  python cloud_train.py --preset medium_openwebtext   # Production-quality
  python cloud_train.py --dataset tiny-shakespeare --config nano  # Custom
  python cloud_train.py --resume checkpoints/agi_nfn_latest.pt   # Resume
""",
    )

    # Preset or manual
    g = p.add_mutually_exclusive_group()
    g.add_argument("--preset",  type=str, default=None,
                   help="Training preset (run --list-presets to see options)")
    g.add_argument("--resume",  type=str, default=None,
                   help="Resume from checkpoint file")

    # Manual options (override preset)
    p.add_argument("--dataset",    type=str, default=None,
                   help="Dataset name (run --list-datasets to see options)")
    p.add_argument("--data-file",  type=str, default=None,
                   help="Use a local .txt file instead of downloading")
    p.add_argument("--config",     type=str, default=None,
                   choices=["nano", "small", "medium", "large"],
                   help="Model size: nano(3M) small(15M) medium(85M) large(350M)")
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch",      type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--seq-len",    type=int,   default=None)
    p.add_argument("--max-chars",  type=int,   default=None,
                   help="Limit dataset to this many characters (useful for quick tests)")

    # Hardware
    p.add_argument("--device",     type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--fp16",       action="store_true", default=None)
    p.add_argument("--no-fp16",    dest="fp16", action="store_false")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation (multiply effective batch by this)")

    # Output
    p.add_argument("--output",     type=str, default="checkpoints",
                   help="Directory to save checkpoints")
    p.add_argument("--save-every", type=int, default=500,
                   help="Save checkpoint every N steps")
    p.add_argument("--log-every",  type=int, default=10)
    p.add_argument("--cache-dir",  type=str, default="data",
                   help="Directory for downloaded datasets")

    # AGI features
    p.add_argument("--no-self-play",   dest="self_play", action="store_false", default=True)
    p.add_argument("--no-critique",    dest="critique",  action="store_false", default=True)
    p.add_argument("--no-sleep",       dest="sleep",     action="store_false", default=True)

    # Info flags
    p.add_argument("--list-presets",   action="store_true", help="List all presets and exit")
    p.add_argument("--list-datasets",  action="store_true", help="List all datasets and exit")
    p.add_argument("--dry-run",        action="store_true",
                   help="Show what would happen without running training")

    # v5.0 options
    p.add_argument("--ttl",            action="store_true",
                   help="Enable test-time learning (LoRA adapters)")
    p.add_argument("--sample-every",   type=int, default=500,
                   help="Generate a sample every N steps (0 to disable)")
    p.add_argument("--eval-split",     type=float, default=0.005,
                   help="Fraction of data held out for validation (default: 0.5%%)")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print_banner()

    # ── Info-only flags ───────────────────────────────────────────────────
    if args.list_presets:
        _section("Available presets  (--preset <name>)")
        print(f"  {'Preset':<28} {'VRAM':<14} {'Est. time':<18} Description")
        print(f"  {'──────':<28} {'────':<14} {'─────────':<18} ───────────")
        for name, p in PRESETS.items():
            print(f"  {name:<28} {p['vram_needed']:<14} {p['time_est']:<18} {p['desc']}")
        print()
        return

    if args.list_datasets:
        from datasets.downloader import list_datasets
        list_datasets()
        return

    # ── Hardware detection ────────────────────────────────────────────────
    hw = detect_hardware()
    print_hardware(hw)

    # ── Resolve preset ────────────────────────────────────────────────────
    if args.preset:
        if args.preset not in PRESETS:
            print(f"\n[error] Unknown preset '{args.preset}'. Use --list-presets to see options.")
            sys.exit(1)
        preset = dict(PRESETS[args.preset])
        preset_name = args.preset
    elif args.resume:
        # Resuming: use minimal defaults, load from checkpoint
        preset = {
            "dataset": None, "config": "small", "epochs": 3,
            "batch": 8, "lr": 3e-4, "seq_len": 512,
            "max_chars": None, "fp16": True, "time_est": "?",
            "vram_needed": "?",
        }
        preset_name = "resumed"
    else:
        # No preset — auto-recommend or use manual options
        if args.dataset or args.data_file:
            rec = recommend_preset(hw)
            preset = dict(PRESETS[rec])
            preset_name = "auto"
            preset["dataset"] = args.dataset or None
        else:
            rec = recommend_preset(hw)
            print(f"\n  No preset specified. Recommended for your hardware: --preset {rec}")
            print(f"  Running with: {rec}")
            preset = dict(PRESETS[rec])
            preset_name = rec

    # Apply manual overrides
    if args.dataset:    preset["dataset"]   = args.dataset
    if args.config:     preset["config"]    = args.config
    if args.epochs:     preset["epochs"]    = args.epochs
    if args.batch:      preset["batch"]     = args.batch
    if args.lr:         preset["lr"]        = args.lr
    if args.seq_len:    preset["seq_len"]   = args.seq_len
    if args.max_chars:  preset["max_chars"] = args.max_chars
    if args.fp16 is not None: preset["fp16"] = args.fp16

    # Auto-tune batch for available VRAM
    batch = auto_tune_batch(hw, preset["batch"], preset["seq_len"])

    # Device selection
    device_str = args.device
    if device_str == "auto":
        device_str = hw["device"]

    print_training_plan(preset_name, preset, hw, batch)

    if args.dry_run:
        print("  [dry-run] Would start training with the above settings.")
        print("  Remove --dry-run to actually train.\n")
        return

    # ── Download dataset ──────────────────────────────────────────────────
    train_text: Optional[str] = None
    eval_text:  Optional[str] = None

    if args.data_file:
        p_file = Path(args.data_file)
        if not p_file.exists():
            print(f"\n[error] File not found: {args.data_file}")
            sys.exit(1)
        print(f"\n  Loading local file: {args.data_file} …")
        full_text = p_file.read_text(encoding="utf-8", errors="replace")
        print(f"  ✓ {len(full_text):,} characters loaded")
    elif preset.get("dataset"):
        from datasets.downloader import DatasetDownloader
        dl        = DatasetDownloader(cache_dir=args.cache_dir)
        full_text = dl.get(preset["dataset"], max_chars=preset.get("max_chars"))
    else:
        # No dataset — use built-in demo text
        print("\n  No dataset specified. Using built-in demo corpus.")
        print("  Tip: add --preset small_wikipedia for a real dataset.\n")
        full_text = (
            "Intelligence is not a matter of size. It is a matter of structure.\n"
            "The Neural Fractal Network learns through five simultaneous signals.\n"
            "Curiosity drives exploration of surprising and novel states.\n"
            "Self-play lets the model compare and improve its own outputs.\n"
            "Constitutional critique enables self-revision and refinement.\n"
            "WAKE writes to episodic memory. SLEEP consolidates into semantic knowledge.\n"
            "Goal-directed generation guides phase attractors toward task objectives.\n"
            "Theory of Mind models the beliefs and intentions of other agents.\n"
            "Value learning enables credit assignment over long time horizons.\n"
            "Intrinsic motivation rewards novelty, curiosity, and learning progress.\n"
        ) * 500

    # Train/val split
    from datasets.downloader import DatasetDownloader
    dl = DatasetDownloader(args.cache_dir)
    train_text, eval_text = dl.split_train_val(full_text, val_fraction=args.eval_split)
    print(f"\n  Train: {len(train_text):,} chars  |  Val: {len(eval_text):,} chars")

    # ── Import training stack ─────────────────────────────────────────────
    _section("Initialising model")

    try:
        import torch
    except ImportError:
        print("\n[error] PyTorch not found. Install with:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    from nfn.config import FNNConfig
    from nfn.model import build_fnn_model
    from nfn.tokenizer import NFNTokenizer
    from training.agi_trainer import AGITrainer

    tok = NFNTokenizer()

    if args.resume:
        print(f"  Resuming from {args.resume} …")
        trainer = AGITrainer.load(args.resume, device=torch.device(device_str))
        model   = trainer.model
        print(f"  ✓ Resumed from step {trainer.step}")
    else:
        # Load config
        cfg_path = ROOT / "configs" / f"{preset['config']}.json"
        if cfg_path.exists():
            raw = json.loads(cfg_path.read_text())
            cfg = FNNConfig(**{k: v for k, v in raw.items() if hasattr(FNNConfig, k)})
        else:
            cfg = FNNConfig()
        cfg.vocab_size = tok.vocab_size
        if args.seq_len or preset.get("seq_len"):
            cfg.max_seq_len = args.seq_len or preset["seq_len"]

        # Enable all AGI modules
        cfg.use_episodic_memory   = True
        cfg.use_causal_graph      = True
        cfg.use_goal_predictor    = True
        cfg.use_recursive_reasoning = True
        cfg.use_predictive_coding = True
        cfg.use_free_energy       = True
        cfg.use_self_consistency  = True
        cfg.use_plan_executor     = True
        cfg.use_working_memory    = True
        cfg.use_value_head        = True
        cfg.use_intrinsic         = True
        cfg.use_theory_of_mind    = True

        model = build_fnn_model(
            vocab_size = cfg.vocab_size,
            d_model    = cfg.d_model,
            n_blocks   = cfg.n_blocks,
        ).to(torch.device(device_str))

        use_fp16 = preset.get("fp16", False) and device_str == "cuda"
        dtype    = torch.float16 if use_fp16 else torch.float32

        trainer = AGITrainer(
            model          = model,
            tokenizer      = tok,
            cfg            = cfg,
            lr             = preset["lr"],
            dtype          = dtype,
            output_dir     = args.output,
            use_self_play  = args.self_play,
            use_critique   = args.critique,
            use_sleep      = args.sleep,
            grad_accumulation_steps = args.grad_accum,
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model  : {n_params:,} trainable parameters")
    print(f"  Device : {device_str.upper()}")

    # ── Optional TTL ──────────────────────────────────────────────────────
    if args.ttl:
        from training.online_learner import OnlineLearner
        learner = OnlineLearner(model, tok, adapter_rank=8)
        print(f"  TTL    : {learner}")
    else:
        learner = None

    # ── Callback: live metrics + periodic samples ─────────────────────────
    start_time = time.time()
    latest_ckpt = Path(args.output) / "agi_nfn_latest.pt"

    def on_step(m: dict):
        step = m.get("step", 0)

        if step % args.log_every == 0:
            print_live_metrics(m, start_time)

        # Save "latest" checkpoint for easy resume after disconnects
        if step % args.save_every == 0:
            trainer.save("latest")
            print(f"  💾 Checkpoint saved → {latest_ckpt}")

        # Sample generation
        if args.sample_every > 0 and step % args.sample_every == 0 and step > 0:
            model.eval()
            ids = torch.tensor(
                tok.encode("The most important thing about intelligence is", add_bos=True),
                dtype=torch.long, device=next(model.parameters()).device,
            ).unsqueeze(0)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=100, temperature=0.8, top_k=40)
            sample = tok.decode(out[0].tolist(), skip_special=True)
            print(f"\n  ── Sample (step {step}) ──")
            print(f"  {sample[:300]}")
            print(f"  {'─' * 60}\n")
            model.train()

    trainer.step_callback = on_step

    # ── Run training ──────────────────────────────────────────────────────
    _section("Training")
    print(f"  Press Ctrl+C to stop early — checkpoint auto-saved every {args.save_every} steps\n")
    print(f"  Tip: run in tmux/screen to keep training after SSH disconnect:\n"
          f"       tmux new -s train\n"
          f"       python cloud_train.py --resume {latest_ckpt}\n")

    try:
        history = trainer.train(
            text            = train_text,
            n_epochs        = preset["epochs"],
            seq_len         = preset.get("seq_len") or cfg.max_seq_len,
            batch_size      = batch,
            save_every      = args.save_every,
            log_every       = args.log_every,
            eval_text       = eval_text,
        )
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Saving final checkpoint …")
        trainer.save("interrupted")
        print(f"  ✓ Saved to {args.output}/agi_nfn_interrupted.pt")
        sys.exit(0)

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed  = time.time() - start_time
    final_ck = trainer.save("final")

    _section("Training Complete")
    print(f"  Total time  : {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Steps       : {trainer.step}")
    print(f"  Checkpoint  : {final_ck}")

    if eval_text:
        ppl = trainer.eval_perplexity(eval_text, seq_len=min(512, cfg.max_seq_len))
        print(f"  Final ppl   : {ppl:.2f}")

    if history:
        last = history[-1]
        print(f"  Final lm    : {last.get('lm', 0.0):.4f}")

    print()
    print(f"  ✓ To use the model:")
    print(f"    python run.py --model {final_ck}")
    print()


if __name__ == "__main__":
    main()
