#!/usr/bin/env python3
"""
NFN AGI Training — v5.0

Trains AGINFNModel beyond next-token prediction with five concurrent signals:

  1. Language modelling       — standard cross-entropy with curiosity weighting
  2. Multi-objective AGI loss — causal / goal / coherence / ponder / pred /
                                free_energy / consistency
  3. Self-play DPO-lite       — generate N candidates, rank by LM loss,
                                train DPO preference + winner distillation
  4. Constitutional critique  — generate → critique → revise → train on revision
  5. WAKE/SLEEP memory cycle  — episodic writes every step, consolidation +
                                replay every --sleep-every steps

Optional: --ttl (test-time learning)
    Wraps the model with LoRA fast-weight adapters. During each sampling
    callback the adapter updates on the batch context — the model adapts to
    its own training distribution in real time without touching main weights.

Usage:
    python train_agi.py --text data/corpus.txt --config nano --epochs 5
    python train_agi.py --text data/corpus.txt --config medium --lr 1e-4
    python train_agi.py --resume checkpoints/agi_nfn_step500.pt
    python train_agi.py --text data/corpus.txt --no-self-play --no-critique
    python train_agi.py --text data/corpus.txt --ttl --adapter-rank 8
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel, build_agi_model
from nfn.tokenizer import NFNTokenizer, load_tokenizer
from nfn.online_learner import OnlineLearner
from training.agi_trainer import AGITrainer


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NFN AGI Training — trains AGINFNModel beyond next-token prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data / model
    p.add_argument("--text",    type=str, default=None,
                   help="Path to training text file")
    p.add_argument("--config",  type=str, default="nano",
                   choices=["nano", "small", "medium", "large"],
                   help="Model size preset (loads from configs/<name>.json)")
    p.add_argument("--resume",  type=str, default=None,
                   help="Resume from checkpoint path (sets model + optimizer state)")

    # Training loop
    p.add_argument("--epochs",  type=int,   default=3,    help="Number of epochs")
    p.add_argument("--batch",   type=int,   default=4,    help="Batch size")
    p.add_argument("--lr",      type=float, default=3e-4, help="Learning rate")
    p.add_argument("--seq-len", type=int,   default=None,
                   help="Sequence length (default: from config)")
    p.add_argument("--output",  type=str,   default="checkpoints",
                   help="Directory to save checkpoints")
    p.add_argument("--save-every",  type=int, default=500,
                   help="Save checkpoint every N steps")
    p.add_argument("--log-every",   type=int, default=10,
                   help="Log metrics every N steps")
    p.add_argument("--warmup-steps",type=int, default=100,
                   help="LR warmup steps")
    p.add_argument("--grad-accum",  type=int, default=1,
                   help="Gradient accumulation steps")

    # Hardware
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--fp16",   action="store_true",
                   help="Use fp16 mixed precision (CUDA only)")

    # AGI feature flags
    p.add_argument("--no-self-play", dest="self_play", action="store_false",
                   help="Disable self-play + DPO-lite loop")
    p.add_argument("--no-critique",  dest="critique",  action="store_false",
                   help="Disable constitutional self-critique")
    p.add_argument("--no-sleep",     dest="sleep",     action="store_false",
                   help="Disable WAKE/SLEEP memory consolidation cycle")
    p.add_argument("--no-curiosity", dest="curiosity", action="store_false",
                   help="Disable curiosity-driven loss weighting")
    p.set_defaults(self_play=True, critique=True, sleep=True, curiosity=True)

    # Self-play tuning
    p.add_argument("--self-play-every", type=int,   default=50)
    p.add_argument("--n-candidates",    type=int,   default=4)
    p.add_argument("--dpo-beta",        type=float, default=0.1)
    p.add_argument("--dpo-weight",      type=float, default=0.3)

    # Constitutional critique tuning
    p.add_argument("--critique-every",  type=int,   default=100)
    p.add_argument("--critique-weight", type=float, default=2.0)

    # SLEEP tuning
    p.add_argument("--sleep-every",        type=int, default=200)
    p.add_argument("--sleep-replay-steps", type=int, default=10)

    # Curiosity tuning
    p.add_argument("--curiosity-tau",    type=float, default=1.0)
    p.add_argument("--curiosity-weight", type=float, default=0.3)

    # Curriculum tuning
    p.add_argument("--agi-start",  type=int, default=200,
                   help="Step at which AGI losses begin ramping in")
    p.add_argument("--agi-ramp",   type=int, default=100,
                   help="Steps over which AGI losses ramp from 0 to 1")

    # Sampling / eval
    p.add_argument("--sample-every",  type=int, default=200,
                   help="Generate a sample text every N steps (0 = disable)")
    p.add_argument("--sample-prompt", type=str,
                   default="The neural fractal network",
                   help="Prompt for text samples during training")
    p.add_argument("--eval-text",     type=str, default=None,
                   help="Path to held-out text for perplexity evaluation")
    p.add_argument("--eval-every",    type=int, default=0,
                   help="Evaluate held-out perplexity every N steps (0 = end only)")

    # Test-time learning (LoRA fast-weight adapters)
    p.add_argument("--ttl",            action="store_true",
                   help="Enable test-time learning: LoRA adapters update at inference "
                        "time without modifying base weights")
    p.add_argument("--adapter-rank",   type=int,   default=8,
                   help="LoRA adapter rank for test-time learning")
    p.add_argument("--online-lr",      type=float, default=2e-4,
                   help="Learning rate for test-time adapter updates")
    p.add_argument("--online-steps",   type=int,   default=4,
                   help="Gradient steps per test-time adapt() call")
    p.add_argument("--ppl-gate",       type=float, default=30.0,
                   help="Skip adapter update when context ppl < this (model already knows it)")
    p.add_argument("--adapter-decay",  type=float, default=0.97,
                   help="Exponential decay applied to adapters after each update "
                        "(1.0 = no forgetting, 0.9 = fast forgetting)")
    p.add_argument("--save-adapters",  type=str,   default=None,
                   help="Save LoRA adapter weights to this path at end of training")
    p.add_argument("--load-adapters",  type=str,   default=None,
                   help="Load LoRA adapter weights from this path at start")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────

def select_device(pref: str) -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


# ─────────────────────────────────────────────────────────────────────────────
# Model / tokenizer construction
# ─────────────────────────────────────────────────────────────────────────────

def build_model_from_config(
    config_name: str,
    tokenizer:   NFNTokenizer,
    seq_len:     int = None,
) -> AGINFNModel:
    """Build an AGINFNModel from a named config preset."""
    cfg_path = ROOT / "configs" / f"{config_name}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}\n"
            f"Available configs: {[p.stem for p in (ROOT/'configs').glob('*.json')]}"
        )
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    # Enable all AGI modules for the AGI trainer
    cfg_dict.update(
        use_episodic_memory     = True,
        use_working_memory      = True,
        use_causal_graph        = True,
        use_goal_predictor      = True,
        use_recursive_reasoning = True,
        use_predictive_coding   = True,
        use_free_energy         = True,
        use_self_consistency    = True,
        use_plan_executor       = True,
        use_bayesian_decoder    = False,   # BayesianDecoder adds overhead; off by default
        use_mixture_of_depths   = True,
        use_multi_token_pred    = True,
        use_hyper_net           = True,
        use_ssm                 = False,   # opt-in: high memory cost
    )

    cfg = NFNConfig(**{k: v for k, v in cfg_dict.items()
                       if k in NFNConfig.__dataclass_fields__})
    cfg.vocab_size    = tokenizer.vocab_size
    cfg.pad_token_id  = tokenizer.pad_token_id
    cfg.bos_token_id  = tokenizer.bos_token_id
    cfg.eos_token_id  = tokenizer.eos_token_id

    if seq_len is not None:
        cfg.max_seq_len = seq_len

    return AGINFNModel(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Startup info
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_millions(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def print_startup_info(
    model:   AGINFNModel,
    args:    argparse.Namespace,
    device:  torch.device,
):
    """Print model summary, training config, and estimated compute."""
    cfg = model.cfg
    pc  = model.param_count()

    sep = "=" * 65
    print(f"\n{sep}")
    print("  NFN AGI Training System v5.0")
    print(sep)

    # ── Architecture summary ──────────────────────────────────────────────
    print("\n  Model Architecture")
    print(f"    d_model   : {cfg.d_model}")
    print(f"    n_blocks  : {cfg.n_blocks}")
    print(f"    vocab     : {cfg.vocab_size:,}")
    print(f"    seq_len   : {cfg.max_seq_len}")
    print(f"    params    : {_fmt_millions(pc['total'])} "
          f"({pc['total']:,} total)")
    print(f"      embed   : {_fmt_millions(pc['embed'])}")
    print(f"      blocks  : {_fmt_millions(pc['blocks'])}")
    print(f"      head    : {_fmt_millions(pc['lm_head'])}")

    # ── Active modules ────────────────────────────────────────────────────
    modules = []
    if cfg.use_episodic_memory:     modules.append("episodic+semantic mem")
    if cfg.use_working_memory:      modules.append(f"working mem ({cfg.wm_n_slots} slots)")
    if cfg.use_causal_graph:        modules.append("causal DAG")
    if cfg.use_goal_predictor:      modules.append("goal forcing")
    if cfg.use_recursive_reasoning: modules.append(f"ACT (max {cfg.reasoning_max_steps} steps)")
    if cfg.use_predictive_coding:   modules.append("predictive coding")
    if cfg.use_free_energy:         modules.append("free energy")
    if cfg.use_self_consistency:    modules.append("self-consistency")
    if cfg.use_plan_executor:       modules.append(f"planner ({cfg.plan_n_subgoals} subgoals)")
    if cfg.use_mixture_of_depths:   modules.append(f"MoD ({cfg.mod_capacity_factor:.0%} tokens)")
    if cfg.use_multi_token_pred:    modules.append(f"MTP (N={cfg.mtp_n_heads})")
    if cfg.use_hyper_net:           modules.append(f"hyper-net (r={cfg.hyper_rank})")
    if cfg.use_ssm:                 modules.append(f"SSM (N={cfg.ssm_d_state})")
    print(f"\n  Active AGI modules:")
    for m in modules:
        print(f"    + {m}")

    # ── Training config ───────────────────────────────────────────────────
    print(f"\n  Training Config")
    print(f"    device       : {device}")
    print(f"    epochs       : {args.epochs}")
    print(f"    batch        : {args.batch}")
    print(f"    lr           : {args.lr}")
    print(f"    fp16         : {args.fp16 and device.type == 'cuda'}")
    print(f"    grad_accum   : {args.grad_accum}")

    # ── Curriculum ────────────────────────────────────────────────────────
    print(f"\n  Curriculum Schedule")
    print(f"    LM-only phase: steps 0 → {args.agi_start}")
    print(f"    AGI ramp     : steps {args.agi_start} → {args.agi_start + args.agi_ramp}")
    print(f"    Full AGI     : steps > {args.agi_start + args.agi_ramp}")
    losses = ["lm", "causal", "goal", "coherence", "ponder", "pred",
              "free_energy", "consistency"]
    print(f"    Loss signals : {', '.join(losses)}")

    # ── AGI features ──────────────────────────────────────────────────────
    print(f"\n  AGI Training Features")
    print(f"    self-play    : {'ON' if args.self_play else 'OFF'}"
          + (f"  (every {args.self_play_every} steps,"
             f" {args.n_candidates} candidates,"
             f" beta={args.dpo_beta})" if args.self_play else ""))
    print(f"    critique     : {'ON' if args.critique else 'OFF'}"
          + (f"  (every {args.critique_every} steps,"
             f" weight={args.critique_weight}x)" if args.critique else ""))
    print(f"    wake/sleep   : {'ON' if args.sleep else 'OFF'}"
          + (f"  (sleep every {args.sleep_every} steps,"
             f" {args.sleep_replay_steps} replay steps)" if args.sleep else ""))
    print(f"    curiosity    : {'ON' if args.curiosity else 'OFF'}"
          + (f"  (tau={args.curiosity_tau},"
             f" weight={args.curiosity_weight})" if args.curiosity else ""))

    # ── FLOPs estimate ────────────────────────────────────────────────────
    # Rough estimate: 6 * n_params * seq_len per forward pass
    seq = cfg.max_seq_len
    flops_per_step = 6 * pc["total"] * seq
    print(f"\n  Estimated FLOPs/step : ~{_fmt_millions(flops_per_step)} "
          f"(6 * {_fmt_millions(pc['total'])} params * {seq} tokens)")
    if args.self_play:
        sp_overhead = 100 / args.self_play_every * args.n_candidates
        print(f"  Self-play overhead   : ~{sp_overhead:.0f}% "
              f"({args.n_candidates} extra passes every {args.self_play_every} steps)")

    # ── Test-time learning ────────────────────────────────────────────────
    if getattr(args, "ttl", False):
        print(f"\n  Test-Time Learning (LoRA fast weights)")
        print(f"    adapter rank : {args.adapter_rank}")
        print(f"    online lr    : {args.online_lr}")
        print(f"    steps/call   : {args.online_steps}")
        print(f"    ppl gate     : < {args.ppl_gate} → skip (already known)")
        print(f"    decay/call   : ×{args.adapter_decay}")
        print(f"    (base weights remain frozen; adapters update at each sample)")

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = select_device(args.device)
    dtype  = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    tokenizer = NFNTokenizer()

    # ── Build or resume model ─────────────────────────────────────────────
    if args.resume:
        print(f"Resuming from {args.resume} …")
        trainer = AGITrainer.load(args.resume, device=device)
        model   = trainer.model
        print(f"  Resumed from step {trainer.step}")
    else:
        if args.text is None:
            print("Warning: no --text provided. Using built-in demo corpus.")
        model   = build_model_from_config(args.config, tokenizer, seq_len=args.seq_len)
        model   = model.to(device)
        trainer = AGITrainer(
            model                = model,
            tokenizer            = tokenizer,
            cfg                  = model.cfg,
            lr                   = args.lr,
            dtype                = dtype,
            output_dir           = args.output,
            grad_accumulation_steps = args.grad_accum,
            # Curriculum
            agi_loss_start_step  = args.agi_start,
            agi_loss_ramp_steps  = args.agi_ramp,
            # Self-play
            use_self_play        = args.self_play,
            self_play_every      = args.self_play_every,
            n_candidates         = args.n_candidates,
            dpo_beta             = args.dpo_beta,
            dpo_loss_weight      = args.dpo_weight,
            # Constitutional critique
            use_critique         = args.critique,
            critique_every       = args.critique_every,
            critique_weight      = args.critique_weight,
            # SLEEP
            use_sleep            = args.sleep,
            sleep_every          = args.sleep_every,
            sleep_replay_steps   = args.sleep_replay_steps,
            # Curiosity
            use_curiosity        = args.curiosity,
            curiosity_tau        = args.curiosity_tau,
            curiosity_weight     = args.curiosity_weight,
        )

    # ── Startup info ──────────────────────────────────────────────────────
    print_startup_info(model, args, device)

    # ── Load training text ────────────────────────────────────────────────
    if args.text:
        text_path = Path(args.text)
        if not text_path.exists():
            print(f"Error: text file not found: {args.text}")
            sys.exit(1)
        text = text_path.read_text(encoding="utf-8")
        print(f"Corpus: {len(text):,} characters from {args.text}")
    else:
        text = (
            "The Neural Fractal Network is a new architecture.\n"
            "It combines fractal topology with parametric sinusoidal connections.\n"
            "Each node has a phase θ and frequency Ω learned through back-propagation.\n"
            "The sinusoidal connection: Γ(t) = A·sin(ω·t + φ) where A, ω, φ are learned.\n"
            "The system achieves long-range dependency modelling via phase synchronisation.\n"
            "Episodic memory consolidates into semantic knowledge over time.\n"
            "Goal-directed learning guides phase attractors toward task objectives.\n"
            "Self-play allows the model to improve by comparing its own outputs.\n"
            "Constitutional critique enables self-revision of generated text.\n"
        ) * 100
        print(f"Using built-in demo corpus ({len(text):,} chars). Use --text for real data.")

    eval_text: str = None
    if args.eval_text:
        eval_path = Path(args.eval_text)
        if eval_path.exists():
            eval_text = eval_path.read_text(encoding="utf-8")
            print(f"Eval corpus: {len(eval_text):,} characters from {args.eval_text}")
        else:
            print(f"Warning: eval text not found at {args.eval_text}")

    # ── Test-time learner (optional) ──────────────────────────────────────
    online_learner: Optional[OnlineLearner] = None
    if args.ttl:
        online_learner = OnlineLearner(
            model            = model,
            tokenizer        = tokenizer,
            adapter_rank     = args.adapter_rank,
            online_lr        = args.online_lr,
            n_steps          = args.online_steps,
            decay_factor     = args.adapter_decay,
            ppl_gate         = args.ppl_gate,
            max_adapt_tokens = min(256, model.cfg.max_seq_len),
        )
        if args.load_adapters:
            lpath = Path(args.load_adapters)
            if lpath.exists():
                online_learner.load_adapters(str(lpath))
                print(f"Loaded adapters from {lpath}")
            else:
                print(f"Warning: --load-adapters path not found: {lpath}")
        print(f"Test-time learning: {online_learner}")

    # ── Sampling callback ─────────────────────────────────────────────────
    def on_step(metrics: dict):
        step = metrics["step"]

        # Periodic held-out perplexity
        if eval_text and args.eval_every > 0 and step % args.eval_every == 0:
            ppl = trainer.eval_perplexity(eval_text,
                                          seq_len=min(512, model.cfg.max_seq_len))
            print(f"  [eval ppl @ step {step}]: {ppl:.2f}")

        if args.sample_every > 0 and step % args.sample_every == 0:
            model.eval()

            # If TTL is on, adapt the learner to the last batch context first
            if online_learner is not None and "context_ids" in metrics:
                adapt_stats = online_learner.adapt(metrics["context_ids"])
                if not adapt_stats["skipped"]:
                    print(f"  [TTL] adapted in {adapt_stats['steps']} steps "
                          f"(ppl {adapt_stats['ppl']:.1f} → loss {adapt_stats['loss']:.4f}, "
                          f"norm {adapt_stats['adapter_norm']:.4f})")

            prompt_ids = torch.tensor(
                tokenizer.encode(args.sample_prompt, add_bos=True),
                dtype=torch.long, device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                out_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=80,
                    temperature=0.8,
                    top_k=40,
                    top_p=0.95,
                )
            sample_text = tokenizer.decode(out_ids[0].tolist(), skip_special=True)
            print(f"\n── Sample (step {step}) ──")
            print(sample_text)
            if online_learner is not None:
                s = online_learner.stats()
                print(f"  [TTL stats] calls={s['adapt_calls']} "
                      f"skipped={s['skipped']} norm={s['mean_adapter_norm']:.4f}")
            print("─" * 40 + "\n")
            model.train()

    trainer.step_callback = on_step

    # ── Train ─────────────────────────────────────────────────────────────
    print("Starting AGI training …\n")
    t_start = time.time()

    history = trainer.train(
        text            = text,
        n_epochs        = args.epochs,
        seq_len         = args.seq_len,
        batch_size      = args.batch,
        n_warmup_steps  = args.warmup_steps,
        save_every      = args.save_every,
        log_every       = args.log_every,
        eval_text       = eval_text,
    )

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed:.0f}s "
          f"({elapsed / 60:.1f} min).")
    print(f"Final checkpoint saved to {args.output}/agi_nfn_final.pt")

    # ── Final eval ────────────────────────────────────────────────────────
    if eval_text:
        ppl = trainer.eval_perplexity(eval_text, seq_len=min(512, model.cfg.max_seq_len))
        print(f"Final eval perplexity: {ppl:.2f}")

    if history:
        last = history[-1]
        print(f"Final step {last.get('step')}: "
              f"lm={last.get('lm', 0.0):.4f} "
              f"agi_w={last.get('agi_weight', 0.0):.2f}")

    # ── Save adapters ─────────────────────────────────────────────────────
    if online_learner is not None:
        save_path = args.save_adapters or str(Path(args.output) / "adapters_final.pt")
        online_learner.save_adapters(save_path)
        s = online_learner.stats()
        print(f"\nTest-time learning summary:")
        print(f"  Adapter params : {s['adapter_params']:,} ({s['adapter_ratio']} of model)")
        print(f"  Adapt calls    : {s['adapt_calls']} ({s['skipped']} skipped)")
        print(f"  Avg update loss: {s['avg_loss']:.4f}")
        print(f"  Adapters saved : {save_path}")


if __name__ == "__main__":
    main()
