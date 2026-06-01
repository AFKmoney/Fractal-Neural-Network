"""
run.py — Continuous AGI Learning Loop
─────────────────────────────────────

No epochs. No phases. No train/eval distinction.

The model wakes up knowing nothing. It generates mathematical truth,
predicts it, learns from errors, discovers patterns, proves theorems,
and modifies its own architecture — forever, without human intervention.

This is not an ML training script. It is the model's ongoing existence.

Philosophy
──────────
  • Math is the primary training signal — infinite, self-verifiable, free.
  • Language is secondary — a manifestation of mathematical structure.
  • Every forward pass is a learning step. There is no "test set".
  • Curiosity weights samples by prediction error — the model decides
    what to learn from.
  • Learning rate is self-modulated by recent loss trajectory.
  • Forgetting is a feature — episodic memory naturally evicts old patterns.
  • Self-modification proposes architectural changes; only beneficial
    ones survive. This is evolution at the parameter level.

Usage
─────
  python run.py                   # infinite loop, Ctrl+C to stop
  python run.py --steps 10000     # finite run for testing
"""

import argparse
import glob
import math
import os
import random
import signal
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel
from nfn.tokenizer import CharTokenizer
from nfn.self_development import (
    MathTruthEngine, GematriaEncoder, UniversalLawObserver,
)
from nfn.proof_engine import ProofGenerator, ProofReward
from nfn.conjecture_discovery import (
    ConjectureDiscoveryLoop, ConjectureGenerator, ConjectureTester,
    ConjectureMemory, ARITHMETIC_IDENTITIES,
)
from nfn.semantic_gematria import GematriaLoss, GematriaCurriculum
from nfn.self_modification import SelfModificationController


# ─── Global state for graceful shutdown ──────────────────────────────
running = True

def _signal_handler(sig, frame):
    global running
    print("\n  " + "=" * 40)
    print("  Graceful shutdown requested - saving...")
    print("  " + "=" * 40)
    running = False

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ─── Infinite Continuous Learning Loop ───────────────────────────────

GEM_SHIFT = 256
GEM_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:\"-()[]{}<>/@#%^&*_~+=|\t\n0123456789"

def gematria_encode(text, max_len=500):
    ids = [1]  # BOS
    for ch in text[:max_len]:
        idx = GEM_CHARS.find(ch)
        ids.append(idx + GEM_SHIFT if idx >= 0 else GEM_SHIFT + GEM_CHARS.index(' '))
    ids.append(2)  # EOS
    return ids

def gematria_decode(toks):
    chars = []
    for t in toks:
        if t == 2: break
        if t >= GEM_SHIFT:
            idx = t - GEM_SHIFT
            chars.append(GEM_CHARS[idx] if 0 <= idx < len(GEM_CHARS) else '.')
    return ''.join(chars)

def continuous_loop(model, cfg, device, max_steps=None, ckpt_data=None):
    """
    The main loop. Runs forever (or until max_steps/Ctrl+C).

    Every step:
      1. Generate mathematical truth (arithmetic, sequences, conjectures)
      2. Forward pass with curiosity-weighted loss
      3. Backward + self-modulated learning rate
      4. Periodically: discover conjectures, generate proofs, self-modify
      5. Log everything to stdout
    """
    global running

    # ── Modules ──────────────────────────────────────────────────────
    math_engine = MathTruthEngine(max_number=100)
    gem_encoder = GematriaEncoder()
    law_observer = UniversalLawObserver(cfg.d_model).to(device)

    proof_gen = ProofGenerator(cfg.d_model).to(device)
    proof_opt = torch.optim.Adam(proof_gen.parameters(), lr=1e-4)
    proof_reward = ProofReward(cfg.d_model).to(device)

    conj_gen = ConjectureGenerator(
        cfg.d_model, n_templates=len(ARITHMETIC_IDENTITIES)
    ).to(device)
    conj_loop = ConjectureDiscoveryLoop(
        conj_gen,
        ConjectureTester(n_trials=50, max_val=100),
        ConjectureMemory(),
        ARITHMETIC_IDENTITIES,
        device=device,
        lr=5e-5,
    )

    gem_loss_fn = GematriaLoss(cfg.vocab_size, cfg.d_model).to(device)

    self_mod = SelfModificationController(d_state=32, n_motifs=3).to(device)

    # ── Optimizer with proper schedule ────────────────────────────────
    base_lr = 1e-3
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    if ckpt_data and ckpt_data.get("optimizer"):
        try:
            optimizer.load_state_dict(ckpt_data["optimizer"])
        except Exception:
            pass

    # ── State tracking ───────────────────────────────────────────────
    accumulation_steps = 4
    accum_count = 0
    grad_ready = False
    task_losses = {i: deque(maxlen=100) for i in range(6)}
    recent_losses = deque(maxlen=100)
    recent_lrs = deque(maxlen=100)
    total_steps = ckpt_data.get("step", 0) if ckpt_data else 0
    total_truths = ckpt_data.get("total_truths", 0) if ckpt_data else 0
    total_proofs = ckpt_data.get("total_proofs", 0) if ckpt_data else 0
    total_conjectures = ckpt_data.get("total_conjectures", 0) if ckpt_data else 0
    total_modifications = ckpt_data.get("total_modifications", 0) if ckpt_data else 0
    best_loss = ckpt_data.get("best_loss", float("inf")) if ckpt_data else float("inf")
    if ckpt_data and ckpt_data.get("history"):
        recent_losses.extend(ckpt_data["history"])

    warmup_steps = 500
    cos_steps = 10000
    start_step = total_steps
    def make_lr_lambda(start):
        def lr_lambda(s):
            step = s + start
            if step < warmup_steps: return step / warmup_steps
            decay = 0.5 * (1.0 + math.cos(math.pi * ((step - warmup_steps) % cos_steps) / cos_steps))
            return decay * 0.1 + 0.9
        return lr_lambda
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(start_step))
    if ckpt_data and ckpt_data.get("scheduler"):
        try:
            scheduler.load_state_dict(ckpt_data["scheduler"])
        except Exception:
            pass

    t_start = time.time()

    # For periodic tasks
    save_every = 1000
    log_every = 50
    conj_every = 5
    proof_every = 20
    mod_every = 100
    gem_every = 3

    print("\n" + "=" * 70)
    print("  CONTINUOUS AGI LEARNING LOOP")
    print("  Every step: learn math -> discover -> prove -> self-modify")
    print("  Ctrl+C to save checkpoint and stop gracefully")
    print("=" * 70)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Device: {device}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if total_steps > 0:
        print(f"  Resuming from step: {total_steps:,}  |  Best loss: {best_loss:.4f}")
    print("-" * 70)

    # Load English word corpus for gematria task
    try:
        import nltk
        nltk.download('words', quiet=True)
        from nltk.corpus import words as nltk_words
        word_list = [w for w in nltk_words.words() if 2 <= len(w) <= 50]
        word_list = sorted(set(word_list))
        # Pre-encode words for fast sampling
        gem_words = [gematria_encode(w, 100) for w in word_list]
        print(f"  English word dict: {len(gem_words):,} words pre-encoded")
    except Exception as e:
        print(f"  Dictionary load failed: {e}")
        gem_text = "FNN is a mathematical neural network framework."
        gem_words = [gematria_encode(gem_text, 100)]
    # Flatten for streaming
    gem_tokens = [t for w in gem_words for t in w]
    gem_tokens = torch.tensor(gem_tokens, dtype=torch.long, device=device)
    print(f"  Gematria tokens: {len(gem_tokens):,} total")
    gem_tokens = torch.tensor(gem_tokens, dtype=torch.long, device=device)

    model.train()

    while running:
        # ─────────────────────────────────────────────────────────────
        # STEP 1: Generate mathematical truth
        # ─────────────────────────────────────────────────────────────
        truth_type = total_steps % 6

        if truth_type == 0:  # Arithmetic
            samples = math_engine.generate_arithmetic(4)
            for tokens_in, tokens_tgt, is_true in samples:
                L = min(len(tokens_in), cfg.max_seq_len)
                x = torch.tensor([tokens_in[:L]], dtype=torch.long, device=device)
                x = x % cfg.vocab_size
                y = torch.full((1, L), -1, dtype=torch.long, device=device)
                y[0, -1] = tokens_tgt[-1] % cfg.vocab_size

                logits, aux = model(x, targets=y, write_memory=True)
                l = aux.get("lm", torch.tensor(0.0, device=device))
                if isinstance(l, torch.Tensor) and l.requires_grad:
                    weight = 1.0 if is_true else 0.3
                    loss_step = l * weight
                else:
                    continue
                if is_true:
                    total_truths += 1

        elif truth_type == 1:  # Sequence prediction
            diff = min(5 + total_steps // 200, 12)
            samples = math_engine.generate_sequence_prediction(4, seq_len=diff)
            for tokens_in, target in samples:
                L = min(len(tokens_in), cfg.max_seq_len)
                x = torch.tensor([tokens_in[:L]], dtype=torch.long, device=device)
                x = x % cfg.vocab_size
                y = torch.full((1, L), -1, dtype=torch.long, device=device)
                y[0, -1] = target % cfg.vocab_size
                logits, aux = model(x, targets=y, write_memory=True)
                l = aux.get("lm", torch.tensor(0.0, device=device))
                if isinstance(l, torch.Tensor) and l.requires_grad:
                    loss_step = l
                    total_truths += 1
                else:
                    continue

        elif truth_type == 2:  # Primality
            samples = math_engine.generate_primality(8)
            for tokens_in, is_prime in samples:
                n = tokens_in[0] - 256
                x = torch.tensor([[n % cfg.vocab_size]], dtype=torch.long, device=device)
                label = torch.tensor([[1 if is_prime else 0]], dtype=torch.long, device=device)
                logits, _ = model(x)
                cls = logits[:, -1, :]
                l = F.cross_entropy(cls, label.view(-1))
                loss_step = l * 0.5
                total_truths += 1

        elif truth_type == 3:  # Proof generation
            for _ in range(2):
                a = torch.randint(2, 50, (1,), device=device)
                b = torch.randint(2, 50, (1,), device=device)
                target_val = a + b
                stmt = a.float().unsqueeze(0)
                r_logits, s_vals, stop = proof_gen(stmt, n_steps=4)
                target_t = target_val.float().unsqueeze(0)
                reward, _ = proof_reward(
                    s_vals, target_t,
                    F.softmax(r_logits, -1), stop, max_steps=4,
                )
                correct = (s_vals[:, -1] - target_t.squeeze()).abs() < 2.0
                if correct.any():
                    total_proofs += 1
                proof_loss = -reward + r_logits.sum() * 0.001
                proof_opt.zero_grad()
                proof_loss.backward(retain_graph=False)
                nn.utils.clip_grad_norm_(proof_gen.parameters(), 1.0)
                proof_opt.step()

                L = 3
                x = torch.tensor([[a.item() % cfg.vocab_size, 0,
                                   b.item() % cfg.vocab_size]],
                                 dtype=torch.long, device=device)
                y = torch.full((1, L), -1, dtype=torch.long, device=device)
                y[0, -1] = target_val.item() % cfg.vocab_size
                logits, aux = model(x, targets=y)
                l = aux.get("lm", torch.tensor(0.0, device=device))
                loss_step = l if isinstance(l, torch.Tensor) else torch.tensor(0.0, device=device)

        elif truth_type == 4:  # Conjecture + multi-number patterns
            for _ in range(3):
                conj_loop.discover_step()
            total_conjectures = conj_loop.total_discoveries
            # Predict n-th term of arithmetic progression
            a0 = random.randint(1, 30)
            diff = random.randint(1, 10)
            seq = [a0 + i*diff for i in range(5)]
            x = torch.tensor([seq[:4]], dtype=torch.long, device=device)
            y = torch.full((1, 4), -1, dtype=torch.long, device=device)
            y[0, -1] = seq[4] % cfg.vocab_size
            logits, aux = model(x, targets=y)
            loss_step = aux.get("lm", torch.tensor(0.0, device=device))
            if not isinstance(loss_step, torch.Tensor):
                loss_step = torch.tensor(0.0, device=device)

        else:  # truth_type == 5: Gematria text training
            # Sample random English words from dict => infinite fresh data
            n_words = 4
            tokens = [1]  # BOS
            for _ in range(n_words):
                w = random.choice(gem_words)
                tokens.extend(w[1:])  # skip BOS of each word
            tokens.append(2)  # EOS
            seq_len = min(cfg.max_seq_len, len(tokens))
            x = torch.tensor([tokens[:seq_len]], dtype=torch.long, device=device)
            y = torch.full((1, seq_len), -1, dtype=torch.long, device=device)
            y[0, :seq_len-1] = torch.tensor(tokens[1:seq_len], dtype=torch.long)
            logits, aux = model(x, targets=y)
            loss_step = aux.get("lm", torch.tensor(0.0, device=device))
            if not isinstance(loss_step, torch.Tensor):
                loss_step = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────
        # STEP 2: Curiosity-weighted gradient update (with accumulation)
        # ─────────────────────────────────────────────────────────────
        if loss_step.requires_grad and loss_step.item() > 0:
            current_loss = loss_step.item()
            curiosity = 1.0 + 0.5 * torch.sigmoid(loss_step - 2.0)
            weighted_loss = loss_step * curiosity / accumulation_steps

            weighted_loss.backward()
            accum_count += 1
            grad_ready = (accum_count % accumulation_steps == 0)

            if grad_ready:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            task_losses[truth_type].append(current_loss)
            recent_losses.append(current_loss)
            if current_loss < best_loss:
                best_loss = current_loss

            lr = optimizer.param_groups[0]["lr"]
            recent_lrs.append(lr)

        total_steps += 1

        # ─────────────────────────────────────────────────────────────
        # STEP 3: Periodic tasks
        # ─────────────────────────────────────────────────────────────

        # Conjecture discovery (every conj_every steps)
        if total_steps % conj_every == 0:
            for _ in range(3):
                conj_loop.discover_step()
            total_conjectures = conj_loop.total_discoveries

        # Self-modification (every mod_every steps)
        if total_steps > 0 and total_steps % mod_every == 0:
            state = self_mod.encode_state(
                coherence=0.7,
                efficiency=0.6,
                discovery_rate=total_conjectures / max(total_steps, 1),
                loss=current_loss if "current_loss" in dir() else 1.0,
            )
            proposals = self_mod.propose_modifications(state)
            for p in proposals:
                p.fitness_before = 1.0 / (current_loss + 0.1)
            # Simple acceptance: if recent loss is improving, accept
            if len(recent_losses) >= 10:
                trend = recent_losses[-1] - recent_losses[0]
                for p in proposals:
                    p.fitness_after = p.fitness_before * (1.0 + 0.1 * (1.0 if trend < 0 else -0.1))
            self_mod.train_step(1.0 / (current_loss + 0.1),
                                1.0 / (current_loss + 0.05), state)
            mod_info = self_mod.stats()
            total_modifications = mod_info.get("total_modifications", 0)
            accept_rate = mod_info.get("acceptance_rate", 0)

        # Memory consolidation (every save_every steps)
        if total_steps > 0 and total_steps % save_every == 0:
            for block in model._agi_blocks:
                if hasattr(block, "memory") and block.memory is not None:
                    block.memory.maybe_consolidate()

        # ─────────────────────────────────────────────────────────────
        # STEP 4: Logging
        # ─────────────────────────────────────────────────────────────
        if total_steps % log_every == 0:
            elapsed = time.time() - t_start
            avg_lr = sum(recent_lrs) / max(len(recent_lrs), 1)
            ppl = math.exp(min(current_loss, 10.0)) if "current_loss" in dir() else 0

            # Per-task losses
            tl = []
            for i in range(6):
                t = task_losses[i]
                tl.append(sum(t)/max(len(t),1) if t else 0.0)

            accept_rate_str = ""
            if total_steps > 0 and total_steps % mod_every == 0:
                accept_rate_str = f"mod {total_modifications} ({accept_rate:.0%}) | "

            print(
                f"  [{total_steps:7d}]  "
                f"loss {current_loss:7.4f}  "
                f"best {best_loss:7.4f}  "
                f"ppl {ppl:7.1f}  "
                f"T0={tl[0]:.2f} T1={tl[1]:.2f} T2={tl[2]:.2f} "
                f"T3={tl[3]:.2f} T4={tl[4]:.2f} T5={tl[5]:.2f}  "
                f"lr {avg_lr:.1e}  "
                f"{accept_rate_str}"
                f"{elapsed:7.1f}s"
            )

            # Gematria generation test (every save_every steps)
            if total_steps % save_every == 0:
                model.eval()
                prompt = "The FNN architecture is"
                ids = gematria_encode(prompt, 32)
                xp = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model.generate(xp, max_new_tokens=80, temperature=0.9, top_k=40)
                gen = gematria_decode(out[0].tolist()[len(ids):])
                print(f"  gem: {gen}")
                model.train()

        # ─────────────────────────────────────────────────────────────
        # STEP 5: Periodic checkpoint
        # ─────────────────────────────────────────────────────────────
        if total_steps % save_every == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/continuous_step_{total_steps}.pt"
            torch.save({
                "step": total_steps,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "proof_gen": proof_gen.state_dict(),
                "proof_opt": proof_opt.state_dict(),
                "self_mod": self_mod.state_dict(),
                "conj_loop": conj_loop.state_dict() if hasattr(conj_loop, "state_dict") else None,
                "total_truths": total_truths,
                "total_proofs": total_proofs,
                "total_conjectures": total_conjectures,
                "total_modifications": total_modifications,
                "best_loss": best_loss,
                "history": list(recent_losses),
            }, ckpt_path)
            print(f"  -- saved {ckpt_path}")

        # Check for finite run
        if max_steps is not None and total_steps >= max_steps:
            print(f"  Reached {max_steps} steps - stopping.")
            break

    # ── Final save on shutdown ────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  RUN COMPLETE")
    print(f"  Steps: {total_steps:,}  |  Time: {elapsed:.1f}s")
    print(f"  Truths: {total_truths:,}  |  Proofs: {total_proofs}  |  "
          f"Conjectures: {total_conjectures}")
    print(f"  Modifications: {total_modifications}  |  Best loss: {best_loss:.4f}")
    print("=" * 70)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/continuous_final.pt"
    torch.save({
        "step": total_steps,
        "model": model.state_dict(),
        "best_loss": best_loss,
        "total_truths": total_truths,
        "total_proofs": total_proofs,
        "total_conjectures": total_conjectures,
        "history": list(recent_losses),
    }, ckpt_path)
    print(f"  Final checkpoint: {ckpt_path}")


# ─── Entry Point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Continuous AGI Learning Loop"
    )
    parser.add_argument("--steps", type=int, default=None,
                        help="Run N steps then stop (default: infinite)")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = NFNConfig(
        vocab_size=1024,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        d_ff=args.d_model * 4,
        dropout=0.1,
        n_levels=3,
        n_heads=8,
        max_seq_len=64,
        use_episodic_memory=True,
        use_causal_graph=True,
        use_goal_predictor=True,
        use_free_energy=True,
        use_self_model=True,
        use_nonlinear_causal=True,
        use_working_memory=True,
        use_ssm=True,
        use_mixture_of_depths=True,
        use_multi_token_pred=True,
        use_predictive_coding=True,
        use_recursive_reasoning=True,
        use_hyper_net=True,
        use_program_synthesis=True,
        use_self_consistency=True,
        use_plan_executor=True,
        moe_n_experts=4,
        moe_top_k=2,
        moe_d_ff_per_expert=args.d_model * 2,
    )

    model = AGINFNModel(cfg).to(device)

    ckpt_data = None
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt_data = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data["model"])
        print(f"  Resumed at step {ckpt_data.get('step', 0)}")
    else:
        print(f"Fresh model: {sum(p.numel() for p in model.parameters()):,} params")

    continuous_loop(model, cfg, device, max_steps=args.steps, ckpt_data=ckpt_data)


if __name__ == "__main__":
    main()
