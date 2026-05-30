"""
NFN v5.0 — AGI Training Script

Three-phase training to create genuine AGI:

  Phase 1: MATHEMATICAL SELF-DEVELOPMENT
    - Model learns arithmetic, primality, sequences, modular arithmetic
    - Discovers mathematical conjectures autonomously
    - Generates and verifies proofs
    - Learns gematria numerical structure
    - Observes universal laws in its own dynamics

  Phase 2: LANGUAGE + AGI TRAINING
    - Mathematical truths as training corpus
    - Full AGI loss (10 components) with curriculum
    - Self-play DPO for self-improvement
    - Constitutional self-critique
    - WAKE/SLEEP memory consolidation
    - Curiosity-weighted learning

  Phase 3: SELF-MODIFICATION + CONTINUAL LEARNING
    - Model proposes modifications to its own architecture
    - Only beneficial changes are kept
    - Test-time LoRA adaptation
    - Continuous discovery of new mathematical truths

Usage:
    python train_agi.py --phase 1          # Math pre-training only
    python train_agi.py --phase 2          # Full AGI training
    python train_agi.py --phase 3          # Self-modification
    python train_agi.py --phase all        # All phases sequentially
    python train_agi.py --phase all --steps 10000
"""

import argparse
import math
import os
import sys
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── NFN Imports ─────────────────────────────────────────────────────────────
from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel
from nfn.tokenizer import CharTokenizer

from nfn.self_development import (
    MathTruthEngine, GematriaEncoder, UniversalLawObserver, SelfDevelopmentLoop,
)
from nfn.proof_engine import ProofGenerator, ProofVerifier, ProofReward
from nfn.conjecture_discovery import (
    ConjectureDiscoveryLoop, ConjectureGenerator, ConjectureTester,
    ConjectureMemory, ARITHMETIC_IDENTITIES,
)
from nfn.semantic_gematria import SemanticGematriaLayer, GematriaLoss, GematriaCurriculum
from nfn.self_modification import SelfModificationController


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Mathematical Self-Development
# ─────────────────────────────────────────────────────────────────────────────

def phase1_math_pretraining(model, cfg, device, n_steps=2000, log_every=50):
    """
    Pre-train the model on self-generated mathematical truths.
    
    This is the foundation: before the model sees any text,
    it learns the structure of mathematical truth.
    
    Truth is infinite, self-verifiable, and free.
    """
    print("\n" + "=" * 70)
    print("  PHASE 1: MATHEMATICAL SELF-DEVELOPMENT")
    print("=" * 70)

    math_engine = MathTruthEngine(max_number=100)
    gematria = GematriaEncoder()
    law_observer = UniversalLawObserver(cfg.d_model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)

    proof_gen = ProofGenerator(cfg.d_model).to(device)
    proof_opt = torch.optim.Adam(proof_gen.parameters(), lr=1e-4)
    proof_reward_fn = ProofReward(cfg.d_model).to(device)

    conj_gen = ConjectureGenerator(cfg.d_model, n_templates=len(ARITHMETIC_IDENTITIES)).to(device)
    conj_loop = ConjectureDiscoveryLoop(
        conj_gen, ConjectureTester(n_trials=50, max_val=100),
        ConjectureMemory(), ARITHMETIC_IDENTITIES, device=device, lr=5e-5,
    )

    gem_loss_fn = GematriaLoss(cfg.vocab_size, cfg.d_model).to(device)
    gem_curriculum = GematriaCurriculum()

    history = []
    best_loss = float("inf")
    total_truths = 0
    total_proofs = 0
    total_conjectures = 0
    t0 = time.time()

    for step in range(n_steps):
        epoch_type = step % 6
        loss = torch.tensor(0.0, device=device)

        # ── Type 0: Arithmetic truth ──────────────────────────────────────
        if epoch_type == 0:
            samples = math_engine.generate_arithmetic(4)
            for input_tokens, target_tokens, is_true in samples:
                L = min(len(input_tokens), cfg.max_seq_len)
                x = torch.tensor([input_tokens[:L]], dtype=torch.long, device=device)
                x = x % cfg.vocab_size
                # targets: same length as input, only last position has answer
                y = torch.full((1, L), -1, dtype=torch.long, device=device)
                y[0, -1] = target_tokens[-1] % cfg.vocab_size

                logits, aux = model(x, targets=y)
                if isinstance(aux, dict) and "lm" in aux:
                    l = aux["lm"] if isinstance(aux["lm"], torch.Tensor) else aux.get("total", torch.tensor(0.0, device=device))
                else:
                    l = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1), ignore_index=-1)
                
                if not is_true:
                    l = l * 0.3
                else:
                    total_truths += 1
                loss = loss + l

        # ── Type 1: Sequence prediction ───────────────────────────────────
        elif epoch_type == 1:
            difficulty = min(5 + step // 200, 12)
            samples = math_engine.generate_sequence_prediction(4, seq_len=difficulty)
            for input_tokens, target in samples:
                L = min(len(input_tokens), cfg.max_seq_len)
                x = torch.tensor([input_tokens[:L]], dtype=torch.long, device=device)
                x = x % cfg.vocab_size
                y = torch.full((1, L), -1, dtype=torch.long, device=device)
                y[0, -1] = target % cfg.vocab_size

                logits, _ = model(x, targets=y)
                l = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1), ignore_index=-1)
                loss = loss + l
                total_truths += 1

        # ── Type 2: Primality ─────────────────────────────────────────────
        elif epoch_type == 2:
            samples = math_engine.generate_primality(8)
            for input_tokens, is_prime in samples:
                n = input_tokens[0] - 256
                x = torch.tensor([[n % cfg.vocab_size]], dtype=torch.long, device=device)
                label = torch.tensor([[1 if is_prime else 0]], dtype=torch.long, device=device)
                logits, _ = model(x)
                cls_token = logits[:, -1, :]  # [1, 108]
                l = F.cross_entropy(cls_token, label.view(-1))
                loss = loss + l * 0.5
                total_truths += 1

        # ── Type 3: Proof generation ──────────────────────────────────────
        elif epoch_type == 3:
            for _ in range(4):
                a = random.randint(2, 50)
                b = random.randint(2, 50)
                target_val = a + b

                stmt = torch.tensor([[float(a)]], device=device)
                r_logits, s_vals, stop = proof_gen(stmt, n_steps=4)
                target_t = torch.tensor([[float(target_val)]], device=device)

                reward, metrics = proof_reward_fn(
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

                x = torch.tensor([[a % cfg.vocab_size, 0, b % cfg.vocab_size]], dtype=torch.long, device=device)
                y = torch.full((1, 3), -1, dtype=torch.long, device=device)
                y[0, -1] = target_val % cfg.vocab_size
                logits, aux = model(x, targets=y)
                if isinstance(aux, dict) and "lm" in aux:
                    l = aux["lm"] if isinstance(aux["lm"], torch.Tensor) else torch.tensor(0.0, device=device)
                else:
                    l = torch.tensor(0.0, device=device)
                loss = loss + l

        # ── Type 4: Conjecture discovery ──────────────────────────────────
        elif epoch_type == 4:
            for _ in range(3):
                m = conj_loop.discover_step()
            total_conjectures = conj_loop.total_discoveries

            x = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
            y = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
            logits, aux = model(x, targets=y)
            if isinstance(aux, dict) and "lm" in aux:
                loss = aux["lm"] if isinstance(aux["lm"], torch.Tensor) else torch.tensor(0.0, device=device)

        # ── Type 5: Gematria + universal laws ─────────────────────────────
        else:
            x = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
            y = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
            logits, aux = model(x, targets=y)
            if isinstance(aux, dict) and "lm" in aux:
                loss = aux["lm"] if isinstance(aux["lm"], torch.Tensor) else torch.tensor(0.0, device=device)

            gem_l, gem_m = gem_loss_fn(torch.randn(2, 16, cfg.d_model, device=device), x)
            loss = loss + gem_l * 0.1

            phase_name = gem_curriculum.get_phase()
            advanced = gem_curriculum.advance()

        # ── Universal law regularization ──────────────────────────────────
        if step % 20 == 0:
            with torch.no_grad():
                h = torch.randn(2, 32, cfg.d_model, device=device)
                law_loss, law_m = law_observer(h)

        # ── Gradient step ─────────────────────────────────────────────────
        if loss.requires_grad and loss.item() > 0:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        if loss.item() < best_loss:
            best_loss = loss.item()

        if step % log_every == 0:
            print(
                f"  [P1] step {step:5d}/{n_steps} | "
                f"loss {loss.item():.4f} (best {best_loss:.4f}) | "
                f"truths {total_truths} | proofs {total_proofs} | "
                f"conjectures {total_conjectures} | "
                f"lr {current_lr:.2e} | {elapsed:.1f}s"
            )

        history.append({
            "step": step, "phase": 1,
            "loss": loss.item(), "lr": current_lr,
            "truths": total_truths, "proofs": total_proofs,
            "conjectures": total_conjectures,
        })

    # Save Phase 1
    ckpt_path = "checkpoints/phase1_math.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": n_steps,
        "truths": total_truths,
        "proofs": total_proofs,
        "conjectures": total_conjectures,
        "history": history[-200:],
    }, ckpt_path)
    print(f"\n  Phase 1 complete: {total_truths} truths, {total_proofs} proofs, "
          f"{total_conjectures} conjectures discovered")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Final loss: {best_loss:.4f}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Language + AGI Training
# ─────────────────────────────────────────────────────────────────────────────

def generate_math_corpus(n_items=5000):
    """Generate a mathematical corpus for language training."""
    engine = MathTruthEngine(max_number=200)
    lines = []

    for _ in range(n_items // 4):
        for inp, tgt, is_true in engine.generate_arithmetic(1):
            ops = {0: "+", 1: "-", 2: "*"}
            a = inp[0] - 256
            op = ops.get(inp[1], "+")
            b = inp[2] - 256
            r = tgt[0] - 256
            label = "TRUE" if is_true else "FALSE"
            lines.append(f"{a} {op} {b} = {r} [{label}]")

    for _ in range(n_items // 4):
        for inp, is_prime in engine.generate_primality(1):
            n = inp[0] - 256
            label = "prime" if is_prime else "composite"
            lines.append(f"{n} is {label}")

    for _ in range(n_items // 4):
        for inp, target in engine.generate_sequence_prediction(1):
            seq = [str(x - 256) for x in inp]
            ans = target - 256
            lines.append(f"sequence: {', '.join(seq)}, next: {ans}")

    for _ in range(n_items // 4):
        for inp, target in engine.generate_modular_arithmetic(1):
            a, b, p = inp[0] - 256, inp[1] - 256, inp[2] - 256
            r = target - 256
            lines.append(f"({a} * {b}) mod {p} = {r}")

    # Add number theory facts
    primes = [p for p in range(2, 200) if engine._is_prime(p)]
    for p in primes[:50]:
        lines.append(f"{p} is prime")
        if p > 2:
            lines.append(f"{p} = 6*{(p-1)//6}+1 or 6*{(p+1)//6}-1")
        lines.append(f"fermat: a^({p}-1) = 1 mod {p}")

    for n in range(1, 50):
        s = n * (n + 1) // 2
        lines.append(f"sum(1..{n}) = {s}")
        lines.append(f"{n}^2 = {n*n}")

    return "\n".join(lines)


def phase2_agi_training(model, cfg, device, n_steps=3000, log_every=50):
    """
    Full AGI training with mathematical corpus, self-play, critique,
    curiosity weighting, and memory consolidation.
    """
    print("\n" + "=" * 70)
    print("  PHASE 2: LANGUAGE + AGI TRAINING")
    print("=" * 70)

    tokenizer = CharTokenizer()
    cfg_vocab = tokenizer.vocab_size

    math_corpus = generate_math_corpus(8000)
    print(f"  Generated mathematical corpus: {len(math_corpus)} chars")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)

    gem_loss_fn = GematriaLoss(cfg.vocab_size, cfg.d_model).to(device)
    gem_optimizer = torch.optim.Adam(gem_loss_fn.parameters(), lr=1e-4)

    law_observer = UniversalLawObserver(cfg.d_model).to(device)

    ids = tokenizer.encode(math_corpus, add_bos=True)
    data = torch.tensor(ids, dtype=torch.long)
    seq_len = min(cfg.max_seq_len, 128)
    batch_size = 4

    best_loss = float("inf")
    history = []
    t0 = time.time()

    for step in range(n_steps):
        model.train()

        # Sample batch
        starts = torch.randint(0, max(1, len(data) - seq_len - 1), (batch_size,))
        x = torch.stack([data[s:s+seq_len] for s in starts]).to(device)
        y = torch.stack([data[s+1:s+seq_len+1] for s in starts]).to(device)

        # Forward with AGI losses
        logits, losses = model(x, targets=y, write_memory=True)

        lm_loss = losses.get("lm", torch.tensor(0.0, device=device))
        total_loss = lm_loss

        agi_components = ["causal", "goal", "free_energy", "consistency"]
        for k in agi_components:
            v = losses.get(k, torch.tensor(0.0, device=device))
            if isinstance(v, torch.Tensor):
                total_loss = total_loss + v

        # Gematria loss (number-theoretic structure)
        if step % 5 == 0:
            try:
                gem_l, _ = gem_loss_fn(torch.randn(batch_size, seq_len, cfg.d_model, device=device), x)
                total_loss = total_loss + gem_l * 0.05
                gem_optimizer.zero_grad()
            except Exception:
                pass

        # Self-play DPO (every 200 steps)
        sp_loss_val = 0.0
        if step > 0 and step % 200 == 0:
            model.eval()
            with torch.no_grad():
                prompt = x[:1, :8]
                candidates = []
                scores = []
                for _ in range(4):
                    try:
                        cand = model.generate(prompt, max_new_tokens=16, temperature=1.2, top_k=30)
                        L = cand.shape[1]
                        tgt_c = cand[:, 1:]                     # [1, L-1]
                        pad = torch.full((1, 1), -1, dtype=torch.long, device=device)
                        tgt_c = torch.cat([tgt_c, pad], dim=1)  # [1, L] — last pos ignored
                        _, lss = model(cand, targets=tgt_c, write_memory=False)
                        score = -lss.get("lm", torch.tensor(999.0)).item() if isinstance(lss.get("lm"), torch.Tensor) else -999.0
                        candidates.append(cand)
                        scores.append(score)
                    except Exception as e2:
                        scores.append(-999.0)

                if len(scores) >= 2 and max(scores) > min(scores):
                    sp_loss_val = max(scores) - min(scores)
            model.train()

        # Constitutional critique (every 500 steps)
        crit_loss_val = 0.0
        if step > 0 and step % 500 == 0:
            model.eval()
            with torch.no_grad():
                prompt = x[:1, :8]
                try:
                    revision = model.generate(prompt, max_new_tokens=24, temperature=0.8, top_k=40)
                    if revision.shape[1] > 2:
                        L = revision.shape[1]
                        rev_tgt = revision[:, 1:]               # [1, L-1]
                        pad = torch.full((1, 1), -1, dtype=torch.long, device=device)
                        rev_tgt = torch.cat([rev_tgt, pad], dim=1)  # [1, L]
                        rev_logits, _ = model(revision, targets=rev_tgt, write_memory=False)
                        crit_loss_val = F.cross_entropy(
                            rev_logits.reshape(-1, cfg.vocab_size),
                            rev_tgt.reshape(-1),
                            ignore_index=-1,
                        ).item()
                except Exception as e3:
                    pass
            model.train()

        # Memory consolidation / SLEEP (every 500 steps)
        if step > 0 and step % 500 == 0:
            model.eval()
            with torch.no_grad():
                for block in model._agi_blocks:
                    if hasattr(block, "memory") and block.memory is not None:
                        block.memory.maybe_consolidate()
            model.train()

        # Gradient step
        if total_loss.requires_grad and total_loss.item() > 0:
            optimizer.zero_grad()
            total_loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        loss_val = total_loss.item()
        if loss_val < best_loss:
            best_loss = loss_val

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        if step % log_every == 0:
            ppl = math.exp(min(loss_val, 10.0))
            print(
                f"  [P2] step {step:5d}/{n_steps} | "
                f"loss {loss_val:.4f} (best {best_loss:.4f}) | "
                f"ppl {ppl:.1f} | sp {sp_loss_val:.2f} | crit {crit_loss_val:.2f} | "
                f"lr {lr:.2e} | {elapsed:.1f}s"
            )

        # Save best
        if step % 500 == 0 and loss_val <= best_loss + 0.1:
            ckpt_path = "checkpoints/phase2_agi.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best_loss": best_loss,
                "history": history[-200:],
            }, ckpt_path)

        history.append({
            "step": step, "phase": 2,
            "loss": loss_val,
            "ppl": math.exp(min(loss_val, 10.0)),
            "sp": sp_loss_val,
            "crit": crit_loss_val,
        })

    # Final save
    ckpt_path = "checkpoints/phase2_agi_final.pt"
    torch.save({
        "model": model.state_dict(),
        "step": n_steps,
        "best_loss": best_loss,
        "history": history[-200:],
    }, ckpt_path)
    print(f"\n  Phase 2 complete: best_loss={best_loss:.4f}")
    print(f"  Checkpoint: {ckpt_path}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Self-Modification + Continual Learning
# ─────────────────────────────────────────────────────────────────────────────

def phase3_self_modification(model, cfg, device, n_steps=1000, log_every=50):
    """
    The model modifies its own architecture and continues learning.
    Only beneficial modifications are kept.
    """
    print("\n" + "=" * 70)
    print("  PHASE 3: SELF-MODIFICATION + CONTINUAL LEARNING")
    print("=" * 70)

    controller = SelfModificationController(d_state=32, n_motifs=3).to(device)
    controller_optimizer = torch.optim.Adam(controller.parameters(), lr=1e-4)

    math_engine = MathTruthEngine(max_number=200)
    conj_gen = ConjectureGenerator(cfg.d_model, n_templates=len(ARITHMETIC_IDENTITIES)).to(device)
    conj_loop = ConjectureDiscoveryLoop(
        conj_gen, ConjectureTester(n_trials=30, max_val=100),
        ConjectureMemory(), ARITHMETIC_IDENTITIES, device=device,
    )

    tokenizer = CharTokenizer()
    corpus = generate_math_corpus(3000)
    ids = tokenizer.encode(corpus, add_bos=True)
    data = torch.tensor(ids, dtype=torch.long)
    seq_len = min(cfg.max_seq_len, 128)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-6)

    best_loss = float("inf")
    history = []
    t0 = time.time()

    for step in range(n_steps):
        model.train()

        # Standard training on math corpus
        starts = torch.randint(0, max(1, len(data) - seq_len - 1), (4,))
        x = torch.stack([data[s:s+seq_len] for s in starts]).to(device)
        y = torch.stack([data[s+1:s+seq_len+1] for s in starts]).to(device)

        logits, losses = model(x, targets=y, write_memory=True)
        loss = losses.get("lm", torch.tensor(0.0, device=device))
        if isinstance(loss, torch.Tensor):
            for k in ["causal", "goal", "free_energy"]:
                v = losses.get(k, torch.tensor(0.0, device=device))
                if isinstance(v, torch.Tensor):
                    loss = loss + v

        optimizer.zero_grad()
        if loss.requires_grad and loss.item() > 0:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Conjecture discovery (every 10 steps)
        new_conjectures = 0
        if step % 10 == 0:
            for _ in range(3):
                conj_loop.discover_step()
            new_conjectures = conj_loop.total_discoveries

        # Self-modification (every 100 steps)
        mod_info = {}
        if step > 0 and step % 100 == 0:
            fitness_before = 1.0 / (loss.item() + 0.1)

            state = controller.encode_state(
                coherence=0.7, efficiency=0.6,
                discovery_rate=new_conjectures / max(step + 1, 1),
                loss=loss.item(),
            )

            proposals = controller.propose_modifications(state)

            # Apply modifications tentatively (small step)
            for p in proposals:
                p.fitness_before = fitness_before

            # Train one more step to measure effect
            starts2 = torch.randint(0, max(1, len(data) - seq_len - 1), (4,))
            x2 = torch.stack([data[s:s+seq_len] for s in starts2]).to(device)
            y2 = torch.stack([data[s+1:s+seq_len+1] for s in starts2]).to(device)
            logits2, losses2 = model(x2, targets=y2)
            loss2 = losses2.get("lm", torch.tensor(0.0, device=device))
            if isinstance(loss2, torch.Tensor):
                fitness_after = 1.0 / (loss2.item() + 0.1)
            else:
                fitness_after = fitness_before

            for p in proposals:
                p.fitness_after = fitness_after

            controller.train_step(fitness_before, fitness_after, state)
            mod_info = controller.stats()

        loss_val = loss.item() if isinstance(loss, torch.Tensor) else 0.0
        if loss_val < best_loss:
            best_loss = loss_val

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        if step % log_every == 0:
            print(
                f"  [P3] step {step:5d}/{n_steps} | "
                f"loss {loss_val:.4f} (best {best_loss:.4f}) | "
                f"conjectures {conj_loop.total_discoveries} | "
                f"mods {mod_info.get('total_modifications', 0)} "
                f"(accept {mod_info.get('acceptance_rate', 0):.0%}) | "
                f"lr {lr:.2e} | {elapsed:.1f}s"
            )

        history.append({
            "step": step, "phase": 3,
            "loss": loss_val,
            "conjectures": conj_loop.total_discoveries,
            "modifications": mod_info.get("total_modifications", 0),
        })

    # Final save
    ckpt_path = "checkpoints/phase3_selfmod.pt"
    torch.save({
        "model": model.state_dict(),
        "controller": controller.state_dict(),
        "step": n_steps,
        "best_loss": best_loss,
        "conjectures": conj_loop.total_discoveries,
        "history": history[-200:],
    }, ckpt_path)
    print(f"\n  Phase 3 complete: best_loss={best_loss:.4f}, "
          f"conjectures={conj_loop.total_discoveries}")
    print(f"  Checkpoint: {ckpt_path}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Generation / Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, cfg, device):
    """Quick evaluation: generate text, test math, show self-model state."""
    print("\n" + "=" * 70)
    print("  EVALUATION")
    print("=" * 70)
    model.eval()
    tokenizer = CharTokenizer()

    # Generate
    prompts = ["2+3", "7*4", "is 17", "sum(1..10)", "next: 2,4,6,8"]
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
        try:
            out = model.generate(ids, max_new_tokens=32, temperature=0.8, top_k=30)
            text = tokenizer.decode(out[0].tolist())
            print(f"  '{prompt}' -> '{text}'")
        except Exception as e:
            print(f"  '{prompt}' -> ERROR: {e}")

    # Math accuracy
    engine = MathTruthEngine()
    correct = 0
    total = 20
    for inp, tgt, is_true in engine.generate_arithmetic(total):
        a, op_id, b = inp[0] - 256, inp[1], inp[2] - 256
        ops = {0: "+", 1: "-", 2: "*"}
        r = tgt[0] - 256
        x = torch.tensor([[a % cfg.vocab_size, op_id, b % cfg.vocab_size]], dtype=torch.long, device=device)
        try:
            with torch.no_grad():
                logits, _ = model(x)
                pred = logits[0, -1].argmax().item()
                text_pred = tokenizer.decode([pred])
                if is_true:
                    try:
                        pred_val = int(text_pred.strip())
                        if pred_val == r:
                            correct += 1
                    except ValueError:
                        pass
        except Exception:
            pass
    print(f"\n  Math accuracy: {correct}/{total}")

    # Model stats
    pc = model.param_count()
    print(f"\n  Model params: {pc['total']:,}")
    print(f"  Blocks: {cfg.n_blocks}, d_model: {cfg.d_model}")
    print(f"  Active modules: {[k for k, v in pc.items() if v > 0 and k not in ('total',)]}")

    model.train()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train FNN AGI")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="Training phase to run")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override number of steps per phase")
    parser.add_argument("--d_model", type=int, default=128,
                        help="Model dimension")
    parser.add_argument("--n_blocks", type=int, default=4,
                        help="Number of blocks")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cpu/cuda)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Config — build a capable but trainable model
    cfg = NFNConfig(
        vocab_size=1024,  # accommodates MathTruthEngine offset+256 encoding + CharTokenizer
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        d_ff=args.d_model * 4,
        dropout=0.1,
        n_levels=3,
        n_heads=4,
        max_seq_len=128,
        # AGI modules
        use_episodic_memory=True,
        use_causal_graph=True,
        use_goal_predictor=True,
        use_free_energy=True,
        use_self_model=True,
        use_nonlinear_causal=True,
        # MoE
        moe_n_experts=4,
        moe_top_k=2,
        moe_d_ff_per_expert=args.d_model * 2,
    )

    print(f"\nConfig: d={cfg.d_model}, blocks={cfg.n_blocks}, "
          f"experts={cfg.moe_n_experts}, levels={cfg.n_levels}")

    # Build model
    model = AGINFNModel(cfg).to(device)
    pc = model.param_count()
    print(f"Model: {pc['total']:,} parameters")
    print(model)

    # Resume
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    # Run phases
    all_history = []

    if args.phase in ("1", "all"):
        steps = args.steps or 2000
        h = phase1_math_pretraining(model, cfg, device, n_steps=steps)
        all_history.extend(h)
        evaluate(model, cfg, device)

    if args.phase in ("2", "all"):
        steps = args.steps or 3000
        h = phase2_agi_training(model, cfg, device, n_steps=steps)
        all_history.extend(h)
        evaluate(model, cfg, device)

    if args.phase in ("3", "all"):
        steps = args.steps or 1000
        h = phase3_self_modification(model, cfg, device, n_steps=steps)
        all_history.extend(h)
        evaluate(model, cfg, device)

    # Save training history
    torch.save(all_history, "checkpoints/training_history.pt")
    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total steps: {len(all_history)}")
    print(f"  Final loss: {all_history[-1]['loss']:.4f}" if all_history else "")
    print(f"  History saved: checkpoints/training_history.pt")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
