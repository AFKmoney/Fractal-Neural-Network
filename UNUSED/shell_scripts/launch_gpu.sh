#!/bin/bash
set -e
cd /root/FNN

cat > train_gpu.py << 'TRAINEOF'
"""NFN 2.59B GPU Training — A6000/24GB optimized."""
import os, sys, glob, random, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel

# ─── Gematria Tokenizer ───────────────────────────────────
GEM_SHIFT = 256
GEM_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:\"-()[]{}<>/@#%^&*_~+=|\t\n0123456789"

def gematria_encode(text, max_len=500):
    ids = [1]
    for ch in text[:max_len]:
        idx = GEM_CHARS.find(ch)
        ids.append(idx + GEM_SHIFT if idx >= 0 else GEM_SHIFT)
    ids.append(2)
    return ids

def gematria_decode(toks):
    chars = []
    for t in toks:
        if t == 2: break
        if t >= GEM_SHIFT:
            idx = t - GEM_SHIFT
            chars.append(GEM_CHARS[idx] if 0 <= idx < len(GEM_CHARS) else '.')
    return ''.join(chars)

# ─── Config ───────────────────────────────────────────────
device = torch.device("cuda")
cfg = NFNConfig(
    vocab_size=1024, d_model=1280, n_blocks=12, d_ff=5120,
    dropout=0.1, n_levels=3, n_heads=20, max_seq_len=128,
    use_episodic_memory=True, use_causal_graph=True, use_goal_predictor=True,
    use_free_energy=True, use_self_model=True, use_nonlinear_causal=True,
    use_working_memory=True, use_ssm=True, use_mixture_of_depths=True,
    use_multi_token_pred=True, use_predictive_coding=True,
    use_recursive_reasoning=True, use_hyper_net=True,
    use_program_synthesis=True, use_self_consistency=True, use_plan_executor=True,
    moe_n_experts=4, moe_top_k=2, moe_d_ff_per_expert=2560,
)
print(f"Building model...")
model = AGINFNModel(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} params ({n_params/1e9:.2f}B)")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB")

# ─── Dictionary ───────────────────────────────────────────
import nltk
nltk.download('words', quiet=True)
from nltk.corpus import words
word_list = sorted(set(w for w in words.words() if 2 <= len(w) <= 50))
gem_words = [gematria_encode(w, 100) for w in word_list]
print(f"Dictionary: {len(gem_words):,} words encoded")

# ─── Optimizer ────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
scaler = GradScaler()

warmup = 500
cos_cycle = 10000
def lr_schedule(step):
    if step < warmup: return step / warmup
    s = (step - warmup) % cos_cycle
    return 0.5 * (1 + math.cos(math.pi * s / cos_cycle))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_schedule(s) * 0.9 + 0.1)

# ─── Training Loop ────────────────────────────────────────
TOTAL_STEPS = 100000
save_every = 2000
log_every = 100
best_loss = float("inf")
accum_steps = 2
os.makedirs("/workspace/checkpoints", exist_ok=True)

task_names = ["Arith", "Seq", "Prime", "Proof", "Conj", "Gematria"]
task_losses = {i: [] for i in range(6)}
t0 = time.time()

print(f"\n{'='*60}")
print(f"TRAINING 2.59B — {TOTAL_STEPS:,} steps — GPU")
print(f"Checkpoints: /workspace/checkpoints/")
print(f"{'='*60}\n")

model.train()
opt_step = 0

for step in range(1, TOTAL_STEPS + 1):
    task = step % 6

    if task == 0:  # Arithmetic
        a, b = random.randint(1, 100), random.randint(1, 100)
        op = random.choice([("+", a+b), ("*", a*b), ("-", a-b)])
        seq = [a % 1024, ord(op[0]) % 1024, b % 1024, op[1] % 1024]
        x = torch.tensor([seq[:3]], dtype=torch.long, device=device)
        y = torch.full((1, 3), -1, dtype=torch.long, device=device)
        y[0, -1] = seq[3]

    elif task == 1:  # Sequence
        a0, diff = random.randint(1, 30), random.randint(1, 10)
        seq = [a0 + i*diff for i in range(5)]
        x = torch.tensor([seq[:4]], dtype=torch.long, device=device)
        y = torch.full((1, 4), -1, dtype=torch.long, device=device)
        y[0, -1] = seq[4] % 1024

    elif task == 2:  # Primality
        n = random.randint(2, 200)
        is_p = all(n % i != 0 for i in range(2, int(n**0.5)+1))
        x = torch.tensor([[n % 1024]], dtype=torch.long, device=device)
        y = torch.tensor([[1 if is_p else 0]], dtype=torch.long, device=device)

    elif task == 3:  # Operations
        a, b = random.randint(1, 50), random.randint(1, 50)
        r1, r2 = a+b, a*b
        seq = [a % 1024, b % 1024, r1 % 1024]
        x = torch.tensor([seq[:2]], dtype=torch.long, device=device)
        y = torch.full((1, 2), -1, dtype=torch.long, device=device)
        y[0, -1] = seq[2]

    elif task == 4:  # Arithmetic progression
        a0, d = random.randint(1, 30), random.randint(1, 10)
        seq = [(a0 + i*d) % 1024 for i in range(6)]
        x = torch.tensor([seq[:5]], dtype=torch.long, device=device)
        y = torch.full((1, 5), -1, dtype=torch.long, device=device)
        y[0, -1] = seq[5]

    else:  # Gematria text
        tokens = [1]  # BOS
        for _ in range(random.randint(3, 8)):
            w = random.choice(gem_words)
            tokens.extend(w[1:])  # skip BOS
        tokens.append(2)  # EOS
        sl = min(cfg.max_seq_len, len(tokens))
        x = torch.tensor([tokens[:sl]], dtype=torch.long, device=device)
        y = torch.full((1, sl), -1, dtype=torch.long, device=device)
        y[0, :sl-1] = torch.tensor(tokens[1:sl], dtype=torch.long)

    # Forward with mixed precision
    with autocast():
        logits, aux = model(x, targets=y, write_memory=(step % 4 == 0))
        lm_loss = aux.get("lm", torch.tensor(0.0, device=device))
        loss = lm_loss
        # Add auxiliary losses
        for k in ["causal", "goal", "free_energy", "pred", "consistency", "coherence", "ponder"]:
            v = aux.get(k, torch.tensor(0.0, device=device))
            if isinstance(v, torch.Tensor) and v.requires_grad:
                loss = loss + v * 0.001

    if loss.requires_grad and loss.item() > 0:
        loss = loss / accum_steps
        scaler.scale(loss).backward()

        if (step % accum_steps) == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            opt_step += 1

        cl = loss.item() * accum_steps
        task_losses[task].append(cl)
        if cl < best_loss:
            best_loss = cl

    # Log
    if step % log_every == 0:
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        avg_losses = []
        for i in range(6):
            tl = task_losses[i]
            avg_losses.append(sum(tl[-50:]) / max(1, len(tl[-50:])) if tl else 0)

        print(f"[{step:6d}/{TOTAL_STEPS}] "
              f"loss={cl:.3f} best={best_loss:.4f} "
              f"lr={lr:.1e} "
              f"A={avg_losses[0]:.2f} S={avg_losses[1]:.2f} "
              f"P={avg_losses[2]:.2f} O={avg_losses[3]:.2f} "
              f"C={avg_losses[4]:.2f} G={avg_losses[5]:.2f} "
              f"| {elapsed:.0f}s {step/(elapsed+1e-9):.1f}st/s")

    # Save
    if step % save_every == 0:
        path = f"/workspace/checkpoints/nfn_{step}.pt"
        torch.save({
            "step": step, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
        }, path)
        # Also save current as NFNmini
        torch.save({
            "model": model.state_dict(),
            "step": step, "best_loss": best_loss,
        }, "/workspace/checkpoints/NFNmini.pt")

        # Gematria generation test
        model.eval()
        prompt = "The FNN architecture is"
        ids = gematria_encode(prompt, 32)
        xp = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad(), autocast():
            out = model.generate(xp, max_new_tokens=60, temperature=0.9, top_k=40)
        gen = gematria_decode(out[0].tolist()[len(ids):])
        print(f"  GEN [{step}]: {gen}")
        model.train()

        # Save to workspace
        size_mb = os.path.getsize(path) / 1e6
        print(f"  Saved {path} ({size_mb:.0f}MB)")

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE — {TOTAL_STEPS} steps")
print(f"Best loss: {best_loss:.4f}")
print(f"Final model: /workspace/checkpoints/NFNmini.pt")
print(f"{'='*60}")
TRAINEOF

echo "train_gpu.py written."
python3 train_gpu.py
