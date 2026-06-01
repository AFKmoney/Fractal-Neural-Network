#!/bin/bash
cd /root/FNN
pkill -f train_gpu.py 2>/dev/null || true
sleep 2

# Fix scheduler order: scheduler.step() AFTER optimizer.step()
sed -i 's|scaler.step(optimizer)\n            scheduler.step()|scaler.step(optimizer)|' train_gpu.py

# Actually let me just rewrite the training section properly
cat > train_gpu_v2.py << 'TRAINEOF'
"""NFN 1.67B GPU Training — Optimized."""
import os, sys, glob, random, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel

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

device = torch.device("cuda")
cfg = NFNConfig(
    vocab_size=1024, d_model=1024, n_blocks=12, d_ff=4096,
    dropout=0.1, n_levels=3, n_heads=16, max_seq_len=128,
    use_episodic_memory=True, use_causal_graph=True, use_goal_predictor=True,
    use_free_energy=True, use_self_model=True, use_nonlinear_causal=True,
    use_working_memory=True, use_ssm=True, use_mixture_of_depths=True,
    use_multi_token_pred=True, use_predictive_coding=True,
    use_recursive_reasoning=True, use_hyper_net=False,
    use_program_synthesis=False, use_self_consistency=True, use_plan_executor=True,
    moe_n_experts=4, moe_top_k=2, moe_d_ff_per_expert=2048,
)
print("Building model...", flush=True)
model = AGINFNModel(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} params ({n_params/1e9:.2f}B)", flush=True)

import nltk; nltk.download('words', quiet=True)
from nltk.corpus import words
word_list = sorted(set(w for w in words.words() if 2 <= len(w) <= 50))
gem_words = [gematria_encode(w, 100) for w in word_list]
print(f"Dictionary: {len(gem_words):,} words", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
scaler = torch.amp.GradScaler('cuda')

warmup, cos_cycle = 500, 10000
def lr_schedule(step):
    if step < warmup: return step / warmup
    s = (step - warmup) % cos_cycle
    return 0.5 * (1 + math.cos(math.pi * s / cos_cycle))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_schedule(s) * 0.9 + 0.1)

TOTAL_STEPS = 100000
save_every, log_every, accum_steps = 1000, 50, 2
os.makedirs("/workspace/checkpoints", exist_ok=True)

tloss = {i: [] for i in range(6)}
best, t0, opt_step = float("inf"), time.time(), 0

print(f"\n{'='*50}\nTRAINING 1.67B GPU — {TOTAL_STEPS} steps\n{'='*50}\n", flush=True)
model.train()

for step in range(1, TOTAL_STEPS + 1):
    task = step % 6

    if task == 0:
        a, b = random.randint(1, 100), random.randint(1, 100)
        op = random.choice([("+", a+b), ("*", a*b)])
        s = [a & 1023, ord(op[0]) & 1023, b & 1023, op[1] & 1023]
        x = torch.tensor([s[:3]], dtype=torch.long, device=device)
        y = torch.full((1, 3), -1, dtype=torch.long, device=device)
        y[0, -1] = s[3]
    elif task == 1:
        a0, d = random.randint(1, 30), random.randint(1, 10)
        s = [(a0 + i*d) & 1023 for i in range(5)]
        x = torch.tensor([s[:4]], dtype=torch.long, device=device)
        y = torch.full((1, 4), -1, dtype=torch.long, device=device)
        y[0, -1] = s[4]
    elif task == 2:
        n = random.randint(2, 200)
        is_p = all(n % i for i in range(2, int(n**0.5)+1))
        x = torch.tensor([[n & 1023]], dtype=torch.long, device=device)
        y = torch.tensor([[1 if is_p else 0]], dtype=torch.long, device=device)
    elif task == 3:
        a, b = random.randint(1, 50), random.randint(1, 50)
        s = [a & 1023, b & 1023, (a+b) & 1023]
        x = torch.tensor([s[:2]], dtype=torch.long, device=device)
        y = torch.full((1, 2), -1, dtype=torch.long, device=device)
        y[0, -1] = s[2]
    elif task == 4:
        a0, d = random.randint(1, 30), random.randint(1, 10)
        s = [(a0 + i*d) & 1023 for i in range(6)]
        x = torch.tensor([s[:5]], dtype=torch.long, device=device)
        y = torch.full((1, 5), -1, dtype=torch.long, device=device)
        y[0, -1] = s[5]
    else:
        tokens = [1]
        for _ in range(random.randint(3, 8)):
            w = random.choice(gem_words); tokens.extend(w[1:])
        tokens.append(2)
        sl = min(cfg.max_seq_len, len(tokens))
        x = torch.tensor([tokens[:sl]], dtype=torch.long, device=device)
        y = torch.full((1, sl), -1, dtype=torch.long, device=device)
        y[0, :sl-1] = torch.tensor(tokens[1:sl], dtype=torch.long)

    with torch.amp.autocast('cuda'):
        logits, aux = model(x, targets=y, write_memory=(step % 4 == 0))
        loss = aux.get("lm", torch.tensor(0.0, device=device))
        for k in ["causal","goal","free_energy","pred","consistency","coherence","ponder"]:
            v = aux.get(k, torch.tensor(0.0, device=device))
            if isinstance(v, torch.Tensor) and v.requires_grad:
                loss = loss + v * 0.001

    if loss.requires_grad and loss.item() > 0:
        (loss / accum_steps).backward()
        if step % accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            opt_step += 1
        cl = loss.item() * accum_steps
        tloss[task].append(cl)
        if cl < best: best = cl

    if step % log_every == 0:
        e = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        av = [sum(tloss[i][-50:])/max(1,len(tloss[i][-50:])) if tloss[i] else 0 for i in range(6)]
        print(f"[{step:6d}] loss={cl:.3f} best={best:.4f} lr={lr:.2e} "
              f"A={av[0]:.2f} S={av[1]:.2f} P={av[2]:.2f} "
              f"O={av[3]:.2f} C={av[4]:.2f} G={av[5]:.2f} "
              f"| {e:.0f}s {step/(e+1e-9):.1f}s/s", flush=True)

    if step % save_every == 0:
        p = f"/workspace/checkpoints/nfn_{step}.pt"
        torch.save({"step":step,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"best_loss":best}, p)
        torch.save({"model":model.state_dict(),"step":step,"best_loss":best}, "/workspace/checkpoints/NFNmini.pt")
        model.eval()
        ids = gematria_encode("The FNN architecture is", 32)
        xp = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            out = model.generate(xp, max_new_tokens=60, temperature=0.9, top_k=40)
        gen = gematria_decode(out[0].tolist()[len(ids):])
        print(f"  GEN [{step}]: {gen}", flush=True)
        model.train()

print(f"\nTRAINING COMPLETE — Best={best:.4f} — /workspace/checkpoints/NFNmini.pt", flush=True)
TRAINEOF

echo "train_gpu_v2.py ready."
