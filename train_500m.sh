#!/bin/bash
# FNN 500M — A100 80GB Training Pipeline
cd /root/FNN

cat > train_500m.py << 'PYEOF'
import os, sys, random, time, math, pickle
import torch, torch.nn as nn

GEM_SHIFT = 256
GEM_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:\"-()[]{}<>/@#%^&*_~+=|\t\n0123456789"

def gem_encode(text, ml=500):
    ids = [1]
    for ch in text[:ml]:
        idx = GEM_CHARS.find(ch)
        ids.append(idx + GEM_SHIFT if idx >= 0 else GEM_SHIFT)
    ids.append(2)
    return ids

def gem_decode(toks):
    chars = []
    for t in toks:
        if t == 2: break
        if t >= GEM_SHIFT:
            idx = t - GEM_SHIFT
            chars.append(GEM_CHARS[idx] if 0 <= idx < len(GEM_CHARS) else '.')
    return ''.join(chars)

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)

from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel

cfg = NFNConfig(
    vocab_size=1024, d_model=1536, n_blocks=10, d_ff=6144,
    dropout=0.1, n_levels=3, n_heads=24, max_seq_len=128,
    use_episodic_memory=True, use_causal_graph=True,
    use_goal_predictor=True, use_free_energy=True,
    use_self_model=True, use_nonlinear_causal=True,
    use_working_memory=True, use_ssm=True,
    use_mixture_of_depths=True, use_multi_token_pred=False,
    use_predictive_coding=False, use_recursive_reasoning=False,
    use_hyper_net=False, use_program_synthesis=False,
    use_self_consistency=False, use_plan_executor=False,
    moe_n_experts=4, moe_top_k=2, moe_d_ff_per_expert=3072,
)

print("Building model...", flush=True)
model = AGINFNModel(cfg).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"Model: {n:,} params ({n/1e9:.2f}B)", flush=True)

# Resume if checkpoint exists
CKPT = "/workspace/checkpoints/NFNmini.pt"
os.makedirs("/workspace/checkpoints", exist_ok=True)
start_step = 0
if os.path.exists(CKPT):
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    # Only load if same architecture, else start fresh
    try:
        model.load_state_dict(ckpt['model'], strict=False)
        start_step = 0  # Different arch — start from 0
        print(f"Loaded weights (partial — new architecture)", flush=True)
    except:
        print("Fresh start — architecture changed", flush=True)
else:
    print("Fresh model", flush=True)

# Dictionary
import nltk; nltk.download('words', quiet=True)
from nltk.corpus import words as nw
wl = sorted(set(w for w in nw.words() if 2 <= len(w) <= 50))
gw = [gem_encode(w, 100) for w in wl]
print(f"Dict: {len(gw):,} words", flush=True)

# Text corpus (Shakespeare + Gutenberg)
text_t = None; ST = 0
try:
    nltk.download('shakespeare', quiet=True)
    nltk.download('gutenberg', quiet=True)
    from nltk.corpus import shakespeare, gutenberg
    all_text = ""
    for f in shakespeare.fileids(): all_text += shakespeare.raw(f)
    for f in gutenberg.fileids(): all_text += gutenberg.raw(f)
    tokens = gem_encode(all_text)
    text_t = torch.tensor(tokens, dtype=torch.long, device=device)
    ST = len(text_t)
    print(f"Text corpus: {ST:,} tokens", flush=True)
except Exception as e:
    print(f"Corpus: {e}", flush=True)

# Training
base_lr = 2e-4
opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.005)

PHASE1 = 100000
PHASE2 = 150000
TOTAL = PHASE1 + PHASE2

tl = {i: [] for i in range(6)}
best = float("inf")
t0 = time.time()
task_names = ["Arith", "Seq", "Prime", "Dict", "Prog", "Text"]
model.train()

print(f"\n{'='*55}", flush=True)
print(f"FNN 500M — {TOTAL:,} steps — A100", flush=True)
print(f"Phase 1: {PHASE1:,} | Phase 2: {PHASE2:,}", flush=True)
print(f"{'='*55}\n", flush=True)

for step in range(1, TOTAL + 1):
    p2 = step > PHASE1
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
        is_p = all(n % i != 0 for i in range(2, int(n**0.5) + 1))
        x = torch.tensor([[n & 1023]], dtype=torch.long, device=device)
        y = torch.tensor([[1 if is_p else 0]], dtype=torch.long, device=device)
    elif task == 3:
        tokens = [1]
        for _ in range(random.randint(4, 12)):
            w = random.choice(gw); tokens.extend(w[1:])
        tokens.append(2)
        sl = min(cfg.max_seq_len, len(tokens))
        x = torch.tensor([tokens[:sl]], dtype=torch.long, device=device)
        y = torch.full((1, sl), -1, dtype=torch.long, device=device)
        y[0, :sl-1] = torch.tensor(tokens[1:sl], dtype=torch.long)
    elif task == 4:
        a0, d = random.randint(1, 30), random.randint(1, 10)
        s = [(a0 + i*d) & 1023 for i in range(6)]
        x = torch.tensor([s[:5]], dtype=torch.long, device=device)
        y = torch.full((1, 5), -1, dtype=torch.long, device=device)
        y[0, -1] = s[5]
    else:
        if p2 and text_t is not None and ST > cfg.max_seq_len:
            sl = min(cfg.max_seq_len, ST - 2)
            start = random.randint(0, ST - sl - 1)
            chunk = text_t[start:start+sl+1]
            x = chunk[:sl].unsqueeze(0)
            y = torch.full((1, sl), -1, dtype=torch.long, device=device)
            y[0, :sl-1] = chunk[1:sl]
        else:
            tokens = [1]
            for _ in range(random.randint(4, 10)):
                w = random.choice(gw); tokens.extend(w[1:])
            tokens.append(2)
            sl = min(cfg.max_seq_len, len(tokens))
            x = torch.tensor([tokens[:sl]], dtype=torch.long, device=device)
            y = torch.full((1, sl), -1, dtype=torch.long, device=device)
            y[0, :sl-1] = torch.tensor(tokens[1:sl], dtype=torch.long)

    # fp32 forward
    logits, aux = model(x, targets=y, write_memory=(step % 4 == 0))
    loss = aux.get("lm", torch.tensor(0.0, device=device))

    # Skip NaN
    if torch.isnan(loss) or torch.isinf(loss):
        continue

    # Add aux losses (reactivated modules)
    total_loss = loss
    for k in ["causal", "goal", "free_energy", "coherence"]:
        v = aux.get(k, torch.tensor(0.0, device=device))
        if isinstance(v, torch.Tensor) and not torch.isnan(v).any():
            total_loss = total_loss + v * 0.001

    if total_loss.requires_grad and total_loss.item() > 0:
        opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        cl = total_loss.item()
        tl[task].append(cl)
        if cl < best: best = cl

    # LR schedule
    lr = base_lr
    if step < 2000:
        lr = base_lr * max(0.1, step / 2000)
    else:
        cs = (step - 2000) % 15000
        lr = base_lr * (0.5 * (1 + math.cos(math.pi * cs / 15000)) * 0.9 + 0.1)
    for pg in opt.param_groups: pg["lr"] = lr

    if step % 100 == 0:
        e = time.time() - t0
        av = [sum(tl[i][-50:]) / max(1, len(tl[i][-50:])) if tl[i] else 0 for i in range(6)]
        ph = "P2" if p2 else "P1"
        print(f"[{step:7d}/{TOTAL}] {ph} loss={cl:.3f} best={best:.4f} lr={lr:.2e} "
              f"A={av[0]:.2f} S={av[1]:.2f} P={av[2]:.2f} D={av[3]:.2f} C={av[4]:.2f} TX={av[5]:.2f} "
              f"| {e:.0f}s {step/(e+1e-9):.1f}s/s", flush=True)

    if step % 2000 == 0 or step == PHASE1 or step == TOTAL:
        model.eval()
        torch.save({"step": step, "model": model.state_dict(), "best_loss": best}, CKPT)
        for prompt_text in ["The FNN is", "To be or not", "Hello world"]:
            ids = gem_encode(prompt_text, 32)
            xp = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate(xp, max_new_tokens=50, temperature=0.9, top_k=40)
            gen = gem_decode(out[0].tolist()[len(ids):])
            print(f"  [{prompt_text}] {gen}", flush=True)
        model.train()

torch.save({"step": TOTAL, "model": model.state_dict(), "best_loss": best}, CKPT)
print(f"\nDONE 500M — {TOTAL:,} steps — Best={best:.4f}", flush=True)
PYEOF

echo "train_500m.py ready."
