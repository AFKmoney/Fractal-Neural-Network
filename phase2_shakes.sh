#!/bin/bash
# Phase 2: Shakespeare + Gutenberg gematria training
# Runs AFTER phase 1 completes

cd /root/FNN
sleep 5

cat > train_phase2.py << 'PYEOF'
import os, sys, random, time, math, pickle
import torch, torch.nn as nn
import torch.nn.functional as F
from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel

GEM_SHIFT = 256
GEM_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:\"-()[]{}<>/@#%^&*_~+=|\\t\\n0123456789"

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

# Load dictionary
import nltk; nltk.download('words', quiet=True)
from nltk.corpus import words as nw
wl = sorted(set(w for w in nw.words() if 2 <= len(w) <= 50))
gw = [gem_encode(w, 100) for w in wl]
print(f"Dict: {len(gw):,} words", flush=True)

# Load Shakespeare corpus
print("Loading Shakespeare corpus...", flush=True)
with open('/workspace/shakespeare_gematria.pkl', 'rb') as f:
    shakes_toks = pickle.load(f)
shakes_t = torch.tensor(shakes_toks, dtype=torch.long, device=device)
print(f"Shakespeare: {len(shakes_toks):,} tokens", flush=True)

# Build model
cfg = NFNConfig(
    vocab_size=1024, d_model=768, n_blocks=8, d_ff=3072,
    dropout=0.1, n_levels=2, n_heads=12, max_seq_len=128,
    use_episodic_memory=False, use_causal_graph=False,
    use_goal_predictor=False, use_free_energy=False,
    use_self_model=False, use_nonlinear_causal=False,
    use_working_memory=False, use_ssm=False,
    use_mixture_of_depths=False, use_multi_token_pred=False,
    use_predictive_coding=False, use_recursive_reasoning=False,
    use_hyper_net=False, use_program_synthesis=False,
    use_self_consistency=False, use_plan_executor=False,
    moe_n_experts=4, moe_top_k=2, moe_d_ff_per_expert=1536,
)
model = AGINFNModel(cfg).to(device)

# Load checkpoint
ckpt_path = "/workspace/checkpoints/nfn_20000.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    start_best = ckpt.get('best_loss', float('inf'))
    print(f"Resumed from {ckpt_path}, best_loss={start_best:.4f}", flush=True)
else:
    print("WARNING: no checkpoint, starting fresh", flush=True)
    start_best = float('inf')

n = sum(p.numel() for p in model.parameters())
print(f"Model: {n:,} params ({n/1e9:.2f}B)", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9,0.95), weight_decay=0.005)
STEPS = 50000
save_every, log_every = 2000, 100
os.makedirs("/workspace/checkpoints", exist_ok=True)

tl = {i: [] for i in range(6)}
best = start_best
t0 = time.time()
model.train()

# Prepend BOS to Shakespeare for text chunk sampling
ST = len(shakes_t)

print(f"\n{'='*50}\nPHASE 2: Shakespeare Gematria — {STEPS} steps\n{'='*50}", flush=True)

for step in range(1, STEPS + 1):
    task = step % 6

    if task == 0:  # Arithmetic
        a,b = random.randint(1,100), random.randint(1,100)
        op = random.choice([("+", a+b), ("*", a*b)])
        s = [a&1023, ord(op[0])&1023, b&1023, op[1]&1023]
        x = torch.tensor([s[:3]], dtype=torch.long, device=device)
        y = torch.full((1,3), -1, dtype=torch.long, device=device)
        y[0,-1] = s[3]
    elif task == 1:  # Sequence
        a0,d = random.randint(1,30), random.randint(1,10)
        s = [(a0+i*d)&1023 for i in range(5)]
        x = torch.tensor([s[:4]], dtype=torch.long, device=device)
        y = torch.full((1,4), -1, dtype=torch.long, device=device)
        y[0,-1] = s[4]
    elif task == 2:  # Primality
        n = random.randint(2,200)
        is_p = all(n%i for i in range(2,int(n**0.5)+1))
        x = torch.tensor([[n&1023]], dtype=torch.long, device=device)
        y = torch.tensor([[1 if is_p else 0]], dtype=torch.long, device=device)
    elif task == 3:  # Dict words
        tokens = [1]
        for _ in range(random.randint(4,10)):
            w = random.choice(gw); tokens.extend(w[1:])
        tokens.append(2)
        sl = min(cfg.max_seq_len, len(tokens))
        x = torch.tensor([tokens[:sl]], dtype=torch.long, device=device)
        y = torch.full((1,sl), -1, dtype=torch.long, device=device)
        y[0,:sl-1] = torch.tensor(tokens[1:sl], dtype=torch.long)
    elif task == 4:  # Progression
        a0,d = random.randint(1,30), random.randint(1,10)
        s = [(a0+i*d)&1023 for i in range(6)]
        x = torch.tensor([s[:5]], dtype=torch.long, device=device)
        y = torch.full((1,5), -1, dtype=torch.long, device=device)
        y[0,-1] = s[5]
    else:  # Shakespeare text chunks
        sl = min(cfg.max_seq_len, ST-2)
        start = random.randint(0, ST - sl - 1)
        chunk = shakes_t[start:start+sl+1]
        x = chunk[:sl].unsqueeze(0)
        y = torch.full((1,sl), -1, dtype=torch.long, device=device)
        y[0,:sl-1] = chunk[1:sl]

    logits, aux = model(x, targets=y, write_memory=(step%4==0))
    loss = aux.get("lm", torch.tensor(0.0, device=device))
    if isinstance(loss, torch.Tensor) and loss.requires_grad and loss.item() > 0 and not torch.isnan(loss):
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        cl = loss.item()
        tl[task].append(cl)
        if cl < best: best = cl

    if step < 1000:
        lr = 2e-4 * max(0.1, step/1000)
    else:
        cs = (step-1000) % 10000
        lr = 2e-4 * (0.5*(1+math.cos(math.pi*cs/10000))*0.9+0.1)
    for pg in opt.param_groups: pg['lr'] = lr

    if step % log_every == 0:
        e = time.time()-t0
        av = [sum(tl[i][-50:])/max(1,len(tl[i][-50:])) if tl[i] else 0 for i in range(6)]
        print(f"[{step:6d}] loss={cl:.3f} best={best:.4f} lr={lr:.2e} "
              f"A={av[0]:.2f} S={av[1]:.2f} P={av[2]:.2f} D={av[3]:.2f} C={av[4]:.2f} SH={av[5]:.2f} "
              f"| {e:.0f}s {step/(e+1e-9):.1f}s/s", flush=True)

    if step % save_every == 0:
        p = f"/workspace/checkpoints/shakes_{step}.pt"
        torch.save({"step":step,"model":model.state_dict(),"best_loss":best}, p)
        torch.save({"model":model.state_dict(),"step":step}, "/workspace/checkpoints/NFNmini_shakes.pt")
        model.eval()
        for prompt_text in ["To be or not", "The king", "My lord"]:
            ids = gem_encode(prompt_text, 32)
            xp = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate(xp, max_new_tokens=60, temperature=0.9, top_k=40)
            gen = gem_decode(out[0].tolist()[len(ids):])
            print(f"  [{prompt_text}] {gen}", flush=True)
        model.train()

print(f"\nDONE — Best={best:.4f}", flush=True)
PYEOF

echo "Phase 2 script ready."
