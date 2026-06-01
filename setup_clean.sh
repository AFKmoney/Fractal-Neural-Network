#!/bin/bash
# Phase 1 — Clean training, saves only 1 model file
cd /root/FNN

cat > train_phase1.py << 'PYEOF'
import os, sys, random, time, math
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

from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel

cfg = NFNConfig(
    vocab_size=1024, d_model=768, n_blocks=8, d_ff=3072,
    dropout=0.1, n_levels=2, n_heads=12, max_seq_len=64,
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

print("Building model...", flush=True)
model = AGINFNModel(cfg).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"Model: {n:,} params ({n/1e9:.2f}B)", flush=True)

import nltk; nltk.download('words', quiet=True)
from nltk.corpus import words as nw
wl = sorted(set(w for w in nw.words() if 2 <= len(w) <= 50))
gw = [gem_encode(w, 100) for w in wl]
print(f"Dict: {len(gw):,} words", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), weight_decay=0.01)

STEPS = 100000
SAVE_MODEL = "/workspace/checkpoints/NFNmini.pt"
os.makedirs("/workspace/checkpoints", exist_ok=True)

tl = {i: [] for i in range(6)}
best, t0 = float("inf"), time.time()
model.train()

# For Phase 2+, load text corpora if available
text_t = None
for fname in ["/workspace/texts_gematria.pkl", "/workspace/code_gematria.pkl"]:
    try:
        import pickle
        with open(fname, 'rb') as f: data = pickle.load(f)
        if text_t is None: text_t = torch.tensor(data, dtype=torch.long, device=device)
        else: text_t = torch.cat([text_t, torch.tensor(data, dtype=torch.long, device=device)])
        print(f"Loaded {fname}: {len(data):,} tokens", flush=True)
    except: pass
if text_t is not None:
    ST = len(text_t)
    print(f"Combined text corpus: {ST:,} tokens", flush=True)

task_names = ["Arith", "Seq", "Prime", "Dict", "Prog", "Text"]

for step in range(1, STEPS + 1):
    task = step % 6

    if task == 0:
        a,b = random.randint(1,100), random.randint(1,100)
        op = random.choice([("+", a+b), ("*", a*b)])
        s = [a&1023, ord(op[0])&1023, b&1023, op[1]&1023]
        x = torch.tensor([s[:3]], dtype=torch.long, device=device)
        y = torch.full((1,3), -1, dtype=torch.long, device=device)
        y[0,-1] = s[3]
    elif task == 1:
        a0,d = random.randint(1,30), random.randint(1,10)
        s = [(a0+i*d)&1023 for i in range(5)]
        x = torch.tensor([s[:4]], dtype=torch.long, device=device)
        y = torch.full((1,4), -1, dtype=torch.long, device=device)
        y[0,-1] = s[4]
    elif task == 2:
        n = random.randint(2,200)
        is_p = all(n%i for i in range(2,int(n**0.5)+1))
        x = torch.tensor([[n&1023]], dtype=torch.long, device=device)
        y = torch.tensor([[1 if is_p else 0]], dtype=torch.long, device=device)
    elif task == 3:
        tokens = [1]
        for _ in range(random.randint(3,8)):
            w = random.choice(gw); tokens.extend(w[1:])
        tokens.append(2)
        sl = min(cfg.max_seq_len, len(tokens))
        x = torch.tensor([tokens[:sl]], dtype=torch.long, device=device)
        y = torch.full((1,sl), -1, dtype=torch.long, device=device)
        y[0,:sl-1] = torch.tensor(tokens[1:sl], dtype=torch.long)
    elif task == 4:
        a0,d = random.randint(1,30), random.randint(1,10)
        s = [(a0+i*d)&1023 for i in range(6)]
        x = torch.tensor([s[:5]], dtype=torch.long, device=device)
        y = torch.full((1,5), -1, dtype=torch.long, device=device)
        y[0,-1] = s[5]
    else:
        if text_t is not None and ST > cfg.max_seq_len:
            sl = min(cfg.max_seq_len, ST-2)
            start = random.randint(0, ST - sl - 1)
            chunk = text_t[start:start+sl+1]
            x = chunk[:sl].unsqueeze(0)
            y = torch.full((1,sl), -1, dtype=torch.long, device=device)
            y[0,:sl-1] = chunk[1:sl]
        else:
            tokens = [1]
            for _ in range(random.randint(3,8)):
                w = random.choice(gw); tokens.extend(w[1:])
            tokens.append(2)
            sl = min(cfg.max_seq_len, len(tokens))
            x = torch.tensor([tokens[:sl]], dtype=torch.long, device=device)
            y = torch.full((1,sl), -1, dtype=torch.long, device=device)
            y[0,:sl-1] = torch.tensor(tokens[1:sl], dtype=torch.long)

    logits, aux = model(x, targets=y, write_memory=(step%4==0))
    loss = aux.get("lm", torch.tensor(0.0, device=device))
    if isinstance(loss, torch.Tensor) and loss.requires_grad and loss.item() > 0 and not torch.isnan(loss):
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        cl = loss.item(); tl[task].append(cl)
        if cl < best: best = cl

    if step < 1000:
        lr = 3e-4 * max(0.1, step/1000)
    else:
        cs = (step-1000) % 10000
        lr = 3e-4 * (0.5*(1+math.cos(math.pi*cs/10000))*0.9+0.1)
    for pg in opt.param_groups: pg['lr'] = lr

    if step % 100 == 0:
        e = time.time()-t0
        av = [sum(tl[i][-50:])/max(1,len(tl[i][-50:])) if tl[i] else 0 for i in range(6)]
        print(f"[{step:6d}] loss={cl:.3f} best={best:.4f} lr={lr:.2e} "
              f"{task_names[0]}={av[0]:.2f} {task_names[1]}={av[1]:.2f} {task_names[2]}={av[2]:.2f} "
              f"{task_names[3]}={av[3]:.2f} {task_names[4]}={av[4]:.2f} {task_names[5]}={av[5]:.2f} "
              f"| {e:.0f}s", flush=True)

    # Save ONE model file every 2000 steps (overwrites)
    if step % 2000 == 0:
        model.eval()
        torch.save({"step":step, "model":model.state_dict(), "best_loss":best}, SAVE_MODEL)
        # Quick gen test
        ids = gem_encode("The FNN is", 32)
        xp = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(xp, max_new_tokens=40, temperature=0.9, top_k=40)
        gen = gem_decode(out[0].tolist()[len(ids):])
        print(f"  SAVED + GEN [{step}]: {gen}", flush=True)
        model.train()

# Final save
torch.save({"step":STEPS, "model":model.state_dict(), "best_loss":best}, SAVE_MODEL)
print(f"\nDONE — {STEPS} steps — Best={best:.4f} — {SAVE_MODEL}", flush=True)
PYEOF

echo "train_phase1.py ready — single file save mode."
