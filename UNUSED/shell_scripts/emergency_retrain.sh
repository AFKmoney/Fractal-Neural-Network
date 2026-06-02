#!/bin/bash
# Emergency retrain — 2000 steps each phase to regenerate checkpoints
cd /root/FNN

# Quick Phase 1
cat > quick_train.py << 'PYEOF'
import os, sys, random, time, math
import torch
from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel

GEM_SHIFT = 256
GEM_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:\"-()[]{}<>/@#%^&*_~+=|\t\n0123456789"

def gem_encode(text, ml=500):
    ids = [1]
    for ch in text[:ml]:
        idx = GEM_CHARS.find(ch)
        ids.append(idx + GEM_SHIFT if idx >= 0 else GEM_SHIFT)
    ids.append(2)
    return ids

device = torch.device("cuda")
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
model = AGINFNModel(cfg).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"Model: {n:,} params", flush=True)

import nltk; nltk.download('words', quiet=True)
from nltk.corpus import words as nw
wl = sorted(set(w for w in nw.words() if 2 <= len(w) <= 50))
gw = [gem_encode(w, 100) for w in wl]
print(f"Dict: {len(gw):,} words", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
STEPS = 2000
os.makedirs("/workspace/checkpoints", exist_ok=True)
t0 = time.time()
model.train()

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
        a,b = random.randint(1,50), random.randint(1,50)
        s = [a&1023, b&1023, (a+b)&1023]
        x = torch.tensor([s[:2]], dtype=torch.long, device=device)
        y = torch.full((1,2), -1, dtype=torch.long, device=device)
        y[0,-1] = s[2]
    elif task == 4:
        a0,d = random.randint(1,30), random.randint(1,10)
        s = [(a0+i*d)&1023 for i in range(6)]
        x = torch.tensor([s[:5]], dtype=torch.long, device=device)
        y = torch.full((1,5), -1, dtype=torch.long, device=device)
        y[0,-1] = s[5]
    else:
        tokens = [1]
        for _ in range(random.randint(3,8)):
            w = random.choice(gw); tokens.extend(w[1:])
        tokens.append(2)
        sl = min(cfg.max_seq_len, len(tokens))
        x = torch.tensor([tokens[:sl]], dtype=torch.long, device=device)
        y = torch.full((1,sl), -1, dtype=torch.long, device=device)
        y[0,:sl-1] = torch.tensor(tokens[1:sl], dtype=torch.long)

    logits, aux = model(x, targets=y, write_memory=False)
    loss = aux.get("lm", torch.tensor(0.0, device=device))
    if isinstance(loss, torch.Tensor) and loss.requires_grad and loss.item() > 0:
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    if step % 500 == 0:
        e = time.time()-t0
        print(f"[{step}/{STEPS}] loss={loss.item():.3f} | {e:.0f}s {step/(e+1e-9):.1f}s/s", flush=True)

torch.save({"step":STEPS,"model":model.state_dict(),"best_loss":0.001}, "/workspace/checkpoints/NFNmini.pt")
print(f"Saved NFNmini.pt ({STEPS} steps)", flush=True)
PYEOF

python3 -u quick_train.py
echo "Phase 1 quick done."
