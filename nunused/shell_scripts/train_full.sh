#!/bin/bash
# FNN Unified Training Pipeline — Phase 1 + Phase 2 merged
# Usage: bash train_full.sh
cd /root/FNN

cat > train_full.py << 'PYEOF'
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

from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel

CFG = dict(
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

# ─── Build or Resume ──────────────────────────────────────
cfg = NFNConfig(**CFG)
model = AGINFNModel(cfg).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"Model: {n:,} params ({n/1e9:.2f}B)", flush=True)

CKPT = "/workspace/checkpoints/NFNmini.pt"
start_step = 0
if os.path.exists(CKPT):
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    start_step = ckpt.get('step', 0)
    print(f"Resumed from step {start_step:,}", flush=True)
else:
    print("Fresh model — starting Phase 1", flush=True)

os.makedirs("/workspace/checkpoints", exist_ok=True)

# ─── Dictionary ───────────────────────────────────────────
import nltk; nltk.download('words', quiet=True)
from nltk.corpus import words as nw
wl = sorted(set(w for w in nw.words() if 2 <= len(w) <= 50))
gw = [gem_encode(w, 100) for w in wl]
print(f"Dict: {len(gw):,} words", flush=True)

# ─── Text Corpus (Shakespeare + Gutenberg) ─────────────────
text_t = None
ST = 0
TEXT_PATH = "/workspace/texts_gematria.pkl"
if os.path.exists(TEXT_PATH):
    with open(TEXT_PATH, 'rb') as f:
        data = pickle.load(f)
    text_t = torch.tensor(data, dtype=torch.long, device=device)
    ST = len(text_t)
    print(f"Text corpus: {ST:,} tokens", flush=True)
else:
    print("No text corpus — downloading...", flush=True)
    try:
        nltk.download('shakespeare', quiet=True)
        nltk.download('gutenberg', quiet=True)
        from nltk.corpus import shakespeare, gutenberg
        all_text = ""
        for f in shakespeare.fileids(): all_text += shakespeare.raw(f)
        for f in gutenberg.fileids(): all_text += gutenberg.raw(f)
        tokens = gem_encode(all_text)
        with open(TEXT_PATH, 'wb') as f: pickle.dump(tokens, f)
        text_t = torch.tensor(tokens, dtype=torch.long, device=device)
        ST = len(text_t)
        print(f"Corpus generated: {ST:,} tokens", flush=True)
    except Exception as e:
        print(f"Corpus download failed: {e}", flush=True)

# ─── Training ─────────────────────────────────────────────
PHASE1_STEPS = 100000
PHASE2_STEPS = 50000
TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
tl = {i: [] for i in range(6)}
best = float("inf")
t0 = time.time()
task_names = ["Arith", "Seq", "Prime", "Dict", "Prog", "Text"]
model.train()

for step in range(start_step + 1, TOTAL_STEPS + 1):
    # Phase detection
    in_phase2 = step > PHASE1_STEPS
    if in_phase2:
        # Phase 2: lower LR, text-heavy
        base_lr = 2e-4
    else:
        base_lr = 3e-4

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
        for _ in range(random.randint(4, 10)):
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
        # Phase 1: dictionary words | Phase 2: Shakespeare text
        if in_phase2 and text_t is not None and ST > cfg.max_seq_len:
            sl = min(cfg.max_seq_len, ST - 2)
            start = random.randint(0, ST - sl - 1)
            chunk = text_t[start:start+sl+1]
            x = chunk[:sl].unsqueeze(0)
            y = torch.full((1, sl), -1, dtype=torch.long, device=device)
            y[0, :sl-1] = chunk[1:sl]
        else:
            tokens = [1]
            for _ in range(random.randint(3, 8)):
                w = random.choice(gw); tokens.extend(w[1:])
            tokens.append(2)
            sl = min(cfg.max_seq_len, len(tokens))
            x = torch.tensor([tokens[:sl]], dtype=torch.long, device=device)
            y = torch.full((1, sl), -1, dtype=torch.long, device=device)
            y[0, :sl-1] = torch.tensor(tokens[1:sl], dtype=torch.long)

    logits, aux = model(x, targets=y, write_memory=(step % 4 == 0))
    loss = aux.get("lm", torch.tensor(0.0, device=device))
    if isinstance(loss, torch.Tensor) and loss.requires_grad and loss.item() > 0 and not torch.isnan(loss):
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        cl = loss.item(); tl[task].append(cl)
        if cl < best: best = cl

    # LR schedule
    warmup = 1000
    if step < warmup:
        lr = base_lr * max(0.1, step / warmup)
    else:
        cs = (step - warmup) % 10000
        lr = base_lr * (0.5 * (1 + math.cos(math.pi * cs / 10000)) * 0.9 + 0.1)
    for pg in opt.param_groups: pg["lr"] = lr

    if step % 100 == 0:
        e = time.time() - t0
        av = [sum(tl[i][-50:]) / max(1, len(tl[i][-50:])) if tl[i] else 0 for i in range(6)]
        ph = "P2" if in_phase2 else "P1"
        print(f"[{step:6d}/{TOTAL_STEPS}] {ph} loss={cl:.3f} best={best:.4f} lr={lr:.2e} "
              f"A={av[0]:.2f} S={av[1]:.2f} P={av[2]:.2f} D={av[3]:.2f} C={av[4]:.2f} TX={av[5]:.2f} "
              f"| {e:.0f}s", flush=True)

    if step % 2000 == 0 or step == PHASE1_STEPS or step == TOTAL_STEPS:
        model.eval()
        torch.save({"step": step, "model": model.state_dict(), "best_loss": best}, CKPT)
        ids = gem_encode("The FNN is", 32)
        xp = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(xp, max_new_tokens=40, temperature=0.9, top_k=40)
        gen = gem_decode(out[0].tolist()[len(ids):])
        saved_as = f"step {step}"
        if step == PHASE1_STEPS: saved_as = "PHASE 1 COMPLETE"
        if step == TOTAL_STEPS: saved_as = "PHASE 2 COMPLETE"
        print(f"  SAVED [{saved_as}] GEN: {gen}", flush=True)
        model.train()

print(f"\n{'='*50}", flush=True)
print(f"TRAINING COMPLETE — {TOTAL_STEPS:,} steps", flush=True)
print(f"Best loss: {best:.4f}", flush=True)
print(f"Model: {CKPT}", flush=True)
print(f"{'='*50}", flush=True)
PYEOF

echo "train_full.py ready — Phase 1+2 unified."
