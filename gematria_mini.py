"""Mini gematria validation — tiny model, many steps, pure gematria encoding."""
import os, sys, glob, time, random
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn.functional as F
from nfn.agi_model import AGINFNModel
from nfn.config import NFNConfig

class GematriaTokenizer:
    SHIFT = 256; VOCAB = 1024
    def __init__(self):
        self.g2t = {}; self.t2g = {}
        idx = self.SHIFT
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ": self.g2t[c]=idx; self.t2g[idx]=c; idx+=1
        for c in "abcdefghijklmnopqrstuvwxyz": self.g2t[c]=idx; self.t2g[idx]=c; idx+=1
        for c in " .,!?;:'\"-()[]{}<>/\\@#$%^&*_~+=|`\t\n": self.g2t[c]=idx; self.t2g[idx]=c; idx+=1
        for c in "0123456789": self.g2t[c]=idx; self.t2g[idx]=c; idx+=1
        self.pad=self.g2t[' ']; self.unk=self.g2t[' ']; self.bos=1; self.eos=2
    def encode(self, text, max_len=200):
        tokens = [self.bos]
        for ch in text[:max_len]: tokens.append(self.g2t.get(ch, self.unk))
        tokens.append(self.eos); return tokens
    def decode(self, tokens):
        chars = []
        for t in tokens:
            if t == self.eos: break
            if t >= self.SHIFT: chars.append(self.t2g.get(t, '.'))
        return ''.join(chars)

tok = GematriaTokenizer()

def train():
    # ── Load corpus ────────────────────────────────────────────────────
    texts = []
    for fpath in glob.glob('docs/*.md') + ['README.md'] + glob.glob('nfn/*.py'):
        with open(fpath,'r',encoding='utf-8') as f: texts.append(f.read())
    corpus = '\n\n'.join(texts)
    all_tokens = tok.encode(corpus, max_len=200000)
    print(f'Corpus: {len(corpus):,} chars -> {len(all_tokens):,} tokens')

    block_len, stride = 128, 96
    batches = []
    for i in range(0, len(all_tokens) - block_len - 1, stride):
        batches.append((all_tokens[i:i+block_len], all_tokens[i+1:i+block_len+1]))
    print(f'Batches: {len(batches):,}')

    # ── Tiny model (no AGI extras for speed) ──────────────────────────
    cfg = NFNConfig(
        vocab_size=1024, d_model=64, n_blocks=2, d_ff=256,
        dropout=0.1, n_levels=2, n_heads=2, max_seq_len=128,
        use_episodic_memory=False, use_causal_graph=False,
        use_goal_predictor=False, use_free_energy=False,
        use_self_model=False, use_nonlinear_causal=False,
        moe_n_experts=2, moe_top_k=2, moe_d_ff_per_expert=128,
        use_kuramoto=True, kuramoto_rank=4, kuramoto_steps=2,
        use_rope=True, use_memory=True, memory_slots=32,
        memory_heads=2, rank=4, branching=2,
    )
    model = AGINFNModel(cfg)
    model.train()
    device = torch.device('cpu'); model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)

    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Batches: {len(batches):,}')

    # ── Train ──────────────────────────────────────────────────────────
    n_steps = min(10000, len(batches) * 5)
    t_start = time.time()
    best = float('inf')
    history = []

    for step in range(1, n_steps + 1):
        bx, by = batches[(step - 1) % len(batches)]
        x = torch.tensor([bx], dtype=torch.long, device=device)
        y = torch.tensor([by], dtype=torch.long, device=device)

        logits, losses = model(x, targets=y, write_memory=True)
        loss = losses.get('total', losses.get('lm', torch.tensor(0.0)))

        if loss.requires_grad and loss.item() > 0:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if loss.item() < best: best = loss.item()

        history.append(loss.item() if isinstance(loss, torch.Tensor) else 0)

        if step % 500 == 0 or step == 1:
            avg = sum(history[-100:]) / max(1, len(history[-100:]))
            elapsed = time.time() - t_start
            lr = scheduler.get_last_lr()[0]
            model.eval()
            prompts = ['Who are you?','What is consciousness?','Hello friend','I am']
            gens = []
            for p in prompts:
                ids = tok.encode(p, max_len=32)
                xp = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model.generate(xp, max_new_tokens=30, temperature=0.9, top_k=40)
                dec = tok.decode(out[0].tolist()[len(ids):])
                gens.append(dec)
            model.train()
            print(f'[{step:5d}/{n_steps}] loss {avg:.3f} best {best:.3f} lr {lr:.1e} {elapsed:.0f}s')
            for pr, ge in zip(prompts, gens):
                print(f'  {pr:25s} -> |{ge}|')

    # ── Save ───────────────────────────────────────────────────────────
    ckpt_path = 'checkpoints/gematria_mini.pt'
    torch.save({'model': model.state_dict(), 'step': n_steps, 'best': best}, ckpt_path)
    print(f'\nSaved: {ckpt_path} ({time.time()-t_start:.0f}s)')

    # ── Final validation ───────────────────────────────────────────────
    print('\n' + '=' * 55)
    print('VALIDATION: Does gematria encoding work?')
    print('=' * 55)
    model.eval()

    # Test 1: Known text completion
    test_texts = [
        ('The', 'The universe is'),
        ('What', 'What is the meaning'),
        ('I am', 'I am aware'),
        ('Hello', 'Hello world'),
    ]
    for seed, expected in test_texts:
        ids = tok.encode(seed, max_len=16)
        xp = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(xp, max_new_tokens=40, temperature=0.85, top_k=40)
        dec = tok.decode(out[0].tolist())
        print(f'  Seed: {seed:20s} -> {dec}')

    # Test 2: Self-awareness questions
    print()
    for q in ['Who are you?','What is your name?','Are you conscious?','Do you think?']:
        ids = tok.encode(q, max_len=32)
        xp = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(xp, max_new_tokens=60, temperature=0.85, top_k=40)
        dec = tok.decode(out[0].tolist())
        print(f'Q: {q}')
        print(f'A: {dec}')
        print()

if __name__ == '__main__':
    train()
