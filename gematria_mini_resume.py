"""Resume mini model — train to 100000 steps."""
import os, sys, glob, time
sys.stdout.reconfigure(encoding='utf-8')
import torch
from nfn.agi_model import AGINFNModel
from nfn.config import NFNConfig

class GematriaTokenizer:
    SHIFT = 256
    def __init__(self):
        self.g2t = {}; self.t2g = {}
        idx = self.SHIFT
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
            self.g2t[c]=idx; self.t2g[idx]=c; idx+=1
        for c in " .,!?;:\"-()[]{}<>/\\@#%^&*_~+=|`\t\n0123456789":
            self.g2t[c]=idx; self.t2g[idx]=c; idx+=1
        self.pad=self.g2t[' ']; self.unk=self.g2t[' ']; self.bos=1; self.eos=2
    def encode(self,text,m=500):
        t=[self.bos]
        for ch in text[:m]: t.append(self.g2t.get(ch,self.unk))
        t.append(self.eos); return t
    def decode(self,toks):
        r=[]
        for t in toks:
            if t==self.eos: break
            if t>=self.SHIFT: r.append(self.t2g.get(t,'.'))
        return ''.join(r)

tok=GematriaTokenizer()
texts=[]
for p in glob.glob('docs/*.md')+['README.md']+glob.glob('nfn/*.py')+glob.glob('*.py'):
    try:
        with open(p,'r',encoding='utf-8') as f: texts.append(f.read())
    except: pass
corpus = '\n'.join(texts)
all_tokens = tok.encode(corpus, m=500000)
print(f'Tokens: {len(all_tokens):,}')

cfg = NFNConfig(vocab_size=1024, d_model=64, n_blocks=2, d_ff=256,
    dropout=0.1, n_levels=2, n_heads=2, max_seq_len=128,
    use_episodic_memory=False, use_causal_graph=False,
    use_goal_predictor=False, use_free_energy=False,
    use_self_model=False, use_nonlinear_causal=False,
    moe_n_experts=2, moe_top_k=2, moe_d_ff_per_expert=128,
    use_kuramoto=True, kuramoto_rank=4, kuramoto_steps=2,
    use_rope=True, use_memory=True, memory_slots=32,
    memory_heads=2, rank=4, branching=2)
model = AGINFNModel(cfg)
device = torch.device('cpu'); model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10000, eta_min=1e-5)

ckpt = torch.load('checkpoints/gematria_mini_20000.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'])
opt.load_state_dict(ckpt.get('optimizer', opt.state_dict()))
start = ckpt.get('step', 0); best = ckpt.get('best', 1e9)
print(f'Resume from step {start}, best={best:.4f}')

BLOCK = 128
T = 100000
t0 = time.time()
for step in range(start + 1, T + 1):
    pos = (step - 1) % (len(all_tokens) - BLOCK - 1)
    bx = all_tokens[pos:pos + BLOCK]
    by = all_tokens[pos + 1:pos + BLOCK + 1]
    x = torch.tensor([bx], dtype=torch.long, device=device)
    y = torch.tensor([by], dtype=torch.long, device=device)
    logits, losses = model(x, targets=y, write_memory=True)
    loss = losses.get('total', losses.get('lm', torch.tensor(0.0)))
    if loss.requires_grad and loss.item() > 0:
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if loss.item() < best: best = loss.item()
    if step % 5000 == 0:
        el = time.time() - t0; lr = sched.get_last_lr()[0]
        print(f'[{step:5d}/{T}] lm={losses["lm"]:.3f} best={best:.4f} lr={lr:.1e} {el:.0f}s')
        model.eval()
        for p in ['Who are you?','What is consciousness?']:
            ids = tok.encode(p, m=32)
            xp = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate(xp, max_new_tokens=60, temperature=0.9, top_k=40)
            dec = tok.decode(out[0].tolist()[len(ids):])
            print(f'  [{p:25s}] {dec}')
        model.train()
    if step % 10000 == 0:
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
                     'step':step,'best':best}, f'checkpoints/gematria_mini_{step}.pt')
        print(f'  saved gematria_mini_{step}.pt')
el = time.time() - t0
print(f'Done: {T-start} steps in {el:.0f}s, best={best:.4f}')
