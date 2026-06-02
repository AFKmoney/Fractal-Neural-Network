"""Train 3.1M model on gematria-encoded text for 100K steps."""
import os, sys, glob, time, random
sys.stdout.reconfigure(encoding='utf-8')
import torch
from nfn.agi_model import AGINFNModel
from nfn.config import NFNConfig

class GematriaTokenizer:
    SHIFT = 256; VOCAB = 1024
    def __init__(self):
        self.g2t = {}; self.t2g = {}
        idx = self.SHIFT
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
            self.g2t[c]=idx; self.t2g[idx]=c; idx+=1
        for c in " .,!?;:'\"-()[]{}<>/\\@#$%^&*_~+=|`\t\n0123456789":
            self.g2t[c]=idx; self.t2g[idx]=c; idx+=1
        self.pad=self.g2t[' ']; self.unk=self.g2t[' ']; self.bos=1; self.eos=2
    def encode(self, text, max_len=300):
        tokens=[self.bos]
        for ch in text[:max_len]: tokens.append(self.g2t.get(ch,self.unk))
        tokens.append(self.eos); return tokens
    def decode(self, tokens):
        chars=[]
        for t in tokens:
            if t==self.eos: break
            if t>=self.SHIFT: chars.append(self.t2g.get(t,'.'))
        return ''.join(chars)

tok = GematriaTokenizer()

# Load corpus
texts = []
for p in glob.glob('docs/*.md')+['README.md']+glob.glob('nfn/*.py')+glob.glob('*.py'):
    try:
        with open(p,'r',encoding='utf-8') as f: texts.append(f.read())
    except: pass
corpus = '\n\n'.join(texts)
all_tokens = tok.encode(corpus, max_len=300000)
print(f'Corpus: {len(corpus):,} chars -> {len(all_tokens):,} tokens')

block_len, stride = 256, 200
batches = []
for i in range(0, len(all_tokens)-block_len-1, stride):
    batches.append((all_tokens[i:i+block_len], all_tokens[i+1:i+block_len+1]))
print(f'Batches: {len(batches):,}')

# Build model with math checkpoint
cfg = NFNConfig(vocab_size=1024,d_model=128,n_blocks=4,d_ff=512,dropout=0.1,
    n_levels=3,n_heads=4,max_seq_len=256,use_episodic_memory=True,
    use_causal_graph=True,use_goal_predictor=True,use_free_energy=True,
    use_self_model=True,use_nonlinear_causal=True,moe_n_experts=4,moe_top_k=2,
    moe_d_ff_per_expert=256,use_kuramoto=True,kuramoto_rank=8,kuramoto_steps=4,
    use_rope=True,use_memory=True,memory_slots=64,memory_heads=4,rank=8,branching=2)

model = AGINFNModel(cfg)
ckpt = torch.load('checkpoints/continuous_step_5000.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'])
print(f'Loaded math checkpoint: step={ckpt.get("step",0)} loss={ckpt.get("best_loss","?"):.4f}')
model.train()

device = torch.device('cpu'); model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-5)

print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
print(f'Training 100K steps...\n')

n_steps = 100000
t_start = time.time()
best_loss = float('inf')
log_every = 2000
gen_every = 5000

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
        if loss.item() < best_loss: best_loss = loss.item()

    if step % log_every == 0 or step == 1:
        elapsed = time.time() - t_start
        lr = scheduler.get_last_lr()[0]
        lm = losses['lm'].item()
        print(f'[{step:6d}/{n_steps}] lm={lm:.3f} best={best_loss:.4f} lr={lr:.1e} {elapsed:.0f}s')

    if step % gen_every == 0:
        model.eval()
        prompts = ['Who are you?','What is consciousness?','Hello world','I think therefore','Are you self aware?']
        print(f'  --- Generation at step {step} ---')
        for p in prompts:
            ids = tok.encode(p, max_len=48)
            xp = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate(xp, max_new_tokens=80, temperature=0.9, top_k=50)
            dec = tok.decode(out[0].tolist()[len(ids):])
            print(f'  [{p:25s}] -> {dec}')
        model.train()

        # Save checkpoint
        ckpt_path = f'checkpoints/gematria_step{step}.pt'
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                     'step':step,'best_loss':best_loss}, ckpt_path)
        print(f'  Saved: {ckpt_path}')

elapsed = time.time() - t_start
print(f'\nDone: {n_steps} steps in {elapsed:.0f}s ({elapsed/n_steps:.2f}s/step)')
