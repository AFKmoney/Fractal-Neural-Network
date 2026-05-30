"""Mini model trained ONLY on English text (docs/*.md)."""
import os, sys, glob, time
sys.stdout.reconfigure(encoding='utf-8')
import torch
from nfn.agi_model import AGINFNModel
from nfn.config import NFNConfig

class GematriaTokenizer:
    SHIFT = 256
    def __init__(self):
        self.g2t={}; self.t2g={}
        idx = self.SHIFT
        for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
            self.g2t[c]=idx; self.t2g[idx]=c; idx+=1
        for c in ' .,!?;:"-()[]{}<>/@#%^&*_~+=|\t\n0123456789':
            self.g2t[c]=idx; self.t2g[idx]=c; idx+=1
        self.pad=self.g2t[' ']; self.unk=self.g2t[' ']; self.bos=1; self.eos=2
    def encode(self,text,m=500000):
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

# Only English docs
docs = []
for p in glob.glob('docs/*.md'):
    try:
        with open(p,'r',encoding='utf-8') as f: docs.append(f.read())
    except: pass
with open('README.md','r',encoding='utf-8') as f: docs.append(f.read())
corpus = '\n'.join(docs)
all_tokens = tok.encode(corpus, m=200000)
print(f'English corpus: {len(corpus):,} chars -> {len(all_tokens):,} tokens')

cfg = NFNConfig(vocab_size=1024, d_model=64, n_blocks=2, d_ff=256,
    dropout=0.1, n_levels=2, n_heads=2, max_seq_len=128,
    use_episodic_memory=False, use_causal_graph=False,
    use_goal_predictor=False, use_free_energy=False,
    use_self_model=False, use_nonlinear_causal=False,
    moe_n_experts=2, moe_top_k=2, moe_d_ff_per_expert=128,
    use_kuramoto=True, kuramoto_rank=4, kuramoto_steps=2,
    use_rope=True, use_memory=True, memory_slots=32,
    memory_heads=2, rank=4, branching=2)
model = AGINFNModel(cfg).to('cpu')
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

BLOCK=128
T=min(100000, len(all_tokens)-BLOCK-1)
t0=time.time(); best=1e9
print(f'Training: {T} steps (each sees unique window position)')
for step in range(1, T+1):
    pos = (step-1) % (len(all_tokens)-BLOCK-1)
    bx = all_tokens[pos:pos+BLOCK]
    by = all_tokens[pos+1:pos+BLOCK+1]
    x=torch.tensor([bx],dtype=torch.long); y=torch.tensor([by],dtype=torch.long)
    logits, losses = model(x, targets=y, write_memory=True)
    loss = losses.get('total', losses.get('lm', torch.tensor(0.0)))
    if loss.requires_grad and loss.item()>0:
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if loss.item()<best: best=loss.item()
    if step%5000==0:
        el=time.time()-t0
        print(f'[{step:5d}/{T}] lm={losses["lm"]:.3f} best={best:.4f} {el:.0f}s')
        model.eval()
        for p in ['The FNN architecture','The training process','The key insight','FNN is a']:
            ids=tok.encode(p,m=32)
            xp=torch.tensor([ids],dtype=torch.long)
            with torch.no_grad():
                out=model.generate(xp, max_new_tokens=60, temperature=0.9, top_k=30)
            dec=tok.decode(out[0].tolist()[len(ids):])
            print(f'  [{p:25s}] {dec}')
        model.train()
    if step%25000==0:
        torch.save({'model':model.state_dict(),'step':step,'best':best},
            f'checkpoints/gematria_en_{step}.pt')
        print(f'  saved gematria_en_{step}.pt')
print(f'Done: {T} steps, best={best:.4f}')
