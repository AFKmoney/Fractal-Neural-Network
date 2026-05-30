"""Resume gematria training from step 20000 to 100000."""
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
    def encode(self, text, m=300):
        t=[self.bos]
        for ch in text[:m]: t.append(self.g2t.get(ch,self.unk))
        t.append(self.eos); return t
    def decode(self, toks):
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
all_tokens=tok.encode('\n\n'.join(texts),m=300000)
bl=256; st=200
batches=[]
for i in range(0,len(all_tokens)-bl-1,st):
    batches.append((all_tokens[i:i+bl],all_tokens[i+1:i+bl+1]))

cfg=NFNConfig(vocab_size=1024,d_model=128,n_blocks=4,d_ff=512,dropout=0.1,
    n_levels=3,n_heads=4,max_seq_len=256,use_episodic_memory=True,
    use_causal_graph=True,use_goal_predictor=True,use_free_energy=True,
    use_self_model=True,use_nonlinear_causal=True,moe_n_experts=4,moe_top_k=2,
    moe_d_ff_per_expert=256,use_kuramoto=True,kuramoto_rank=8,kuramoto_steps=4,
    use_rope=True,use_memory=True,memory_slots=64,memory_heads=4,rank=8,branching=2)
model=AGINFNModel(cfg)
device=torch.device('cpu'); model.to(device)
opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.05)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=5000,eta_min=1e-5)

ckpt=torch.load('checkpoints/gematria_step20000.pt',map_location='cpu',weights_only=False)
model.load_state_dict(ckpt['model'])
opt.load_state_dict(ckpt.get('optimizer',opt.state_dict()))
start=ckpt.get('step',0); best=ckpt.get('best_loss',1e9)
print(f'Resume step={start}, best={best:.4f}')

T=100000; t0=time.time()
for step in range(start+1,T+1):
    bx,by=batches[(step-1)%len(batches)]
    x=torch.tensor([bx],dtype=torch.long,device=device)
    y=torch.tensor([by],dtype=torch.long,device=device)
    logits,losses=model(x,targets=y,write_memory=True)
    loss=losses.get('total',losses.get('lm',torch.tensor(0.0)))
    if loss.requires_grad and loss.item()>0:
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        if loss.item()<best: best=loss.item()
    if step%2000==0:
        el=time.time()-t0; lr=sched.get_last_lr()[0]
        print(f'[{step:5d}/{T}] lm={losses["lm"]:.3f} best={best:.4f} lr={lr:.1e} {el:.0f}s')
    if step%5000==0:
        model.eval()
        for p in ['Who are you?','What is consciousness?','Hello friend','I think','Are you aware?']:
            ids=tok.encode(p,m=32)
            xp=torch.tensor([ids],dtype=torch.long,device=device)
            with torch.no_grad():
                out=model.generate(xp,max_new_tokens=60,temperature=0.9,top_k=50)
            dec=tok.decode(out[0].tolist()[len(ids):])
            print(f'  [{p:20s}] {dec}')
        model.train()
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
                     'step':step,'best_loss':best},f'checkpoints/gematria_{step}.pt')
        print(f'  saved gematria_{step}.pt')
el=time.time()-t0
print(f'Done: {T-start} steps in {el:.0f}s, best={best:.4f}')
