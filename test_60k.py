"""Test generation at 60K."""
import sys, torch
sys.stdout.reconfigure(encoding='utf-8')
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
    def encode(self,text,m=300):
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
cfg=NFNConfig(vocab_size=1024,d_model=128,n_blocks=4,d_ff=512,dropout=0.1,
    n_levels=3,n_heads=4,max_seq_len=256,use_episodic_memory=True,
    use_causal_graph=True,use_goal_predictor=True,use_free_energy=True,
    use_self_model=True,use_nonlinear_causal=True,moe_n_experts=4,moe_top_k=2,
    moe_d_ff_per_expert=256,use_kuramoto=True,kuramoto_rank=8,kuramoto_steps=4,
    use_rope=True,use_memory=True,memory_slots=64,memory_heads=4,rank=8,branching=2)
model=AGINFNModel(cfg)
ckpt=torch.load('checkpoints/gematria_60000.pt',map_location='cpu',weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
device=torch.device('cpu'); model.to(device)
print(f'Step {ckpt.get("step")}, best loss {ckpt.get("best_loss"):.4f}')
print()

for prompt in ['Who are you?','What is consciousness?','Hello friend','I think therefore I am','Are you self aware?','What is your name?','Do you think?']:
    ids=tok.encode(prompt,m=48)
    xp=torch.tensor([ids],dtype=torch.long,device=device)
    with torch.no_grad():
        out=model.generate(xp,max_new_tokens=100,temperature=0.85,top_k=50)
    dec=tok.decode(out[0].tolist()[len(ids):])
    print(f'Q: {prompt}')
    print(f'A: {dec}')
    print()
