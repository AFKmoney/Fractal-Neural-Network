"""Test mini model completions on doc prompts."""
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
        for c in ' .,!?;:"-()[]{}<>/\\@#%^&*_~+=|\`\t\n0123456789':
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
cfg = NFNConfig(vocab_size=1024, d_model=64, n_blocks=2, d_ff=256,
    dropout=0.1,n_levels=2,n_heads=2,max_seq_len=128,
    use_episodic_memory=False, use_causal_graph=False, use_goal_predictor=False,
    use_free_energy=False, use_self_model=False, use_nonlinear_causal=False,
    moe_n_experts=2, moe_top_k=2, moe_d_ff_per_expert=128,
    use_kuramoto=True, kuramoto_rank=4, kuramoto_steps=2,
    use_rope=True, use_memory=True, memory_slots=32, memory_heads=2, rank=4, branching=2)
model = AGINFNModel(cfg)
ckpt = torch.load('checkpoints/gematria_mini_40000.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model']); model.eval()
device = torch.device('cpu'); model.to(device)

prompts = [
    'The FNN architecture is',
    'This model implements',
    'The training process',
    'FNN is a',
    'The framework',
    'The key innovation',
]

print('=== Mini Model completions at 40K ===')
for p in prompts:
    ids = tok.encode(p, m=32)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=60, temperature=0.9, top_k=30)
    dec = tok.decode(out[0].tolist()[len(ids):])
    print(f'  [{p:25s}] {dec}')
