import os,random,time,math,pickle
import torch,torch.nn as nn
device=torch.device('cuda')

GEM_SHIFT=256
CH='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:"-()[]{}<>/@#%^&*_~+=|\t\n0123456789'
def ge(t,ml=500):
    ids=[1]
    for ch in t[:ml]:
        idx=CH.find(ch)
        ids.append(idx+GEM_SHIFT if idx>=0 else GEM_SHIFT)
    ids.append(2)
    return ids
def gd(toks):
    chars=[]
    for t in toks:
        if t==2: break
        if t>=GEM_SHIFT:
            idx=t-GEM_SHIFT
            chars.append(CH[idx] if 0<=idx<len(CH) else '.')
    return ''.join(chars)

from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel
cfg=NFNConfig(vocab_size=1024,d_model=768,n_blocks=8,d_ff=3072,dropout=0.1,n_levels=2,n_heads=12,max_seq_len=64,
    use_episodic_memory=False,use_causal_graph=False,use_goal_predictor=False,use_free_energy=False,
    use_self_model=False,use_nonlinear_causal=False,use_working_memory=False,use_ssm=False,
    use_mixture_of_depths=False,use_multi_token_pred=False,use_predictive_coding=False,
    use_recursive_reasoning=False,use_hyper_net=False,use_program_synthesis=False,
    use_self_consistency=False,use_plan_executor=False,
    moe_n_experts=4,moe_top_k=2,moe_d_ff_per_expert=1536)

m=AGINFNModel(cfg).to(device)
ckpt=torch.load('/workspace/checkpoints/NFNmini.pt',map_location=device,weights_only=False)
m.load_state_dict(ckpt['model'])
print(f"Resumed step {ckpt.get('step',0)} best={ckpt.get('best_loss',0):.4f}",flush=True)

import nltk;nltk.download('words',quiet=True)
from nltk.corpus import words as nw
wl=sorted(set(w for w in nw.words() if 2<=len(w)<=50))
gw=[ge(w,100) for w in wl]

with open('/workspace/texts_gematria.pkl','rb') as f:
    data=pickle.load(f)
text_t=torch.tensor(data,dtype=torch.long,device=device)
ST=len(text_t)
print(f"Dict: {len(gw)} words, Text: {ST:,} tokens",flush=True)

opt=torch.optim.AdamW(m.parameters(),lr=2e-4,betas=(0.9,0.95),weight_decay=0.005)
STEPS=50000
os.makedirs('/workspace/checkpoints',exist_ok=True)
SAVE='/workspace/checkpoints/NFNmini_phase2.pt'
tl={i:[] for i in range(6)}
best=float('inf')
t0=time.time()
m.train()

for step in range(1,STEPS+1):
    task=step%6
    if task==0:
        a,b=random.randint(1,100),random.randint(1,100)
        op=random.choice([('+',a+b),('*',a*b)])
        s=[a&1023,ord(op[0])&1023,b&1023,op[1]&1023]
        x=torch.tensor([s[:3]],dtype=torch.long,device=device)
        y=torch.full((1,3),-1,dtype=torch.long,device=device)
        y[0,-1]=s[3]
    elif task==1:
        a0,d=random.randint(1,30),random.randint(1,10)
        s=[(a0+i*d)&1023 for i in range(5)]
        x=torch.tensor([s[:4]],dtype=torch.long,device=device)
        y=torch.full((1,4),-1,dtype=torch.long,device=device)
        y[0,-1]=s[4]
    elif task==2:
        n=random.randint(2,200)
        is_p=all(n%i for i in range(2,int(n**0.5)+1))
        x=torch.tensor([[n&1023]],dtype=torch.long,device=device)
        y=torch.tensor([[1 if is_p else 0]],dtype=torch.long,device=device)
    elif task==3:
        tokens=[1]
        for _ in range(random.randint(4,10)):
            w=random.choice(gw);tokens.extend(w[1:])
        tokens.append(2)
        sl=min(cfg.max_seq_len,len(tokens))
        x=torch.tensor([tokens[:sl]],dtype=torch.long,device=device)
        y=torch.full((1,sl),-1,dtype=torch.long,device=device)
        y[0,:sl-1]=torch.tensor(tokens[1:sl],dtype=torch.long)
    elif task==4:
        a0,d=random.randint(1,30),random.randint(1,10)
        s=[(a0+i*d)&1023 for i in range(6)]
        x=torch.tensor([s[:5]],dtype=torch.long,device=device)
        y=torch.full((1,5),-1,dtype=torch.long,device=device)
        y[0,-1]=s[5]
    else:
        sl=min(cfg.max_seq_len,ST-2)
        start=random.randint(0,ST-sl-1)
        chunk=text_t[start:start+sl+1]
        x=chunk[:sl].unsqueeze(0)
        y=torch.full((1,sl),-1,dtype=torch.long,device=device)
        y[0,:sl-1]=chunk[1:sl]

    logits,aux=m(x,targets=y,write_memory=(step%4==0))
    loss=aux.get('lm',torch.tensor(0.0,device=device))
    if isinstance(loss,torch.Tensor) and loss.requires_grad and loss.item()>0 and not torch.isnan(loss):
        opt.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        cl=loss.item();tl[task].append(cl)
        if cl<best:best=cl

    if step<1000:lr=2e-4*max(0.1,step/1000)
    else:cs=(step-1000)%10000;lr=2e-4*(0.5*(1+math.cos(math.pi*cs/10000))*0.9+0.1)
    for pg in opt.param_groups:pg['lr']=lr

    if step%100==0:
        e=time.time()-t0
        av=[sum(tl[i][-50:])/max(1,len(tl[i][-50:])) if tl[i] else 0 for i in range(6)]
        print(f"[{step:6d}] loss={cl:.3f} best={best:.4f} lr={lr:.2e} A={av[0]:.2f} S={av[1]:.2f} P={av[2]:.2f} D={av[3]:.2f} C={av[4]:.2f} TX={av[5]:.2f} | {e:.0f}s",flush=True)

    if step%2000==0:
        m.eval()
        torch.save({'step':step,'model':m.state_dict(),'best_loss':best},SAVE)
        ids=ge('The FNN is',32)
        xp=torch.tensor([ids],dtype=torch.long,device=device)
        with torch.no_grad():
            out=m.generate(xp,max_new_tokens=40,temperature=0.9,top_k=40)
        gen=gd(out[0].tolist()[len(ids):])
        print(f"  SAVED+GEN [{step}]: {gen}",flush=True)
        m.train()

torch.save({'step':STEPS,'model':m.state_dict(),'best_loss':best},SAVE)
print(f"\nDONE Phase2 best={best:.4f} — {SAVE}",flush=True)
