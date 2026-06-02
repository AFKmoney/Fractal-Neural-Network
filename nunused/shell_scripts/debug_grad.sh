#!/bin/bash
python3 << 'EOF'
import torch, sys
sys.path.insert(0, '/root/FNN')
from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel

cfg = NFNConfig(vocab_size=1024, d_model=768, n_blocks=8, d_ff=3072, dropout=0.1,
    n_levels=3, n_heads=12, max_seq_len=64,
    use_episodic_memory=False, use_causal_graph=True, use_goal_predictor=True,
    use_free_energy=True, use_self_model=True, use_nonlinear_causal=True,
    use_working_memory=True, use_ssm=True,
    use_multi_token_pred=False, use_predictive_coding=False,
    use_recursive_reasoning=False, use_hyper_net=False,
    use_program_synthesis=False, use_self_consistency=False,
    use_mixture_of_depths=False, use_plan_executor=False,
    moe_n_experts=4, moe_top_k=2, moe_d_ff_per_expert=1536)

m = AGINFNModel(cfg).cuda()
n = sum(p.numel() for p in m.parameters())
print(f'Params: {n:,} ({n/1e9:.2f}B)')

p0 = next(m.parameters()).clone()
x = torch.randint(0, 1024, (1, 32)).cuda()
y = torch.randint(0, 1024, (1, 32)).cuda()

with torch.amp.autocast('cuda'):
    logits, aux = m(x, targets=y, write_memory=False)
    lm = aux.get('lm', torch.tensor(0.0))
    print(f'lm loss: {lm.item() if hasattr(lm,"item") else lm}, requires_grad: {lm.requires_grad if hasattr(lm,"requires_grad") else "N/A"}')
    print(f'Aux keys: {list(aux.keys())}')
    for k,v in aux.items():
        if hasattr(v, 'requires_grad') and v.requires_grad:
            print(f'  {k}: {v.item():.3f} grad=True')
        elif hasattr(v, 'item'):
            print(f'  {k}: {v.item():.3f}')

if lm.requires_grad:
    lm.backward()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    opt.step()
    p1 = next(m.parameters())
    delta = (p0 - p1).abs().max().item()
    print(f'Max param change after 1 step: {delta:.6f}')
else:
    print('ERROR: lm loss has no grad!')
EOF
