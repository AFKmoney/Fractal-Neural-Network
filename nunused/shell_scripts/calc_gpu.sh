#!/bin/bash
cd /root/FNN
python3 -c "
import torch, sys
sys.path.insert(0, '.')
from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel

for d_model, n_blocks in [(1024,12),(1280,12),(1536,10),(1280,14)]:
    try:
        cfg = NFNConfig(
            vocab_size=1024, d_model=d_model, n_blocks=n_blocks,
            d_ff=d_model*4, dropout=0.1, n_levels=3,
            n_heads=max(2,d_model//64), max_seq_len=128,
            use_episodic_memory=True, use_causal_graph=True, use_goal_predictor=True,
            use_free_energy=True, use_self_model=True, use_nonlinear_causal=True,
            use_working_memory=True, use_ssm=True, use_mixture_of_depths=True,
            use_multi_token_pred=True, use_predictive_coding=True,
            use_recursive_reasoning=True, use_hyper_net=True,
            use_program_synthesis=True, use_self_consistency=True, use_plan_executor=True,
            moe_n_experts=4, moe_top_k=2, moe_d_ff_per_expert=d_model*2,
        )
        m = AGINFNModel(cfg)
        p = sum(p.numel() for p in m.parameters())
        mem_gb = p * 10 / (1024**3)  # fp16 model + fp32 AdamW
        print(f'd={d_model:4d} B={n_blocks:2d} | {p/1e9:.2f}B params | ~{mem_gb:.1f}GB train')
        del m, cfg
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'd={d_model:4d} B={n_blocks:2d} | ERROR: {str(e)[:80]}')
        torch.cuda.empty_cache()
"
