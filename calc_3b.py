"""Test config sizes for 3-4B params."""
import torch
from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel

configs = [
    # d_model, n_blocks, d_ff, moe_experts
    (1024, 12, 4096, 4),
    (1536, 16, 6144, 4),
    (2048, 18, 8192, 4),
    (2048, 24, 8192, 4),
    (2560, 24, 10240, 4),
    (3072, 32, 12288, 4),
]

for d_model, n_blocks, d_ff, moe_n in configs:
    try:
        cfg = NFNConfig(
            vocab_size=1024, d_model=d_model, n_blocks=n_blocks,
            d_ff=d_ff, dropout=0.1, n_levels=3,
            n_heads=max(2, d_model // 64), max_seq_len=128,
            use_episodic_memory=True, use_causal_graph=True,
            use_goal_predictor=True, use_free_energy=True,
            use_self_model=True, use_nonlinear_causal=True,
            use_working_memory=True, use_ssm=True,
            use_mixture_of_depths=True, use_multi_token_pred=True,
            use_predictive_coding=True, use_recursive_reasoning=True,
            use_hyper_net=True, use_program_synthesis=True,
            use_self_consistency=True, use_plan_executor=True,
            moe_n_experts=moe_n, moe_top_k=2,
            moe_d_ff_per_expert=d_model * 2,
        )
        m = AGINFNModel(cfg)
        p = sum(p.numel() for p in m.parameters())
        mem_fp16 = p * 2 / (1024**3)
        mem_adamw = p * 8 / (1024**3)
        print(f'd={d_model} B={n_blocks:2d} | {p:>12,} params ({p/1e9:.2f}B) | fp16:{mem_fp16:.1f}GB + AdamW:{mem_adamw:.1f}GB = {mem_fp16+mem_adamw:.1f}GB')
    except Exception as e:
        print(f'd={d_model} B={n_blocks:2d} | ERROR: {e}')
