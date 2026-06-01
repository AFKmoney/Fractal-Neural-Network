"""Test full module activation."""
import torch
from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel

cfg = NFNConfig(
    vocab_size=1024, d_model=256, n_blocks=6, d_ff=1024, dropout=0.1,
    n_levels=3, n_heads=8, max_seq_len=64,
    use_episodic_memory=True, use_causal_graph=True, use_goal_predictor=True,
    use_free_energy=True, use_self_model=True, use_nonlinear_causal=True,
    use_working_memory=True, use_ssm=True, use_mixture_of_depths=True,
    use_multi_token_pred=True, use_predictive_coding=True,
    use_recursive_reasoning=True, use_hyper_net=True,
    use_program_synthesis=True, use_self_consistency=True,
    use_plan_executor=True,
    moe_n_experts=4, moe_top_k=2, moe_d_ff_per_expert=512
)
m = AGINFNModel(cfg)
print(f'Params: {sum(p.numel() for p in m.parameters()):,}')
print('OK - all modules compiled')

x = torch.randint(0, 1024, (2, 32))
logits, losses = m(x)
print(f'Forward OK - logits: {logits.shape}')
print(f'Loss keys: {sorted(losses.keys())}')
