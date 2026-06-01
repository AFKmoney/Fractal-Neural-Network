#!/bin/bash
set -e
cd /root/FNN

echo "=== Patching run.py for 2.59B GPU training ==="

# Update config in run.py for 2.59B
python3 << 'PYEOF'
import re

with open('run.py', 'r') as f:
    code = f.read()

# Update defaults
code = code.replace("default=256", "default=1280")
code = code.replace("default=6", "default=12")

# Update n_heads in config (should be d_model/64 = 20)
code = code.replace("n_heads=8,", "n_heads=20,")

# Update max_seq_len for better context
code = code.replace("max_seq_len=64,", "max_seq_len=128,")

# Add mixed precision + gradient checkpointing
# After model.to(device), add autocast and checkpoint
old = "model = AGINFNModel(cfg).to(device)"
new = '''model = AGINFNModel(cfg).to(device)
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, '_agi_blocks'):
        for block in model._agi_blocks:
            if hasattr(block, 'core'):
                block.core.gradient_checkpointing = True'''
code = code.replace(old, new)

# Add GPU-specific imports at top
if 'torch.cuda.amp' not in code:
    code = code.replace('import torch', 'import torch\nfrom torch.cuda.amp import autocast, GradScaler')

# Add scaler and autocast in the gradient step
# Find where weighted_loss is computed and wrap in autocast
scaler_insert = '''
    # Mixed precision scaler
    scaler = GradScaler()'''
code = code.replace("# ── State tracking", scaler_insert + "\n\n    # ── State tracking")

with open('run.py', 'w') as f:
    f.write(code)

print("Config patched. Checking params...")
PYEOF

# Quick param count check
python3 -c "
import torch, sys
sys.path.insert(0, '.')
from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel
cfg = NFNConfig(
    vocab_size=1024, d_model=1280, n_blocks=12, d_ff=5120, dropout=0.1,
    n_levels=3, n_heads=20, max_seq_len=128,
    use_episodic_memory=True, use_causal_graph=True, use_goal_predictor=True,
    use_free_energy=True, use_self_model=True, use_nonlinear_causal=True,
    use_working_memory=True, use_ssm=True, use_mixture_of_depths=True,
    use_multi_token_pred=True, use_predictive_coding=True,
    use_recursive_reasoning=True, use_hyper_net=True,
    use_program_synthesis=True, use_self_consistency=True, use_plan_executor=True,
    moe_n_experts=4, moe_top_k=2, moe_d_ff_per_expert=2560,
)
m = AGINFNModel(cfg)
p = sum(x.numel() for x in m.parameters())
print(f'Model: {p/1e9:.2f}B params')
del m
torch.cuda.empty_cache()
print('Ready for training.')
"
