"""
Test LEAC v2 — Forward Pass + Singularite Divine
"""
import sys
sys.path.insert(0, "C:/Users/PHIL/Desktop/fnn/FNN")

import torch
from nfn.model import build_leac_model

print("=" * 60)
print("LEAC v2 — Forward Pass Test")
print("=" * 60)

# Moteur Ontologique: forward pass
print("\n[1] Moteur Ontologique - Forward Pass")
m = build_leac_model(512, preset="moteur_ontologique", nfmc_n_scales=8)
m.eval()

x = torch.randint(0, 512, (1, 64))
y = torch.randint(0, 512, (1, 64))

with torch.no_grad():
    logits, losses = m(x, targets=y)

print(f"  Logits shape: {logits.shape}")
print(f"  Total loss:   {losses.get('total', losses.get('lm', 0)):.4f}")
loss_keys = [k for k in sorted(losses.keys()) if isinstance(losses[k], torch.Tensor) and losses[k].item() != 0]
print(f"  Non-zero loss components: {loss_keys}")

# Singularite Divine: just check it builds
print("\n[2] Singularite Divine - Build Only")
print("  Building... (this is the 750M param configuration)")
cfg_div = type(m.cfg).singularite_divine()
cfg_div.nfmc_n_scales = 8
cfg_div.vocab_size = 512
# Limit blocks for faster test
cfg_div.n_blocks = 4
m_div = build_leac_model(512, preset="singularite_divine", nfmc_n_scales=8,
                         n_blocks=4)
print(f"  Params (4 blocks): {m_div.param_count()['total']:,}")

# Estimate at 24 blocks
full_params = m_div.param_count()['total'] * (24 // 4)
print(f"  Params (est. 24 blocks): ~{full_params:,}")

print("\n" + "=" * 60)
print("LEAC v2 — Forward Pass OK!")
print("=" * 60)
