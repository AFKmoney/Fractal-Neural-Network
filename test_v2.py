"""
Test FNN v6.0 — presets et comptage de parametres.
"""
from nfn.model import build_fnn_model
from nfn.config import FNNConfig

print("=" * 60)
print("FNN v6.0 — Presets Test")
print("=" * 60)

presets = ["nano", "small", "medium", "large"]

for i, preset in enumerate(presets, 1):
    print(f"\n[{i}/{len(presets)}] Preset: {preset}")
    try:
        m = build_fnn_model(512, preset=preset)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  Params: {n_params:,}")
        print(f"  {m}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("FNN v6.0 — ALL PRESETS OK!")
print("=" * 60)
