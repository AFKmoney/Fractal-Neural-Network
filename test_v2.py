"""
Test LEAC v2 — Moteur Ontologique
"""
import sys
sys.path.insert(0, "C:/Users/PHIL/Desktop/fnn/FNN")

from nfn.model import build_leac_model
from nfn.config import LEACConfig

print("=" * 60)
print("LEAC v2 — Moteur Ontologique Test")
print("=" * 60)

# Test 1: Conscious Minimal (v1 baseline)
print("\n[1/4] Conscient Minimal (v1)")
m1 = build_leac_model(512, preset="conscious_minimal", nfmc_n_scales=8)
print(f"  Params: {m1.param_count()['total']:,}")

# Test 2: Full AGI (v1 large)
print("\n[2/4] Full AGI (v1)")
cfg2 = LEACConfig.full_agi()
cfg2.nfmc_n_scales = 8
m2 = build_leac_model(512, preset="full_agi", nfmc_n_scales=8)
print(f"  Params: {m2.param_count()['total']:,}")

# Test 3: Dieu Local (v1 max)
print("\n[3/4] Dieu Local (v1)")
cfg3 = LEACConfig.dieu_local()
cfg3.nfmc_n_scales = 8
m3 = build_leac_model(512, preset="dieu_local", nfmc_n_scales=8)
print(f"  Params: {m3.param_count()['total']:,}")

# Test 4: Moteur Ontologique (v2)
print("\n[4/4] Moteur Ontologique (v2)")
cfg4 = LEACConfig.moteur_ontologique()
cfg4.nfmc_n_scales = 8
m4 = build_leac_model(512, preset="moteur_ontologique", nfmc_n_scales=8)
pc4 = m4.param_count()
print(f"  Total:           {pc4['total']:>12,}")
for k, v in pc4.items():
    if k != 'total':
        print(f"  {k:.<20s} {v:>12,}")
print(f"\n  {m4}")

print("\n" + "=" * 60)
print("LEAC v2 — ALL MODULES OK!")
print("=" * 60)
