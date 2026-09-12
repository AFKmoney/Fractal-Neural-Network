# Quarantine — 2026-09-11

Nothing erased from `main`. This branch documents the split.
No rewrite of `nfn/block.py` / `nfn/moe.py` / `nfn/phase_ode.py`.

## Why

GitHub `main` = version saved after Claude sessions (May 2026).
The FNN block is still fractal-linear + phase + MoE.
The volume around it is graft + dead code.

## Stay on this branch but unused / graft (still in tree until deleted)

- `nfn/prism/` — second architecture
- `interface/` — FastAPI + remote SSH
- `train_agi.py`, `cloud_train.py`, `training/agi_trainer.py`
- `nfn/multimodal.py`, `multi_token_pred.py`, `program_synthesis.py`, `proof_engine.py`, `conjecture_discovery.py`

## Kept on purpose (model.py / causal.py import them)

- `nfn/flash_attn.py` — used by `causal.py` and `self_model.py`
- ads / godel / rg / mera / hyperbolic — imported by `model.py` for preset `large`

## Mini train 2026-09-11

CPU, d=64, 2 blocks, ~111k params, 40 steps, char vocab 110.
loss 4.66 → 2.75. Generation still salad. Smoke test only.
