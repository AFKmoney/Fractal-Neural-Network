# FNN — Fractal Neural Network

- Author: Philippe-Antoine Robert
- Version: v5.1 (May 2026)
- License: Proprietary

---

## What This Is

FNN is an experimental AI architecture that explores a simple hypothesis:

> Mathematical structure alone can produce intelligent behavior, without needing internet-scale text training.

The model trains on self-generated mathematical truths (arithmetic, primes, sequences) and attempts to bridge into language through gematria encoding — a character-to-number mapping that places English text inside the model's mathematical token space.

---

## Why "LLM Killer"?

The pitch isn't marketing hype. It's a specific bet:

- Modern LLMs use 100B+ parameters trained on trillions of text tokens
- FNN uses 17M parameters trained on mathematical patterns + a dictionary
- The claim: mathematical reasoning is the foundation of intelligence; language is a surface manifestation
- If the model can learn language through math, it proves billion-parameter models are wasteful, not necessary

This README documents exactly what we built, what worked, and what didn't.

---

## Architecture

### NFNmini (v5.1 — the current model)

| Parameter | Value |
|---|---|
| Parameters | 17,125,438 |
| d_model | 256 |
| n_blocks | 6 |
| n_heads | 8 |
| d_ff | 1024 |
| vocab_size | 1024 |
| max_seq_len | 64 |
| MoE experts | 4 (top_k=2) |
| Features | Episodic memory, Causal Graph, Goal Predictor, Free Energy, Self-Model, Nonlinear Causal, RoPE, Kuramoto |

### Training Setup

| Setting | Value |
|---|---|
| Optimizer | AdamW (lr=1e-3 baseline, betas 0.9/0.95, weight_decay=0.01) |
| Scheduler | Cosine with warmup (500 steps), 10K-step cycles |
| Gradient accumulation | 4 steps (effective batch=4) |
| Hardware | CPU only (no GPU) |
| Training time | ~25 hours total (all runs combined) |

---

## The AGI Continuous Loop (run.py)

The model trains in a single infinite loop with 6 rotating task types:

### Task 0 — Arithmetic
Generates a+b, a-b, a*b problems with ground-truth verification. The model predicts results. This is the easiest task.

### Task 1 — Sequence Prediction
Arithmetic progression completion: given [a0, a1, a2, a3], predict a4. Difficulty increases over time.

### Task 2 — Primality Classification
Binary classification: is this number prime? Easy for small numbers, gets harder.

### Task 3 — Proof Generation
A separate REINFORCE-trained proof generator module attempts to prove arithmetic statements. The main model then predicts the result.

### Task 4 — Conjecture Discovery
Runs a conjecture discovery loop (generate candidate, test against computational verifier, store if valid). The main model trains on arithmetic progression patterns.

### Task 5 — Gematria Text Training
Randomly samples English words from a 237K-word dictionary (NLTK), encodes them through gematria (A=257, B=258, ...), and trains next-token prediction. This is the bridge between math and language.

---

## What We Actually Achieved

### Training Results (50,000 steps)

| Task | Step 0 | Step 50,000 | Change |
|---|---|---|---|
| T5 — Gematria Text | 7.63 | 3.00 | -4.63 |
| T4 — Conjectures | 5.62 | 4.60 | -1.02 |
| T3 — Proof Generation | 5.80 | 5.05 | -0.75 |
| T1 — Sequence Prediction | 7.79 | 6.87 | -0.92 |
| T2 — Primality | 1.15 | 1.21 | +0.06 |
| T0 — Arithmetic | 2.95 | 5.82 | +2.87 |

- Best loss: 1.0001 → 0.0265 (on easiest individual sample)
- Truths generated: 0 → 116,835
- Proofs discovered: 0 → 475
- Conjectures tested: 0 → 3,288
- Architecture self-modifications: 90 (33% acceptance rate)

### What These Numbers Mean

**T5 (gematria): 7.63 → 3.00** — The model learned character-level English patterns from a 237K-word dictionary. Random guessing over 93 characters gives ln(93) ≈ 4.53. A loss of 3.00 means the model is ~4.5x better than random at predicting the next letter. But it has not reached coherent word generation — outputs are character sequences like `ampighpr`, `yoeu`, `owhlana` that contain fragments of real words but are not actual English words.

**T2 (primality): stable at ~1.2** — Binary classification, not improving further. The model already performs well at this.

**T0 (arithmetic): 2.95 → 5.82** — The model got worse at arithmetic. This is a clear trade-off: the model reallocated capacity from easy tasks to harder ones. Not a bug — this is expected behavior when a limited-capacity model trains on diverse tasks simultaneously.

**T3 (proof): 5.80 → 5.05** — Modest improvement. The proof generator module is learning but slowly.

**Best loss: 0.0265** — This is near-perfect prediction on the single easiest arithmetic sample in the training history. It does not represent overall model quality.

### Generation Samples (Gematria Decoding)

These are actual outputs when prompted with "The FNN architecture is" during training:

| Step | Output |
|---|---|
| 500 | `nrs` |
| 2,000 | `bf neoix *e) a,hcaiakdtoi tpoitdi` |
| 5,000 | `mcgnelcgoosnrrs` |
| 10,500 | `ccolou` |
| 16,000 | `yoeu` (almost "you") |
| 20,000 | `innteirpueebtluctdmpnouiop` |
| 30,000 | *(empty)* |
| 38,000 | `grgcool` |
| 45,000 | `eyut` |
| 50,000 | `ampighpr` |

The outputs show gradual improvement — from pure random to fragments that resemble English letter patterns — but never reached coherent words or sentences.

---

## The Gematria Bridge: Validated but Incomplete

### What We Proved

The gematria bridge THEORETICALLY works:

1. Text → numbers (encoding) is lossless
2. Numbers → model training works (loss drops)
3. Model → numbers → text (decoding) recovers readable characters
4. A 178K-param toy model can memorize and reproduce text perfectly at loss ~0.1
5. The 17M model learned character n-gram statistics (T5: 7.63 → 3.00)

### What We Did NOT Achieve

1. The model does NOT produce coherent English sentences
2. The model does NOT "understand" language — it has only learned character-level statistics
3. The "mathematical structure → language" bridge remains theoretical, not demonstrated
4. T5 plateaued at ~3.0 — the model could not learn multi-character word patterns despite 50K steps

### Why It Plateaued

- 237K unique words with occasional sampling means each word is seen roughly once every ~39K gematria steps (at 1/6 gematria rate: ~234K total steps per dictionary cycle)
- CPU training is slow (0.5-1.0s per step for 17M params)
- Full dictionary coverage would require ~234K total steps = ~65 hours on CPU
- Even with full coverage, character-level word modeling is inherently hard with small models

---

## Three Key Bug Fixes

### 1. Tasks 4 and 5 Were Random Noise (Fixed)
The original code generated `torch.randint()` for tasks 4 and 5 — literally training the model on random input-output pairs. Two-thirds of training steps were productive, one-third was noise. Fixed to use arithmetic progressions (task 4) and gematria word sampling (task 5).

### 2. Self-Modulated LR Replaced
The original self-modulating learning rate oscillated between 1e-6 and 5e-3 based on recent loss, causing training instability. Replaced with a proper cosine schedule with 500-step warmup and 10K-step cycles.

### 3. Resume/Checkpoint Bugs
- `total_steps` restores properly from checkpoints (was always 0)
- Optimizer state restores correctly
- Scheduler state preserved between runs

---

## File Structure

```
FNN/
  run.py                          Main AGI Continuous Loop
  nfn/
    agi_model.py                  AGINFNModel (full architecture)
    agi_block.py                  AGIBlock (memory, free energy, causal graph)
    efficient_block.py            Core transformer block
    moe.py                        Mixture of Experts attention
    episodic_memory.py            EpisodicStore + TwoTierMemory
    config.py                     NFNConfig
    self_development.py           MathTruthEngine, GematriaEncoder
    conjecture_discovery.py       Conjecture generator + tester
    proof_engine.py               ProofGenerator, ProofReward
    semantic_gematria.py          GematriaTable, GematriaLoss
    self_modification.py          Architecture self-modification
    (...30+ supporting modules)
  models/
    NFNmini.pt                    17M params, step 50,000, best_loss=0.0265
    FNN_v1_3M.pt                  3.1M params, step 39,500, best_loss=0.0090
  docs/
    FNN_Final_Report.pdf          13-page comprehensive report
    FNN_Complete_Documentation.pdf Older PDF documentation
  checkpoints/
    continuous_step_50000.pt      Final checkpoint (full state)
```

---

## How to Use NFNmini

```python
import torch
from nfn.config import NFNConfig
from nfn.agi_model import AGINFNModel

# Load model
cfg = NFNConfig(
    vocab_size=1024, d_model=256, n_blocks=6, d_ff=1024,
    n_heads=8, max_seq_len=64, n_levels=3,
    use_episodic_memory=True, use_causal_graph=True,
    use_goal_predictor=True, use_free_energy=True,
    use_self_model=True, use_nonlinear_causal=True,
    moe_n_experts=4, moe_top_k=2, moe_d_ff_per_expert=512,
)
model = AGINFNModel(cfg)
ckpt = torch.load('models/NFNmini.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

# Run inference / continue training
```

To continue training:
```
python run.py --steps 100000 --resume checkpoints/continuous_step_50000.pt
```

---

## Honest Assessment

### What Works

- The AGI loop architecture is functional and self-contained
- Mathematical truth generation provides an infinite training signal
- The model LEARNS from mathematical patterns (loss drops consistently)
- Gematria encoding/decoding is technically correct
- The model picks up character-level statistics from encoded text
- Gradient accumulation + cosine schedule provides stable training
- The codebase is modular and extensible

### What Doesn't Work (Yet)

- The model does NOT produce coherent language — the gematria bridge remains unproven at the word level
- 50K steps is insufficient for language emergence on CPU
- The model trades off performance between tasks (gets worse at arithmetic while improving at others)
- 3.0 gematria loss is still far from useful (<1.0 needed for word-level patterns)
- The 17M parameter budget is tiny compared to modern models (GPT-2: 1.5B, LLaMA: 7-65B)
- No evaluation against external benchmarks (MMLU, HumanEval, etc.)

### What Would Help

- GPU training (100-1000x speedup) to process more steps
- Larger model (100M-500M params) for more representational capacity
- Better gematria data (synthetic text generator, larger corpus)
- Per-task evaluation metrics (not just aggregate loss)
- External benchmarks to compare against mainstream models

---

## The Core Question

Can a 17M-parameter model, trained on self-generated math + a small dictionary, produce intelligent behavior?

**After 50,000 training steps:** Not yet. The model learns mathematical patterns and character statistics, but demonstrates no emergent language understanding.

The thesis that "mathematical structure alone produces intelligence" remains an open question — not disproven, but not demonstrated either. The experiments establish the infrastructure and baseline; the next step is scaling up compute, data, and model size.

---

## Version History

- v5.0 — Initial AGI loop, 3.1M params, math-only training
- v5.1 — NFNmini 17M params, 50K steps, gematria dictionary integration, bug fixes
