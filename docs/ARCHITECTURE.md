# NFN — Complete Technical Architecture

**Author:** Philippe-Antoine Robert
**Version:** 3.2
**Date:** 2026-05-03 07:22:48 UTC

---

## Table of Contents

1. [Overview](#1-overview)
2. [Parametric Sinusoidal Connections](#2-parametric-sinusoidal-connections)
3. [Fractal Topology](#3-fractal-topology)
4. [Phase Dynamics — Kuramoto ODE](#4-phase-dynamics--kuramoto-ode)
5. [Flash Attention + Long-Context RoPE](#5-flash-attention--long-context-rope)
6. [Fractal KV-Cache](#6-fractal-kv-cache)
7. [Persistent Cross-Context Memory](#7-persistent-cross-context-memory)
8. [NFMC v3.0 — Condensed Fractal Kernel](#8-nfmc-v30--condensed-fractal-kernel)
9. [v3.1 — ZeroShotNFMC](#9-v31--zeroshotnfmc)
10. [v3.2 — EfficientNFN](#10-v32--efficientnfn)
11. [BPTP Training](#11-bptp-training)
12. [Three-Tier Tokenizer](#12-three-tier-tokenizer)
13. [Multi-GPU DDP / FSDP](#13-multi-gpu-ddp--fsdp)

---

## 1. Overview

NFN is a causal language model whose architecture rests on four fundamental principles absent from standard transformers:

| Principle | Implementation | File |
|-----------|----------------|------|
| Fractal self-similarity | `MotifBranch` with `SinusoidalAggregator × K` | `network.py`, `connections.py` |
| Parametric sinusoidal coupling | `Γ(t) = A·exp(−γt)·sin(ω·t+φ)` learned | `connections.py` |
| Phase synchronization | Differentiable Kuramoto ODE (RK4) | `phase_ode.py` |
| Condensed a priori knowledge | NFMC multidimensional fractal kernel | `condensate.py`, `nfmc.py` |

---

## 2. Parametric Sinusoidal Connections

### Definition

Each connection between nodes is a **learned temporal function**:

```
Γ(t) = A · exp(−γt) · sin(ω·t + φ)
```

Learned parameters: `A` (amplitude), `ω` (frequency), `φ` (phase), `γ` (damping).

### Implementation — `SinusoidalAggregator`

```python
# connections.py
class SinusoidalGate(nn.Module):
    # A, omega, phi, log_gamma: [out_channels, rank] — learned via SGD
    def forward(self, t):
        angle = t.view(-1,1,1) * self.omega.view(1,1,-1) + self.phi.unsqueeze(0)
        sin_val = torch.sin(angle)              # [N, out_channels, rank]
        gate = (self.A.unsqueeze(0) * sin_val).sum(-1)  # [N, out_channels]
        if self.damping:
            gate = gate * torch.exp(-F.softplus(self.log_gamma).unsqueeze(0) * t.abs().unsqueeze(-1))
        return gate
```

### Advantages vs Scalar Weights

- Encodes **temporal relationships** — connections are stronger at certain frequencies.
- **Natural damping** — distant connections weaken exponentially.
- **Stable gradients** — sinusoids have bounded derivatives, unlike ReLU.
- **Inductive bias** — the learned frequencies correspond to the scales of language.

---

## 3. Fractal Topology

### Supported Motifs

| Motif | Branching `b` | Structure | Use Case |
|-------|---------------|-----------|----------|
| `binary_tree` | 2 | Binary hierarchical | Semantic structure |
| `cantor` | 3 | Cantor set | Multiresolution |

### Bottom-Up / Top-Down Hierarchy

```
Level K (top):  L/b^K nodes   ← Flash Self-Attention here
Level K-1:      L/b^(K-1) nodes
    ...
Level 0 (base): L nodes       ← input tokens

Bottom-up: sinusoidal aggregation (L → L/b → ... → L/b^K)
Top-down: sinusoidal broadcast    (L/b^K → ... → L/b → L)
```

### Complexity per NFN Block

```
Bottom-up: Σ_{k=0}^{K-1} (L/b^k) · O(d²) = O(L · b/(b-1) · d²) = O(L·d²)
Attention: O((L/b^K)² · d)  — context reduced at the top level
Top-down:  O(L·d²)  (symmetric)
Total:     O(L·d² + (L/b^K)²·d)
```

For `b=2, K=4, L=4096`: `O(4096·d² + 256²·d)` vs `O(4096²·d)` standard → **16× cheaper**.

---

## 4. Phase Dynamics — Kuramoto ODE

### Model Equation

```
dθᵢ/dt = Ωᵢ + Σⱼ Kⱼᵢ · sin(θⱼ − θᵢ + φⱼᵢ)
```

where:
- `θᵢ`: phase of node i
- `Ωᵢ`: natural frequency (learned)
- `Kⱼᵢ`: rank `r` coupling matrix (learned)
- `φⱼᵢ`: phase shift (learned)

### Differentiable RK4 Integration

```python
# phase_ode.py
def rk4_step(f, y, t, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt*k1/2)
    k3 = f(t + dt/2, y + dt*k2/2)
    k4 = f(t + dt, y + dt*k3)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

**Key concept**: The RK4 integration is fully differentiable → gradients flow through the ODE via the phases.

### Role in the Network

Phases `θ` modulate hidden representations:

```
h_out = h + α · tanh(W_phase · [cos(θ), sin(θ)])
```

Nodes that synchronize (`θᵢ ≈ θⱼ`) form **conceptual clusters** — this is the mechanism of temporal binding.

---

## 5. Flash Attention + Long-Context RoPE

### Flash Attention

Uses `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.0+):
- IO-aware algorithm (Dao et al., 2022).
- No attention matrix in memory: O(L) memory, O(L²) FLOPs.
- `is_causal=True` for training, `False` with KV-cache.

### RoPE with NTK Extension

```python
# rope.py
def precompute_freqs_cis(dim, max_seq_len, base=10000., scale_factor=1.0):
    if scale_factor != 1.0:
        # NTK-aware scaling (bloc et al., 2023)
        base = base * (scale_factor ** (dim / (dim - 2)))
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, theta)
    return torch.polar(torch.ones_like(angles), angles)
```

Dynamic extension: If `seq_len > max_seq_len`, the NTK scale factor is automatically recomputed → **unlimited context without quality loss**.

| Config | `max_seq_len` | `context_len` (NTK) |
|--------|---------------|---------------------|
| nano | 512 | 4K |
| small | 2048 | 32K |
| medium | 4096 | 128K |
| large | 8192 | 128K+ |

---

## 6. Fractal KV-Cache

### Principle

Higher levels of the fractal hierarchy change more slowly than lower levels. The fractal cache exploits this property:

```
Level k recomputes every b^k tokens
Level 0: every token
Level 1: every 2 tokens
Level 2: every 4 tokens
Level K: every 16 tokens (for b=2, K=4)
```

### Implementation

```python
# kv_cache.py
class FractalStateCache:
    def should_recompute(self, step: int, level: int) -> bool:
        return step % (self.branching ** level) == 0
```

**Result**: Autoregressive inference in O(1) per token (instead of O(L) to recompute the entire context).

---

## 7. Persistent Cross-Context Memory

### Architecture

```
Memory [B, M, d]  ← M slots randomly initialized (small values)

READ  : Q = LayerNorm(ctx) @ W_q
        K, V = Memory @ W_k, Memory @ W_v
        output = softmax(QKᵀ/√d) @ V   ← cross-attention

WRITE : summary = mean(ctx, dim=1)       ← context summary
        gate = sigmoid(summary @ W_gate) ← [M] importance per slot
        candidate = summary @ W_write    ← [M, d] candidate values
        Memory ← (1−gate)·Memory + gate·candidate  ← EMA gated
```

### FractalMemoryBank

For models with `memory_per_level=True`, each fractal level has its own bank:

```
Level 0: episodic memory (8 slots, frequent updates)
Level 1: working memory (16 slots)
Level 2: semantic memory (32 slots)
Level K: encyclopedic memory (64+ slots, rare updates)
```

**Memory survives between conversations** — it can be saved to disk and reloaded (`save_memory()` / `load_memory()`).

---

## 8. NFMC v3.0 — Condensed Fractal Kernel

### The Universal Kernel

```
K(x, y) = ∫_Ω exp(i·Φ_ω(x,y)) dμ(ω)
```

Approximated by fractal Random Fourier Features:

```
K(x,y) ≈ φ(x)ᵀφ(y)

φ(x) = [cos(W·x + b), sin(W·x + b)] · √(2/r)
W: 1/f fractal frequencies — band k: W_k ~ N(0, 2^(k/n_scales)·I)
```

### Spectral Condensation (one-shot, zero SGD)

```python
# condensate.py
class SpectralCondensate:
    def condense(self, features):   # [N, rff_dim]
        features -= features.mean(0)
        _, S, Vh = torch.linalg.svd(features, full_matrices=False)
        self.U = Vh[:self.rank].T   # top-r eigenvectors
        self.S = S[:self.rank] / S[0]  # normalized eigenvalues
```

### Helmholtz Phase Locking

```
E(θ) = −½ Σᵢⱼ K̃(xᵢ,xⱼ) cos(θᵢ − θⱼ)   [Helmholtz free energy]

Kuramoto update:
θᵢ ← θᵢ + η Σⱼ K̃(xᵢ,xⱼ) sin(θⱼ − θᵢ)
```

The **phase attractors** correspond to syntactic and semantic categories.

---

## 9. v3.1 — ZeroShotNFMC

### Analytic Embedding (0 parameters)

```python
# analytic_embed.py
# e_k(t) = cos(ω_k · t/V · 2π) for k ∈ [0, d/2)
# ω_k = base^(k / (d/2))  — fractal frequency network

FractalCodepointEmbedding:  [V, d]  ← 0 parameters, buffer table
CharClassEmbedding:         [V, 16] ← vowel/consonant/digit/punct
AnalyticTokenEmbedding:     fusion via fixed orthogonal projection (QR)
```

### Modern Hopfield Memory (exponential capacity)

Ramsauer et al., 2020 — capacity O(exp(d/2)):

```
x_new = Xᵀ · softmax(β · X · ξ / √d)
```

Patterns seeded from:
1. Fourier vectors at **Farey/Mandelbrot frequencies**
2. Random vectors with **Zipf** weighting
3. Orthogonal vectors for uniform coverage

### Mandelbrot Frequencies (Farey Sequence)

The Mandelbrot set is parameterized by the external angle θ ∈ [0,1). The angles p/q (Farey fraction) correspond to the **parabolic points of period q**:

```
1/2  → period 2 (left main bulb)
1/3  → period 3
1/4  → period 4
2/5  → period 5
...  [Stern-Brocot Sequence]
```

These frequencies correspond exactly to the temporal scales of language:
- Period 2: binary subject/predicate
- Period 3: SVO triplet (Subject-Verb-Object)
- Period 4: quaternary structures (determiner-noun-verb-complement)

### Zipfian Decoder

Zipf's law: `P(rank=k) ∝ k^{−α}` is universal for natural language.
Decoder initialization:

```python
# hopfield.py
# W[k,:] = singular_vector_k * k^{-α/2}  — Zipf spectral structure
# bias[k] = -α · log(k)                  — correct marginal distribution
```

**Result**: Correct distribution from the first pass, without any examples.

### Causal Phase Predictor

```
dθₜ/dt = Ω(xₜ) + K(xₜ) ⊙ Σⱼ<ₜ sin(θⱼ − θₜ)
```

- `Ω(xₜ) = W_Ω · xₜ`: natural frequencies conditioned on input.
- Strict causality verified (diff = 0.000000 on future inputs).
- Initial frequencies = Mandelbrot angles (fixed).

---

## 10. v3.2 — EfficientNFN

### FractalLinearAttention — O(L·d²)

Kernelized identity (Katharopoulos et al., 2020):

```
(φ(Q)φ(K)ᵀ)V = φ(Q)(φ(K)ᵀV)    [associativity]
O(L²d)         O(Ld²)
```

Causal implementation via cumulative sum:

```python
for i in range(L):
    kv_sum += k[i].outer(v[i])   # [d, d] — no L×L matrix
    k_sum  += k[i]               # [d]
    out[i] = (q[i] @ kv_sum) / (q[i] · k_sum)
```

Multi-scale feature maps: `φ_k(x) = elu(x + freq_Mandelbrot_k) + 1`

| L | Standard Attn | FractalLinearAttn | Gain |
|---|---------------|-------------------|------|
| 512 | 33.6M FLOPs | 8.4M | 4× |
| 4096 | 2.15B | 134M | 16× |
| 32768 | 137B | 537M | **255×** |

### PhaseRoutedMoE — Continuous von Mises Routing

```
gate_e(x) = exp(κ · cos(θ_x − θ_e)) / Z     [von Mises distribution]

θ_x = atan2(W_im·x, W_re·x)   ← phase encoded from input
θ_e = Mandelbrot angles       ← fixed expert phases (seeded)
```

**Verified Properties:**
- Expert loads without auxiliary loss: 0.296 / 0.267 / 0.243 / 0.193 ≈ 0.25
- Continuous gradients everywhere (no discrete argmax)
- K=2/E=8 active experts = 25% of the FLOPs of a dense FFN

### PhaseSoliton — Long-Range Coherence

```
gain(x) = sigmoid(W_gain · θ(x) + coherence(x) / τ)
out = LayerNorm(x + gain · x)
```

Amplifies coherent phase patterns, suppresses incoherent noise.
Prevents the erasure of long-range dependencies without quadratic attention.

---

## 11. BPTP Training

### Back-Propagation Through Phase

The total loss:

```
L = L_task + λ_phase · L_phase + λ_freq · L_freq + λ_spectral · L_spectral

L_task     = CrossEntropy(logits, targets)
L_phase    = ||phases − phases_target||²   [target synchronization]
L_freq     = ||FFT(phases)||²_out_of_band  [spectral purity]
L_spectral = ||W||²_spectral               [spectral norm regularization]
```

### Differentiated Optimizer

```python
# trainer.py
# Sinusoidal parameters (A, ω, φ, γ): LR × 0.3, weight_decay=0
# Other parameters: Standard LR, weight_decay=0.1
```

Sinusoidal parameters have a reduced LR because their gradients are naturally larger (periodic functions with large derivatives).

### Features

| Feature | Parameter | Notes |
|---------|-----------|-------|
| Gradient accumulation | `grad_accumulation_steps` | Simulate large batches |
| Mixed precision | `dtype=torch.bfloat16` | AMP with GradScaler |
| Gradient checkpointing | `use_grad_checkpointing=True` | −50% memory |
| torch.compile | `compile_model=True` | 2-3× on Ampere+ |
| Scheduler | cosine + warmup | ratio min_lr=0.1 |

---

## 12. Three-Tier Tokenizer

```
Tier 1 — TiktokenTokenizer: cl100k_base (100K vocab, GPT-4 quality)
Tier 2 — BPETokenizer: Pure Python BPE, trainable from scratch (32K)
Tier 3 — CharTokenizer: 110 tokens, always available (fallback)

Auto-selection: tiktoken > char (if tiktoken is not installed)
```

Special tokens: `<pad>` `<bos>` `<eos>` `<unk>` `<sep>` `<sys>` `<usr>` `<ast>` `<code>` `</code>` `<think>` `</think>`

---

## 13. Multi-GPU DDP / FSDP

```bash
# DDP — model replicated on each GPU
torchrun --nproc_per_node=4 train.py --distributed ddp

# FSDP — sharded model (for very large models)
torchrun --nproc_per_node=8 train.py --distributed fsdp
```

**FSDP** uses `MixedPrecision(param_dtype=bfloat16, reduce_dtype=float32)` and `ShardingStrategy.FULL_SHARD` with `BackwardPrefetch.BACKWARD_PRE` for optimal performance.

The `NFNBlock` layers are automatically wrapped as FSDP units via `transformer_auto_wrap_policy`.

---

*Philippe-Antoine Robert — 2026-05-03 07:22:48 UTC*
