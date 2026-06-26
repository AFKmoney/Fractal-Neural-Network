"""
Fast CPU inference via NumPy — bypasses PyTorch's ~70ms per-op overhead.

On this CPU, every PyTorch tensor operation costs ~70ms regardless of size.
NumPy operations cost <1ms. By reimplementing the forward pass in NumPy
we reduce per-token latency from ~5 seconds to ~50ms.

Supported architecture: FNNModel with FNNBlocks
(all optional AGI features disabled — memory, causal, goal, MoD, etc.)
"""
import os
# Single-threaded BLAS: prevents the ~100ms thread-spawn overhead per matmul.
# Must be set before numpy imports MKL/OpenBLAS.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


# ─── NumPy primitives ─────────────────────────────────────────────────────────

def _ln(x, w, b=None, eps=1e-5):
    """LayerNorm. w/b may be None (elementwise_affine=False)."""
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    xn = (x - mean) / np.sqrt(var + eps)
    if w is not None:
        xn = xn * w
    if b is not None:
        xn = xn + b
    return xn


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _elu_p1(x):
    """elu(x) + 1 — the linear-attention feature map."""
    return np.where(x >= 0, x + 1.0, np.exp(np.clip(x, -80, 0)))


# ─── Block components ─────────────────────────────────────────────────────────

def _fractal_attn(x, w):
    """FractalLinearAttention. x: [L, d] → [L, d] (residual NOT added here)."""
    L, d = x.shape
    H, d_h = w['H'], w['d_h']

    x_n = _ln(x, w['nq_w'], w['nq_b'])          # norm_q

    qkv = x_n @ w['qkv'].T                       # [L, 3d]
    q = qkv[:, :d].reshape(L, H, d_h).transpose(1, 0, 2)   # [H, L, d_h]
    k = qkv[:, d:2*d].reshape(L, H, d_h).transpose(1, 0, 2)
    v = qkv[:, 2*d:].reshape(L, H, d_h).transpose(1, 0, 2)

    freq = w['freq']                              # [d_h]
    q_f = _elu_p1(q + freq)
    k_f = _elu_p1(k + freq)

    # Causal linear attention via cumulative outer-product sum
    kv = k_f[:, :, :, np.newaxis] * v[:, :, np.newaxis, :]  # [H, L, d_h, d_h]
    kv_cs = np.cumsum(kv, axis=1)                            # [H, L, d_h, d_h]
    k_cs  = np.cumsum(k_f, axis=1)                           # [H, L, d_h]

    num   = np.einsum('hld,hldv->hlv', q_f, kv_cs)          # [H, L, d_h]
    denom = (q_f * k_cs).sum(-1, keepdims=True)              # [H, L, 1]
    out   = num / np.maximum(denom, 1e-6)                    # [H, L, d_h]

    out = out.transpose(1, 0, 2).reshape(L, d)               # [L, d]
    return out @ w['out'].T                                   # [L, d]


def _soliton(x, w):
    """PhaseSoliton. x: [L, d] → [L, d]."""
    n_ph = w['n_phases']
    h = x @ w['ph_proj'].T                                   # [L, 2*n_ph]
    theta = np.arctan2(h[:, n_ph:], h[:, :n_ph])             # [L, n_ph]

    theta_mean = theta.mean(0, keepdims=True)                 # [1, n_ph]
    coherence  = np.cos(theta - theta_mean).mean(-1, keepdims=True)  # [L, 1]

    gain_in = theta @ w['gain_w'].T + w['gain_b']             # [L, 1]
    gain    = 1.0 / (1.0 + np.exp(-(gain_in + coherence / w['tau'])))

    return _ln(x + gain * x, w['norm_w'], w['norm_b'])


def _moe(x, w):
    """
    PhaseRoutedMoE. x: [L, d] (already pre-normed by block's norm2).
    Returns: LN(x + expert_outputs(x))  — same as the module's forward.
    """
    L, d = x.shape
    E, K = w['E'], w['K']
    n_ph = w['n_phases']
    d_ff = w['W1'].shape[1]

    # Routing: phase-space von-Mises gate
    h = x @ w['ph_proj'].T                                    # [L, 2*n_ph]
    theta_x = np.arctan2(h[:, n_ph:], h[:, :n_ph])            # [L, n_ph]
    diff    = theta_x[:, np.newaxis, :] - w['expert_phases']   # [L, E, n_ph]
    energy  = w['kappa'] * np.cos(diff).mean(-1)               # [L, E]
    g       = _softmax(energy)                                  # [L, E]

    # Top-k sparse weights
    topk    = np.argpartition(-g, K, axis=-1)[:, :K]           # [L, K]
    g_sp    = np.zeros_like(g)
    np.put_along_axis(g_sp, topk, np.take_along_axis(g, topk, axis=1), axis=1)
    g_sp   /= g_sp.sum(-1, keepdims=True) + 1e-9               # [L, E]

    # Vectorized expert FFN: one large matmul (no Python loop)
    h1 = (x @ w['W1'].reshape(E * d_ff, d).T).reshape(L, E, d_ff)  # [L, E, d_ff]
    h1 = _gelu(h1)
    out_e = np.einsum('nef,ekf->nek', h1, w['W2'])             # [L, E, d]
    out   = (g_sp[:, :, np.newaxis] * out_e).sum(1)            # [L, d]

    return _ln(x + out, w['norm_w'], w['norm_b'])


def _block(x, w):
    """FNNBlock. x: [L, d] → [L, d]."""
    L = x.shape[0]
    x = x + _fractal_attn(_ln(x, w['n1_w'], w['n1_b']), w['attn'])
    x = _soliton(x, w['sol'])
    x = _moe(_ln(x, w['n2_w'], w['n2_b']), w['moe'])
    return x


# ─── Weight extraction ────────────────────────────────────────────────────────

def extract_weights(model):
    """
    Extract all model weights to numpy arrays for fast CPU inference.
    model: FNNModel with FNNBlocks (all optional features disabled).
    """
    w = {}

    # ── Embedding ─────────────────────────────────────────────────────────
    embed = model.embed
    if hasattr(embed, 'fourier') and hasattr(embed, 'charclass'):
        # AnalyticTokenEmbedding: precompute full [V, d] table
        V = embed.fourier.vocab_size
        fourier_np   = embed.fourier.table.detach().cpu().float().numpy()   # [V, d]
        charclass_np = embed.charclass.table.detach().cpu().float().numpy() # [V, 16]
        W_fuse_np    = embed.W_fuse.detach().cpu().float().numpy()          # [d+16, d]
        raw = np.concatenate([fourier_np, charclass_np], axis=-1) @ W_fuse_np  # [V, d]
        # LayerNorm (elementwise_affine=False — no gamma/beta)
        mean  = raw.mean(-1, keepdims=True)
        var   = ((raw - mean) ** 2).mean(-1, keepdims=True)
        embed_table = (raw - mean) / np.sqrt(var + 1e-5)
        if embed.bias is not None:
            embed_table += embed.bias.detach().cpu().float().numpy()
        w['embed'] = embed_table.astype(np.float32)
    elif hasattr(embed, 'table'):
        w['embed'] = embed.table.detach().cpu().float().numpy()
    else:
        w['embed'] = embed.weight.detach().cpu().float().numpy()

    # ── Blocks ────────────────────────────────────────────────────────────
    w['blocks'] = []
    for i, agi_block in enumerate(model.blocks):
        # Unwrap MoD if present
        core = agi_block.core
        if hasattr(core, 'block'):           # MixtureOfDepths
            core = core.block

        attn = core.attn
        sol  = core.soliton
        moe  = core.moe

        # level_freqs index: block_idx % n_levels
        lf = attn.level_freqs[core.block_idx % attn.n_levels].detach().cpu().float().numpy()

        bw = {
            'n1_w': core.norm1.weight.detach().cpu().float().numpy(),
            'n1_b': core.norm1.bias.detach().cpu().float().numpy(),
            'n2_w': core.norm2.weight.detach().cpu().float().numpy(),
            'n2_b': core.norm2.bias.detach().cpu().float().numpy(),
            'attn': {
                'H': attn.H, 'd_h': attn.d_head,
                'nq_w': attn.norm_q.weight.detach().cpu().float().numpy(),
                'nq_b': attn.norm_q.bias.detach().cpu().float().numpy(),
                'qkv':  attn.qkv.weight.detach().cpu().float().numpy(),  # [3d, d]
                'out':  attn.out.weight.detach().cpu().float().numpy(),  # [d, d]
                'freq': lf,                                              # [d_h]
            },
            'sol': {
                'n_phases': sol.phase_enc.n_phases,
                'ph_proj':  sol.phase_enc.proj.weight.detach().cpu().float().numpy(),
                'gain_w':   sol.gain.weight.detach().cpu().float().numpy(),
                'gain_b':   sol.gain.bias.detach().cpu().float().numpy(),
                'tau':      float(sol.tau),
                'norm_w':   sol.norm.weight.detach().cpu().float().numpy(),
                'norm_b':   sol.norm.bias.detach().cpu().float().numpy(),
            },
            'moe': {
                'E': moe.E, 'K': moe.K,
                'n_phases': moe.n_phases,
                'kappa':    float(moe.kappa),
                'ph_proj':  moe.phase_enc.proj.weight.detach().cpu().float().numpy(),
                'expert_phases': moe.expert_phases.detach().cpu().float().numpy(),  # [E, n_ph]
                'W1':       moe.W1.detach().cpu().float().numpy(),  # [E, d_ff, d]
                'W2':       moe.W2.detach().cpu().float().numpy(),  # [E, d, d_ff]
                'norm_w':   moe.norm.weight.detach().cpu().float().numpy(),
                'norm_b':   moe.norm.bias.detach().cpu().float().numpy(),
            },
        }
        w['blocks'].append(bw)

    # ── Final norm + LM head ──────────────────────────────────────────────
    ln_f = getattr(model, 'ln_f', None) or getattr(model, 'out_norm', None)
    w['ln_w'] = ln_f.weight.detach().cpu().float().numpy()
    w['ln_b'] = ln_f.bias.detach().cpu().float().numpy()

    # ZipfianDecoder or BayesianZipfianDecoder → both have .proj
    lm = model.lm_head
    head = lm.proj if hasattr(lm, 'proj') else lm
    w['lm_w'] = head.weight.detach().cpu().float().numpy()  # [V, d]
    w['lm_b'] = head.bias.detach().cpu().float().numpy()    # [V]

    return w


# ─── Full model forward ───────────────────────────────────────────────────────

def numpy_forward(input_ids_np, weights):
    """
    input_ids_np: [B, L] int32/int64 numpy array
    Returns: logits [B, L, V] float32 numpy array
    """
    B, L = input_ids_np.shape
    results = []

    for b in range(B):
        x = weights['embed'][input_ids_np[b]]      # [L, d]
        for bw in weights['blocks']:
            x = _block(x, bw)
        x = _ln(x, weights['ln_w'], weights['ln_b'])
        logits = x @ weights['lm_w'].T + weights['lm_b']  # [L, V]
        results.append(logits)

    return np.stack(results, axis=0).astype(np.float32)   # [B, L, V]


# ─── Model patcher ────────────────────────────────────────────────────────────

def apply_numpy_patch(model):
    """
    Monkey-patch model.forward() to use numpy during eval mode.
    Falls back to normal PyTorch forward during training.

    Call this after building the model in _init_default_model().
    """
    import torch
    _cache = {'weights': None}

    _original_forward = model.forward

    def _fast_forward(input_ids, write_memory=False, targets=None, **kwargs):
        if model.training or targets is not None:
            return _original_forward(input_ids, write_memory=write_memory,
                                     targets=targets, **kwargs)

        # Lazy weight extraction
        if _cache['weights'] is None:
            _cache['weights'] = extract_weights(model)

        ids_np = input_ids.detach().cpu().numpy().astype(np.int32)
        logits_np = numpy_forward(ids_np, _cache['weights'])
        logits = torch.from_numpy(logits_np).to(input_ids.device)
        return logits, {}

    model.forward = _fast_forward

    def _invalidate_numpy_cache():
        """Call after model weights change (e.g. after training step)."""
        _cache['weights'] = None

    model.invalidate_numpy_cache = _invalidate_numpy_cache
    return model
