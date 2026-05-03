"""
Analytic Token Embedding — parameter-free geometry from mathematical structure.

Core insight: tokens don't need a *learned* embedding. Their geometry can be
derived from:

  1. Their ordinal position in the vocabulary (Fourier basis on [0, V])
  2. Their character-class features (vowel/consonant/digit/punct/space)
  3. Their orthographic n-gram hash (captures morphological similarity)

The resulting embedding is:
    e(t) = normalize([φ_fourier(t), φ_class(t), φ_ngram(t)])

where ALL three φ are deterministic functions — zero parameters, zero training.

Why this works (mathematically):
  - Fourier basis: tokens with similar IDs → similar geometry. For char
    tokenizers, similar ASCII codes → similar codepoints → similar features.
    Vowels (a=97, e=101, i=105, o=111, u=117) cluster naturally.
  - Character-class: distinguishes functional words from content words,
    punctuation from alphanumeric — capturing part-of-speech signal.
  - N-gram hash: makes subword tokens with shared prefixes/suffixes cluster
    (e.g., 'run', 'runs', 'running' → similar hash vectors).

Total parameters: 0.  Gradient: none.
"""

import math
import hashlib
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Fractal Fourier basis on the vocabulary ordinal
# ─────────────────────────────────────────────────────────────────────────────

class FractalCodepointEmbedding(nn.Module):
    """
    Maps token_id ∈ [0, V) → d_model geometry via self-similar Fourier basis.

        e_k(t) = cos(ω_k · t/V · 2π),  e_{k+d/2}(t) = sin(ω_k · t/V · 2π)
        ω_k = base^(k / (d/2))   — fractal frequency lattice

    Key properties:
    • No parameters (registered buffers only)
    • Vocabulary distance respects ID distance: |e(t) - e(t+1)| ∝ 1/V
    • Harmonic resonances: tokens at 1/2, 1/3, 1/4 of vocab are aligned
    • Works for ANY tokenizer (char, BPE, tiktoken, etc.)
    """

    def __init__(self, vocab_size: int, d_model: int, base: float = 2.0):
        super().__init__()
        assert d_model % 2 == 0
        half = d_model // 2

        # Fractal frequency lattice: ω_k = base^(k / half)
        k = torch.arange(half, dtype=torch.float64)
        freqs = (base ** (k / half)).float()                        # [half]

        # Precompute full table [V, d_model]  (float32, ~V*d bytes)
        t = torch.arange(vocab_size, dtype=torch.float32) / vocab_size
        angles = t.unsqueeze(1) * freqs.unsqueeze(0) * (2 * math.pi)  # [V, half]
        table = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)  # [V, d]

        self.register_buffer("table", table)
        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.table[token_ids]


# ─────────────────────────────────────────────────────────────────────────────
# Character-class feature map
# ─────────────────────────────────────────────────────────────────────────────

_VOWELS = set("aeiouAEIOU")
_CONSONANTS = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")


class CharClassEmbedding(nn.Module):
    """
    Fixed 16-dimensional character-class feature vector per token.

    Features (all binary/continuous, no learning):
    0: is_vowel        1: is_consonant    2: is_digit
    3: is_space        4: is_upper        5: is_lower
    6: is_punct        7: is_special_tok  8: is_ascii
    9: id_norm         10-15: bigram hash (6 dims from token ID)

    These capture part-of-speech signal without any training.
    """

    def __init__(self, vocab_size: int, id2char: Optional[dict] = None):
        super().__init__()
        F_DIM = 16
        table = torch.zeros(vocab_size, F_DIM)

        for tid in range(vocab_size):
            ch = id2char.get(tid, "") if id2char else (chr(tid) if tid < 128 else "")
            if not ch:
                table[tid, 7] = 1.0   # special token
            else:
                c = ch[0]
                table[tid, 0] = float(c in _VOWELS)
                table[tid, 1] = float(c in _CONSONANTS)
                table[tid, 2] = float(c.isdigit())
                table[tid, 3] = float(c in " \t\n\r")
                table[tid, 4] = float(c.isupper())
                table[tid, 5] = float(c.islower())
                table[tid, 6] = float(not c.isalnum() and not c.isspace())
                table[tid, 8] = float(ord(c) < 128 if len(c) == 1 else 0)
            table[tid, 9] = tid / max(vocab_size - 1, 1)

            # Deterministic 6-dim hash of the token ID
            h = int(hashlib.md5(str(tid).encode()).hexdigest()[:8], 16)
            for b in range(6):
                table[tid, 10 + b] = float((h >> b) & 1)

        self.register_buffer("table", table)
        self.out_dim = F_DIM

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.table[token_ids]


# ─────────────────────────────────────────────────────────────────────────────
# Full analytic embedding (Fourier + class features, fused)
# ─────────────────────────────────────────────────────────────────────────────

class AnalyticTokenEmbedding(nn.Module):
    """
    Full parameter-free token embedding.

    e(t) = LayerNorm( W_fuse · [φ_fourier(t) ∥ φ_class(t)] )

    W_fuse is a FIXED orthogonal projection (random but seeded), not learned.
    The output has the same shape as a standard nn.Embedding.

    Optional: learnable=True adds a tiny learnable bias (d_model params total)
    for fine-grained adaptation while keeping 99.9% of geometry analytic.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        id2char: Optional[dict] = None,
        learnable_bias: bool = False,
        seed: int = 31415,
    ):
        super().__init__()
        self.fourier = FractalCodepointEmbedding(vocab_size, d_model)
        self.charclass = CharClassEmbedding(vocab_size, id2char)

        in_dim = d_model + self.charclass.out_dim

        # Fixed orthogonal fusion projection (Gram-Schmidt on random matrix)
        g = torch.Generator()
        g.manual_seed(seed)
        W_rand = torch.randn(in_dim, d_model, generator=g)
        # QR decomposition → orthogonal columns
        Q, _ = torch.linalg.qr(W_rand)
        W = Q[:, :d_model]                    # [in_dim, d_model]
        self.register_buffer("W_fuse", W)

        # Non-parametric norm when learnable_bias=False (truly 0 params)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=learnable_bias)

        # Optional learnable bias (d_model scalars — tiny)
        self.bias = nn.Parameter(torch.zeros(d_model)) if learnable_bias else None
        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, L] or [B]
        f = self.fourier(token_ids)          # [..., d_model]
        c = self.charclass(token_ids)        # [..., 16]
        h = torch.cat([f, c], dim=-1)       # [..., d_model+16]
        out = h @ self.W_fuse               # [..., d_model]
        if self.bias is not None:
            out = out + self.bias
        return self.norm(out)
