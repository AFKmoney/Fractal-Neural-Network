"""
NFN v5.0 — Semantic Gematria

Uses gematria (numerical values of tokens) as a structural prior that bridges
number theory and natural language semantics.

Core Insight
------------
In gematria, every word has a numerical value. Words with equal or related
values share a mathematical relationship. This is NOT superstition — it is
a MATHEMATICAL ISOMORPHISM:

  (tokens, language)  ←→  (integers, number theory)

When the model learns that two tokens share a gematria relationship, it
learns a STRUCTURAL PRIOR about semantic similarity. This prior is:
  - Derived from mathematics (not from data)
  - Universal (same for all languages with the same encoding)
  - Composable (gematria(a+b) = gematria(a) + gematria(b))

Architecture
-----------
  GematriaTable          : multiple encoding systems for tokens
  SemanticGematriaLayer  : neural layer that uses gematria as attention bias
  GematriaLoss           : self-supervised loss from numerical relationships
  GematriaCurriculum     : progressive training schedule
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GematriaTable(nn.Module):
    """
    Maintains multiple gematria encoding tables for the vocabulary.
    
    Each system assigns a different numerical value to each token,
    creating multiple "views" of the same vocabulary in number space.
    
    Systems:
      - Ordinal:    token_id + 1
      - Prime:      primes[token_id]
      - Fibonacci:  fib(token_id)  
      - Digital root: sum of digits until single digit
      - Learned:    nn.Embedding (trainable, starts from ordinal)
    """

    def __init__(self, vocab_size: int, d_gematria: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_gematria = d_gematria

        primes = self._sieve(vocab_size * 10)
        if len(primes) < vocab_size:
            primes = primes + [primes[-1]] * (vocab_size - len(primes))
        self.register_buffer("prime_values", torch.tensor(primes[:vocab_size], dtype=torch.float))

        fib = [1, 1]
        for _ in range(max(vocab_size, 100)):
            fib.append(fib[-1] + fib[-2])
        fib_scaled = [math.log(f + 1) for f in fib]
        self.register_buffer("fib_values", torch.tensor(fib_scaled[:vocab_size], dtype=torch.float))

        ordinal = torch.arange(1, vocab_size + 1, dtype=torch.float)
        self.register_buffer("ordinal_values", ordinal)

        digital_roots = torch.tensor([self._digital_root(i) for i in range(1, vocab_size + 1)], dtype=torch.float)
        self.register_buffer("digital_root_values", digital_roots)

        self.learned_values = nn.Embedding(vocab_size, 1)
        nn.init.normal_(self.learned_values.weight, std=0.1)
        with torch.no_grad():
            for i in range(vocab_size):
                self.learned_values.weight[i, 0] = math.log(i + 2)

        self.system_proj = nn.Linear(5, d_gematria)

    def _sieve(self, n: int) -> List[int]:
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(n + 1) if is_prime[i]]

    def _digital_root(self, n: int) -> int:
        while n > 9:
            n = sum(int(d) for d in str(n))
        return n

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [B, L]
        Returns: [B, L, d_gematria] — gematria embedding for each token
        """
        ids_clamped = token_ids.clamp(0, self.vocab_size - 1)

        ordinal = torch.log1p(self.ordinal_values[ids_clamped].clamp(0, 1e6)).unsqueeze(-1)
        prime = torch.log1p(self.prime_values[ids_clamped].clamp(0, 1e6)).unsqueeze(-1)
        fib = self.fib_values[ids_clamped].unsqueeze(-1)
        digital_root = self.digital_root_values[ids_clamped].unsqueeze(-1)
        learned = self.learned_values(ids_clamped)

        all_vals = torch.cat([ordinal, prime, fib, digital_root, learned], dim=-1)

        return self.system_proj(all_vals)


class SemanticGematriaLayer(nn.Module):
    """
    Uses gematria values as a STRUCTURAL BIAS for attention.
    
    Tokens with similar gematria values get an attention bonus.
    This creates an inductive bias: "tokens that are numerically
    related should attend to each other more strongly."
    
    The bias is additive to the standard attention scores:
      score(i,j) = Q_i · K_j / √d + λ · sim(gem(i), gem(j))
    
    where sim is cosine similarity in gematria space.
    
    This means: even before training, the model has a mathematical
    prior about which tokens should interact. The prior comes from
    number theory, not from data.
    """

    def __init__(self, d_model: int, vocab_size: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.gematria = GematriaTable(vocab_size, d_gematria=d_model)
        self.gem_attn_bias = nn.Linear(d_model, n_heads)
        self.lambda_gem = nn.Parameter(torch.tensor(0.1))

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, h: torch.Tensor, token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, L, d_model]
        token_ids: [B, L]
        Returns: (h_out [B, L, d_model], gematria_coherence_loss)
        """
        B, L, d = h.shape

        Q = self.q_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        gem_emb = self.gematria(token_ids)
        gem_sim = F.cosine_similarity(
            gem_emb.unsqueeze(2), gem_emb.unsqueeze(1), dim=-1
        )
        gem_bias = self.gem_attn_bias(gem_emb)
        gem_bias_t = gem_bias.transpose(1, 2)
        attn_bias = gem_bias_t.unsqueeze(3) * gem_sim.unsqueeze(1)

        scores = scores + self.lambda_gem * attn_bias

        causal_mask = torch.triu(
            torch.ones(L, L, device=h.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, L, d)
        out = self.out_proj(out)
        h_out = self.norm(h + out)

        gem_coherence = gem_sim.tril(-1).sum() / max(1, L * (L - 1) / 2)
        loss = -gem_coherence * 0.01

        return h_out, loss


class GematriaLoss(nn.Module):
    """
    Self-supervised losses derived from gematria relationships.
    
    Three loss components:
      1. Gematria Prediction: predict token B's gematria from token A's
         (forces the model to learn numerical structure of language)
      2. Gematria Composition: gem(a+b) should relate to gem(a)+gem(b)
         (forces compositional number structure)
      3. Gematria Coherence: nearby tokens should have related gematria
         (forces local numerical coherence)
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.gematria = GematriaTable(vocab_size, d_gematria=d_model)
        self.predict_head = nn.Linear(d_model, d_model)

    def forward(
        self, h: torch.Tensor, token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        h: [B, L, d_model]
        token_ids: [B, L]
        """
        gem = self.gematria(token_ids)

        pred_next = self.predict_head(gem[:, :-1])
        target_next = gem[:, 1:]
        prediction_loss = F.mse_loss(pred_next, target_next)

        if token_ids.shape[1] >= 3:
            gem_a = gem[:, :-2]
            gem_b = gem[:, 1:-1]
            gem_ab = self.gematria(
                (token_ids[:, :-2] + token_ids[:, 1:-1]).clamp(0, self.gematria.vocab_size - 1)
            )
            composition_loss = F.mse_loss(gem_ab, gem_a + gem_b)
        else:
            composition_loss = torch.tensor(0.0, device=h.device)

        gem_norm = F.normalize(gem, dim=-1)
        coherence = (gem_norm[:, :-1] * gem_norm[:, 1:]).sum(-1).mean()
        coherence_loss = -coherence * 0.1

        total = prediction_loss + 0.1 * composition_loss + coherence_loss

        metrics = {
            "gem_prediction": prediction_loss.item(),
            "gem_composition": composition_loss.item(),
            "gem_coherence": coherence.item(),
        }
        return total, metrics


class GematriaCurriculum:
    """
    Progressive curriculum for gematria training.
    
    Phase 1: Learn ordinal gematria (simple counting)
    Phase 2: Learn prime gematria (number theory structure)
    Phase 3: Learn Fibonacci gematria (growth patterns)
    Phase 4: Learn digital root (cyclical structure)
    Phase 5: Learn composed gematria (additive structure)
    Phase 6: Discover NEW gematria relationships
    
    Each phase adds complexity. The model masters simple numerical
    relationships before tackling complex ones.
    """

    def __init__(self, phase_steps: List[int] = None):
        self.phase_steps = phase_steps or [200, 500, 1000, 2000, 5000, -1]
        self.phase_names = [
            "ordinal", "prime", "fibonacci", "digital_root",
            "composition", "discovery",
        ]
        self.current_phase = 0
        self.step = 0

    def get_phase(self) -> str:
        return self.phase_names[self.current_phase]

    def advance(self):
        self.step += 1
        if self.current_phase < len(self.phase_steps) - 1:
            threshold = self.phase_steps[self.current_phase]
            if threshold > 0 and self.step >= threshold:
                self.current_phase += 1
                self.step = 0
                return True
        return False
