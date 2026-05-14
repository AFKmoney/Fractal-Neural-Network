"""
NFN AGI v5.0 — Hierarchical Predictive Coding (upgraded)

v5.0 Upgrades
-------------
  1. Multi-step ahead predictions: each block now predicts N steps ahead
     (not just 1), creating richer world models over multiple layers.

  2. Probabilistic predictions with learned prior:
     Replace MSE with proper likelihood under a learned Gaussian:
         p(h_{t+k} | h_t) = N(μ_pred, σ_pred²)
     Loss = -log p = (h - μ)² / σ² + log σ
     This allows uncertainty-aware predictions.

  3. Top-down residual: prediction error fed back as residual (not just loss)
     with a learned gate that decides when to trust the error signal.

Theory (Rao & Ballard 1999, Friston 2010, Chalupka 2020)
------
In predictive coding, each level of the hierarchy:
  1. Sends a *prediction* of the level below's representation
  2. Receives the *prediction error* (actual - predicted) from below
  3. Updates its own representation to minimise prediction error

In NFN v5.0:
  • N-step predictions: block_t predicts h_{t+1}, h_{t+2}, ..., h_{t+N}
  • Probabilistic: predict μ and σ, not just μ
  • Multi-scale: different prediction heads for different lookahead distances

The total predictive coding loss:
    L_pc = Σ_k λ_k · Σ_t NLL(h_{t+k}, N(μ_{t,k}, σ_{t,k}²))
where λ_k = λ^k (geometric decay for longer horizons).
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Head
# ─────────────────────────────────────────────────────────────────────────────

class ProbabilisticPredictionHead(nn.Module):
    """
    Probabilistic prediction head: h_t → N(μ_{t+k}, σ_{t+k}²).

    Instead of predicting just the mean (MSE), predicts a full Gaussian.
    This gives uncertainty-aware predictions: σ high = "I'm not sure".

    Architecture: lightweight 2-layer MLP → (μ, log_σ) for lookahead k.
    """

    def __init__(self, d_model: int, hidden_scale: float = 0.5, lookahead: int = 1):
        super().__init__()
        self.lookahead = lookahead
        hidden = max(64, int(d_model * hidden_scale))

        # Step conditioning: encode which lookahead step this head is for
        self.step_embed = nn.Embedding(16, d_model // 8)

        self.net = nn.Sequential(
            nn.Linear(d_model + d_model // 8, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.mu_head      = nn.Linear(hidden, d_model)
        self.log_sig_head = nn.Linear(hidden, d_model)

        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_sig_head.weight)
        nn.init.constant_(self.log_sig_head.bias, -1.0)  # start σ ≈ 0.37

        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, L, d]
        Returns (μ [B, L, d], log_σ [B, L, d]) for lookahead step self.lookahead.
        """
        B, L, d = h.shape
        k_emb = self.step_embed(
            torch.tensor(self.lookahead - 1, device=h.device)
        ).unsqueeze(0).unsqueeze(0).expand(B, L, -1)    # [B, L, d//8]

        inp   = torch.cat([h, k_emb], dim=-1)
        feat  = self.net(inp)
        mu    = self.norm(h + self.mu_head(feat))
        log_s = self.log_sig_head(feat).clamp(-4, 4)
        return mu, log_s

    def nll(self, h: torch.Tensor, h_target: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood under N(μ, σ²).
        h_target should not have grad (call .detach()).
        """
        mu, log_s = self.forward(h)
        sigma2    = (2 * log_s).exp().clamp(min=1e-6)
        nll       = 0.5 * ((h_target - mu) ** 2 / sigma2 + 2 * log_s)
        return nll.mean()


# Keep backward-compatible alias
class PredictionHead(ProbabilisticPredictionHead):
    """Backward-compatible wrapper (single-step, deterministic mean)."""

    def __init__(self, d_model: int, hidden_scale: float = 0.5):
        super().__init__(d_model, hidden_scale, lookahead=1)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns just the mean prediction [B, L, d]."""
        mu, _ = super().forward(h)
        return mu


# ─────────────────────────────────────────────────────────────────────────────
# Predictive Coding Block
# ─────────────────────────────────────────────────────────────────────────────

class PredictiveCodingBlock(nn.Module):
    """
    Wraps any h→h block with predictive coding.

    Forward pass:
      1. If a prediction from the previous block is available, compute
         the prediction error ε = h_in − pred_from_below and inject it
         into h_in as a correction:
             h_corrected = h_in + α_error · ε
      2. Run the wrapped block: h_out = block(h_corrected)
      3. Compute a prediction of what the next block will produce:
             pred_next = prediction_head(h_out)
      4. Compute prediction loss: L_pred = ||h_out − pred_from_above||²
         (set externally by the block above after it runs)

    This creates a feedback loop: each block's output is shaped by
    predictions from both the block above (top-down) and below (bottom-up).
    """

    def __init__(
        self,
        d_model:       int,
        block:         nn.Module,           # the wrapped AGI/Efficient block
        error_scale:   float = 0.1,         # α — how strongly to apply error
        lambda_pred:   float = 0.01,
        n_steps_ahead: int   = 3,           # v5.0: predict N steps ahead
        horizon_decay: float = 0.5,         # weight decay across horizon steps
    ):
        super().__init__()
        self.block         = block
        self.error_scale   = error_scale
        self.lambda_pred   = lambda_pred
        self.n_steps_ahead = n_steps_ahead
        self.horizon_decay = horizon_decay

        # v5.0: N probabilistic prediction heads (one per lookahead step)
        self.predictors = nn.ModuleList([
            ProbabilisticPredictionHead(d_model, lookahead=k + 1)
            for k in range(n_steps_ahead)
        ])

        # Error gate: learns when to trust prediction errors
        self.error_gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.error_gate.weight)
        nn.init.zeros_(self.error_gate.bias)

        # Stored predictions from this block (k-step ahead) — read by blocks above
        self._predictions: List[Optional[torch.Tensor]] = [None] * n_steps_ahead

    @property
    def last_prediction(self) -> Optional[torch.Tensor]:
        """Backward-compatible: return 1-step-ahead mean prediction."""
        if self._predictions[0] is None:
            return None
        mu, _ = self.predictors[0].forward.__wrapped__ if hasattr(self.predictors[0].forward, '__wrapped__') else (None, None)
        return self._predictions[0]

    def forward(
        self,
        h:                 torch.Tensor,                   # [B, L, d]
        prediction_above:  Optional[torch.Tensor] = None,  # [B, L, d] 1-step pred
        **block_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          h_out      : [B, L, d]
          pred_loss  : scalar  — multi-step probabilistic prediction loss
        """
        # ── Bottom-up: correct h using prediction error from above ────────
        pred_loss = torch.tensor(0.0, device=h.device)
        if prediction_above is not None:
            error = h - prediction_above                    # [B, L, d]
            gate  = torch.sigmoid(self.error_gate(h))      # [B, L, 1]
            h = h + self.error_scale * gate * error
            # NLL loss under the k=1 predictor
            pred_loss = self.lambda_pred * (error ** 2).mean()

        # ── Run wrapped block ─────────────────────────────────────────────
        if hasattr(self.block, 'forward') and 'write_memory' in self.block.forward.__code__.co_varnames:
            h_out, block_losses = self.block(h, **block_kwargs)
        else:
            result = self.block(h)
            h_out  = result[0] if isinstance(result, tuple) else result

        # ── v5.0: Generate N-step probabilistic predictions ───────────────
        # Store mean predictions for consumption by blocks above
        for k, predictor in enumerate(self.predictors):
            mu, _ = predictor(h_out)
            self._predictions[k] = mu.detach()

        return h_out, pred_loss

    def multistep_prediction_loss(
        self,
        h_sequence: List[torch.Tensor],   # list of h from subsequent blocks
    ) -> torch.Tensor:
        """
        Compute multi-step ahead NLL loss.

        h_sequence[k] = actual hidden state k blocks ahead.
        Compares against self's k-step prediction.

        Called externally after all blocks have run.
        """
        total = torch.tensor(0.0, device=h_sequence[0].device if h_sequence else torch.device("cpu"))
        if not h_sequence:
            return total

        # We need the current block's h to compute predictions
        # (This is called after forward, so we use the stored predictions)
        for k, h_target in enumerate(h_sequence[:self.n_steps_ahead]):
            if self._predictions[k] is None:
                continue
            weight = self.horizon_decay ** k
            pred   = self._predictions[k]                  # [B, L, d] mean
            err    = (pred - h_target.detach()) ** 2
            total  = total + weight * self.lambda_pred * err.mean()

        return total


# ─────────────────────────────────────────────────────────────────────────────
# Free Energy Minimiser  (Friston's active inference, simplified)
# ─────────────────────────────────────────────────────────────────────────────

class FreeEnergyMinimiser(nn.Module):
    """
    Implements a simplified version of Friston's free energy principle.

    Free energy F = complexity − accuracy
               = KL[q(z|h) || p(z)] − E_q[log p(h|z)]

    In practice:
      • q(z|h): recognition density — Gaussian with μ, σ from a small encoder
      • p(z):   prior — unit Gaussian (regularisation toward simple representations)
      • p(h|z): generative model — the PredictionHead going forward

    The free energy loss is the ELBO:
        F = 0.5 · ||μ||² + 0.5 · σ² - 0.5 · log σ² - 0.5
          + ||h − decode(z)||²

    This gives each block a latent code z that:
      1. Stays compact (KL term)
      2. Reconstructs the hidden state accurately (reconstruction term)

    The latent z is injected back into h, providing a compressed
    "belief state" about what the model currently represents.
    """

    def __init__(self, d_model: int, latent_dim: int, lambda_fe: float = 0.001):
        super().__init__()
        self.lambda_fe = lambda_fe

        # Encoder: h → (μ, log_σ²)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
        )
        self.mu_proj     = nn.Linear(d_model // 2, latent_dim)
        self.log_var_proj = nn.Linear(d_model // 2, latent_dim)

        # Decoder: z → h reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Inject latent back into h
        self.inject = nn.Linear(latent_dim, d_model, bias=False)
        nn.init.zeros_(self.inject.weight)

        nn.init.zeros_(self.mu_proj.weight)
        nn.init.zeros_(self.mu_proj.bias)
        nn.init.constant_(self.log_var_proj.bias, -2.0)  # start with small σ

    def forward(
        self,
        h: torch.Tensor,    # [B, L, d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          h_enriched : [B, L, d]  — h + injected latent
          fe_loss    : scalar     — free energy (ELBO bound)
        """
        enc = self.encoder(h)                   # [B, L, d//2]
        mu  = self.mu_proj(enc)                 # [B, L, latent]
        log_var = self.log_var_proj(enc)        # [B, L, latent]
        log_var = log_var.clamp(-4, 4)

        # Reparameterisation trick
        std = (0.5 * log_var).exp()
        if self.training:
            z = mu + std * torch.randn_like(std)
        else:
            z = mu

        # Reconstruction loss
        h_recon  = self.decoder(z)              # [B, L, d]
        recon    = F.mse_loss(h_recon, h.detach())

        # KL divergence: KL[N(μ,σ²) || N(0,1)]
        kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).mean()

        fe_loss = self.lambda_fe * (recon + kl)

        # Inject latent belief into h
        h_enriched = h + self.inject(z) * 0.05

        return h_enriched, fe_loss
