"""
NFN v4.0 — Hierarchical Predictive Coding

Each block predicts what the next block's output will be.  The
prediction error propagates back as an additional gradient signal —
this is the neuro-scientific mechanism behind how the brain builds
internal world models (Rao & Ballard 1999, Friston 2010).

Theory
------
In predictive coding, each level of the hierarchy:
  1. Sends a *prediction* of the level below's representation
  2. Receives the *prediction error* (actual - predicted) from below
  3. Updates its own representation to minimise prediction error

In NFN v4.0 this becomes:
  • Each AGIBlock predicts the hidden state that the NEXT block will produce
  • The prediction error ε = h_actual − h_predicted is:
      (a) Fed back as an auxiliary loss signal: L_pred = ||ε||²
      (b) Injected into the current block's output as a correction:
          h_corrected = h + α · ε_from_below
  • The model therefore learns to have internally consistent representations
    across layers — each layer's output is what the layer above expects

Implementation
--------------
  PredictionHead     : maps h_t → predicted h_{t+1}  (one block ahead)
  PredictiveCodingBlock: wraps a block, maintains predictions + errors
  HierarchicalPC     : stacks N PredictiveCodingBlocks, propagates errors

The total predictive coding loss across all blocks:
    L_pc = Σ_t λ_pc · ||h_t − pred_{t-1}(h_{t-1})||²_F / (B·L·d)

This is a self-supervised loss that costs nothing extra — it reuses
activations already computed during the forward pass.
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Head
# ─────────────────────────────────────────────────────────────────────────────

class PredictionHead(nn.Module):
    """
    Predicts the next block's hidden state from the current one.

    Architecture: lightweight 2-layer MLP with residual.
    Kept small intentionally — we want to capture the systematic
    component of inter-block transformations, not overfit.
    """

    def __init__(self, d_model: int, hidden_scale: float = 0.5):
        super().__init__()
        hidden = max(64, int(d_model * hidden_scale))
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, L, d] → predicted next h: [B, L, d]"""
        return self.norm(h + self.net(h))


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
        d_model:      int,
        block:        nn.Module,           # the wrapped AGI/Efficient block
        error_scale:  float = 0.1,         # α — how strongly to apply error
        lambda_pred:  float = 0.01,
    ):
        super().__init__()
        self.block        = block
        self.error_scale  = error_scale
        self.lambda_pred  = lambda_pred

        self.predictor    = PredictionHead(d_model)

        # Error gate: learns when to trust prediction errors
        self.error_gate   = nn.Linear(d_model, 1)
        nn.init.zeros_(self.error_gate.weight)
        nn.init.zeros_(self.error_gate.bias)

        # Stored prediction from this block (read by the block above)
        self._prediction: Optional[torch.Tensor] = None

    @property
    def last_prediction(self) -> Optional[torch.Tensor]:
        return self._prediction

    def forward(
        self,
        h:                 torch.Tensor,                   # [B, L, d]
        prediction_above:  Optional[torch.Tensor] = None,  # [B, L, d]
        **block_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          h_out      : [B, L, d]
          pred_loss  : scalar  — prediction error from above
        """
        # ── Bottom-up: correct h using prediction error from above ────────
        pred_loss = torch.tensor(0.0, device=h.device)
        if prediction_above is not None:
            error = h - prediction_above                    # [B, L, d]
            gate  = torch.sigmoid(self.error_gate(h))      # [B, L, 1]
            h = h + self.error_scale * gate * error
            pred_loss = self.lambda_pred * (error ** 2).mean()

        # ── Run wrapped block ─────────────────────────────────────────────
        if hasattr(self.block, 'forward') and 'write_memory' in self.block.forward.__code__.co_varnames:
            h_out, block_losses = self.block(h, **block_kwargs)
        else:
            result = self.block(h)
            h_out  = result[0] if isinstance(result, tuple) else result
            block_losses = {}

        # ── Generate prediction for next block ────────────────────────────
        self._prediction = self.predictor(h_out)

        return h_out, pred_loss


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
