"""
NFN v4.0 — Multimodal Fractal Phase Space

Extends the fractal kernel to handle image patches, audio frames, and sensor
readings in the same Kuramoto phase space as text tokens.

Theory
------
All modalities are projected into a shared fractal RFF space before phase
locking. Cross-modal synchronisation arises naturally from the Kuramoto
coupling: image patches that are semantically consistent with the current
text will phase-lock with text tokens; conflicting patches will de-synchronise.

Modality-specific projections:
  text   : d_model   → d_shared  (learned linear)
  image  : patch_dim → d_shared  (learned linear + 2-D RoPE)
  audio  : frame_dim → d_shared  (learned linear + 1-D temporal RoPE)
  sensor : sensor_dim→ d_shared  (learned linear)

All share the same FractalRFF + SpectralCondensate + HelmholtzPhaseLocking
stack. Cross-modal attention is implicit in the Kuramoto coupling matrix K_sim.

Architecture
------------
  ModalityProjector   : modality-specific linear → d_shared
  MultimodalFractalRFF: projects all modalities, concatenates, runs phase lock
  CrossModalSync      : measures phase coherence between modality streams
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .condensate import FractalRFF, SpectralCondensate, HelmholtzPhaseLocking
from .flash_attn import FlashAttention


# ─────────────────────────────────────────────────────────────────────────────
# 2-D Positional Encoding for image patches  (sinusoidal, no params)
# ─────────────────────────────────────────────────────────────────────────────

def _2d_sinusoidal_pos(h: int, w: int, d: int, device: torch.device) -> torch.Tensor:
    """
    Returns [h*w, d] 2-D sinusoidal position encoding.
    First d//2 dims encode row, last d//2 encode column.
    """
    assert d % 4 == 0, "d must be divisible by 4 for 2-D sinusoidal encoding"
    half = d // 2

    pos_row = torch.arange(h, device=device).float()
    pos_col = torch.arange(w, device=device).float()
    div = torch.exp(torch.arange(0, half, 2, device=device).float() * -(math.log(10000.0) / half))

    row_sin = torch.sin(pos_row[:, None] * div[None, :])   # [h, half//2]
    row_cos = torch.cos(pos_row[:, None] * div[None, :])   # [h, half//2]
    col_sin = torch.sin(pos_col[:, None] * div[None, :])   # [w, half//2]
    col_cos = torch.cos(pos_col[:, None] * div[None, :])   # [w, half//2]

    # Combine: [h, w, d]
    row_enc = torch.cat([row_sin, row_cos], dim=-1).unsqueeze(1).expand(h, w, half)
    col_enc = torch.cat([col_sin, col_cos], dim=-1).unsqueeze(0).expand(h, w, half)
    pos = torch.cat([row_enc, col_enc], dim=-1)   # [h, w, d]
    return pos.reshape(h * w, d)


# ─────────────────────────────────────────────────────────────────────────────
# Modality-specific projectors
# ─────────────────────────────────────────────────────────────────────────────

class ModalityProjector(nn.Module):
    """
    Projects one modality's raw features into the shared fractal phase space.

    For image / audio, adds sinusoidal position encoding before projection.
    """

    MODALITIES = ("text", "image", "audio", "sensor")

    def __init__(
        self,
        modality:   str,
        in_dim:     int,
        d_shared:   int,
        patch_hw:   Tuple[int, int] = (14, 14),  # for image only
    ):
        super().__init__()
        assert modality in self.MODALITIES, f"Unknown modality: {modality}"
        self.modality = modality
        self.patch_hw = patch_hw
        self.d_shared = d_shared

        self.proj = nn.Linear(in_dim, d_shared)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

        self.norm = nn.LayerNorm(d_shared)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        text  : [B, L, in_dim]       → [B, L, d_shared]
        image : [B, n_patches, in_dim] → [B, n_patches, d_shared]
        audio : [B, n_frames, in_dim]  → [B, n_frames, d_shared]
        sensor: [B, n_sensors, in_dim] → [B, n_sensors, d_shared]
        """
        h = self.proj(x)   # [B, T, d_shared]

        if self.modality == "image":
            H, W = self.patch_hw
            n_patches = x.shape[1]
            if n_patches == H * W:
                pos = _2d_sinusoidal_pos(H, W, self.d_shared, x.device)
                h = h + pos.unsqueeze(0)

        elif self.modality == "audio":
            T = x.shape[1]
            # 1-D sinusoidal temporal position
            pos_idx = torch.arange(T, device=x.device).float()
            div = torch.exp(
                torch.arange(0, self.d_shared, 2, device=x.device).float()
                * -(math.log(10000.0) / self.d_shared)
            )
            sin_enc = torch.sin(pos_idx[:, None] * div[None, :])  # [T, d//2]
            cos_enc = torch.cos(pos_idx[:, None] * div[None, :])  # [T, d//2]
            pos = torch.cat([sin_enc, cos_enc], dim=-1)[:, :self.d_shared]
            h = h + pos.unsqueeze(0)

        return self.norm(h)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Modal Phase Coherence
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalSync(nn.Module):
    """
    Measures and enforces phase coherence between modality streams.

    Coherence between modality A (phases [B, L_A, n_phases]) and
    modality B (phases [B, L_B, n_phases]):

        coherence(A, B) = |⟨e^{i(θ_A - θ_B)}⟩|
                        = mean |cos(θ_A_k - θ_B_k)| + |sin(θ_A_k - θ_B_k)|

    Used as an auxiliary loss to encourage aligned modalities to synchronise.
    """

    def __init__(self, n_phases: int, d_shared: int):
        super().__init__()
        self.n_phases = n_phases
        # Cross-modal attention: each modality queries others
        self.cross_attn = FlashAttention(d_shared, n_heads=4)
        self.norm = nn.LayerNorm(d_shared)

    def phase_coherence(
        self,
        theta_a: torch.Tensor,  # [B, L_a, n_phases]
        theta_b: torch.Tensor,  # [B, L_b, n_phases]
    ) -> torch.Tensor:
        """Returns scalar mean coherence in [0, 1]."""
        # Use mean phase of each modality
        mu_a = theta_a.mean(1)   # [B, n_phases]
        mu_b = theta_b.mean(1)   # [B, n_phases]
        diff = mu_a - mu_b
        # Kuramoto order parameter magnitude
        return (torch.cos(diff).abs() + torch.sin(diff).abs()).mean() * 0.5

    def forward(
        self,
        h_query: torch.Tensor,   # [B, L_q, d_shared]  — query modality
        h_key:   torch.Tensor,   # [B, L_k, d_shared]  — key modality
    ) -> torch.Tensor:
        """
        Enriches query modality with cross-modal context from key modality.
        Returns [B, L_q, d_shared].
        """
        attended = self.cross_attn(h_query, h_key, h_key)
        return self.norm(h_query + attended)


# ─────────────────────────────────────────────────────────────────────────────
# MultimodalFractalRFF  (full multimodal phase space)
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalFractalRFF(nn.Module):
    """
    Shared fractal phase space for any combination of modalities.

    Each modality goes through:
      1. ModalityProjector → d_shared
      2. Optional cross-modal enrichment (CrossModalSync)
      3. Shared FractalRFF → SpectralCondensate → HelmholtzPhaseLocking

    The shared Kuramoto dynamics naturally synchronise semantically related
    tokens across modalities — no explicit cross-modal supervision needed.

    Usage:
      model = MultimodalFractalRFF(
          d_text=512, d_image=768, d_audio=256, d_shared=256,
          n_rff=128, rank=32, n_phases=8
      )
      phases = model(text=text_h, image=patch_h, audio=frame_h)
      # phases is a dict: {"text": [B,L,n_phases], "image": [B,P,n_phases], ...}
    """

    def __init__(
        self,
        d_shared:   int = 256,
        d_text:     Optional[int] = None,
        d_image:    Optional[int] = None,
        d_audio:    Optional[int] = None,
        d_sensor:   Optional[int] = None,
        patch_hw:   Tuple[int, int] = (14, 14),
        n_rff:      int = 128,
        n_scales:   int = 6,
        rank:       int = 32,
        n_phases:   int = 8,
        n_iter:     int = 6,
        eta:        float = 0.12,
        seed:       int = 42,
    ):
        super().__init__()
        self.d_shared  = d_shared
        self.n_phases  = n_phases
        self.modalities_active: list[str] = []

        # Modality projectors (only instantiate requested modalities)
        self.projectors = nn.ModuleDict()
        if d_text is not None:
            self.projectors["text"]   = ModalityProjector("text", d_text, d_shared)
            self.modalities_active.append("text")
        if d_image is not None:
            self.projectors["image"]  = ModalityProjector("image", d_image, d_shared, patch_hw)
            self.modalities_active.append("image")
        if d_audio is not None:
            self.projectors["audio"]  = ModalityProjector("audio", d_audio, d_shared)
            self.modalities_active.append("audio")
        if d_sensor is not None:
            self.projectors["sensor"] = ModalityProjector("sensor", d_sensor, d_shared)
            self.modalities_active.append("sensor")

        assert len(self.modalities_active) >= 1, "At least one modality must be specified"

        # Cross-modal synchroniser (text ↔ image primary pair if both active)
        self.cross_sync: Optional[CrossModalSync] = None
        if "text" in self.modalities_active and len(self.modalities_active) > 1:
            self.cross_sync = CrossModalSync(n_phases, d_shared)

        # Shared fractal phase stack (parameter-free once condensed)
        self.rff = FractalRFF(d_shared, n_rff, n_scales=n_scales, seed=seed)
        rff_out  = self.rff.out_dim
        self.condensate  = SpectralCondensate(rff_out, rank)
        self.phase_lock  = HelmholtzPhaseLocking(n_phases, n_iter=n_iter, eta=eta)

        # Per-modality output projections back to d_shared
        self.out_projs = nn.ModuleDict({
            m: nn.Linear(2 * n_phases + rank, d_shared, bias=False)
            for m in self.modalities_active
        })
        for proj in self.out_projs.values():
            nn.init.normal_(proj.weight, std=0.01)

        self.norms = nn.ModuleDict({
            m: nn.LayerNorm(d_shared)
            for m in self.modalities_active
        })

    # ─── condensation helpers ────────────────────────────────────────────────

    def condense_from_hidden(self, h: torch.Tensor):
        """Seed condensate from any hidden state matrix [N, d_shared]."""
        with torch.no_grad():
            phi = self.rff(h.view(-1, h.shape[-1]))
            self.condensate.condense(phi)

    def update_condensate(self, h: torch.Tensor, lr: float = 0.3):
        """Online incremental SVD update with new features."""
        with torch.no_grad():
            phi = self.rff(h.view(-1, h.shape[-1]))
            self.condensate.update_online(phi, lr=lr)

    # ─── core forward ────────────────────────────────────────────────────────

    def _encode_modality(
        self,
        name:  str,
        raw:   torch.Tensor,              # [B, T, d_modal]
        init_phases: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects one modality and runs fractal phase locking.
        Returns (h_proj, theta, z) where:
          h_proj : [B, T, d_shared]
          theta  : [B, T, n_phases]
          z      : [B, T, rank]
        """
        h_proj = self.projectors[name](raw)       # [B, T, d_shared]
        phi    = self.rff(h_proj)                  # [B, T, 2*n_rff]
        z      = self.condensate(phi)              # [B, T, rank]
        theta, _ = self.phase_lock(z, init_phases) # [B, T, n_phases]
        return h_proj, theta, z

    def forward(
        self,
        text:   Optional[torch.Tensor] = None,   # [B, L_text, d_text]
        image:  Optional[torch.Tensor] = None,   # [B, L_img, d_image]
        audio:  Optional[torch.Tensor] = None,   # [B, L_aud, d_audio]
        sensor: Optional[torch.Tensor] = None,   # [B, L_sen, d_sensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict mapping modality name → enriched hidden state [B, T, d_shared].
        Also populates self.last_phases for coherence loss computation.
        """
        inputs = {}
        if text   is not None: inputs["text"]   = text
        if image  is not None: inputs["image"]  = image
        if audio  is not None: inputs["audio"]  = audio
        if sensor is not None: inputs["sensor"] = sensor

        assert inputs, "At least one modality tensor must be provided"

        # Phase-encode each modality
        encoded: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for name, raw in inputs.items():
            h_proj, theta, z = self._encode_modality(name, raw)
            encoded[name] = (h_proj, theta, z)

        # Cross-modal synchronisation: enrich text with visual/audio context
        if self.cross_sync is not None and "text" in encoded:
            h_text = encoded["text"][0]
            for other_name in inputs:
                if other_name != "text":
                    h_other = encoded[other_name][0]
                    h_text = self.cross_sync(h_text, h_other)
            # Re-encode text after cross-modal enrichment
            phi   = self.rff(h_text)
            z_t   = self.condensate(phi)
            theta_t, _ = self.phase_lock(z_t)
            encoded["text"] = (h_text, theta_t, z_t)

        # Store phases for coherence loss
        self.last_phases = {name: enc[1] for name, enc in encoded.items()}

        # Decode back to d_shared enriched features
        outputs: Dict[str, torch.Tensor] = {}
        for name, (h_proj, theta, z) in encoded.items():
            h_frac = torch.cat([torch.cos(theta), torch.sin(theta), z], dim=-1)
            h_out  = self.out_projs[name](h_frac)
            outputs[name] = self.norms[name](h_proj + h_out)

        return outputs

    # ─── auxiliary losses ────────────────────────────────────────────────────

    def loss_coherence(self) -> torch.Tensor:
        """
        Cross-modal phase coherence loss.
        Encourages text and image phases to synchronise on shared semantics.
        Only meaningful when both text and image are present.
        """
        if not hasattr(self, "last_phases"):
            return torch.tensor(0.0)
        phases = self.last_phases
        if "text" not in phases or len(phases) < 2:
            return torch.tensor(0.0, device=next(iter(phases.values())).device)

        theta_text = phases["text"]
        device = theta_text.device
        total = torch.tensor(0.0, device=device)
        count = 0
        for name, theta_other in phases.items():
            if name == "text":
                continue
            if self.cross_sync is not None:
                coherence = self.cross_sync.phase_coherence(theta_text, theta_other)
            else:
                mu_t = theta_text.mean(1)
                mu_o = theta_other.mean(1)
                coherence = torch.cos(mu_t - mu_o).abs().mean()
            # We want coherence → 1, so loss = 1 - coherence
            total = total + (1.0 - coherence)
            count += 1
        return total / max(count, 1)

    def phase_similarity_matrix(self) -> Optional[torch.Tensor]:
        """
        Returns cross-modal phase similarity [n_modalities, n_modalities].
        Useful for inspection / debugging.
        """
        if not hasattr(self, "last_phases") or len(self.last_phases) < 2:
            return None
        names  = list(self.last_phases.keys())
        n      = len(names)
        device = next(iter(self.last_phases.values())).device
        sim    = torch.zeros(n, n, device=device)
        for i, na in enumerate(names):
            mu_a = self.last_phases[na].mean(1)   # [B, n_phases]
            for j, nb in enumerate(names):
                mu_b = self.last_phases[nb].mean(1)
                sim[i, j] = torch.cos(mu_a - mu_b).mean()
        return sim
