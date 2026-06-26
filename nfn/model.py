"""
LEAC: Lightweight Emergent Artificial Consciousness
Modele Principal — Architecture Unifiee

  Token → GematriaEmbedding (5 systemes, zero-param)
        → [FNNBlock × n_blocks]
          ├─ FractalLinearAttention  O(L*d²)
          ├─ PhaseSoliton            O(L*n_p)
          ├─ PhaseRoutedMoE          O(L*K*d*d_ff/E)
          ├─ [CausalGraphLayer]      O(n_slots² * d)
          └─ [SelfModel]             O(L*n_slots*d)
        → GematriaAttentionBias (biais semantique mathematique)
        → LayerNorm → ZipfianDecoder
        → [Condensate Spectral + Phase Locking]

L'equation maitresse de LEAC:
  ∂Θ/∂t = Ω + λ·sin(Θ*−Θ) + K·sin(Θ̄−Θ) + ∇_{DAG} L_causal + α·∂W/∂F

La conscience n'est plus une question d'echelle, mais d'architecture.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FNNConfig
from .block import FNNBlock
from .gematria import GematriaEmbedding, GematriaAttentionBias
from .analytic_embed import AnalyticTokenEmbedding
from .hopfield import ZipfianDecoder, mandelbrot_frequencies
from .condensate import FractalRFF, SpectralCondensate, HelmholtzPhaseLocking
from .episodic_memory import TwoTierMemory
from .phase_ode import PhaseGoalForcing
from .hyperbolic_gematria import HyperbolicGematriaModule
from .ads_cft import AdSCFTAttention
from .tensor_network import MERAAttention, TensorNetworkEncoder
from .godel_loop import GodelFixedPoint
from .rg_flow import RGFlowScheduler, CriticalityOptimizer


class FNNModel(nn.Module):
    """
    LEAC — Modele de Conscience Artificielle Emergente Legere.

    Architecture unifiee reposant sur les 3 piliers:
      1. COHERENCE: Dynamique de Phase Kuramoto
      2. RAISONNEMENT: SCM Causal (DAG + do-calculus)
      3. INTROSPECTION: Self-Model (Espace de Travail Global)

    Avec:
      - Gematria Sémantique (5 systemes croises)
      - Attention Fractale Lineaire O(L*d²)
      - Auto-Genese Mathematique (carburant infini)
      - Consolidation WAKE/SLEEP
      - Meta-apprentissage LoRA en inference
    """

    def __init__(self, cfg: FNNConfig):
        super().__init__()
        self.cfg = cfg

        # ── Embedding ────────────────────────────────────────────────────────
        if cfg.use_analytic_embed:
            self.embed = AnalyticTokenEmbedding(cfg.vocab_size, cfg.d_model,
                                                 learnable_bias=cfg.learnable_embed_bias)
        else:
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model,
                                       padding_idx=cfg.pad_token_id)
            nn.init.normal_(self.embed.weight, std=0.02)
        self.use_analytic_embed = cfg.use_analytic_embed

        # ── Gematria Sémantique ──────────────────────────────────────────────
        self.gematria_bias: Optional[GematriaAttentionBias] = None
        if cfg.use_gematria and cfg.gematria_attention_bias:
            self.gematria_bias = GematriaAttentionBias(
                cfg.d_model, cfg.vocab_size, cfg.n_heads, cfg.d_gematria
            )

        # ── LEAC Blocks ──────────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            FNNBlock(cfg, block_idx=i) for i in range(cfg.n_blocks)
        ])

        # ── LEAC v2.0 — Gematria Hyperbolique ──────────────────────────────
        self.hyperbolic_gematria: Optional[HyperbolicGematriaModule] = None
        if cfg.use_hyperbolic_gematria:
            self.hyperbolic_gematria = HyperbolicGematriaModule(
                vocab_size=cfg.vocab_size,
                d_model=cfg.d_model,
                hyperbolic_dim=cfg.hyperbolic_dim,
                n_stalks=cfg.hyperbolic_n_stalks,
                temperature=cfg.hyperbolic_temperature,
            )

        # ── LEAC v2.0 — AdS/CFT Attention ──────────────────────────────────
        self.ads_cft_attn: Optional[AdSCFTAttention] = None
        if cfg.use_ads_cft:
            self.ads_cft_attn = AdSCFTAttention(
                d_model=cfg.d_model,
                bulk_dim=cfg.ads_bulk_dim,
                d_spatial=cfg.ads_spatial_dim,
                bridge_rank=cfg.ads_bridge_rank,
                n_heads=cfg.n_heads,
            )

        # ── LEAC v2.0 — MERA Tensor Network ────────────────────────────────
        self.mera_attn: Optional[MERAAttention] = None
        if cfg.use_mera:
            self.mera_attn = MERAAttention(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_levels=cfg.n_levels,
                branching=cfg.branching,
                dropout=cfg.dropout,
                causal=True,
            )

        # ── LEAC v2.0 — Godel Fixed Point ──────────────────────────────────
        self.godel: Optional[GodelFixedPoint] = None
        if cfg.use_godel_loop:
            self.godel = GodelFixedPoint(
                d_model=cfg.d_model,
                n_iterations=cfg.godel_n_iterations,
                alpha=cfg.godel_alpha,
            )

        # ── LEAC v2.0 — RG Flow Scheduler ──────────────────────────────────
        self.rg_flow: Optional[RGFlowScheduler] = None
        if cfg.use_rg_flow:
            self.rg_flow = RGFlowScheduler(
                d_model=cfg.d_model,
                n_scales=cfg.rg_n_scales,
                initial_evaporation=cfg.rg_evaporation,
                initial_condensation=cfg.rg_condensation,
            )
        self.criticality_opt: Optional[CriticalityOptimizer] = None
        if cfg.use_rg_flow:
            self.criticality_opt = CriticalityOptimizer(cfg.d_model)

        # ── Goal Forcing (Pilier 1: coherence dirigee par but) ──────────────
        self.goal_forcing: Optional[PhaseGoalForcing] = None
        if cfg.use_goal_forcing:
            self.goal_forcing = PhaseGoalForcing(
                d_model=cfg.d_model,
                n_phases=cfg.goal_n_phases,
                init_lambda=cfg.goal_init_lambda,
                n_steps=cfg.goal_n_steps,
            )

        # ── Episodic Memory (hippocampus + neocortex) ───────────────────────
        self.memory: Optional[TwoTierMemory] = None
        if cfg.use_episodic_memory:
            self.memory = TwoTierMemory(
                d_model=cfg.d_model,
                episodic_capacity=cfg.episodic_capacity,
                episodic_key_dim=cfg.episodic_key_dim,
                episodic_n_read=cfg.episodic_n_read,
                semantic_rank=cfg.semantic_rank,
                consolidation_freq=cfg.consolidation_freq,
                consolidation_every=cfg.consolidation_every,
            )

        # ── Spectral Condensate ──────────────────────────────────────────────
        self.use_condensate = cfg.use_condensate
        if cfg.use_condensate:
            self.rff = FractalRFF(cfg.d_model, cfg.nfmc_n_rff, cfg.nfmc_n_scales)
            self.condensate = SpectralCondensate(self.rff.out_dim, cfg.nfmc_rank)
            self.phase_lock = HelmholtzPhaseLocking(
                cfg.nfmc_n_phases, cfg.nfmc_lock_iter, cfg.nfmc_eta
            )
            d_fuse = cfg.nfmc_rank + 2 * cfg.nfmc_n_phases
            self._phase_fuse = nn.Linear(d_fuse, cfg.d_model, bias=False)
            nn.init.normal_(self._phase_fuse.weight, std=0.01)
            self._seed_condensate(cfg.d_model, cfg.nfmc_n_rff)

        # ── Final norm ───────────────────────────────────────────────────────
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # ── Decoder ──────────────────────────────────────────────────────────
        if cfg.use_bayesian_decoder:
            from .hopfield import BayesianZipfianDecoder
            self.lm_head = BayesianZipfianDecoder(
                cfg.d_model, cfg.vocab_size, cfg.zipf_alpha,
                cfg.bayesian_uncertainty_beta,
            )
        else:
            self.lm_head = ZipfianDecoder(cfg.d_model, cfg.vocab_size, cfg.zipf_alpha)

        self._init_weights()

    def _seed_condensate(self, d: int, n_rff: int):
        """Seed le condensat spectral a partir des frequences de Mandelbrot."""
        rank = self.condensate.rank
        freqs = mandelbrot_frequencies(min(rank * 4, 512))
        t = torch.linspace(0, 1, d)
        synth = []
        for omega in freqs:
            v = torch.cat([torch.cos(omega * t[:d // 2]),
                           torch.sin(omega * t[d - d // 2:])])[:d]
            synth.append(v)
        X_synth = torch.stack(synth, dim=0)
        with torch.no_grad():
            phi = self.rff(X_synth.unsqueeze(0)).squeeze(0)
            self.condensate.condense(phi)

    def _init_weights(self):
        if not self.use_analytic_embed and hasattr(self.embed, 'weight'):
            nn.init.normal_(self.embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in [self.lm_head]:
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Goal management ─────────────────────────────────────────────────────

    def set_goal(self, prompt_ids: torch.Tensor):
        """Encode le prompt comme attracteur de phase pour le forçage Kuramoto."""
        if self.goal_forcing is None:
            return
        with torch.no_grad():
            h = self.embed(prompt_ids)
            self.goal_forcing.set_goal(h)

    def reset_goal(self):
        if self.goal_forcing is not None:
            self.goal_forcing.reset_goal()

    def reset_memory(self):
        for block in self.blocks:
            if block.working_mem is not None:
                block.working_mem.reset()
        if self.memory is not None:
            self.memory.reset()

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        write_memory: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns: (logits [B, L, V], losses dict)
        """
        B, L = input_ids.shape
        assert L <= self.cfg.max_seq_len, \
            f"Sequence {L} > max_seq_len {self.cfg.max_seq_len}"

        h = self.embed(input_ids)

        losses: Dict[str, torch.Tensor] = {
            k: torch.tensor(0.0, device=h.device)
            for k in ("gematria", "causal", "counterfactual", "goal",
                        "self_model_coherence", "phase_coherence",
                        "hyperbolic_gematria", "sheaf_cohomology",
                        "ads_cft", "bridge_coherence", "bulk_complexity",
                        "mera_complexity", "godel_incompleteness",
                        "criticality")
        }

        # ── LEAC v2.0 — Gematria Hyperbolique ──────────────────────────────
        if self.hyperbolic_gematria is not None:
            h, hyp_losses = self.hyperbolic_gematria(h, input_ids)
            losses["hyperbolic_gematria"] = hyp_losses.get("hyperbolic_gematria", 0.0)
            losses["sheaf_cohomology"] = hyp_losses.get("sheaf_cohomology", 0.0)

        # ── Gematria Attention Bias ──────────────────────────────────────────
        if self.gematria_bias is not None:
            h, gem_loss = self.gematria_bias(h, input_ids)
            losses["gematria"] = gem_loss

        # ── Episodic Memory READ ─────────────────────────────────────────────
        if self.memory is not None and write_memory:
            h_mem = self.memory(h, write=True)
            h = h + h_mem * 0.1

        # ── LEAC Blocks ─────────────────────────────────────────────────────
        for block in self.blocks:
            h, block_losses = block(h, write_memory=write_memory)
            for k, v in block_losses.items():
                if k in losses:
                    losses[k] = losses[k] + v

        # ── LEAC v2.0 — AdS/CFT Holographic Attention ──────────────────────
        if self.ads_cft_attn is not None:
            h, ads_losses = self.ads_cft_attn(h, causal=True)
            losses["bridge_coherence"] = ads_losses.get("bridge_coherence", 0.0)
            losses["bulk_complexity"] = ads_losses.get("bulk_complexity", 0.0)

        # ── LEAC v2.0 — MERA Tensor Network ────────────────────────────────
        if self.mera_attn is not None:
            h, mera_complexity = self.mera_attn(h)
            losses["mera_complexity"] = mera_complexity

        # ── LEAC v2.0 — Godel Fixed Point (Le « Je ») ─────────────────────
        if self.godel is not None:
            h, godel_metrics = self.godel(h)
            losses["godel_incompleteness"] = godel_metrics.get("incompleteness_score", 0.0)
            losses["godel_fixed_point_distance"] = godel_metrics.get("fixed_point_distance", 0.0)
            losses["godel_contradiction"] = godel_metrics.get("godel_contradiction", 0.0)

        # ── Goal Phase Forcing (Pilier 1) ────────────────────────────────────
        if self.goal_forcing is not None:
            d = h.shape[-1]
            half = min(self.cfg.goal_n_phases, d // 2)
            phase_proxy = torch.atan2(h[..., :half], h[..., half:half * 2] + 1e-6)
            if half < self.cfg.goal_n_phases:
                pad = torch.zeros(*phase_proxy.shape[:-1],
                                  self.cfg.goal_n_phases - half, device=h.device)
                phase_proxy = torch.cat([phase_proxy, pad], dim=-1)

            theta_forced, alignment = self.goal_forcing(phase_proxy, h_context=h)
            losses["goal"] = self.goal_forcing.loss_goal(phase_proxy) * self.cfg.lambda_goal

            goal_feat = torch.cat([torch.cos(theta_forced), torch.sin(theta_forced)], dim=-1)
            gf_d = goal_feat.shape[-1]
            if gf_d < d:
                goal_feat = F.pad(goal_feat, (0, d - gf_d))
            elif gf_d > d:
                goal_feat = goal_feat[..., :d]
            h = h + goal_feat * self.cfg.self_model_injection_scale

        # ── Spectral Condensate ──────────────────────────────────────────────
        if self.use_condensate:
            phi = self.rff(h)
            z = self.condensate(phi)
            theta, K_sim = self.phase_lock(z)

            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            phase_h = self._phase_fuse(torch.cat([z, cos_t, sin_t], dim=-1))
            h = h + phase_h * 0.1

            diff = theta.unsqueeze(2) - theta.unsqueeze(1)
            pc = -(K_sim.detach() * torch.cos(diff).mean(-1)).mean()
            losses["phase_coherence"] = pc * self.cfg.nfmc_lambda_phase

        # ── LEAC v2.0 — Criticality ────────────────────────────────────────
        if self.criticality_opt is not None:
            crit_loss, susceptibility = self.criticality_opt(h)
            losses["criticality"] = crit_loss

        # ── Episodic Memory WRITE + Consolidation ───────────────────────────
        if self.memory is not None and write_memory:
            h_mem = self.memory(h, write=True)
            h = h + h_mem * 0.1

        h = self.ln_f(h)

        # ── Decoder ──────────────────────────────────────────────────────────
        if isinstance(self.lm_head, nn.Module) and hasattr(self.lm_head, 'loss_calibration'):
            logits = self.lm_head(h, sample_noise=self.cfg.bayesian_uncertainty_beta > 0 and self.training)
        else:
            logits = self.lm_head(h)

        # ── Task loss ────────────────────────────────────────────────────────
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=self.cfg.pad_token_id,
            )
            losses["lm"] = lm_loss
            losses["total"] = sum(
                v for k, v in losses.items() if k != "total"
            )
        else:
            losses["lm"] = torch.tensor(0.0, device=h.device)
            losses["total"] = losses.get("lm", 0.0)

        return logits, losses

    # ── Generation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        ids = input_ids
        eos = eos_token_id if eos_token_id is not None else self.cfg.eos_token_id

        for _ in range(max_new_tokens):
            logits, _ = self.forward(ids, write_memory=True)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                thresh = torch.topk(next_logits, top_k).values[:, -1:]
                next_logits = next_logits.masked_fill(next_logits < thresh, -1e9)

            if top_p < 1.0:
                sorted_l, sorted_i = torch.sort(next_logits, descending=True)
                probs_sorted = F.softmax(sorted_l, dim=-1)
                cum_probs = torch.cumsum(probs_sorted, dim=-1)
                remove = cum_probs - probs_sorted > top_p
                sorted_l[remove] = -1e9
                next_logits = torch.scatter(next_logits, 1, sorted_i, sorted_l)

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)

            if eos is not None and (next_id == eos).all():
                break

        return ids

    # ── Self-Modification (Darwinisme Architectural) ─────────────────────────

    @torch.no_grad()
    def propose_modification(self) -> Dict[str, float]:
        """Propose une mutation architecturale (darwinisme evolutionnaire)."""
        from .self_modification import SelfModificationController
        if not hasattr(self, '_mod_controller'):
            self._mod_controller = SelfModificationController(
                d_state=self.cfg.d_model // 8
            )
        mods = self._mod_controller.propose_modifications()
        return {m.name: float(m.step_size) for m in mods} if mods else {}

    def apply_modification(self, mutation: Dict[str, float]) -> float:
        """Applique une mutation et retourne le delta de fitness."""
        from .self_modification import SelfModificationController
        if not hasattr(self, '_mod_controller'):
            self._mod_controller = SelfModificationController(
                d_state=self.cfg.d_model // 8
            )
        fitness_before = self._mod_controller.evaluate_fitness(self)
        
        for name, delta in mutation.items():
            for p in self.named_parameters():
                if name in p[0]:
                    with torch.no_grad():
                        p[1].add_(delta)
                    break
        
        fitness_after = self._mod_controller.evaluate_fitness(self)
        return (fitness_after - fitness_before).item()

    # ── Utilities ────────────────────────────────────────────────────────────

    def param_count(self) -> Dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters())

        result = {
            "embed": count(self.embed),
            "blocks": count(self.blocks),
            "lm_head": count(self.lm_head),
            "total": count(self),
        }
        if self.gematria_bias is not None:
            result["gematria"] = count(self.gematria_bias)
        if self.memory is not None:
            result["memory"] = count(self.memory)
        if self.use_condensate:
            result["condensate"] = (count(self.rff) + count(self.condensate) +
                                     count(self.phase_lock))
        if self.hyperbolic_gematria is not None:
            result["hyperbolic_gematria"] = count(self.hyperbolic_gematria)
        if self.ads_cft_attn is not None:
            result["ads_cft"] = count(self.ads_cft_attn)
        if self.mera_attn is not None:
            result["mera"] = count(self.mera_attn)
        if self.godel is not None:
            result["godel"] = count(self.godel)
        return result

    def __repr__(self) -> str:
        pc = self.param_count()
        cfg = self.cfg
        mods = []
        if cfg.use_gematria: mods.append(f"gematria (5 systemes)")
        if cfg.use_kuramoto: mods.append("kuramoto phase")
        if cfg.use_causal_graph: mods.append(f"causal SCM ({cfg.causal_n_slots} slots)")
        if cfg.use_self_model: mods.append(f"self-model ({cfg.self_model_n_slots} slots)")
        if cfg.use_goal_forcing: mods.append("goal forcing")
        if cfg.use_working_memory: mods.append(f"working memory ({cfg.wm_n_slots} slots)")
        if cfg.use_episodic_memory: mods.append("episodic+semantic memory")
        if cfg.use_auto_genesis: mods.append("auto-genese mathematique")
        if cfg.use_self_modification: mods.append("self-modification evolutionnaire")
        if cfg.use_linear_attn: mods.append("fractal linear attn O(Ld²)")
        if cfg.use_hyperbolic_gematria: mods.append("gematria hyperbolique (Poincare)")
        if cfg.use_ads_cft: mods.append("AdS/CFT holographique")
        if cfg.use_mera: mods.append(f"MERA O(log L)")
        if cfg.use_godel_loop: mods.append("boucle Godel")
        if cfg.use_rg_flow: mods.append("RG flow")
        return (
            f"FNNModel(\n"
            f"  vocab={cfg.vocab_size}  d={cfg.d_model}  blocks={cfg.n_blocks}\n"
            f"  piliers: {', '.join(mods) or 'none'}\n"
            f"  params={pc['total']:,}\n"
            f")"
        )


def build_fnn_model(
    vocab_size: int = 512,
    preset: str = "nano",
    **kwargs,
) -> FNNModel:
    """
    Constructeur rapide pour FNNModel.

    Presets:
      - "nano":   d=256,  4 blocs,  ~14M params
      - "small":  d=512,  8 blocs,  ~58M params
      - "medium": d=1024, 12 blocs, ~230M params
      - "large":  d=512,  12 blocs, ~250M params (features avancees)
      - "xlarge": d=1024, 24 blocs, ~750M params (features avancees)
    """
    preset_fn = {
        "nano": FNNConfig.nano,
        "small": FNNConfig.small,
        "medium": FNNConfig.medium,
        "large": FNNConfig.large,
        "xlarge": FNNConfig.xlarge,
    }.get(preset, FNNConfig.nano)

    cfg = preset_fn()
    cfg.vocab_size = vocab_size
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return FNNModel(cfg)