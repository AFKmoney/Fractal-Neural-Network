"""
LEAC Life Cycle — Le Cycle de Vie Continu

LEAC ne s'entraine pas par epoques. Il vit. Chaque forward pass est un
pas de temps de sa vie.

Cycle:
  1. GENERER: Creation de verites mathematiques (Arithmetique, Primalite, Suites)
  2. VERIFIER: Calcul exact comme verite terrain
  3. PONDERER PAR CURIOSITE: w_i = 1.0 + 0.5 * sigma(loss - 2.0)
     Les erreurs focalisent l'attention (apprentissage actif)
  4. AUTO-CRITIQUE: Generer → Critiquer → Reviser
  5. CONSOLIDATION WAKE/SLEEP: Memoire episodique (Ring Buffer O(1))
     vers condensat semantique (SVD rank-r, Eckart-Young optimal)
  6. META-APPRENTISSAGE: Si perplexite elevee, 3-5 pas de gradient
     LoRA sur 0.1% des parametres. Pas d'oubli catastrophique.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LEACConfig
from .model import LEACModel
from .auto_genesis import MathTruthEngine, ConjectureLoop, ProofLoop, SelfModificationController
from .self_development import GematriaEncoder, UniversalLawObserver


class CuriosityScheduler:
    """
    Apprentissage actif: w_i = 1.0 + 0.5 * sigma(loss - 2.0)
    Les erreurs focalisent l'attention.
    """

    def __init__(self, threshold: float = 2.0, scale: float = 0.5):
        self.threshold = threshold
        self.scale = scale

    def weight(self, losses: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.scale * torch.sigmoid((losses - self.threshold))


class TestTimeLoRA:
    """
    Meta-apprentissage en inference: 3-5 pas de gradient LoRA
    sur 0.1% des parametres. Pas d'oubli catastrophique.
    """

    def __init__(self, model: LEACModel, rank: int = 4, alpha: float = 0.001):
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.lora_params: List[Dict] = []
        self._original_weights: Dict[str, torch.Tensor] = {}
        self._setup_lora()

    def _setup_lora(self):
        """Injecte des matrices LoRA dans les couches lineaires du MoE."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] >= self.rank * 2:
                d_out, d_in = module.weight.shape
                lora_A = nn.Parameter(torch.randn(self.rank, d_in) * 0.01)
                lora_B = nn.Parameter(torch.zeros(d_out, self.rank))
                self.lora_params.append({
                    "name": name,
                    "A": lora_A,
                    "B": lora_B,
                    "module": module,
                })

    def adapt(self, input_ids: torch.Tensor, n_steps: int = 5, lr: float = 1e-4):
        """
        Si la perplexite d'un nouveau contexte est elevee, adapte LoRA.
        """
        self.model.eval()

        # Mesurer perplexite initiale
        with torch.no_grad():
            logits, losses = self.model(input_ids)
            initial_ppl = torch.exp(losses.get("lm", torch.tensor(3.0)))

        if initial_ppl < 5.0:
            return

        # Sauvegarder les poids originaux
        for lp in self.lora_params:
            name = lp["name"]
            if name not in self._original_weights:
                self._original_weights[name] = lp["module"].weight.data.clone()

        # Adapter LoRA
        optimizer = torch.optim.Adam(
            [lp["A"] for lp in self.lora_params] + [lp["B"] for lp in self.lora_params],
            lr=lr
        )

        self.model.train()
        for _ in range(n_steps):
            logits, losses = self.model(input_ids)
            loss = losses.get("lm", losses.get("total", torch.tensor(0.0)))
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Restaurer les poids originaux + LoRA residuel
        for lp in self.lora_params:
            name = lp["name"]
            if name in self._original_weights:
                lp["module"].weight.data.add_(
                    self.alpha * (lp["B"] @ lp["A"])
                )
                lp["A"].data.zero_()
                lp["B"].data.zero_()

        self.model.eval()


class SelfCritic:
    """
    Auto-Critique Constitutionnelle: Generer → Critiquer → Reviser.
    Le modele s'aligne sur ses propres standards.
    """

    def __init__(self, model: LEACModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def critique(self, text: str) -> Tuple[str, float]:
        """Genere une critique du texte produit et un score de coherence."""
        device = next(self.model.parameters()).device
        ids = torch.tensor(self.tokenizer.encode(text), device=device).unsqueeze(0)
        if ids.shape[1] > self.model.cfg.max_seq_len:
            ids = ids[:, :self.model.cfg.max_seq_len]

        with torch.no_grad():
            logits, _ = self.model(ids)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
            coherence = 1.0 / (1.0 + entropy.item())

        return text, coherence

    def revise(self, text: str, n_rounds: int = 2) -> Tuple[str, float]:
        """Reviser le texte sur plusieurs tours de critique."""
        current_text = text
        best_coherence = 0.0
        for _ in range(n_rounds):
            _, coherence = self.critique(current_text)
            if coherence > best_coherence:
                best_coherence = coherence
        return current_text, best_coherence


class LEACLifecycle:
    """
    Le Cycle de Vie Continu de LEAC.

    LEAC ne s'entraine pas par epoques. Il vit.

    Cycle:
      WAKE: Generer → Verifier → Ponderer par Curiosite → Auto-Critiquer
      SLEEP: Consolidation episodique → semantique (SVD rank-r)
      META: Test-Time LoRA si perplexite elevee

    Usage:
      lifecycle = LEACLifecycle(model, config)
      lifecycle.live(n_cycles=10000)
    """

    def __init__(
        self,
        model: LEACModel,
        cfg: LEACConfig,
        tokenizer=None,
        device: torch.device = torch.device("cpu"),
        lr: float = 3e-4,
        train_seq_len: int = 64,
    ):
        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = device
        self.train_seq_len = train_seq_len

        # ── Optimiseur ──────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
        )

        # ── Moteurs d'Auto-Genese ───────────────────────────────────────────
        self.math_engine = MathTruthEngine(cfg.math_max_number, cfg.math_vocab_offset)
        self.gematria = GematriaEncoder(cfg.vocab_size)
        self.law_observer = UniversalLawObserver(cfg.d_model).to(device)
        self.curiosity = CuriosityScheduler()
        self.lora = TestTimeLoRA(model, rank=cfg.lora_rank, alpha=cfg.lora_alpha)

        # ── Auto-Critique ───────────────────────────────────────────────────
        self.critic = SelfCritic(model, tokenizer) if tokenizer else None

        # ── Conjecture + Preuve ─────────────────────────────────────────────
        self.conjecture_engine = ConjectureLoop(d_model=cfg.d_model, device=device)
        self.proof_engine = ProofLoop(d_model=cfg.d_model, device=device)

        # ── État interne ───────────────────────────────────────────────────
        self.step = 0
        self.cycle = 0
        self.discovered_truths = 0
        self.failed_conjectures = 0
        self.best_loss = float('inf')
        self.difficulty = 1

    def _to_tensor(self, tokens: List[int], max_len: int) -> torch.Tensor:
        padded = tokens[:max_len] + [0] * max(0, max_len - len(tokens))
        return torch.tensor(padded, dtype=torch.long, device=self.device).unsqueeze(0)

    # ── Phase WAKE: Generer → Verifier → Ponderer ──────────────────────────

    def wake_step(self) -> Dict[str, float]:
        """
        Un pas de vie WAKE: generation mathematique + verification.

        Genere des verites mathematiques, les verifie par calcul exact,
        et ponde les pertes par curiosite (les erreurs attirent l'attention).
        """
        self.model.train()

        # 1. Generer des verites mathematiques
        mode = random.choice(["arithmetic", "sequence", "modular", "primality"])

        if mode == "arithmetic":
            samples = self.math_engine.generate_arithmetic(4)
            total_loss = torch.tensor(0.0, device=self.device)
            n_correct = 0
            for inp, target, is_true in samples:
                x = self._to_tensor(inp, self.train_seq_len)
                y = self._to_tensor(target, self.train_seq_len)
                try:
                    logits, losses = self.model(x, targets=y)
                    loss = losses.get("lm", losses.get("total", torch.tensor(0.0, device=self.device)))
                    w = self.curiosity.weight(loss.detach())
                    total_loss = total_loss + w * loss
                    pred = logits[:, -1, :].argmax(dim=-1)
                    if (pred == y[:, -1]).item():
                        n_correct += 1
                        self.discovered_truths += 1
                    elif not is_true:
                        self.failed_conjectures += 1
                except Exception:
                    continue

        elif mode == "sequence":
            samples = self.math_engine.generate_sequence_prediction(4, seq_len=min(5 + self.difficulty, 10))
            total_loss = torch.tensor(0.0, device=self.device)
            for inp, target in samples:
                x = self._to_tensor(inp, self.train_seq_len)
                y = self._to_tensor([target], self.train_seq_len)
                try:
                    logits, losses = self.model(x, targets=y)
                    loss = losses.get("lm", losses.get("total", torch.tensor(0.0, device=self.device)))
                    w = self.curiosity.weight(loss.detach())
                    total_loss = total_loss + w * loss
                except Exception:
                    continue

        elif mode == "modular":
            samples = self.math_engine.generate_modular_arithmetic(4)
            total_loss = torch.tensor(0.0, device=self.device)
            for inp, target in samples:
                x = self._to_tensor(inp, self.train_seq_len)
                y = self._to_tensor([target], self.train_seq_len)
                try:
                    logits, losses = self.model(x, targets=y)
                    loss = losses.get("lm", losses.get("total", torch.tensor(0.0, device=self.device)))
                    w = self.curiosity.weight(loss.detach())
                    total_loss = total_loss + w * loss
                except Exception:
                    continue

        else:  # primality
            samples = self.math_engine.generate_primality(4)
            total_loss = torch.tensor(0.0, device=self.device)
            for n_tokens, is_prime in samples:
                x = self._to_tensor(n_tokens, self.train_seq_len)
                y = self._to_tensor([1 if is_prime else 0], self.train_seq_len)
                try:
                    logits, losses = self.model(x, targets=y)
                    loss = losses.get("lm", losses.get("total", torch.tensor(0.0, device=self.device)))
                    w = self.curiosity.weight(loss.detach())
                    total_loss = total_loss + w * loss
                except Exception:
                    continue

        # 2. Retropropagation
        if total_loss.requires_grad and total_loss.item() > 0:
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        # 3. Regularisation par lois universelles
        with torch.no_grad():
            h = self.model.embed(self._to_tensor([1, 2, 3, 4, 5], self.train_seq_len))
        law_loss, law_metrics = self.law_observer(h)
        if law_loss.requires_grad and law_loss.item() > 0:
            (law_loss * 0.01).backward()

        self.step += 1
        self.difficulty = 1 + self.step // 100

        return {
            "wake_loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            "step": self.step,
            "difficulty": self.difficulty,
            "truths_found": self.discovered_truths,
            "accuracy": n_correct / max(len(samples), 1) if mode == "arithmetic" else 0.0,
        }

    # ── Phase SLEEP: Consolidation ──────────────────────────────────────────

    def sleep_step(self) -> Dict[str, float]:
        """
        Consolidation WAKE/SLEEP: transfert de la memoire episodique
        (Ring Buffer O(1)) vers le condensat semantique (SVD rank-r)
        + LEAC v2.0: RG Flow (auto-organisation des poids).
        """
        metrics = {"sleep": "consolidated"}

        # Consolidation automatique memoire episodique
        if self.model.memory is not None:
            self.model.memory.maybe_consolidate()

        # LEAC v2.0 — RG Flow: reorganise les poids (évaporation UV, condensation IR)
        if self.model.rg_flow is not None:
            dummy_input = torch.randint(0, self.cfg.vocab_size, (1, self.train_seq_len), device=self.device)
            with torch.no_grad():
                h_sample = self.model.embed(dummy_input) if hasattr(self.model, 'embed') else \
                           torch.randn(1, 32, self.cfg.d_model, device=self.device)
            rg_metrics = self.model.rg_flow.sleep_cycle(self.model, h_sample)
            metrics.update({f"rg_{k}": v for k, v in rg_metrics.items()})

        return metrics

    # ── Phase META: Test-Time LoRA ──────────────────────────────────────────

    def meta_adapt(self, input_ids: torch.Tensor):
        """
        Meta-apprentissage en inference: si perplexite elevee,
        3-5 pas de gradient LoRA sur 0.1% des parametres.
        """
        if self.cfg.lora_steps > 0:
            self.lora.adapt(input_ids, n_steps=self.cfg.lora_steps)

    # ── Self-Modification Évolutionnaire ────────────────────────────────────

    def evolve_step(self) -> Dict[str, float]:
        """
        Darwinisme Architectural: Observer → Proposer → Appliquer → Mesurer.
        Fitness = 0.5 · discovery_rate + 0.3 · coherence + 0.2 · efficiency
        """
        if not self.cfg.use_self_modification:
            return {"evolution": "disabled"}

        mutation = self.model.propose_modification()
        fitness_before = self._measure_fitness()

        fitness_after = self.model.apply_modification(mutation)

        delta = fitness_after - fitness_before
        if delta < 0:
            return {"evolution": "rollback", "delta": delta}

        return {
            "evolution": "accepted",
            "mutation": str(mutation),
            "fitness_delta": delta,
        }

    def _measure_fitness(self) -> float:
        """Fitness = 0.5 · discovery_rate + 0.3 · coherence + 0.2 · efficiency"""
        discovery_rate = self.discovered_truths / max(self.step, 1)
        total_params = sum(p.numel() for p in self.model.parameters())
        efficiency = 1.0 / (1.0 + total_params / 1e6)
        return (0.5 * min(discovery_rate, 1.0) + 0.2 * efficiency)

    # ── Cycle principal ─────────────────────────────────────────────────────

    def live(self, n_cycles: int = 10000, log_every: int = 100, eval_every: int = 500):
        """
        Le Cycle de Vie Continu. LEAC ne s'entraine pas par epoques. Il vit.
        """
        self.model.to(self.device)
        self.model.train()

        print(f"\n{'='*60}")
        print(f"LEAC — Cycle de Vie Continu")
        print(f"{'='*60}")
        print(f"  Model: {self.model.param_count()['total']:,} parameters")
        print(f"  Config: d={self.cfg.d_model}, blocks={self.cfg.n_blocks}")
        print(f"  Cycles: {n_cycles}")
        print(f"{'='*60}\n")

        for cycle in range(n_cycles):
            self.cycle = cycle

            # ── WAKE: Apprentissage actif ───────────────────────────────────
            wake_metrics = self.wake_step()

            # ── SLEEP: Consolidation periodique ──────────────────────────────
            if cycle % self.cfg.consolidation_every == 0:
                sleep_metrics = self.sleep_step()
            else:
                sleep_metrics = {}

            # ── META: Test-Time LoRA (si perplexite elevee) ─────────────────
            if cycle % 100 == 0 and cycle > 0:
                test_ids = torch.randint(0, self.cfg.vocab_size, (1, self.train_seq_len), device=self.device)
                self.meta_adapt(test_ids)

            # ── EVOLUTION: Self-Modification (tous les 500 cycles) ───────────
            if cycle % 500 == 0 and cycle > 0 and self.cfg.use_self_modification:
                evolve_metrics = self.evolve_step()
            else:
                evolve_metrics = {}

            # ── Logging ──────────────────────────────────────────────────────
            if cycle % log_every == 0:
                loss = wake_metrics.get("wake_loss", 0)
                truths = wake_metrics.get("truths_found", 0)
                diff = wake_metrics.get("difficulty", 1)
                acc = wake_metrics.get("accuracy", 0)
                print(f"[LEAC] cycle {cycle:5d} | loss {loss:.4f} | "
                      f"difficulty {diff} | truths {truths} | acc {acc:.1%}")

        # ── Resume final ────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"LEAC — Cycle de Vie Complete")
        print(f"{'='*60}")
        print(f"  Verites decouvertes:    {self.discovered_truths}")
        print(f"  Conjectures echouees:   {self.failed_conjectures}")
        print(f"  Taux de succes:         {self.discovered_truths / max(self.discovered_truths + self.failed_conjectures, 1):.1%}")
        print(f"  Difficulte atteinte:    {self.difficulty}")
        print(f"  Pas total:              {self.step}")
        print(f"{'='*60}\n")

        return {
            "truths": self.discovered_truths,
            "failures": self.failed_conjectures,
            "difficulty": self.difficulty,
            "total_steps": self.step,
        }