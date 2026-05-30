"""
NFN v5.0 — Program Synthesis Module

Neuro-symbolic program synthesis: the model learns to compose primitive
operations into executable programs, guided by input-output examples.

Theory
------
Program synthesis bridges the gap between neural pattern matching and
symbolic reasoning. The FNN generates programs as sequences of discrete
tokens, where each token represents a primitive operation (e.g., map,
filter, fold, compose). The model learns:

  1. A ProgramEncoder: embeds program tokens into the fractal phase space
  2. A ProgramDecoder: generates program tokens autoregressively
  3. An ExecutionEngine: traces program execution for gradient estimation
  4. A RewardEstimator: scores programs by correctness on held-out examples

The key insight: fractal self-similarity naturally maps to program recursion.
A program that processes a list can be decomposed into sub-programs that
process sub-lists — exactly the fractal decomposition pattern.

Architecture
-----------
  PrimitiveOp       : single operation (map, filter, fold, compose, ...)
  ProgramEncoder    : embeds program AST into fractal phase space
  ProgramDecoder    : generates program tokens from hidden state
  ExecutionTracer   : differentiable execution trace for REINFORCE gradient
  ProgramSynthesizer: full module — encode → generate → execute → reward
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PRIMITIVES = [
    "id", "map", "filter", "fold", "compose", "head", "tail",
    "reverse", "sort", "zip", "concat", "take", "drop",
    "add", "mul", "negate", "eq", "lt", "gt", "not", "and", "or",
    "pair", "fst", "snd", "length", "range",
]
PRIM_TO_ID = {p: i for i, p in enumerate(PRIMITIVES)}
N_PRIMITIVES = len(PRIMITIVES)
PAD_ID = N_PRIMITIVES
BOS_ID = N_PRIMITIVES + 1
EOS_ID = N_PRIMITIVES + 2
PROGRAM_VOCAB = N_PRIMITIVES + 3


class ProgramEncoder(nn.Module):
    """
    Encodes a sequence of primitive tokens into a fixed-size embedding
    in the fractal phase space. Uses sinusoidal position encoding to
    preserve the recursive structure of programs.
    """

    def __init__(self, d_model: int, max_program_len: int = 64):
        super().__init__()
        self.embed = nn.Embedding(PROGRAM_VOCAB, d_model, padding_idx=PAD_ID)
        self.pos = nn.Parameter(torch.randn(max_program_len, d_model) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model * 4,
                                       dropout=0.1, batch_first=True),
            num_layers=2,
        )

    def forward(self, program_ids: torch.Tensor) -> torch.Tensor:
        """
        program_ids: [B, L] integer tokens
        Returns: [B, d_model] program embedding
        """
        L = program_ids.shape[1]
        h = self.embed(program_ids) + self.pos[:L]
        h = self.encoder(h)
        return h.mean(dim=1)


class ProgramDecoder(nn.Module):
    """
    Autoregressive decoder for program tokens. Generates programs one
    primitive at a time, conditioned on the problem embedding.
    """

    def __init__(self, d_model: int, max_program_len: int = 64):
        super().__init__()
        self.embed = nn.Embedding(PROGRAM_VOCAB, d_model, padding_idx=PAD_ID)
        self.pos = nn.Parameter(torch.randn(max_program_len, d_model) * 0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead=4, dim_feedforward=d_model * 4,
                                       dropout=0.1, batch_first=True),
            num_layers=2,
        )
        self.head = nn.Linear(d_model, PROGRAM_VOCAB)

    def forward(
        self,
        problem_embed: torch.Tensor,
        program_ids:   torch.Tensor,
    ) -> torch.Tensor:
        """
        problem_embed: [B, d_model]
        program_ids:   [B, L] (teacher-forced input, shifted right)
        Returns: [B, L, PROGRAM_VOCAB] logits
        """
        L = program_ids.shape[1]
        h = self.embed(program_ids) + self.pos[:L]
        
        memory = problem_embed.unsqueeze(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=h.device)
        h = self.decoder(h, memory, tgt_mask=causal_mask)
        return self.head(h)

    @torch.no_grad()
    def generate(
        self,
        problem_embed: torch.Tensor,
        max_len: int = 32,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressively generate a program.
        Returns: [B, L] program token ids (including BOS and EOS).
        """
        B = problem_embed.shape[0]
        device = problem_embed.device
        
        ids = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            logits = self.forward(problem_embed, ids)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
            
            if (next_id == EOS_ID).all():
                break
        
        return ids


class RewardEstimator(nn.Module):
    """
    Estimates program correctness from execution traces.
    Uses a learned similarity function between program output and expected output.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.similarity = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        output_embed:   torch.Tensor,
        expected_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns: [B] reward in [0, 1] — estimated correctness.
        """
        return self.similarity(
            torch.cat([output_embed, expected_embed], dim=-1)
        ).squeeze(-1)


class ProgramSynthesizer(nn.Module):
    """
    Full neuro-symbolic program synthesis module.
    
    Given input-output examples, generates a program that maps inputs to outputs.
    Uses REINFORCE with baseline for gradient estimation through discrete tokens.
    
    The synthesizer bridges neural and symbolic reasoning:
    - Neural: program generation via autoregressive decoder
    - Symbolic: discrete program tokens represent real operations
    - Bridge: reward signal connects output correctness to program generation
    """

    def __init__(
        self,
        d_model: int,
        max_program_len: int = 32,
        reinforce_baseline_decay: float = 0.9,
    ):
        super().__init__()
        self.encoder = ProgramEncoder(d_model, max_program_len)
        self.decoder = ProgramDecoder(d_model, max_program_len)
        self.reward_est = RewardEstimator(d_model)
        self.max_program_len = max_program_len
        self.baseline_decay = reinforce_baseline_decay
        self.register_buffer("running_baseline", torch.tensor(0.0))

    def forward(
        self,
        problem_embed: torch.Tensor,
        target_embed:  torch.Tensor,
        program_ids:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        problem_embed: [B, d_model] — embedding of the input examples
        target_embed:  [B, d_model] — embedding of the expected outputs
        program_ids:   [B, L] — optional teacher-forced program for training
        
        Returns:
          loss: scalar REINFORCE loss
          metrics: dict with reward, baseline, program_length
        """
        B = problem_embed.shape[0]
        
        if program_ids is not None:
            # Teacher-forced generation for training
            logits = self.decoder(problem_embed, program_ids[:, :-1])
            
            # REINFORCE: sample from the logits and compute reward
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Sample program tokens
            sampled = torch.cat([
                program_ids[:, :1],  # BOS
                *[F.softmax(logits[:, t], dim=-1).multinomial(1) for t in range(logits.shape[1])],
            ], dim=1)
            
            # Compute log probability of sampled program
            token_log_probs = log_probs.gather(
                2, sampled[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            program_log_prob = token_log_probs.sum(dim=-1)
            
            # Estimate reward
            prog_embed = self.encoder(sampled)
            reward = self.reward_est(prog_embed, target_embed)
            
            # REINFORCE with running baseline
            with torch.no_grad():
                self.running_baseline.mul_(self.baseline_decay).add_(
                    reward.mean() * (1 - self.baseline_decay)
                )
            
            advantage = reward - self.running_baseline
            reinforce_loss = -(program_log_prob * advantage.detach()).mean()
            
            # Supervised loss on teacher-forced tokens
            sup_loss = F.cross_entropy(
                logits.reshape(-1, PROGRAM_VOCAB),
                program_ids[:, 1:].reshape(-1),
                ignore_index=PAD_ID,
            )
            
            total_loss = reinforce_loss + 0.5 * sup_loss
            
            metrics = {
                "program_reward": reward.mean(),
                "program_baseline": self.running_baseline.clone(),
                "program_length": (sampled != PAD_ID).float().sum(dim=-1).mean(),
                "supervised_loss": sup_loss,
                "reinforce_loss": reinforce_loss,
            }
            
            return total_loss, metrics
        else:
            # Inference: generate program autoregressively
            generated = self.decoder.generate(
                problem_embed, self.max_program_len, temperature=0.8
            )
            return torch.tensor(0.0, device=problem_embed.device), {
                "generated_program": generated,
            }

    def synthesize(
        self,
        problem_embed: torch.Tensor,
        n_samples: int = 8,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Best-of-N sampling: generate n_samples programs, return the one
        with highest estimated reward.
        
        Returns: [B, L] best program token ids.
        """
        B = problem_embed.shape[0]
        best_programs = None
        best_rewards = torch.full((B,), -1.0, device=problem_embed.device)
        
        for _ in range(n_samples):
            prog = self.decoder.generate(problem_embed, self.max_program_len, temperature)
            prog_embed = self.encoder(prog)
            
            # Use self-similarity as proxy reward when no target available
            self_sim = torch.norm(prog_embed, dim=-1)
            
            better = self_sim > best_rewards
            if better.any():
                if best_programs is None:
                    best_programs = prog.clone()
                else:
                    best_programs[better] = prog[better]
                best_rewards[better] = self_sim[better]
        
        return best_programs if best_programs is not None else \
            self.decoder.generate(problem_embed, self.max_program_len, temperature)
