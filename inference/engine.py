"""
NFN Inference Engine — streaming text generation with multiple decoding strategies.

Strategies:
  - greedy        : argmax at each step
  - temperature   : sample from softmax(logits / T)
  - top-k         : restrict to top-k tokens
  - top-p (nucleus): restrict to cumulative probability ≥ p
  - beam search   : best-first beam decoding
  - mirostat      : adaptive perplexity-targeting sampler (v2)
"""

import math
import time
from typing import AsyncIterator, Callable, Dict, Generator, Iterator, List, Optional

import torch
import torch.nn.functional as F

from nfn.config import NFNConfig
from nfn.network import NFNLanguageModel
from nfn.tokenizer import NFNTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Sampling utilities
# ─────────────────────────────────────────────────────────────────────────────

def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    threshold = values[:, -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens whose cum-prob exceeds p (shift right by 1 to keep the boundary)
    remove = cumprobs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[remove] = float("-inf")
    return logits.scatter(1, sorted_idx, sorted_logits)


def mirostat_v2(
    logits: torch.Tensor,
    tau: float,
    eta: float,
    mu: float,
) -> tuple:
    """
    Mirostat v2 adaptive sampler.
    Returns (sampled_token_id, updated_mu).
    """
    probs = F.softmax(logits, dim=-1)[0]
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    k = max(1, int(torch.searchsorted(
        torch.cumsum(sorted_probs, dim=0),
        torch.tensor(1 - math.exp(-mu * math.log(2)))
    ).item() + 1))
    top_probs = sorted_probs[:k]
    top_idx = sorted_idx[:k]
    top_probs = top_probs / top_probs.sum()
    sampled = torch.multinomial(top_probs, 1)
    token_id = top_idx[sampled].item()
    surprise = -math.log2(top_probs[sampled].item() + 1e-10)
    mu = mu - eta * (surprise - tau)
    return token_id, mu


# ─────────────────────────────────────────────────────────────────────────────
# Beam search
# ─────────────────────────────────────────────────────────────────────────────

def beam_search(
    model: NFNLanguageModel,
    input_ids: torch.Tensor,   # [1, L]
    max_new_tokens: int,
    beam_width: int = 4,
    length_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    model.eval()
    B = 1
    beams = [(input_ids, 0.0)]   # (sequence, log-prob)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            candidates = []
            for seq, score in beams:
                if eos_token_id and seq[0, -1].item() == eos_token_id:
                    candidates.append((seq, score))
                    continue
                logits, _ = model(seq[:, -model.cfg.max_seq_len:])
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)[0]
                top_lp, top_idx = torch.topk(log_probs, beam_width)
                for lp, idx in zip(top_lp, top_idx):
                    new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    L = new_seq.shape[1]
                    new_score = score + lp.item() / (L ** length_penalty)
                    candidates.append((new_seq, new_score))
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

    return beams[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# Main inference engine
# ─────────────────────────────────────────────────────────────────────────────

class NFNInferenceEngine:
    """
    Wraps an NFNLanguageModel for convenient text generation.

    Usage:
        engine = NFNInferenceEngine(model, tokenizer)
        for token in engine.stream("Hello, I am"):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        model: NFNLanguageModel,
        tokenizer: NFNTokenizer,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = model.cfg
        self.device = device or next(model.parameters()).device
        self.model.eval()

    # ── Low-level token generator ─────────────────────────────────────────────

    @torch.no_grad()
    def _token_stream(
        self,
        input_ids: torch.Tensor,   # [1, L]
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        strategy: str = "top_p",
        beam_width: int = 4,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
    ) -> Iterator[int]:
        generated = input_ids.clone().to(self.device)
        mu = mirostat_tau * math.log(2)

        if strategy == "beam":
            result = beam_search(
                self.model, generated, max_new_tokens,
                beam_width=beam_width, eos_token_id=self.cfg.eos_token_id,
                device=self.device,
            )
            for tok_id in result[0, input_ids.shape[1]:].tolist():
                yield tok_id
            return

        for _ in range(max_new_tokens):
            ctx = generated[:, -self.cfg.max_seq_len:]
            logits, _ = self.model(ctx)
            logits = logits[:, -1, :]  # [1, V]

            if strategy == "greedy":
                next_id = logits.argmax(dim=-1).item()
            elif strategy == "mirostat":
                next_id, mu = mirostat_v2(logits, mirostat_tau, mirostat_eta, mu)
            else:
                # temperature + top_k + top_p
                logits = logits / max(temperature, 1e-6)
                if top_k > 0:
                    logits = top_k_filter(logits, top_k)
                if top_p < 1.0:
                    logits = top_p_filter(logits, top_p)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()

            yield next_id

            if next_id == self.cfg.eos_token_id:
                break

            generated = torch.cat([
                generated,
                torch.tensor([[next_id]], device=self.device)
            ], dim=1)

    # ── Text-level interface ──────────────────────────────────────────────────

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        strategy: str = "top_p",
        **kwargs,
    ) -> Iterator[str]:
        """Yields decoded string tokens one by one (streaming)."""
        ids = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([ids], device=self.device)

        for tok_id in self._token_stream(
            input_ids, max_new_tokens, temperature, top_k, top_p, strategy, **kwargs
        ):
            if tok_id in (self.cfg.pad_token_id, self.cfg.eos_token_id):
                break
            yield self.tokenizer.decode([tok_id], skip_special=True)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        **kwargs,
    ) -> str:
        return "".join(self.stream(prompt, max_new_tokens=max_new_tokens, **kwargs))

    # ── Async streaming (for FastAPI WebSocket) ───────────────────────────────

    async def astream(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        **kwargs,
    ) -> AsyncIterator[str]:
        import asyncio
        for tok in self.stream(prompt, max_new_tokens=max_new_tokens, **kwargs):
            yield tok
            await asyncio.sleep(0)

    # ── Perplexity / evaluation ───────────────────────────────────────────────

    @torch.no_grad()
    def perplexity(self, text: str, stride: int = 64) -> float:
        """Sliding-window perplexity over a text."""
        ids = self.tokenizer.encode(text)
        L = len(ids)
        if L < 2:
            return float("inf")
        max_len = self.cfg.max_seq_len
        nlls = []
        for i in range(0, L - 1, stride):
            chunk_ids = ids[max(0, i - max_len + 1): i + max_len]
            input_t = torch.tensor([chunk_ids[:-1]], device=self.device)
            target_t = torch.tensor([chunk_ids[1:]], device=self.device)
            if input_t.shape[1] == 0:
                continue
            logits, _ = self.model(input_t)
            nll = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                target_t.view(-1),
                ignore_index=self.cfg.pad_token_id,
            )
            nlls.append(nll.item())
        if not nlls:
            return float("inf")
        return math.exp(sum(nlls) / len(nlls))

    # ── Multi-turn chat ───────────────────────────────────────────────────────

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: str = "Tu es un assistant intelligent basé sur le Neural Fractal Network.",
        max_new_tokens: int = 400,
        **kwargs,
    ) -> str:
        """
        Format a multi-turn conversation and generate a response.
        messages: [{"role": "user"|"assistant", "content": "..."}]
        """
        prompt_parts = [f"<sys>{system}</sys>\n"]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt_parts.append(f"<usr>{content}</usr>\n")
            else:
                prompt_parts.append(f"<ast>{content}</ast>\n")
        prompt_parts.append("<ast>")
        prompt = "".join(prompt_parts)
        return self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
