"""
NFN AGI Inference Engine v4.0

Wraps FNNModel with production-grade sampling, streaming, and AGI features:
  - Token streaming (generator + async generator)
  - Speculative decoding (2-4× speedup via MTP heads)
  - Think rounds (internal reasoning before generation)
  - Tool calling (ToolCallingModel integration)
  - Continual learning (learn/retrieve/RAG)
  - Episodic memory write-through during generation

Sampling strategies: greedy, temperature, top-k, top-p (nucleus), mirostat v2
"""

import math
import time
from typing import AsyncIterator, Dict, Generator, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from nfn.config import FNNConfig
from nfn.tokenizer import NFNTokenizer
from interface.tools import ToolRegistry, ToolCallingModel, make_default_registry
from training.continual import ContinualLearner, KnowledgeStore


# ─────────────────────────────────────────────────────────────────────────────
# Sampling utilities
# ─────────────────────────────────────────────────────────────────────────────

def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k == 0:
        return logits
    values, _ = torch.topk(logits, min(k, logits.shape[-1]))
    return logits.masked_fill(logits < values[:, -1, None], float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumprobs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[remove] = float("-inf")
    return logits.scatter(1, sorted_idx, sorted_logits)


def mirostat_v2(logits: torch.Tensor, tau: float, eta: float, mu: float) -> Tuple[int, float]:
    probs = F.softmax(logits, dim=-1)[0]
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    k = max(1, int(torch.searchsorted(
        torch.cumsum(sorted_probs, dim=0),
        torch.tensor(1 - math.exp(-mu * math.log(2))),
    ).item() + 1))
    top_probs = sorted_probs[:k] / sorted_probs[:k].sum()
    top_idx   = sorted_idx[:k]
    sampled   = torch.multinomial(top_probs, 1)
    token_id  = top_idx[sampled].item()
    surprise  = -math.log2(top_probs[sampled].item() + 1e-10)
    mu        = mu - eta * (surprise - tau)
    return token_id, mu


def sample_token(
    logits: torch.Tensor,   # [1, V]
    temperature:  float = 1.0,
    top_k:        int   = 50,
    top_p:        float = 0.95,
    greedy:       bool  = False,
) -> int:
    if greedy:
        return logits.argmax(-1).item()
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-6)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    return torch.multinomial(F.softmax(logits, -1), 1).item()


# ─────────────────────────────────────────────────────────────────────────────
# AGI Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

class AGIInferenceEngine:
    """
    Production inference engine for FNNModel.

    Features:
      - Streaming token generation (sync + async)
      - Speculative decoding via MTP heads
      - think(n_rounds) — internal reasoning before responding
      - tool_call — ToolCallingModel with registered tools
      - learn(text) — online continual learning
      - retrieve(query) — episodic/semantic memory search
      - chat() — multi-turn conversation with persistent memory
      - Memory write-through: every generated token is written to episodic memory

    Usage:
        engine = AGIInferenceEngine(model, tokenizer)
        engine.register_tool("calc", "Evaluate math", {...}, lambda args: str(eval(args["expr"])))

        for tok in engine.stream("What is 2^10?", use_tools=True):
            print(tok, end="", flush=True)
    """

    def __init__(
        self,
        model,                              # FNNModel
        tokenizer: NFNTokenizer,
        registry: Optional[ToolRegistry] = None,
        device: Optional[torch.device]   = None,
        knowledge_store_path: Optional[str] = None,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = model.cfg
        self.device    = device or next(model.parameters()).device

        # Tool calling
        self.registry     = registry or make_default_registry()
        self.tool_model   = ToolCallingModel(model, tokenizer, self.registry)

        # Continual learning
        store = None
        if knowledge_store_path:
            try:
                store = KnowledgeStore.load(knowledge_store_path)
            except Exception:
                store = KnowledgeStore()
        self.learner = ContinualLearner(model, tokenizer, store=store)
        self._store_path = knowledge_store_path

        # Mirostat state per session
        self._mirostat_mu: float = 5.0

        model.eval()

    # ── Encoding / decoding ──────────────────────────────────────────────────

    def _encode(self, text: str, add_bos: bool = True) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_bos=add_bos)
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def _decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def _decode(self, ids: torch.Tensor) -> str:
        flat = ids[0].tolist()
        bos  = getattr(self.tokenizer, "bos_id", self.cfg.bos_token_id)
        if flat and flat[0] == bos:
            flat = flat[1:]
        return self.tokenizer.decode(flat)

    # ── Core streaming generator ─────────────────────────────────────────────

    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int  = 512,
        temperature:    float = 0.8,
        top_k:          int   = 50,
        top_p:          float = 0.95,
        greedy:         bool  = False,
        mirostat:       bool  = False,
        mirostat_tau:   float = 5.0,
        mirostat_eta:   float = 0.1,
        think_rounds:   int   = 0,
        write_memory:   bool  = True,
        stop_tokens:    Optional[List[int]] = None,
    ) -> Iterator[str]:
        """
        Yields decoded tokens one by one as they are generated.
        Writes each generated token to episodic memory if write_memory=True.
        """
        self.model.eval()
        input_ids = self._encode(prompt)
        max_ctx = self.cfg.max_seq_len
        if input_ids.shape[1] > max_ctx:
            input_ids = input_ids[:, -max_ctx:]

        # Optional think rounds (internal reasoning before output)
        if think_rounds > 0 and hasattr(self.model, "think"):
            input_ids = self.model.think(input_ids, n_rounds=think_rounds)

        stop_ids = stop_tokens or [self.cfg.eos_token_id]
        ids = input_ids
        mu  = self._mirostat_mu
        n_generated = 0

        for _ in range(max_new_tokens):
            ctx = ids[:, -max_ctx:]
            logits, _ = self.model(ctx, write_memory=write_memory)
            next_logits = logits[:, -1, :]   # [1, V]

            # Suppress EOS for the first 5 tokens so we always get some output
            if n_generated < 5:
                for sid in stop_ids:
                    next_logits[:, sid] = float("-inf")

            if mirostat:
                tok, mu = mirostat_v2(next_logits, mirostat_tau, mirostat_eta, mu)
            else:
                tok = sample_token(next_logits, temperature, top_k, top_p, greedy)

            self._mirostat_mu = mu
            ids = torch.cat([ids, torch.tensor([[tok]], device=self.device)], dim=1)
            n_generated += 1

            if tok in stop_ids:
                break

            yield self._decode_token(tok)

    async def astream(
        self,
        prompt: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Async wrapper around stream() for WebSocket / SSE use."""
        for token in self.stream(prompt, **kwargs):
            yield token

    # ── Full generation ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int   = 512,
        temperature:    float = 0.8,
        top_k:          int   = 50,
        top_p:          float = 0.95,
        greedy:         bool  = False,
        think_rounds:   int   = 0,
        use_speculative: bool = False,
        write_memory:   bool  = True,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate and return full text string."""
        if use_speculative and hasattr(self.model, "mtp") and self.model.mtp is not None:
            input_ids = self._encode(prompt)
            out_ids = self.model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, speculative=True,
            )
            return self._decode(out_ids[:, input_ids.shape[1]:])

        tokens = []
        for tok in self.stream(
            prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            greedy=greedy, think_rounds=think_rounds, write_memory=write_memory,
        ):
            tokens.append(tok)
            if stop:
                text_so_far = "".join(tokens)
                for s in stop:
                    if s in text_so_far:
                        idx = text_so_far.find(s)
                        return text_so_far[:idx]
        return "".join(tokens)

    # ── Chat (multi-turn with memory) ────────────────────────────────────────

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_new_tokens: int   = 512,
        temperature:    float = 0.8,
        top_k:          int   = 50,
        top_p:          float = 0.95,
        think_rounds:   int   = 0,
        use_tools:      bool  = False,
        write_memory:   bool  = True,
    ) -> str:
        """
        Multi-turn chat with persistent episodic memory.

        Messages format: [{"role": "user"|"assistant"|"system", "content": "..."}]
        """
        prompt = self._build_chat_prompt(messages, system)

        if use_tools and len(self.registry) > 0:
            sys_block = (system or "") + "\n" + self.registry.system_block()
            return self.tool_model.generate(
                prompt="",
                system_prompt=sys_block,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        return self.generate(
            prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            think_rounds=think_rounds, write_memory=write_memory,
        )

    def _build_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
    ) -> str:
        parts = []
        if system:
            parts.append(f"<system>{system}</system>")
        for m in messages:
            role    = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                parts.append(f"<user>{content}</user>")
            elif role == "assistant":
                parts.append(f"<assistant>{content}</assistant>")
            elif role == "system":
                parts.append(f"<system>{content}</system>")
        parts.append("<assistant>")
        return "\n".join(parts)

    # ── Think ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def think(
        self,
        prompt: str,
        n_rounds: int = 3,
        max_tokens_per_round: int = 128,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Run n_rounds of internal reasoning.
        Returns {"thoughts": [str], "final_prompt": str}

        The model thinks about the prompt before answering — each round
        the hidden state is refined via the RecursiveReasoner's goal alignment.
        """
        self.model.eval()
        if hasattr(self.model, "set_goal"):
            input_ids = self._encode(prompt)
            self.model.set_goal(input_ids)

        thoughts = []
        current_prompt = prompt + "\n<think>"

        for i in range(n_rounds):
            thought_tokens = []
            for tok in self.stream(
                current_prompt,
                max_new_tokens=max_tokens_per_round,
                temperature=temperature,
                write_memory=False,
                stop_tokens=[self.cfg.eos_token_id],
            ):
                thought_tokens.append(tok)
                if "</think>" in "".join(thought_tokens):
                    break
            thought = "".join(thought_tokens).replace("</think>", "").strip()
            thoughts.append(thought)
            current_prompt = current_prompt + thought + f"</think>\n<think>"

        if hasattr(self.model, "reset_goal"):
            self.model.reset_goal()

        final_prompt = prompt + "\n" + "\n".join(f"[Thought {i+1}] {t}" for i, t in enumerate(thoughts))
        return {"thoughts": thoughts, "final_prompt": final_prompt}

    # ── Continual learning ────────────────────────────────────────────────────

    def learn(self, text: str, source: str = "") -> Dict:
        """Learn new text. Returns embedding stats."""
        t0   = time.time()
        emb  = self.learner.learn(text, source=source)
        elapsed = time.time() - t0
        if self._store_path:
            self.learner.save_store(self._store_path)
        return {
            "learned": True,
            "chars":   len(text),
            "emb_norm": emb.norm().item(),
            "store_size": len(self.learner.store),
            "elapsed_s":  round(elapsed, 3),
        }

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        """Retrieve top-k relevant memories."""
        results = self.learner.retrieve(query, top_k=top_k)
        return [{"text": t, "score": round(s, 4)} for t, s in results]

    def generate_with_rag(self, prompt: str, top_k: int = 3, **kwargs) -> str:
        """Generate with retrieval-augmented context."""
        return self.learner.generate_with_rag(prompt, top_k_retrieve=top_k, **kwargs)

    # ── Tool management ──────────────────────────────────────────────────────

    def register_tool(self, name: str, description: str, parameters: dict, fn) -> None:
        from interface.tools import ToolSpec
        self.registry.register(ToolSpec(name, description, parameters, fn))
        self.tool_model = ToolCallingModel(self.model, self.tokenizer, self.registry)

    # ── Model info ────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        params = sum(p.numel() for p in self.model.parameters())
        motifs = getattr(self.cfg, "motifs", ["binary_tree"])
        n_levels = getattr(self.cfg, "n_levels", 4)
        tools = []
        try:
            tools = list(self.registry._tools.keys())
        except Exception:
            pass
        return {
            "status": "ready",
            # Nested model object that the JS info tab reads directly
            "model": {
                "params":      f"{params/1e6:.2f}M",
                "device":      str(self.device),
                "vocab_size":  self.cfg.vocab_size,
                "d_model":     self.cfg.d_model,
                "n_blocks":    self.cfg.n_blocks,
                "n_levels":    n_levels,
                "motifs":      motifs,
                "max_seq_len": self.cfg.max_seq_len,
            },
            # Flat fields kept for backwards compat and sidebar badges
            "params_M":     round(params / 1e6, 2),
            "d_model":      self.cfg.d_model,
            "n_blocks":     self.cfg.n_blocks,
            "vocab_size":   self.cfg.vocab_size,
            "max_seq_len":  self.cfg.max_seq_len,
            "tools":        tools,
            "knowledge_entries": len(self.learner.store),
            "features": {
                "episodic_memory":   self.cfg.use_episodic_memory,
                "working_memory":    self.cfg.use_working_memory,
                "causal_graph":      self.cfg.use_causal_graph,
                "goal_predictor":    self.cfg.use_goal_predictor,
                "self_consistency":  self.cfg.use_self_consistency,
                "free_energy":       self.cfg.use_free_energy,
                "mixture_of_depths": self.cfg.use_mixture_of_depths,
                "multi_token_pred":  self.cfg.use_multi_token_pred,
                "hyper_net":         self.cfg.use_hyper_net,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatible alias (old code imports NFNInferenceEngine)
# ─────────────────────────────────────────────────────────────────────────────

class NFNInferenceEngine(AGIInferenceEngine):
    """Alias for backwards compatibility with v3 code."""

    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, **kwargs)

    def generate(self, prompt: str, max_new_tokens: int = 200, **kwargs) -> str:
        return super().generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
