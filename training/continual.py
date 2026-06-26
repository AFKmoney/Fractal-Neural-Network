"""
NFN Continual Learning — Online Knowledge Consolidation

The fundamental limitation of LLMs: everything they know was fixed at training time.
Ask GPT-4 about something that happened after its cutoff — it hallucinates or refuses.

NFN's continual learning gives the model a biological solution:
  - New information → episodic ring buffer (immediate, exact, bounded)
  - Replay → incremental SVD condensate update (no SGD, no catastrophic forgetting)
  - Retrieval → k-NN from episodic + condensate projection (at inference time)

This is the hippocampus-neocortex model of memory (Kumaran et al., 2016):
  Hippocampus (episodic)  : rapid, exact encoding of episodes
  Neocortex   (semantic)  : slow, compressed, robust knowledge
  Consolidation           : sleep replay moves episodic → semantic

For NFN this means:
  - Feed any text → it's immediately in episodic memory, retrievable next token
  - After consolidation → it's compressed into semantic SVD, retrievable forever
  - No gradient update, no retraining, no catastrophic forgetting

Classes:
  ContinualLearner : wraps FNNModel with online learn() / retrieve() API
  KnowledgeStore   : persistent JSON-serialisable store of consolidated knowledge
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Store (persistent)
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeStore:
    """
    Serialisable store for condensed knowledge snippets.

    Each entry = {"text": str, "embedding": List[float], "source": str}

    Embeddings are mean-pooled hidden states — used for nearest-neighbour
    retrieval at inference time.

    Stored as gzipped JSON. Grows incrementally; retrieval is O(N) but
    for N < 100K, this is fast enough (< 1ms on CPU).
    """

    def __init__(self, max_entries: int = 10_000):
        self.max_entries = max_entries
        self.entries: List[Dict] = []
        self._emb_cache: Optional[torch.Tensor] = None   # [N, d]

    def add(self, text: str, embedding: torch.Tensor, source: str = ""):
        """Add one knowledge snippet."""
        self.entries.append({
            "text":      text,
            "embedding": embedding.cpu().tolist(),
            "source":    source,
        })
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        self._emb_cache = None   # invalidate cache

    def retrieve(
        self,
        query_emb: torch.Tensor,   # [d]
        top_k: int = 4,
    ) -> List[Tuple[str, float]]:
        """Return top_k (text, score) pairs most similar to query_emb."""
        if not self.entries:
            return []
        if self._emb_cache is None:
            embs = [e["embedding"] for e in self.entries]
            try:
                self._emb_cache = torch.tensor(embs, dtype=torch.float32)
            except (ValueError, RuntimeError):
                # Stale entries from a different model dimension — discard
                self.entries.clear()
                return []
        cache = self._emb_cache.to(query_emb.device)
        if cache.shape[-1] != query_emb.shape[-1]:
            # Dimension mismatch (model rebuilt with different d_model)
            self.entries.clear()
            self._emb_cache = None
            return []
        q = F.normalize(query_emb.float(), dim=-1).unsqueeze(0)   # [1, d]
        e = F.normalize(cache, dim=-1)                             # [N, d]
        scores = (q @ e.T).squeeze(0)                              # [N]
        k = min(top_k, len(self.entries))
        top_scores, top_idx = scores.topk(k)
        return [(self.entries[i]["text"], top_scores[j].item())
                for j, i in enumerate(top_idx.tolist())]

    def save(self, path: str):
        import gzip
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.entries)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(payload)

    @classmethod
    def load(cls, path: str, max_entries: int = 10_000) -> "KnowledgeStore":
        import gzip
        store = cls(max_entries)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            store.entries = json.load(f)
        return store

    def __len__(self) -> int:
        return len(self.entries)


# ─────────────────────────────────────────────────────────────────────────────
# Continual Learner
# ─────────────────────────────────────────────────────────────────────────────

class ContinualLearner:
    """
    Wraps FNNModel with continual online learning capabilities.

    The model can:
      1. learn(text)         — encode new text into episodic memory immediately
      2. retrieve(query)     — find the most relevant learned snippets
      3. generate_with_rag() — generate with retrieved context prepended
      4. consolidate()       — move episodic → semantic condensate
      5. save/load store     — persist knowledge across sessions

    No gradient updates. No retraining. Truly online.

    The three learning mechanisms at work:
      A. Episodic write-through: every token the model processes is written
         to the ring buffer with its hidden-state key.
      B. Semantic condensate: after consolidation, the SVD condensate
         captures the statistical structure of all seen texts.
      C. Knowledge store: explicit text snippets with mean-pool embeddings
         for fast retrieval via cosine similarity.
    """

    def __init__(
        self,
        model,                   # FNNModel
        tokenizer,               # NFNTokenizer
        store: Optional[KnowledgeStore] = None,
        chunk_size: int = 256,   # tokens per learn() chunk
        consolidate_after: int = 10,  # consolidate after every N learn() calls
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.store     = store or KnowledgeStore()
        self.chunk_size = chunk_size
        self.consolidate_after = consolidate_after
        self._learn_calls = 0
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def learn(
        self,
        text: str,
        source: str = "",
        write_to_store: bool = True,
    ) -> torch.Tensor:
        """
        Encode new text into the model's memory immediately.

        Steps:
          1. Tokenise and chunk
          2. Run forward (write_memory=True) → writes to episodic ring buffers
          3. Extract mean-pool embedding of the text
          4. Add text + embedding to KnowledgeStore for retrieval
          5. Periodically trigger consolidation (episodic → semantic SVD)

        Returns mean-pool embedding [d] for the learned text.
        """
        self.model.eval()
        ids = self.tokenizer.encode(text)
        if not ids:
            return torch.zeros(self.model.cfg.d_model, device=self.device)

        embeddings = []
        for start in range(0, len(ids), self.chunk_size):
            chunk_ids = ids[start: start + self.chunk_size]
            x = torch.tensor(chunk_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            # forward with write_memory=True writes episodic keys
            logits, losses = self.model(x, write_memory=True)
            # Extract hidden state from the model's last block output
            # We use a proxy: the lm_head inverse is hard, so re-run embed+blocks
            h = self.model.embed(x)
            for block in self.model.blocks:
                result = block(h, write_memory=False)
                h = result[0] if isinstance(result, tuple) else result
            h = self.model.ln_f(h)
            embeddings.append(h.mean(1).squeeze(0))   # [d]

        mean_emb = torch.stack(embeddings).mean(0)    # [d]

        if write_to_store:
            # Store in chunks for better retrieval granularity
            chunk_tokens = self.tokenizer.encode(text)
            words = text.split()
            chunk_words = max(1, len(words) // max(1, len(embeddings)))
            for i, emb in enumerate(embeddings):
                chunk_text = " ".join(words[i * chunk_words: (i + 1) * chunk_words])
                if chunk_text:
                    self.store.add(chunk_text, emb, source=source)

        self._learn_calls += 1
        if self._learn_calls % self.consolidate_after == 0:
            self.consolidate()

        return mean_emb

    @torch.no_grad()
    def retrieve(
        self,
        query: str,
        top_k: int = 4,
    ) -> List[Tuple[str, float]]:
        """
        Find the most relevant learned snippets for a query.

        Returns List of (text, similarity_score) pairs.
        """
        ids = self.tokenizer.encode(query)
        if not ids:
            return []
        x = torch.tensor(ids[-self.chunk_size:], dtype=torch.long, device=self.device).unsqueeze(0)
        h = self.model.embed(x)
        for block in self.model.blocks:
            result = block(h, write_memory=False)
            h = result[0] if isinstance(result, tuple) else result
        h = self.model.ln_f(h)
        query_emb = h.mean(1).squeeze(0)
        return self.store.retrieve(query_emb, top_k=top_k)

    @torch.no_grad()
    def generate_with_rag(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        top_k_retrieve: int = 3,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate with Retrieval-Augmented Generation.

        1. Retrieve top-k relevant snippets from the knowledge store
        2. Prepend them as context
        3. Generate with the augmented prompt

        This is how the model uses everything it has learned —
        the retrieved context appears as if the model "remembered" it.
        """
        retrieved = self.retrieve(prompt, top_k=top_k_retrieve)

        if retrieved:
            context_parts = ["Relevant context from memory:"]
            for i, (text, score) in enumerate(retrieved, 1):
                context_parts.append(f"[{i}] (relevance={score:.2f}) {text}")
            context = "\n".join(context_parts)
            augmented_prompt = f"{context}\n\nQuery: {prompt}"
        else:
            augmented_prompt = prompt

        ids = self.tokenizer.encode(augmented_prompt, add_bos=True)
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        max_ctx = self.model.cfg.max_seq_len
        if input_ids.shape[1] > max_ctx:
            input_ids = input_ids[:, -max_ctx:]

        out_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        new_ids  = out_ids[:, input_ids.shape[1]:]
        flat     = new_ids[0].tolist()
        return self.tokenizer.decode(flat)

    def consolidate(self):
        """
        Trigger episodic → semantic consolidation across all AGIBlocks.

        This compresses the episodic ring buffer content into the semantic
        SVD condensate — the 'neocortex' consolidation analogue.
        """
        self.model.eval()
        with torch.no_grad():
            for block in self.model.blocks:
                if hasattr(block, "memory") and block.memory is not None:
                    block.memory.maybe_consolidate()

    def save_store(self, path: str):
        self.store.save(path)

    def load_store(self, path: str):
        self.store = KnowledgeStore.load(path)

    def stats(self) -> Dict:
        return {
            "knowledge_entries": len(self.store),
            "learn_calls":       self._learn_calls,
            "consolidations":    self._learn_calls // max(1, self.consolidate_after),
            "device":            str(self.device),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Continual fine-tuning (for when you DO want gradient updates)
# ─────────────────────────────────────────────────────────────────────────────

class EWCRegularizer:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Prevents catastrophic forgetting when doing continual fine-tuning.

    After training on task A, compute Fisher information F_A.
    When training on task B, add penalty:
        L_EWC = λ · Σ_i F_A_i · (θ_i - θ*_A_i)²

    This anchors important weights (high Fisher) near their task-A values
    while allowing unimportant weights to change freely.

    Usage:
        ewc = EWCRegularizer(model, lambda_ewc=1000)
        # Train on task A...
        ewc.register_task("task_a", dataloader_a)
        # Train on task B with EWC penalty:
        loss = task_b_loss + ewc.penalty(model)
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0):
        self.model      = model
        self.lambda_ewc = lambda_ewc
        self.tasks: Dict[str, Dict[str, torch.Tensor]] = {}  # task → {name: (mean, fisher)}

    @torch.no_grad()
    def register_task(
        self,
        task_name: str,
        loss_samples: List[torch.Tensor],   # list of per-sample losses (already backward'd)
    ):
        """
        Compute and store Fisher information for the current task.

        Call this AFTER training on a task, before starting the next task.
        loss_samples: list of scalar tensors (one per example) with gradients.
        """
        # Save current param values
        means = {n: p.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        # Fisher ≈ E[∇logP² ] — approximated via squared gradients
        fishers = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        self.model.zero_grad()
        for loss in loss_samples:
            loss.backward(retain_graph=True)
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fishers[n] += p.grad.data.clone() ** 2
            self.model.zero_grad()

        n = max(len(loss_samples), 1)
        for n_key in fishers:
            fishers[n_key] /= n

        self.tasks[task_name] = {"means": means, "fishers": fishers}

    def penalty(self, model: Optional[nn.Module] = None) -> torch.Tensor:
        """Return EWC penalty summed across all registered tasks."""
        m = model or self.model
        device = next(m.parameters()).device
        loss = torch.tensor(0.0, device=device)
        for task_data in self.tasks.values():
            means   = task_data["means"]
            fishers = task_data["fishers"]
            for n, p in m.named_parameters():
                if n in means and p.requires_grad:
                    loss = loss + (fishers[n].to(device) * (p - means[n].to(device)) ** 2).sum()
        return self.lambda_ewc * loss * 0.5
