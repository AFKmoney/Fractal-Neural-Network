"""
NFN Tokenizer — three-tier implementation:

  Tier 1 – TiktokenTokenizer  : uses OpenAI's tiktoken (cl100k_base, 100K vocab)
  Tier 2 – BPETokenizer        : trains own BPE from a corpus (no deps)
  Tier 3 – CharTokenizer       : char-level fallback (108 tokens)

Auto-selection at runtime:
  load_tokenizer()  →  tiktoken if available, else CharTokenizer

All tokenizers share the same interface:
  .encode(text, add_bos, add_eos, max_length) → List[int]
  .decode(ids, skip_special) → str
  .vocab_size, .pad_token_id, .bos_token_id, .eos_token_id
  .save(path) / .load(path)
"""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ── Special tokens ────────────────────────────────────────────────────────────
PAD     = "<pad>"
BOS     = "<bos>"
EOS     = "<eos>"
UNK     = "<unk>"
SEP     = "<sep>"
SYS     = "<sys>"
USR     = "<usr>"
AST     = "<ast>"
CODE    = "<code>"
ENDCODE = "</code>"
THINK   = "<think>"
ENDTHINK= "</think>"

SPECIAL_TOKENS = [PAD, BOS, EOS, UNK, SEP, SYS, USR, AST, CODE, ENDCODE, THINK, ENDTHINK]
N_SPECIAL = len(SPECIAL_TOKENS)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: Tiktoken wrapper (100K vocab, best quality)
# ─────────────────────────────────────────────────────────────────────────────

class TiktokenTokenizer:
    """
    Wraps tiktoken's cl100k_base (used by GPT-4 / Claude).
    Vocab size: 100,277 + N_SPECIAL special tokens.
    """

    def __init__(self):
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
            self._ok = True
        except ImportError:
            self._ok = False
            return

        # Reserve low IDs for special tokens; tiktoken IDs are offset by N_SPECIAL
        self._offset = N_SPECIAL
        self._special_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self._id_to_special = {i: t for i, t in enumerate(SPECIAL_TOKENS)}

    @property
    def available(self) -> bool:
        return self._ok

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab + N_SPECIAL if self._ok else 0

    @property
    def pad_token_id(self) -> int:  return self._special_to_id[PAD]
    @property
    def bos_token_id(self) -> int:  return self._special_to_id[BOS]
    @property
    def eos_token_id(self) -> int:  return self._special_to_id[EOS]
    @property
    def unk_token_id(self) -> int:  return self._special_to_id[UNK]

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        # Replace special token strings before tiktoken processing
        chunks = self._split_specials(text)
        ids = []
        if add_bos:
            ids.append(self.bos_token_id)
        for chunk, is_special in chunks:
            if is_special:
                ids.append(self._special_to_id.get(chunk, self.unk_token_id))
            else:
                ids.extend(i + self._offset for i in self._enc.encode(chunk))
        if add_eos:
            ids.append(self.eos_token_id)
        if max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        tiktoken_ids, parts = [], []
        i = 0
        while i < len(ids):
            id_ = ids[i]
            if id_ < N_SPECIAL:
                if tiktoken_ids:
                    parts.append(self._enc.decode(tiktoken_ids))
                    tiktoken_ids = []
                if not skip_special:
                    parts.append(self._id_to_special.get(id_, ""))
            else:
                tiktoken_ids.append(id_ - self._offset)
            i += 1
        if tiktoken_ids:
            parts.append(self._enc.decode(tiktoken_ids))
        return "".join(parts)

    def _split_specials(self, text: str) -> List[Tuple[str, bool]]:
        """Split text into (chunk, is_special) pairs."""
        pattern = "(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")"
        parts = re.split(pattern, text)
        result = []
        for p in parts:
            if p in self._special_to_id:
                result.append((p, True))
            elif p:
                result.append((p, False))
        return result

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"type": "tiktoken", "encoding": "cl100k_base"}, f)

    @classmethod
    def load(cls, path: str) -> "TiktokenTokenizer":
        return cls()


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Pure-Python BPE tokenizer (no external deps)
# ─────────────────────────────────────────────────────────────────────────────

class BPETokenizer:
    """
    Minimal byte-pair encoding tokenizer.

    Trains from scratch on a text corpus in O(V·T) time.
    Default target vocab: 32K tokens.
    """

    def __init__(self, vocab_size: int = 32_000):
        self.target_vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._trained = False
        self._build_base_vocab()

    def _build_base_vocab(self):
        """Start with special tokens + all printable bytes."""
        self.vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        offset = len(SPECIAL_TOKENS)
        # All bytes 0-255 as single-char tokens
        for b in range(256):
            ch = chr(b) if 32 <= b < 127 else f"<0x{b:02X}>"
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def _word_to_chars(self, word: str) -> List[str]:
        chars = []
        for ch in word:
            b = ord(ch)
            if 32 <= b < 127:
                chars.append(ch)
            else:
                chars.append(f"<0x{b:02X}>")
        return chars + ["</w>"]

    def train(self, text: str, n_merges: Optional[int] = None):
        """Train BPE on a raw text string."""
        if n_merges is None:
            n_merges = self.target_vocab_size - len(self.vocab)
        n_merges = max(0, n_merges)

        # Build word frequency table
        words = re.findall(r'\w+|[^\w\s]|\s+', text)
        word_freq: Counter = Counter(words)

        # Represent each word as tuple of char-tokens
        vocab_table: Dict[Tuple, int] = {}
        for word, freq in word_freq.items():
            key = tuple(self._word_to_chars(word))
            vocab_table[key] = vocab_table.get(key, 0) + freq

        for _ in range(n_merges):
            # Count all adjacent pairs
            pairs: Counter = Counter()
            for word, freq in vocab_table.items():
                for a, b in zip(word, word[1:]):
                    pairs[(a, b)] += freq
            if not pairs:
                break

            # Find most frequent pair
            best = max(pairs, key=pairs.__getitem__)
            a, b = best
            merged = a + b

            # Add to vocab
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)
                self.id_to_token[self.vocab[merged]] = merged
            self.merges.append((a, b))

            # Update vocab_table
            new_table: Dict[Tuple, int] = {}
            for word, freq in vocab_table.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_table[tuple(new_word)] = freq
            vocab_table = new_table

        self._trained = True

    def _apply_merges(self, chars: List[str]) -> List[str]:
        for a, b in self.merges:
            i = 0
            new = []
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(chars[i])
                    i += 1
            chars = new
        return chars

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:  return self.vocab[PAD]
    @property
    def bos_token_id(self) -> int:  return self.vocab[BOS]
    @property
    def eos_token_id(self) -> int:  return self.vocab[EOS]
    @property
    def unk_token_id(self) -> int:  return self.vocab[UNK]

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
        # Split on special tokens first
        pattern = "(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")"
        parts = re.split(pattern, text)
        for part in parts:
            if not part:
                continue
            if part in self.vocab and part in SPECIAL_TOKENS:
                ids.append(self.vocab[part])
            else:
                words = re.findall(r'\w+|[^\w\s]|\s+', part)
                for word in words:
                    chars = self._word_to_chars(word)
                    if self._trained:
                        chars = self._apply_merges(chars)
                    for ch in chars:
                        ids.append(self.vocab.get(ch, self.unk_token_id))
        if add_eos:
            ids.append(self.eos_token_id)
        if max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, UNK)
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        # Decode byte tokens
        for b in range(256):
            if 32 <= b < 127:
                continue
            text = text.replace(f"<0x{b:02X}>", chr(b))
        return text

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "type": "bpe",
                "vocab": self.vocab,
                "merges": self.merges,
                "target_vocab_size": self.target_vocab_size,
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data.get("target_vocab_size", 32_000))
        tok.vocab = data["vocab"]
        tok.id_to_token = {int(v) if isinstance(v, str) else v: k
                           for k, v in data["vocab"].items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        tok._trained = len(tok.merges) > 0
        return tok


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3: Character-level fallback (always available)
# ─────────────────────────────────────────────────────────────────────────────

class CharTokenizer:
    """Character-level tokenizer. 108 tokens. Always available."""

    def __init__(self):
        chars = [chr(i) for i in range(32, 127)]
        extra = ["\n", "\t", "\r"]
        vocab_list = SPECIAL_TOKENS + extra + [c for c in chars if c not in extra]
        self.tok2id: Dict[str, int] = {t: i for i, t in enumerate(vocab_list)}
        self.id2tok: Dict[int, str] = {i: t for t, i in self.tok2id.items()}

    @property
    def vocab_size(self) -> int:   return len(self.tok2id)
    @property
    def pad_token_id(self) -> int: return self.tok2id[PAD]
    @property
    def bos_token_id(self) -> int: return self.tok2id[BOS]
    @property
    def eos_token_id(self) -> int: return self.tok2id[EOS]
    @property
    def unk_token_id(self) -> int: return self.tok2id[UNK]

    def encode(self, text: str, add_bos=False, add_eos=False,
               max_length: Optional[int] = None) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_token_id)
        for ch in text:
            ids.append(self.tok2id.get(ch, self.unk_token_id))
        if add_eos:
            ids.append(self.eos_token_id)
        if max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = []
        for i in ids:
            tok = self.id2tok.get(i, UNK)
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def batch_encode(self, texts, padding=True, max_length=None,
                     add_bos=False, add_eos=False):
        encoded = [self.encode(t, add_bos=add_bos, add_eos=add_eos,
                               max_length=max_length) for t in texts]
        if padding:
            max_len = max(len(e) for e in encoded)
            masks = []
            for e in encoded:
                pad_n = max_len - len(e)
                masks.append([1]*len(e) + [0]*pad_n)
                e += [self.pad_token_id] * pad_n
            return {"input_ids": encoded, "attention_mask": masks}
        return {"input_ids": encoded}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "tok2id": self.tok2id}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.tok2id = data["tok2id"]
        tok.id2tok = {int(v) if isinstance(v, str) else v: k
                      for k, v in data["tok2id"].items()}
        return tok


# ─────────────────────────────────────────────────────────────────────────────
# Auto-select best available tokenizer
# ─────────────────────────────────────────────────────────────────────────────

NFNTokenizer = CharTokenizer   # backwards compat alias


def load_tokenizer(
    path: Optional[str] = None,
    prefer: str = "auto",   # "tiktoken" | "bpe" | "char" | "auto"
) -> "CharTokenizer | BPETokenizer | TiktokenTokenizer":
    """
    Load a tokenizer from path or create the best available.

    auto priority: tiktoken > bpe (if trained) > char
    """
    if path and os.path.isfile(path):
        with open(path) as f:
            meta = json.load(f)
        t = meta.get("type", "char")
        if t == "tiktoken":
            tok = TiktokenTokenizer.load(path)
            if tok.available:
                return tok
        elif t == "bpe":
            return BPETokenizer.load(path)
        return CharTokenizer.load(path)

    if prefer in ("auto", "tiktoken"):
        tok = TiktokenTokenizer()
        if tok.available:
            return tok

    if prefer in ("auto", "bpe"):
        pass   # untrained BPE — caller should call .train() first

    return CharTokenizer()
