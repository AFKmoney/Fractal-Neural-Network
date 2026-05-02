"""
NFN Tokenizer — two-tier implementation:
  1. CharTokenizer  : character-level (128 printable ASCII + specials)
  2. BPETokenizer   : byte-pair encoding (wraps HuggingFace tokenizers if available)

The tokenizer is serialisable to JSON so models can be shipped standalone.
"""

import json
import os
import re
from typing import Dict, List, Optional, Union


# Special token definitions
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SEP = "<sep>"
SYS = "<sys>"
USR = "<usr>"
AST = "<ast>"
CODE = "<code>"
ENDCODE = "</code>"

SPECIAL_TOKENS = [PAD, BOS, EOS, UNK, SEP, SYS, USR, AST, CODE, ENDCODE]


class CharTokenizer:
    """
    Character-level tokenizer.
    Vocabulary: special tokens + printable ASCII (32-126) + extended Latin.
    Default vocab size ≈ 140.
    """

    def __init__(self):
        self.special_tokens = SPECIAL_TOKENS
        self._build_vocab()

    def _build_vocab(self):
        # Printable ASCII (space to ~)
        chars = [chr(i) for i in range(32, 127)]
        # Add newline, tab explicitly
        extra = ["\n", "\t", "\r"]
        vocab = self.special_tokens + extra + [c for c in chars if c not in extra]

        self.tok2id: Dict[str, int] = {t: i for i, t in enumerate(vocab)}
        self.id2tok: Dict[int, str] = {i: t for t, i in self.tok2id.items()}
        self.vocab_size = len(vocab)

    # ── property accessors ────────────────────────────────────────────────────

    @property
    def pad_token_id(self) -> int:
        return self.tok2id[PAD]

    @property
    def bos_token_id(self) -> int:
        return self.tok2id[BOS]

    @property
    def eos_token_id(self) -> int:
        return self.tok2id[EOS]

    @property
    def unk_token_id(self) -> int:
        return self.tok2id[UNK]

    # ── encode / decode ───────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_token_id)
        for ch in text:
            ids.append(self.tok2id.get(ch, self.unk_token_id))
        if add_eos:
            ids.append(self.eos_token_id)
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = []
        for i in ids:
            tok = self.id2tok.get(i, UNK)
            if skip_special and tok in self.special_tokens:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        max_length: Optional[int] = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> Dict[str, List[List[int]]]:
        encoded = [self.encode(t, add_bos=add_bos, add_eos=add_eos, max_length=max_length)
                   for t in texts]
        if padding:
            max_len = max(len(e) for e in encoded)
            masks = []
            for e in encoded:
                pad_len = max_len - len(e)
                masks.append([1] * len(e) + [0] * pad_len)
                e += [self.pad_token_id] * pad_len
            return {"input_ids": encoded, "attention_mask": masks}
        return {"input_ids": encoded}

    # ── serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "tok2id": self.tok2id}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.special_tokens = SPECIAL_TOKENS
        tok.tok2id = data["tok2id"]
        tok.id2tok = {int(v): k for k, v in data["tok2id"].items()}
        tok.vocab_size = len(tok.tok2id)
        return tok


class BPETokenizer:
    """
    Thin wrapper around HuggingFace `tokenizers` library for BPE.
    Falls back to CharTokenizer if the library is not available.
    """

    def __init__(self, vocab_size: int = 8000, lowercase: bool = False):
        try:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
            self._hf = True
            self._vocab_size = vocab_size
            self._lowercase = lowercase
            self._tokenizer = None  # Built lazily on train()
        except ImportError:
            self._hf = False
            self._char = CharTokenizer()

    @property
    def vocab_size(self) -> int:
        if not self._hf or self._tokenizer is None:
            return self._char.vocab_size if not self._hf else self._vocab_size
        return self._tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def bos_token_id(self) -> int:
        return 1

    @property
    def eos_token_id(self) -> int:
        return 2

    def train_from_files(self, files: List[str]) -> None:
        if not self._hf:
            return
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        tokenizer = Tokenizer(models.BPE(unk_token=UNK))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        trainer = trainers.BpeTrainer(
            vocab_size=self._vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=2,
        )
        tokenizer.train(files, trainer)
        self._tokenizer = tokenizer

    def encode(self, text: str, add_bos: bool = False,
               add_eos: bool = False, max_length: Optional[int] = None) -> List[int]:
        if not self._hf or self._tokenizer is None:
            return self._char.encode(text, add_bos=add_bos, add_eos=add_eos, max_length=max_length)
        enc = self._tokenizer.encode(text)
        ids = enc.ids
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        if max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if not self._hf or self._tokenizer is None:
            return self._char.decode(ids, skip_special=skip_special)
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special)


# ── Convenience alias ─────────────────────────────────────────────────────────

NFNTokenizer = CharTokenizer


def load_tokenizer(path: Optional[str] = None) -> NFNTokenizer:
    if path and os.path.isfile(path):
        return CharTokenizer.load(path)
    return CharTokenizer()
