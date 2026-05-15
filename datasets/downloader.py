"""
NFN — Public Dataset Downloader

Downloads and preprocesses public text datasets for NFN training.
No external dependencies beyond the Python standard library + requests/urllib.

Supported datasets
------------------
  wikipedia-en-simple  : English Simple Wikipedia (~120 MB, perfect for nano/small)
  wikipedia-en          : Full English Wikipedia (~20 GB, for medium/large)
  gutenberg-top100      : Top 100 Project Gutenberg books (~20 MB, literary style)
  openwebtext-10pct     : 10% sample of OpenWebText via HuggingFace (~2 GB)
  cc-news-10pct         : CC-News 10% sample via HuggingFace (~1 GB)
  tiny-shakespeare      : The complete works of Shakespeare (~1 MB, quick tests)
  custom                : Local file or directory of .txt files

Usage
-----
  from datasets.downloader import DatasetDownloader

  dl = DatasetDownloader(cache_dir="data")
  text = dl.get("wikipedia-en-simple", max_chars=50_000_000)
  print(f"Got {len(text):,} characters")

  # Or list what's available
  from datasets.downloader import list_datasets
  list_datasets()
"""

import gzip
import html
import json
import math
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────

DATASETS: Dict[str, dict] = {
    "tiny-shakespeare": {
        "desc":   "Complete works of Shakespeare — 1 MB, great for quick tests",
        "size":   "~1 MB",
        "url":    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "type":   "direct",
        "good_for": ["nano"],
    },
    "wikipedia-en-simple": {
        "desc":   "Simple English Wikipedia — clean, educational — 120 MB",
        "size":   "~120 MB",
        "url":    "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-abstract.xml.gz",
        "type":   "wikipedia_xml_gz",
        "good_for": ["nano", "small"],
    },
    "gutenberg-top100": {
        "desc":   "Top 100 Project Gutenberg books — literary English — 20 MB",
        "size":   "~20 MB",
        "type":   "gutenberg",
        "good_for": ["nano", "small"],
    },
    "openwebtext-10pct": {
        "desc":   "10% sample of OpenWebText (web pages) — 2 GB",
        "size":   "~2 GB",
        "type":   "huggingface",
        "hf_dataset": "Skylion007/openwebtext",
        "hf_split": "train[:10%]",
        "hf_field": "text",
        "good_for": ["small", "medium"],
    },
    "cc-news": {
        "desc":   "CC-News sample (news articles, 2016–2019) — 1 GB",
        "size":   "~1 GB",
        "type":   "huggingface",
        "hf_dataset": "cc_news",
        "hf_split": "train[:5%]",
        "hf_field": "text",
        "good_for": ["small", "medium"],
    },
    "wikipedia-en": {
        "desc":   "Full English Wikipedia — 20 GB — needs medium/large",
        "size":   "~20 GB",
        "type":   "huggingface",
        "hf_dataset": "wikipedia",
        "hf_config": "20220301.en",
        "hf_split": "train",
        "hf_field": "text",
        "good_for": ["medium", "large"],
    },
    "pile-10pct": {
        "desc":   "The Pile 10% sample — diverse English text — 8 GB",
        "size":   "~8 GB",
        "type":   "huggingface",
        "hf_dataset": "monology/pile-uncopyrighted",
        "hf_split": "train[:10%]",
        "hf_field": "text",
        "good_for": ["medium", "large"],
    },
}

GUTENBERG_URLS = [
    ("https://www.gutenberg.org/cache/epub/1342/pg1342.txt", "Pride and Prejudice"),
    ("https://www.gutenberg.org/cache/epub/11/pg11.txt",    "Alice in Wonderland"),
    ("https://www.gutenberg.org/cache/epub/2701/pg2701.txt","Moby Dick"),
    ("https://www.gutenberg.org/cache/epub/1661/pg1661.txt","Sherlock Holmes"),
    ("https://www.gutenberg.org/cache/epub/84/pg84.txt",    "Frankenstein"),
    ("https://www.gutenberg.org/cache/epub/98/pg98.txt",    "A Tale of Two Cities"),
    ("https://www.gutenberg.org/cache/epub/1232/pg1232.txt","The Prince"),
    ("https://www.gutenberg.org/cache/epub/2554/pg2554.txt","Crime and Punishment"),
    ("https://www.gutenberg.org/cache/epub/4300/pg4300.txt","Ulysses"),
    ("https://www.gutenberg.org/cache/epub/345/pg345.txt",  "Dracula"),
    ("https://www.gutenberg.org/cache/epub/74/pg74.txt",    "The Adventures of Tom Sawyer"),
    ("https://www.gutenberg.org/cache/epub/76/pg76.txt",    "Adventures of Huckleberry Finn"),
    ("https://www.gutenberg.org/cache/epub/1080/pg1080.txt","A Modest Proposal"),
    ("https://www.gutenberg.org/cache/epub/1400/pg1400.txt","Great Expectations"),
    ("https://www.gutenberg.org/cache/epub/2600/pg2600.txt","War and Peace"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Progress bar (no external deps)
# ─────────────────────────────────────────────────────────────────────────────

class _ProgressBar:
    def __init__(self, total: int, desc: str = "", width: int = 40):
        self.total   = total
        self.desc    = desc
        self.width   = width
        self.current = 0
        self.start   = time.time()

    def update(self, n: int):
        self.current = min(self.current + n, self.total)
        self._draw()

    def set(self, n: int):
        self.current = min(n, self.total)
        self._draw()

    def _draw(self):
        frac    = self.current / max(self.total, 1)
        filled  = int(frac * self.width)
        bar     = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.start
        pct     = frac * 100
        if elapsed > 0 and frac > 0:
            eta = elapsed / frac * (1 - frac)
            eta_str = f"ETA {eta:.0f}s"
        else:
            eta_str = ""
        size_str = _fmt_bytes(self.current)
        print(f"\r{self.desc} [{bar}] {pct:5.1f}% {size_str} {eta_str}  ",
              end="", flush=True)

    def close(self, msg: str = ""):
        print(f"\r{self.desc} [{'█' * self.width}] 100.0%  {msg}")


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f}KB"
    elif n < 1024 ** 3:
        return f"{n / 1024**2:.1f}MB"
    else:
        return f"{n / 1024**3:.2f}GB"


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Remove excess whitespace, fix unicode, strip Gutenberg headers/footers."""
    text = html.unescape(text)
    # Remove Gutenberg header/footer
    start_markers = ["*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"]
    end_markers   = ["*** END OF THE PROJECT GUTENBERG",   "*** END OF THIS PROJECT GUTENBERG"]
    for m in start_markers:
        idx = text.find(m)
        if idx != -1:
            text = text[text.find("\n", idx) + 1:]
    for m in end_markers:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove non-printable characters
    text = re.sub(r"[^\x09\x0a\x0d\x20-\x7e\x80-\xff]", "", text)
    return text.strip()


def _parse_wikipedia_xml(xml_text: str, max_chars: int) -> str:
    """Extract clean text from Wikipedia XML abstract dump."""
    # Extract <abstract>...</abstract> blocks
    abstracts = re.findall(r"<abstract>(.*?)</abstract>", xml_text, re.DOTALL)
    parts     = []
    total     = 0
    for ab in abstracts:
        ab = html.unescape(ab).strip()
        if ab and len(ab) > 20:
            parts.append(ab)
            total += len(ab) + 1
            if max_chars and total >= max_chars:
                break
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download url to dest with a progress bar."""
    desc = desc or dest.name
    req  = urllib.request.Request(url, headers={"User-Agent": "NFN-Downloader/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            bar   = _ProgressBar(total or 1, desc=f"  Downloading {desc}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                downloaded = 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        bar.set(downloaded)
                    else:
                        bar.update(len(chunk))
            bar.close(f"→ {_fmt_bytes(dest.stat().st_size)}")
    except urllib.error.URLError as e:
        print(f"\n  [warn] Download failed: {e}")
        return dest
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# DatasetDownloader
# ─────────────────────────────────────────────────────────────────────────────

class DatasetDownloader:
    """
    Downloads and caches public text datasets for NFN training.

    Usage:
        dl   = DatasetDownloader(cache_dir="data")
        text = dl.get("wikipedia-en-simple")
        # → returns a single large string ready for tokenisation

    The cache_dir stores downloaded files. On subsequent calls,
    the cached version is returned instantly.
    """

    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    def get(
        self,
        name:      str,
        max_chars: Optional[int] = None,
        shuffle:   bool          = False,
    ) -> str:
        """
        Return dataset `name` as a single UTF-8 string.

        Parameters
        ----------
        name      : dataset key (see list_datasets())
        max_chars : truncate text to this many characters
        shuffle   : shuffle paragraphs before returning
        """
        if name not in DATASETS:
            raise ValueError(
                f"Unknown dataset '{name}'. Run list_datasets() to see available datasets."
            )

        info = DATASETS[name]
        print(f"\n📦 Dataset: {name}")
        print(f"   {info['desc']}")
        print(f"   Size: {info['size']}")

        text = self._fetch(name, info, max_chars)

        if shuffle:
            import random
            paragraphs = [p for p in text.split("\n\n") if p.strip()]
            random.shuffle(paragraphs)
            text = "\n\n".join(paragraphs)

        if max_chars and len(text) > max_chars:
            text = text[:max_chars]

        print(f"   ✓ Ready: {len(text):,} characters ({_fmt_bytes(len(text.encode()))})")
        return text

    def get_file(self, name: str) -> Path:
        """Return the local cache path for dataset `name` (downloading if needed)."""
        info = DATASETS[name]
        return self._fetch_to_file(name, info)

    # ── Dataset-specific fetchers ─────────────────────────────────────────

    def _fetch(self, name: str, info: dict, max_chars: Optional[int]) -> str:
        dtype = info["type"]

        if dtype == "direct":
            return self._fetch_direct(name, info["url"])

        elif dtype == "wikipedia_xml_gz":
            return self._fetch_wikipedia_gz(name, info["url"], max_chars)

        elif dtype == "gutenberg":
            return self._fetch_gutenberg()

        elif dtype == "huggingface":
            return self._fetch_huggingface(name, info, max_chars)

        else:
            raise ValueError(f"Unknown dataset type: {dtype}")

    def _fetch_direct(self, name: str, url: str) -> str:
        cache = self.cache_dir / f"{name}.txt"
        if cache.exists():
            print(f"   ✓ Using cached file: {cache}")
            return cache.read_text(encoding="utf-8", errors="replace")

        _download_file(url, cache, desc=name)
        text = cache.read_text(encoding="utf-8", errors="replace")
        return _clean_text(text)

    def _fetch_wikipedia_gz(self, name: str, url: str, max_chars: Optional[int]) -> str:
        cache_txt = self.cache_dir / f"{name}.txt"
        if cache_txt.exists():
            print(f"   ✓ Using cached file: {cache_txt}")
            return cache_txt.read_text(encoding="utf-8", errors="replace")

        cache_gz = self.cache_dir / f"{name}.xml.gz"
        _download_file(url, cache_gz, desc=name)

        print(f"   Decompressing …", end="", flush=True)
        with gzip.open(cache_gz, "rt", encoding="utf-8", errors="replace") as f:
            xml_text = f.read()
        print(f" {_fmt_bytes(len(xml_text.encode()))}")

        print(f"   Parsing XML …", end="", flush=True)
        text = _parse_wikipedia_xml(xml_text, max_chars or 0)
        print(f" {len(text):,} characters")

        cache_txt.write_text(text, encoding="utf-8")
        cache_gz.unlink(missing_ok=True)  # free disk space
        return text

    def _fetch_gutenberg(self) -> str:
        cache = self.cache_dir / "gutenberg-top100.txt"
        if cache.exists():
            print(f"   ✓ Using cached file: {cache}")
            return cache.read_text(encoding="utf-8", errors="replace")

        parts = []
        print(f"   Downloading {len(GUTENBERG_URLS)} books from Project Gutenberg …")
        for url, title in GUTENBERG_URLS:
            tmp = self.cache_dir / f"_gutenberg_{Path(url).name}"
            try:
                _download_file(url, tmp, desc=title[:40])
                text = tmp.read_text(encoding="utf-8", errors="replace")
                parts.append(_clean_text(text))
                tmp.unlink(missing_ok=True)
            except Exception as e:
                print(f"   [skip] {title}: {e}")

        combined = "\n\n".join(parts)
        cache.write_text(combined, encoding="utf-8")
        return combined

    def _fetch_huggingface(
        self,
        name:      str,
        info:      dict,
        max_chars: Optional[int],
    ) -> str:
        cache = self.cache_dir / f"{name}.txt"
        if cache.exists():
            print(f"   ✓ Using cached file: {cache}")
            return cache.read_text(encoding="utf-8", errors="replace")

        try:
            from datasets import load_dataset
        except ImportError:
            print(
                "\n  [error] The 'datasets' library is required for HuggingFace datasets.\n"
                "  Install it with:  pip install datasets\n"
            )
            sys.exit(1)

        hf_name   = info["hf_dataset"]
        hf_split  = info.get("hf_split", "train")
        hf_config = info.get("hf_config")
        hf_field  = info.get("hf_field", "text")

        print(f"   Loading {hf_name} [{hf_split}] from HuggingFace …")
        if hf_config:
            ds = load_dataset(hf_name, hf_config, split=hf_split, streaming=False)
        else:
            ds = load_dataset(hf_name, split=hf_split, streaming=False)

        print(f"   Processing {len(ds):,} examples …")
        parts  = []
        total  = 0
        for ex in ds:
            text = ex.get(hf_field, "")
            if text and len(text) > 50:
                parts.append(text.strip())
                total += len(text)
                if max_chars and total >= max_chars:
                    break

        combined = "\n\n".join(parts)
        cache.write_text(combined, encoding="utf-8")
        print(f"   Saved to {cache}")
        return combined

    # ── Utility ───────────────────────────────────────────────────────────

    def split_train_val(
        self,
        text:    str,
        val_fraction: float = 0.005,
    ) -> Tuple[str, str]:
        """Split text into train and validation sets.

        Always reserves at least 1 character per side. For very small corpora,
        the validation set may shrink below val_fraction × len(text).
        """
        n = len(text)
        if n < 2:
            return text, ""
        val_size = max(1, int(n * val_fraction))
        train_size = max(1, n - val_size)
        return text[:train_size], text[train_size:]


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def list_datasets():
    """Print all available datasets with descriptions."""
    print("\n Available datasets for NFN training")
    print(" ─────────────────────────────────────────────────────────")
    print(f"  {'Name':<24} {'Size':<12} {'Good for':<20} Description")
    print(f"  {'────':<24} {'────':<12} {'────────':<20} ───────────")
    for name, info in DATASETS.items():
        good = ", ".join(info.get("good_for", []))
        print(f"  {name:<24} {info['size']:<12} {good:<20} {info['desc'][:60]}")
    print()


def download(
    dataset:   str,
    cache_dir: str = "data",
    max_chars: Optional[int] = None,
) -> str:
    """Convenience wrapper — download and return dataset text."""
    return DatasetDownloader(cache_dir).get(dataset, max_chars=max_chars)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="NFN Dataset Downloader")
    p.add_argument("dataset", nargs="?", default=None, help="Dataset name (or omit to list)")
    p.add_argument("--cache-dir", default="data", help="Cache directory")
    p.add_argument("--max-chars", type=int, default=None, help="Max characters to load")
    p.add_argument("--output", type=str, default=None, help="Save text to this file")
    args = p.parse_args()

    if args.dataset is None:
        list_datasets()
    else:
        dl   = DatasetDownloader(args.cache_dir)
        text = dl.get(args.dataset, max_chars=args.max_chars)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"Saved to {args.output}")
