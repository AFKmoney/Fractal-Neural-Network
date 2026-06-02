"""
WebExplorer — Autonomous web fetching and exploration for NFN AGI

Uses only stdlib: urllib.request, urllib.parse, html.parser.
No requests, no BeautifulSoup, no extra dependencies.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Dict, Iterator, List, Optional, Set


# ─────────────────────────────────────────────────────────────────────────────
# HTML parser — strips tags, extracts title and clean body text
# ─────────────────────────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """
    Single-pass HTML parser that:
    - Extracts the <title> tag content
    - Collects visible text, skipping <script>, <style>, <nav>,
      <footer>, <header>, <noscript>, <aside>
    - Extracts <a href="…"> links
    """

    _SKIP_TAGS = frozenset({"script", "style", "nav", "footer", "header",
                            "noscript", "aside"})
    # Void elements that appear in _SKIP_TAGS must be listed here so that
    # handle_starttag never increments _in_skip for them (they have no end tag).
    _VOID_SKIP = frozenset({"meta", "link", "br", "hr", "input", "img",
                            "area", "base", "col", "embed", "param",
                            "source", "track", "wbr"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str = ""
        self.text_parts: List[str] = []
        self.links: List[str] = []

        self._in_skip: int = 0
        self._in_title: bool = False

    # ── HTMLParser overrides ──────────────────────────────────────────────────

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        # Only non-void tags can have matching end tags, so only increment
        # the skip counter for those.  Void elements like <meta> and <link>
        # have no </meta> / </link> end tag and would permanently raise the
        # counter if we incremented here.
        if tag in self._SKIP_TAGS and tag not in self._VOID_SKIP:
            self._in_skip += 1

        if tag == "title" and self._in_skip == 0:
            self._in_title = True

        if tag == "a":
            attr_map = dict(attrs)
            href = attr_map.get("href", "")
            if href:
                self.links.append(href)

        # Treat block-level elements as implicit whitespace
        if tag in {"br", "p", "div", "li", "h1", "h2", "h3", "h4", "h5",
                   "h6", "tr", "td", "th", "blockquote", "article", "section"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._in_skip = max(0, self._in_skip - 1)
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._in_skip == 0:
            self.text_parts.append(data)

    # ── Post-process ─────────────────────────────────────────────────────────

    def get_text(self) -> str:
        raw = "".join(self.text_parts)
        # Collapse multiple blank lines to at most two
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        # Strip trailing/leading whitespace on each line
        lines = [ln.strip() for ln in raw.splitlines()]
        # Drop blank-only lines that add no value (keep at most one blank line)
        result: List[str] = []
        prev_blank = False
        for ln in lines:
            if ln:
                result.append(ln)
                prev_blank = False
            elif not prev_blank:
                result.append("")
                prev_blank = True
        return "\n".join(result).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Internal fetch helper
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_raw(url: str, timeout: int = 10) -> Dict:
    """
    Fetch *url* and return a dict with keys:
      url, title, text, n_chars, _html, _links, error (optional)
    """
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; NFN-AGI-Explorer/1.0; "
            "+https://github.com/nfn-agi)"
        ),
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
    }

    result: Dict = {
        "url": url, "title": "", "text": "", "n_chars": 0,
        "_html": "", "_links": [],
    }

    try:
        req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            if "html" not in content_type and "xhtml" not in content_type:
                result["error"] = f"Non-HTML content-type: {content_type}"
                return result
            raw_bytes = resp.read(2 * 1024 * 1024)

        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=")[-1].split(";")[0].strip()
        try:
            html = raw_bytes.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            html = raw_bytes.decode("utf-8", errors="replace")

        parser = _TextExtractor()
        parser.feed(html)

        result["title"]   = parser.title.strip()
        result["text"]    = parser.get_text()
        result["n_chars"] = len(result["text"])
        result["_html"]   = html
        result["_links"]  = parser.links

    except urllib.error.HTTPError as e:
        result["error"] = f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        result["error"] = f"URL error: {e.reason}"
    except TimeoutError:
        result["error"] = "Timeout"
    except Exception as e:  # noqa: BLE001
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class WebExplorer:
    """
    Fetches web pages and extracts clean text for NFN self-learning.

    All network I/O is done with urllib (stdlib only) so no extra packages
    are required.
    """

    # ── fetch ─────────────────────────────────────────────────────────────────

    def fetch(self, url: str, timeout: int = 10) -> Dict:
        """
        Fetch *url* and return a dict:
          {url, title, text, n_chars, error?}

        Internal keys (_html, _links) are stripped from the returned dict.
        """
        result = _fetch_raw(url, timeout)
        # Strip internal keys before returning
        result.pop("_html", None)
        result.pop("_links", None)
        return result

    # ── extract_links ─────────────────────────────────────────────────────────

    def extract_links(self, html: str, base_url: str) -> List[str]:
        """
        Return a deduplicated list of absolute http/https URLs extracted
        from *html* relative to *base_url*.
        """
        parser = _TextExtractor()
        parser.feed(html)

        seen: Set[str] = set()
        links: List[str] = []
        for href in parser.links:
            href = href.strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            abs_url = urllib.parse.urljoin(base_url, href)
            abs_url = abs_url.split("#")[0]
            parsed = urllib.parse.urlparse(abs_url)
            if parsed.scheme not in ("http", "https"):
                continue
            if abs_url not in seen:
                seen.add(abs_url)
                links.append(abs_url)
        return links

    # ── score_link ────────────────────────────────────────────────────────────

    def score_link(self, url: str, text: str, keywords: List[str]) -> float:
        """
        Simple keyword-overlap score for autonomous navigation.

        Score = (2 * keyword hits in url + keyword hits in link text)
                normalised by number of keywords.
        Returns a float in [0, 2].  When no keywords given, returns 1.0.
        """
        if not keywords:
            return 1.0

        url_lower  = url.lower()
        text_lower = text.lower()

        hits_url  = sum(1 for kw in keywords if kw.lower() in url_lower)
        hits_text = sum(1 for kw in keywords if kw.lower() in text_lower)

        return (hits_url * 2.0 + hits_text) / max(len(keywords), 1)

    # ── explore ───────────────────────────────────────────────────────────────

    def explore(
        self,
        seed_url: str,
        n_pages: int = 5,
        keywords: Optional[List[str]] = None,
        visited: Optional[Set[str]] = None,
    ) -> Iterator[Dict]:
        """
        Generator that explores the web starting from *seed_url*.

        Algorithm:
          1. Fetch current URL, yield the page dict (public keys only).
          2. Extract all outgoing links, score by *keywords*.
          3. Follow the best unvisited link.
          4. Repeat until *n_pages* have been fetched or no new links remain.

        Yielded dicts include all keys from fetch() plus 'score'.
        """
        if keywords is None:
            keywords = []
        if visited is None:
            visited = set()

        queue: List[tuple] = [(1.0, seed_url)]  # (score, url)
        fetched = 0

        while queue and fetched < n_pages:
            score, current_url = queue.pop(0)

            if current_url in visited:
                # Try next in queue
                continue
            visited.add(current_url)

            raw = _fetch_raw(current_url)
            html   = raw.pop("_html", "")
            links  = raw.pop("_links", [])
            raw["score"] = score
            fetched += 1

            yield raw  # public dict: url, title, text, n_chars, score, error?

            if raw.get("error"):
                continue

            # Resolve and score outgoing links
            candidates: List[tuple] = []
            abs_links = self.extract_links(html, current_url)
            for link in abs_links:
                if link not in visited:
                    link_score = self.score_link(link, link, keywords)
                    candidates.append((link_score, link))

            # Sort descending by score, insert best candidates at front of queue
            candidates.sort(key=lambda x: x[0], reverse=True)
            # Prepend so best links are explored next
            queue = candidates[:10] + queue  # keep queue bounded
