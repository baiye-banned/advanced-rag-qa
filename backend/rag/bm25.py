"""Pure-Python BM25 keyword ranking (issue #288).

The keyword route used to rank windows by a hand-tuned hit count. That score knows
nothing about document frequency or document length, so a window stuffed with a
common word (「制度」「员工」) could outrank the one window that actually contains the
asked-for clause number or error code. BM25 adds exactly those two signals:

* inverse document frequency -- a term that appears in every window is worth little;
* length normalisation -- a hit in a 40-character window is worth more than the same
  hit buried in a 900-character one.

The document-frequency corpus is **every retrievable keyword window of the knowledge
base**, not the SQL ``LIKE`` candidate set. Building it from the candidate set would
make IDF depend on the current question and quietly reward whatever happened to be
pre-filtered. ``retrieval.py`` owns that corpus scan and its cache; this module only
tokenises documents and accumulates corpus statistics.

Tokenisation rules (kept dependency-free on purpose):

* whitespace is ignored and text is lower-cased first;
* consecutive Chinese characters become every unigram and every adjacent bigram of
  the run, so two-character words work without a dictionary;
* a contiguous run of ASCII letters/digits is one token, and runs joined by ``-``,
  ``_`` or ``.`` (``TS-999``, ``E-4021``) keep the whole spelling in addition to its
  parts;
* numeric tokens (``42``, ``5000``) are therefore preserved verbatim, which is what
  clause-number and amount questions need.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Mapping

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

_CJK = "㐀-䶿一-鿿豈-﫿"
# One CJK run, allowing whitespace between characters: normalisation removes spaces,
# so 「考 假」 and 「考假」 must yield the same bigram.
_CJK_RUN_RE = re.compile(rf"[{_CJK}]+(?:\s*[{_CJK}]+)*")
# ASCII letters/digits, optionally joined by - _ . (error codes, clause ids).
_ASCII_WORD_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_WORD_PART_RE = re.compile(r"[-_.]+")
_WHITESPACE_RE = re.compile(r"\s+")

__all__ = [
    "BM25CorpusTooLarge",
    "BM25Index",
    "DEFAULT_B",
    "DEFAULT_K1",
    "query_terms",
    "tokenize",
]


class BM25CorpusTooLarge(ValueError):
    """Raised when the corpus exceeds the configured window limit.

    Callers must fall back to the legacy score and record an explicit degradation;
    silently swapping the DF corpus for the candidate set is not an option.
    """

    def __init__(self, doc_count: int, max_docs: int):
        super().__init__(f"BM25 corpus exceeds {max_docs} windows (saw at least {doc_count})")
        self.doc_count = doc_count
        self.max_docs = max_docs


def tokenize(text) -> list[str]:
    """Return BM25 tokens for ``text`` (see the module docstring for the rules)."""
    value = "" if text is None else str(text).lower()
    tokens: list[str] = []

    for match in _ASCII_WORD_RE.finditer(value):
        word = match.group(0)
        tokens.append(word)
        if any(separator in word for separator in "-_."):
            tokens.extend(part for part in _WORD_PART_RE.split(word) if part)

    for match in _CJK_RUN_RE.finditer(value):
        run = _WHITESPACE_RE.sub("", match.group(0))
        tokens.extend(run)
        if len(run) > 1:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))

    return tokens


def query_terms(text) -> tuple[str, ...]:
    """Deduplicated, sorted query terms; sorting keeps the score reproducible."""
    return tuple(sorted(set(tokenize(text))))


class BM25Index:
    """Corpus statistics for BM25.

    Only corpus-level numbers are retained (window count, average length and document
    frequency per term). Per-window term frequencies are recomputed from the text of
    the few windows that survive the ``LIKE`` pre-filter, so a large knowledge base
    does not require storing a Counter for every window.
    """

    def __init__(
        self,
        documents: Iterable[str],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        max_docs: int = 0,
    ):
        self.k1 = max(0.0, float(k1))
        self.b = min(1.0, max(0.0, float(b)))
        self.max_docs = max(0, int(max_docs or 0))
        self._df: Counter[str] = Counter()
        self._doc_count = 0
        self._total_length = 0

        for text in documents:
            if self.max_docs and self._doc_count >= self.max_docs:
                raise BM25CorpusTooLarge(self._doc_count + 1, self.max_docs)
            counts = Counter(tokenize(text))
            self._df.update(counts)
            self._doc_count += 1
            self._total_length += sum(counts.values())

        self._avgdl = self._total_length / self._doc_count if self._doc_count else 0.0

    @property
    def doc_count(self) -> int:
        return self._doc_count

    @property
    def avgdl(self) -> float:
        return self._avgdl

    @property
    def df(self) -> Mapping[str, int]:
        return self._df

    def idf(self, term: str) -> float:
        """Non-negative BM25 IDF for one term."""
        document_frequency = self._df.get(term, 0)
        if self._doc_count <= 0 or document_frequency <= 0:
            return 0.0
        return math.log(
            1.0 + (self._doc_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def score_counts(
        self, counts: Mapping[str, int], length: int, terms: Iterable[str]
    ) -> float:
        """Score pre-tokenised content; ``length`` is its token count."""
        if self._doc_count <= 0 or length <= 0:
            return 0.0
        average = self._avgdl
        normalisation = self.k1 * (
            1.0 - self.b + (self.b * (length / average) if average > 0 else 0.0)
        )
        score = 0.0
        # Sorted iteration keeps floating-point addition order stable across processes.
        for term in sorted(set(terms)):
            frequency = counts.get(term, 0)
            if frequency <= 0:
                continue
            weight = self.idf(term)
            if weight <= 0:
                continue
            score += weight * (frequency * (self.k1 + 1.0)) / (frequency + normalisation)
        return score

    def score_text(self, text, terms: Iterable[str]) -> float:
        """Score a single candidate window against the corpus statistics."""
        tokens = tokenize(text)
        return self.score_counts(Counter(tokens), len(tokens), terms)
