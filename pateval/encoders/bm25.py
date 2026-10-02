"""BM25 over the full texts: the lexical baseline every dense number should be read against.

    score(q, d) = sum over terms t of q:  qtf(t) * idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

This is the Lucene / `bm25s` formulation with Lucene's non-negative idf. Tokenisation is lower-casing and the
word pattern ``\\b\\w\\w+\\b`` (single characters dropped), no stemming and, by default, no stop words; the
Lucene English stop list is available with ``stopwords="lucene"``. A ``Ranker``: it reads the whole text of
both sides and needs no vectors, so the harness records both readings as full text. Pure numpy; a 45,000
document full-text corpus indexes in a couple of minutes on a laptop.
"""

from __future__ import annotations

import collections
import dataclasses
import re

from typing import Any

import numpy as np

TOKEN = re.compile(r"(?u)\b\w\w+\b")
LUCENE_STOP_WORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their then there these "
    "they this to was will with".split()
)
STOP_LISTS = {"none": frozenset(), "lucene": LUCENE_STOP_WORDS}


def tokenize(text: str, stop_words: frozenset[str] = frozenset()) -> list[str]:
    tokens = TOKEN.findall(text.lower())
    return [token for token in tokens if token not in stop_words] if stop_words else tokens


@dataclasses.dataclass
class BM25:
    """`Ranker` for the harness: ``pateval run --encoder bm25`` or ``--encoder pateval.encoders.bm25:BM25``.

    ``--encoder-arg k1=1.5 --encoder-arg b=0.75 --encoder-arg stopwords=lucene`` set the parameters.
    """

    k1: float = 1.2
    b: float = 0.75
    stopwords: str = "none"
    name: str = "bm25"

    def __post_init__(self) -> None:
        if self.stopwords not in STOP_LISTS:
            raise ValueError(f"stopwords must be one of {sorted(STOP_LISTS)}, got {self.stopwords!r}")
        self.k1, self.b = float(self.k1), float(self.b)

    def identity(self) -> dict[str, Any]:
        return {"k1": self.k1, "b": self.b, "stopwords": self.stopwords, "tokenizer": "lowercase \\b\\w\\w+\\b"}

    def rank(self, queries: dict[str, str], corpus: dict[str, str], *, top_k: int) -> dict[str, dict[str, float]]:
        index = _Index.build(corpus, STOP_LISTS[self.stopwords], self.k1, self.b)
        run: dict[str, dict[str, float]] = {}
        for query_id, text in queries.items():
            scores = index.score(collections.Counter(tokenize(text, STOP_LISTS[self.stopwords])))
            # +1 so that dropping the self-match still leaves top_k candidates
            take = min(top_k + 1, len(scores))
            top = np.argpartition(-scores, take - 1)[:take]
            top = top[np.argsort(-scores[top], kind="stable")]
            run[query_id] = dict(
                [(index.ids[j], float(scores[j])) for j in top if index.ids[j] != query_id and scores[j] > 0][:top_k]
            )
        return run


@dataclasses.dataclass
class _Index:
    """Postings sorted by term: for term ``t`` the documents ``docs[indptr[t]:indptr[t+1]]`` with BM25 weights."""

    ids: list[str]
    vocabulary: dict[str, int]
    indptr: np.ndarray
    docs: np.ndarray
    weights: np.ndarray

    @classmethod
    def build(cls, corpus: dict[str, str], stop_words: frozenset[str], k1: float, b: float) -> "_Index":
        ids = list(corpus)
        vocabulary: dict[str, int] = {}
        term_ids: list[np.ndarray] = []
        doc_ids: list[np.ndarray] = []
        tfs: list[np.ndarray] = []
        lengths = np.zeros(len(ids), dtype=np.float64)
        for doc_index, text in enumerate(corpus.values()):
            counts = collections.Counter(tokenize(text, stop_words))
            lengths[doc_index] = sum(counts.values())
            if not counts:
                continue
            term_ids.append(np.fromiter((vocabulary.setdefault(t, len(vocabulary)) for t in counts), dtype=np.int64, count=len(counts)))
            doc_ids.append(np.full(len(counts), doc_index, dtype=np.int32))
            tfs.append(np.fromiter(counts.values(), dtype=np.float32, count=len(counts)))
        if not vocabulary:
            return cls(ids, {}, np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32))
        term = np.concatenate(term_ids)
        order = np.argsort(term, kind="stable")
        term, docs, tf = term[order], np.concatenate(doc_ids)[order], np.concatenate(tfs)[order]
        indptr = np.searchsorted(term, np.arange(len(vocabulary) + 1))
        df = np.diff(indptr).astype(np.float64)
        idf = np.log1p((len(ids) - df + 0.5) / (df + 0.5))
        avgdl = lengths.mean() if lengths.mean() > 0 else 1.0
        norm = k1 * (1.0 - b + b * lengths / avgdl)
        weights = (idf[term] * tf * (k1 + 1.0) / (tf + norm[docs])).astype(np.float32)
        return cls(ids, vocabulary, indptr, docs, weights)

    def score(self, query_counts: collections.Counter[str]) -> np.ndarray:
        scores = np.zeros(len(self.ids), dtype=np.float32)
        for term, qtf in query_counts.items():
            term_id = self.vocabulary.get(term)
            if term_id is None:
                continue
            start, end = self.indptr[term_id], self.indptr[term_id + 1]
            scores[self.docs[start:end]] += qtf * self.weights[start:end]
        return scores
