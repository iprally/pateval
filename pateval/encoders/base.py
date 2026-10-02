"""The encoder contract and the reading-budget runner built on top of it."""

from __future__ import annotations

import dataclasses

from typing import Any, Literal, Protocol, Sequence, runtime_checkable

import numpy as np

from pateval.tasks.base import Reading

Role = Literal["query", "document"]

ACROSS_SPAN_POOLINGS = ("mean", "normalized_mean", "max")


@runtime_checkable
class TextEncoder(Protocol):
    """Anything that turns texts into one L2-normalisable vector each, given a reading.

    This is the injection point for models the harness knows nothing about: implement it in your own
    package and pass `module:attribute` to `load_encoder`. `reading` tells the encoder how much text the
    experiment intends it to read; an encoder that fixes its own reading should validate the request and
    raise rather than silently read something else, because the reading is recorded next to every score.

    An encoder may also define `identity() -> dict`, returning whatever determines its vectors that its name,
    specification and arguments do not show, such as a digest of weights loaded from a path that can be
    overwritten. It is recorded with every score and is part of the vector-cache key.
    """

    name: str

    def encode(self, texts: Sequence[str], *, role: Role, reading: Reading) -> np.ndarray: ...


@runtime_checkable
class Ranker(Protocol):
    """A system that ranks the corpus for every query itself, without vectors: a lexical baseline such as BM25.

    It reads whatever text it is given, so the reading budget does not apply and is recorded as full text.
    `rank` returns, per query id, the top documents with their scores, the query's own id excluded.
    """

    name: str

    def rank(self, queries: dict[str, str], corpus: dict[str, str], *, top_k: int) -> dict[str, dict[str, float]]: ...


@runtime_checkable
class SpanEmbedder(Protocol):
    """A model that embeds one span at a time and owns its tokenizer and within-span readout.

    `tokenize` returns token ids without special tokens or prompts; `overhead_tokens` says how many
    positions of a span the model spends on those for the given role, so the harness can cut the text into
    spans that fit. `embed_spans` receives the raw windows and must add its own special tokens and prompt.
    """

    name: str
    max_span_length: int

    def tokenize(self, texts: Sequence[str]) -> list[list[int]]: ...

    def overhead_tokens(self, role: Role) -> int: ...

    def embed_spans(self, spans: Sequence[list[int]], *, role: Role) -> np.ndarray: ...


def split_into_spans(token_ids: Sequence[int], span_capacity: int, num_spans: int) -> list[list[int]]:
    """Cut a token sequence into at most `num_spans` contiguous windows of at most `span_capacity` tokens.

    An empty text yields one empty span, so that every text produces exactly one vector. Tokens beyond the
    last window are discarded: that is the reading budget, and it is recorded with the result.
    """
    if span_capacity <= 0:
        raise ValueError(f"span_capacity must be positive, got {span_capacity}")
    if num_spans <= 0:
        raise ValueError(f"num_spans must be positive, got {num_spans}")
    ids = list(token_ids)
    if not ids:
        return [[]]
    windows = [ids[start : start + span_capacity] for start in range(0, len(ids), span_capacity)]
    return windows[:num_spans]


def pool_spans(span_vectors: np.ndarray, method: str) -> np.ndarray:
    """Combine the vectors of one text's spans into a single vector.

    `mean` is the plain average, which for a mean-readout model equals whole-document mean pooling
    (late chunking) over the same tokens. `normalized_mean` L2-normalises each span first, so a span
    contributes direction only. `max` is element-wise.
    """
    if span_vectors.ndim != 2 or len(span_vectors) == 0:
        raise ValueError("span_vectors must be a non-empty (num_spans, dim) array")
    if method == "mean":
        return span_vectors.mean(axis=0)
    if method == "normalized_mean":
        norms = np.linalg.norm(span_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (span_vectors / norms).mean(axis=0)
    if method == "max":
        return span_vectors.max(axis=0)
    raise ValueError(f"unknown across-span pooling {method!r}; expected one of {ACROSS_SPAN_POOLINGS}")


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@dataclasses.dataclass
class SpannedEncoder:
    """Turns a single-span model into a reading-aware encoder.

    The text is tokenized once, cut into `reading.num_spans` windows that fit the model's span after its
    prompt and special tokens, every window is embedded independently, and the window vectors are pooled
    with `reading.across_span_pooling`. This is the configuration under which every public baseline in
    the harness is run, so that only the within-span readout is the model's own.
    """

    embedder: SpanEmbedder
    batch_size: int = 32

    @property
    def name(self) -> str:
        return self.embedder.name

    def identity(self) -> dict[str, Any]:
        describe = getattr(self.embedder, "identity", None)
        return dict(describe()) if callable(describe) else {}

    def encode(self, texts: Sequence[str], *, role: Role, reading: Reading) -> np.ndarray:
        if reading.span_length > self.embedder.max_span_length:
            raise ValueError(
                f"{self.embedder.name} cannot read {reading.span_length}-token spans in one pass "
                f"(maximum {self.embedder.max_span_length}); use more, shorter spans instead."
            )
        capacity = reading.span_length - self.embedder.overhead_tokens(role)
        if capacity <= 0:
            raise ValueError(
                f"span_length {reading.span_length} leaves no room for text after the prompt and special tokens"
            )

        spans_per_text: list[list[list[int]]] = [
            split_into_spans(ids, capacity, reading.num_spans) for ids in self.embedder.tokenize(texts)
        ]
        flat_spans = [span for spans in spans_per_text for span in spans]
        span_vectors = self._embed_in_batches(flat_spans, role)

        pooled: list[np.ndarray] = []
        offset = 0
        for spans in spans_per_text:
            pooled.append(pool_spans(span_vectors[offset : offset + len(spans)], reading.across_span_pooling))
            offset += len(spans)
        return l2_normalize(np.stack(pooled).astype(np.float32))

    def _embed_in_batches(self, spans: list[list[int]], role: Role) -> np.ndarray:
        parts = [
            np.asarray(self.embedder.embed_spans(spans[start : start + self.batch_size], role=role), dtype=np.float32)
            for start in range(0, len(spans), self.batch_size)
        ]
        return np.concatenate(parts, axis=0)
