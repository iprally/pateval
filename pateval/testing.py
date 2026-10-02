"""Deterministic, dependency-free encoders for tests and dry runs.

`HashingSpanEmbedder` tokenizes on whitespace, maps each token to a bucket by a stable hash and embeds a
span as its normalised bag of buckets. It is useless for retrieval quality and exact for plumbing:
identical texts give identical vectors, shared vocabulary gives higher cosine, and every span boundary is
observable. `hashing_encoder` is the plugin-style factory used by the tests and by the CLI smoke test.
"""

from __future__ import annotations

import dataclasses
import hashlib

from typing import Sequence

import numpy as np

from pateval.encoders.base import Role, SpannedEncoder, TextEncoder


def _bucket(token: str, num_buckets: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % num_buckets


@dataclasses.dataclass
class HashingSpanEmbedder:
    name: str = "hashing"
    max_span_length: int = 512
    dim: int = 64
    special_tokens: int = 2
    prompt_tokens: dict[str, int] = dataclasses.field(default_factory=lambda: {"query": 0, "document": 0})

    def tokenize(self, texts: Sequence[str]) -> list[list[int]]:
        return [[_bucket(token, self.dim) for token in text.split()] for text in texts]

    def overhead_tokens(self, role: Role) -> int:
        return self.special_tokens + self.prompt_tokens.get(role, 0)

    def embed_spans(self, spans: Sequence[list[int]], *, role: Role) -> np.ndarray:
        vectors = np.zeros((len(spans), self.dim), dtype=np.float32)
        for row, span in enumerate(spans):
            for bucket in span:
                vectors[row, bucket] += 1.0
            if not span:
                vectors[row, 0] = 1.0  # an empty span still needs a direction
        return vectors


def hashing_encoder(dim: int = 64, max_span_length: int = 512, batch_size: int = 32) -> TextEncoder:
    """Factory in the shape a plugin exposes: keyword arguments in, a `TextEncoder` out."""
    return SpannedEncoder(HashingSpanEmbedder(dim=dim, max_span_length=max_span_length), batch_size=batch_size)
