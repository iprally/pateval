"""Encode a task with an encoder and score it, recording the reading next to every number.

`evaluate_encoder` is the one entry point: it encodes both sides at their readings (through the vector
cache when one is given, because a DAPFAM corpus is re-scored under several query views and qrel sets),
builds the ranking, and scores it against the task's qrels and any extra qrel sets that share its
queries and corpus. The result is an `EvaluationRecord` with full provenance, serialisable to one JSON
line, so that a table cell can always be traced back to the configuration that produced it.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import pathlib

from typing import Sequence

import numpy as np

from pateval import scoring
from pateval.encoders.base import Role, TextEncoder
from pateval.tasks.base import Reading, RetrievalTask

Qrels = dict[str, dict[str, int]]


def harness_version() -> str:
    try:
        return importlib.metadata.version("pateval")
    except importlib.metadata.PackageNotFoundError:  # running from a checkout without installation
        return "unknown"


@dataclasses.dataclass
class Score:
    metric: str
    value: float
    num_queries: int
    ci_low: float | None = None
    ci_high: float | None = None

    @classmethod
    def from_result(cls, result: scoring.Result) -> "Score":
        return cls(result.metric, result.value, result.num_queries, result.ci_low, result.ci_high)

    def __str__(self) -> str:
        ci = f" [{self.ci_low:.4f}, {self.ci_high:.4f}]" if self.ci_low is not None else ""
        return f"{self.metric}={self.value:.4f}{ci} ({self.num_queries:,} queries)"


@dataclasses.dataclass
class EvaluationRecord:
    """Everything needed to cite a number: what was scored, with what, read how, and when."""

    task: str
    encoder: str
    query_reading: str
    doc_reading: str
    scores: dict[str, dict[str, Score]]  # qrels name -> metric -> score
    num_queries: int
    num_documents: int
    created_at: str
    harness_version: str
    extra: dict[str, str] = dataclasses.field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "EvaluationRecord":
        raw = json.loads(line)
        raw["scores"] = {
            qrels_name: {metric: Score(**score) for metric, score in metrics.items()}
            for qrels_name, metrics in raw["scores"].items()
        }
        return cls(**raw)

    def main_score(self, qrels_name: str, metric: str) -> Score:
        return self.scores[qrels_name][metric]

    def summary(self) -> str:
        lines = [f"{self.task} | {self.encoder} | q {self.query_reading} / d {self.doc_reading}"]
        for qrels_name, metrics in self.scores.items():
            lines.extend(f"  {qrels_name:>6}: {score}" for score in metrics.values())
        return "\n".join(lines)


class VectorCache:
    """Vectors on disk, keyed by encoder, role, reading and the exact texts encoded.

    The key includes a digest of ids and texts, so a changed corpus rendering or a different truncation
    never reuses stale vectors; the encoder name is part of the key, so an encoder must change its name
    when its weights change.
    """

    def __init__(self, directory: str | pathlib.Path):
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(
        self, encoder_name: str, role: Role, reading: Reading, ids: Sequence[str], texts: Sequence[str]
    ) -> pathlib.Path:
        digest = hashlib.sha256()
        digest.update(f"{encoder_name}|{role}|{reading}|{reading.across_span_pooling}|".encode("utf-8"))
        for identifier, text in zip(ids, texts):
            digest.update(identifier.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(text.encode("utf-8"))
            digest.update(b"\x01")
        return self.directory / f"{digest.hexdigest()}.npz"

    def load(self, path: pathlib.Path) -> np.ndarray | None:
        if not path.exists():
            return None
        return np.load(path)["vectors"]

    def store(self, path: pathlib.Path, ids: Sequence[str], vectors: np.ndarray) -> None:
        np.savez(path, ids=np.array(list(ids), dtype=object), vectors=vectors)


def encode_side(
    encoder: TextEncoder,
    ids: Sequence[str],
    texts: Sequence[str],
    *,
    role: Role,
    reading: Reading,
    cache: VectorCache | None = None,
) -> np.ndarray:
    """Encode one side of a task at `reading`, through the cache when one is given."""
    path = cache.path_for(encoder.name, role, reading, ids, texts) if cache else None
    if cache and path is not None:
        cached = cache.load(path)
        if cached is not None:
            return cached
    vectors = np.asarray(encoder.encode(texts, role=role, reading=reading), dtype=np.float32)
    if vectors.shape[0] != len(ids):
        raise ValueError(f"{encoder.name} returned {vectors.shape[0]} vectors for {len(ids)} texts")
    if cache and path is not None:
        cache.store(path, ids, vectors)
    return vectors


def evaluate_encoder(
    task: RetrievalTask,
    encoder: TextEncoder,
    *,
    query_reading: Reading,
    doc_reading: Reading,
    metrics: Sequence[str] | None = None,
    bootstrap: int = 1000,
    seed: int = 0,
    cache: VectorCache | None = None,
    extra_qrels: dict[str, Qrels] | None = None,
    top_k: int = scoring.DEFAULT_TOP_K,
    extra: dict[str, str] | None = None,
) -> EvaluationRecord:
    """Encode, rank and score `task` with `encoder`; `extra_qrels` are scored from the same ranking."""
    metrics = list(metrics) if metrics else [task.main_metric]
    query_ids, query_texts = list(task.queries), list(task.queries.values())
    corpus_ids, corpus_texts = list(task.corpus), list(task.corpus.values())

    query_vectors = encode_side(encoder, query_ids, query_texts, role="query", reading=query_reading, cache=cache)
    corpus_vectors = encode_side(encoder, corpus_ids, corpus_texts, role="document", reading=doc_reading, cache=cache)
    run = scoring.build_run(query_ids, query_vectors, corpus_ids, corpus_vectors, top_k=top_k)

    qrel_sets: dict[str, Qrels] = {"main": task.qrels, **(extra_qrels or {})}
    scores = {
        qrels_name: {
            metric: Score.from_result(scoring.evaluate(run, qrels, metric=metric, bootstrap=bootstrap, seed=seed))
            for metric in metrics
        }
        for qrels_name, qrels in qrel_sets.items()
    }
    return EvaluationRecord(
        task=task.name,
        encoder=encoder.name,
        query_reading=str(query_reading),
        doc_reading=str(doc_reading),
        scores=scores,
        num_queries=len(query_ids),
        num_documents=len(corpus_ids),
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        harness_version=harness_version(),
        extra=dict(extra or {}),
    )


def append_record(path: str | pathlib.Path, record: EvaluationRecord) -> None:
    """Append one JSON line; a results file is a log, never rewritten."""
    with open(path, "a") as handle:
        handle.write(record.to_json() + "\n")
