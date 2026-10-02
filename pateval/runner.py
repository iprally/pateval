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
import platform
import subprocess

from typing import Any, Sequence

import numpy as np

import pateval

from pateval import scoring
from pateval.encoders.base import Ranker, Role, TextEncoder
from pateval.tasks.base import Reading, RetrievalTask

Qrels = dict[str, dict[str, int]]

# Bump when span splitting, prompting or pooling changes what a cached vector means.
CACHE_FORMAT = 2
# Encoder arguments that change speed, never vectors; left out of the identity.
RUNTIME_ARGUMENTS = frozenset({"batch_size", "device"})
# How a `Ranker` reads: the whole text, no spans, no pooling.
FULL_TEXT_READING = Reading(num_spans=1, span_length=0, across_span_pooling="none")
# Distributions whose versions are recorded with every score.
_RECORDED_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "pytrec_eval": "pytrec-eval-terrier",
    "huggingface_hub": "huggingface-hub",
    "torch": "torch",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "safetensors": "safetensors",
}


def harness_version() -> str:
    return pateval.__version__


def harness_commit() -> str | None:
    """`git describe --always --dirty` of the checkout the package runs from, when it runs from one."""
    root = pathlib.Path(pateval.__file__).resolve().parent.parent
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "describe", "--always", "--dirty"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def environment() -> dict[str, str]:
    """Python, library and harness versions, recorded with every score."""
    env = {"python": platform.python_version(), "pateval": harness_version()}
    for key, distribution in _RECORDED_DISTRIBUTIONS.items():
        try:
            env[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    commit = harness_commit()
    if commit:
        env["pateval_commit"] = commit
    return env


def encoder_identity(
    encoder: TextEncoder | Ranker, spec: str | None = None, arguments: dict[str, Any] | None = None
) -> dict[str, Any]:
    """What determines an encoder's vectors: its name, specification and arguments, plus its own `identity()`."""
    identity: dict[str, Any] = {"name": encoder.name}
    if spec is not None:
        identity["spec"] = spec
    kept = {key: value for key, value in sorted((arguments or {}).items()) if key not in RUNTIME_ARGUMENTS}
    if kept:
        identity["arguments"] = kept
    describe = getattr(encoder, "identity", None)
    if callable(describe):
        identity.update(describe())
    return identity


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
    task_source: dict[str, Any]
    encoder: str
    encoder_identity: dict[str, Any]
    query_reading: dict[str, Any]
    doc_reading: dict[str, Any]
    scores: dict[str, dict[str, Score]]  # qrel set name -> metric -> score
    num_queries: int
    num_documents: int
    settings: dict[str, Any]
    environment: dict[str, str]
    created_at: str
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

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
        prompt = self.encoder_identity.get("prompt")
        lines = [
            f"{self.task} | {self.encoder}{f' (prompt {prompt})' if prompt else ''} | "
            f"q {_reading_label(self.query_reading)} / d {_reading_label(self.doc_reading)}"
        ]
        for qrels_name, metrics in self.scores.items():
            lines.extend(f"  {qrels_name:>6}: {score}" for score in metrics.values())
        return "\n".join(lines)


def _reading_label(reading: dict[str, Any]) -> str:
    if not reading["span_length"]:
        return "full text"
    label = f"{reading['num_spans']}x{reading['span_length']}"
    pooling = reading.get("across_span_pooling", "mean")
    return label if pooling == "mean" else f"{label} {pooling}"


class VectorCache:
    """Vectors on disk, keyed by the encoder's identity, the role, the reading and the exact texts encoded.

    The key includes a digest of ids and texts, so a changed corpus rendering or a different truncation never
    reuses stale vectors, and the encoder identity (a registry model's commit, prompt, dtype and library
    versions), so neither does a new model revision or configuration. Files hold no pickles.
    """

    def __init__(self, directory: str | pathlib.Path):
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(
        self, identity: dict[str, Any], role: Role, reading: Reading, ids: Sequence[str], texts: Sequence[str]
    ) -> pathlib.Path:
        digest = hashlib.sha256()
        header = {"format": CACHE_FORMAT, "encoder": identity, "role": role, "reading": reading.as_dict()}
        digest.update(json.dumps(header, sort_keys=True, default=str).encode("utf-8"))
        for identifier, text in zip(ids, texts):
            digest.update(identifier.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(text.encode("utf-8"))
            digest.update(b"\x01")
        return self.directory / f"{digest.hexdigest()}.npz"

    def load(self, path: pathlib.Path, ids: Sequence[str]) -> np.ndarray | None:
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as stored:
            if stored["ids"].tolist() != list(ids):
                raise ValueError(f"{path} holds vectors for other ids; delete it")
            return stored["vectors"]

    def store(self, path: pathlib.Path, ids: Sequence[str], vectors: np.ndarray) -> None:
        np.savez(path, ids=np.array(list(ids), dtype=str), vectors=vectors)


def encode_side(
    encoder: TextEncoder,
    ids: Sequence[str],
    texts: Sequence[str],
    *,
    role: Role,
    reading: Reading,
    cache: VectorCache | None = None,
    identity: dict[str, Any] | None = None,
) -> np.ndarray:
    """Encode one side of a task at `reading`, through the cache when one is given."""
    path = cache.path_for(identity or encoder_identity(encoder), role, reading, ids, texts) if cache else None
    if cache and path is not None:
        cached = cache.load(path, ids)
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
    encoder: TextEncoder | Ranker,
    *,
    query_reading: Reading,
    doc_reading: Reading,
    metrics: Sequence[str] | None = None,
    bootstrap: int = 1000,
    seed: int = 0,
    cache: VectorCache | None = None,
    extra_qrels: dict[str, Qrels] | None = None,
    top_k: int = scoring.DEFAULT_TOP_K,
    encoder_spec: str | None = None,
    encoder_arguments: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> EvaluationRecord:
    """Encode, rank and score `task` with `encoder`; `extra_qrels` are scored from the same ranking.

    `encoder_spec` and `encoder_arguments` are how the encoder was obtained (see `load_encoder`); they are
    recorded and, apart from runtime-only arguments, part of the vector-cache key. A `Ranker` ranks the
    corpus itself from the full texts: the readings and the cache do not apply and both sides are recorded
    as read in full.
    """
    metrics = list(metrics) if metrics else [task.main_metric]
    if task.qrels_name in (extra_qrels or {}):
        raise ValueError(f"extra qrel set {task.qrels_name!r} has the same name as the task's own")
    identity = encoder_identity(encoder, encoder_spec, encoder_arguments)
    query_ids, query_texts = list(task.queries), list(task.queries.values())
    corpus_ids, corpus_texts = list(task.corpus), list(task.corpus.values())

    if isinstance(encoder, Ranker):
        run = encoder.rank(task.queries, task.corpus, top_k=top_k)
        query_reading = doc_reading = FULL_TEXT_READING
    else:
        query_vectors = encode_side(
            encoder, query_ids, query_texts, role="query", reading=query_reading, cache=cache, identity=identity
        )
        corpus_vectors = encode_side(
            encoder, corpus_ids, corpus_texts, role="document", reading=doc_reading, cache=cache, identity=identity
        )
        run = scoring.build_run(query_ids, query_vectors, corpus_ids, corpus_vectors, top_k=top_k)

    qrel_sets: dict[str, Qrels] = {task.qrels_name: task.qrels, **(extra_qrels or {})}
    scores = {
        qrels_name: {
            metric: Score.from_result(scoring.evaluate(run, qrels, metric=metric, bootstrap=bootstrap, seed=seed))
            for metric in metrics
        }
        for qrels_name, qrels in qrel_sets.items()
    }
    return EvaluationRecord(
        task=task.name,
        task_source=dict(task.source),
        encoder=encoder.name,
        encoder_identity=identity,
        query_reading=query_reading.as_dict(),
        doc_reading=doc_reading.as_dict(),
        scores=scores,
        num_queries=len(query_ids),
        num_documents=len(corpus_ids),
        settings={"metrics": metrics, "top_k": top_k, "bootstrap": bootstrap, "seed": seed},
        environment=environment(),
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        extra=dict(extra or {}),
    )


def append_record(path: str | pathlib.Path, record: EvaluationRecord) -> None:
    """Append one JSON line; a results file is a log, never rewritten."""
    with open(path, "a") as handle:
        handle.write(record.to_json() + "\n")
