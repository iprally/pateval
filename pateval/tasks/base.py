"""The task interface every benchmark adapter normalises to.

A patent retrieval benchmark varies along two axes that published evaluations usually leave implicit,
and both are modelled here rather than baked into an adapter:

*Field view* --- which parts of a patent form the query and the document. DAPFAM offers title+abstract,
title+abstract+claims and full text; a system that reads the description is doing a different task from
one that does not, so the view belongs to the task definition.

*Reading budget* --- how much of the selected fields a system actually encodes. This is **not** part of a
task definition. Published patent results sit at 512 tokens because BERT-derived encoders cannot exceed
their position embeddings, not because any benchmark requires it, and the difference is large enough to
reorder systems. The harness therefore records the budget alongside every score instead of assuming it.
"""

from __future__ import annotations

import dataclasses
import enum

from typing import Any


class FieldView(enum.Enum):
    """Which parts of a patent a side of the task is built from."""

    TITLE_ABSTRACT = "TA"
    TITLE_ABSTRACT_CLAIMS = "TAC"
    CLAIMS = "CLM"
    FULL_TEXT = "FULL"


@dataclasses.dataclass(frozen=True)
class Reading:
    """How much of a text a system encoded, and how the pieces were combined.

    `num_spans=1` with `span_length=512` is the conventional single truncation; `num_spans=8` with
    `span_length=512` reads 4,096 tokens as eight independently encoded pieces.
    """

    num_spans: int
    span_length: int
    across_span_pooling: str = "mean"

    @property
    def budget(self) -> int:
        return self.num_spans * self.span_length

    def __str__(self) -> str:
        return f"{self.num_spans}x{self.span_length}"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class RetrievalTask:
    """A benchmark reduced to the three things scoring needs, plus provenance.

    `qrels` maps query id to {document id: relevance}. Documents absent from a query's entry are treated
    as irrelevant by the metric, which matters for domain-partitioned qrel sets: under an out-of-domain
    partition an in-domain relevant document counts as a false positive, so absolute scores there are
    comparable between systems but not to the unpartitioned number.

    `qrels_name` is the name its scores are recorded under (DAPFAM's scope, PatenTEB's regime); `source` says
    where the data came from (repository and commit, or file and digest) and how it was rendered.
    """

    name: str
    queries: dict[str, str]
    corpus: dict[str, str]
    qrels: dict[str, dict[str, int]]
    query_view: FieldView
    corpus_view: FieldView
    main_metric: str = "ndcg_cut_100"
    qrels_name: str = "main"
    source: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.qrels) - set(self.queries)
        if unknown:
            raise ValueError(f"{self.name}: {len(unknown)} qrel entries have no query, e.g. {sorted(unknown)[:3]}")

    def summary(self) -> str:
        judged = sum(len(v) for v in self.qrels.values())
        return (
            f"{self.name}: {len(self.queries):,} queries, {len(self.corpus):,} documents, "
            f"{judged:,} judgments, {self.query_view.value}->{self.corpus_view.value}"
        )
