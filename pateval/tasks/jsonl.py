"""A task from local files, for custom or private benchmarks and for tests.

Queries and corpus are JSON Lines with `{"id": ..., "text": ...}`; qrels is a JSON object mapping query
id to `{document id: relevance}`.
"""

from __future__ import annotations

import json
import pathlib

from pateval.tasks.base import FieldView, RetrievalTask


def read_jsonl(path: str | pathlib.Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                texts[str(record["id"])] = str(record["text"])
    return texts


def read_qrels(path: str | pathlib.Path) -> dict[str, dict[str, int]]:
    with open(path) as handle:
        raw = json.load(handle)
    return {str(q): {str(d): int(r) for d, r in judged.items()} for q, judged in raw.items()}


def load(
    queries_path: str | pathlib.Path,
    corpus_path: str | pathlib.Path,
    qrels_path: str | pathlib.Path,
    name: str = "custom",
    main_metric: str = "ndcg_cut_100",
    query_view: FieldView = FieldView.FULL_TEXT,
    corpus_view: FieldView = FieldView.FULL_TEXT,
) -> RetrievalTask:
    return RetrievalTask(
        name=name,
        queries=read_jsonl(queries_path),
        corpus=read_jsonl(corpus_path),
        qrels=read_qrels(qrels_path),
        query_view=query_view,
        corpus_view=corpus_view,
        main_metric=main_metric,
    )
