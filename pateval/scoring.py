"""Scoring, with the checks that published patent numbers are easy to get wrong.

Metrics come from `pytrec_eval`, the implementation MTEB and BEIR call, so a number produced here is
comparable to a number produced there. Three behaviours are deliberate:

* **The query's own document is excluded from its ranking.** In family-level patent benchmarks a query is
  usually also a corpus entry; leaving it in inflates every metric uniformly and silently.
* **A reproduction gate.** `check_reproduction` compares a measured value against a published one before
  downstream numbers are trusted. What a published protocol actually was can often only be established by
  reproducing one of its baselines closely; without that anchor a harness bug is indistinguishable from a
  finding.
* **Bootstrap confidence intervals over queries**, since hits are clustered by query and a per-judgment
  interval would overstate confidence.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytrec_eval

DEFAULT_TOP_K = 100


@dataclasses.dataclass
class Result:
    metric: str
    value: float
    per_query: dict[str, float]
    num_queries: int
    ci_low: float | None = None
    ci_high: float | None = None

    def __str__(self) -> str:
        ci = f" [{self.ci_low:.4f}, {self.ci_high:.4f}]" if self.ci_low is not None else ""
        return f"{self.metric}={self.value:.4f}{ci} ({self.num_queries:,} queries)"


def build_run(
    query_ids: list[str],
    query_vectors: np.ndarray,
    corpus_ids: list[str],
    corpus_vectors: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    block: int = 256,
) -> dict[str, dict[str, float]]:
    """Rank the corpus for every query by cosine similarity, excluding the query's own document."""
    if len(query_ids) != len(query_vectors) or len(corpus_ids) != len(corpus_vectors):
        raise ValueError("ids and vectors must be the same length on both sides")
    q = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    c = corpus_vectors / np.linalg.norm(corpus_vectors, axis=1, keepdims=True)

    run: dict[str, dict[str, float]] = {}
    for start in range(0, len(query_ids), block):
        sims = q[start : start + block] @ c.T
        for row, qid in enumerate(query_ids[start : start + block]):
            scores = sims[row]
            # +1 so that dropping the self-match still leaves top_k candidates
            take = min(top_k + 1, len(scores))
            top = np.argpartition(-scores, take - 1)[:take]
            top = top[np.argsort(-scores[top])]
            run[qid] = dict([(corpus_ids[j], float(scores[j])) for j in top if corpus_ids[j] != qid][:top_k])
    return run


def evaluate(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    metric: str = "ndcg_cut_100",
    bootstrap: int = 0,
    seed: int = 0,
) -> Result:
    """Score `run` against `qrels`, restricted to queries the qrel set judges."""
    # pytrec_eval takes a parameterised measure as "ndcg_cut.100" but keys its results "ndcg_cut_100".
    base, _, cutoff = metric.rpartition("_")
    measure = f"{base}.{cutoff}" if cutoff.isdigit() and base else metric
    scoped = {q: r for q, r in run.items() if q in qrels}
    if not scoped:
        raise ValueError("no query in the run is judged by these qrels")
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {measure})
    per_query = {q: v[metric] for q, v in evaluator.evaluate(scoped).items()}
    values = np.fromiter(per_query.values(), dtype=float)
    result = Result(metric=metric, value=float(values.mean()), per_query=per_query, num_queries=len(values))

    if bootstrap:
        # Resample queries, not judgments: hits are clustered by query.
        rng = np.random.default_rng(seed)
        means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(bootstrap)]
        result.ci_low, result.ci_high = (float(x) for x in np.percentile(means, [2.5, 97.5]))
    return result


def check_reproduction(measured: float, published: float, tolerance: float = 0.02) -> tuple[bool, str]:
    """Compare a measured baseline against its published value before trusting anything downstream."""
    if published == 0:
        raise ValueError("published value must be non-zero")
    relative = abs(measured - published) / published
    ok = relative <= tolerance
    verdict = "reproduces" if ok else "DOES NOT REPRODUCE"
    return ok, f"{verdict}: measured {measured:.4f} vs published {published:.4f} ({100 * relative:.2f}% apart)"
