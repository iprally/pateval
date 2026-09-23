"""Tests for scoring, concentrating on the behaviours that silently corrupt patent retrieval numbers."""

import numpy as np
import pytest

from pateval import scoring


def _unit(rows):
    v = np.array(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_run_excludes_the_querys_own_document():
    # In family-level benchmarks a query is also a corpus entry. Retrieving itself is a perfect match and
    # would inflate every metric, so it must never appear in the ranking.
    qids, qv = ["a"], _unit([[1.0, 0.0]])
    cids, cv = ["a", "b"], _unit([[1.0, 0.0], [0.9, 0.1]])
    run = scoring.build_run(qids, qv, cids, cv)
    assert "a" not in run["a"]
    assert "b" in run["a"]


def test_run_keeps_top_k_after_dropping_the_self_match():
    qids, qv = ["q1"], _unit([[1.0, 0.0]])
    cids = ["q1"] + [f"d{i}" for i in range(5)]
    cv = _unit([[1.0, 0.0]] + [[1.0, 0.01 * i] for i in range(5)])
    run = scoring.build_run(qids, qv, cids, cv, top_k=3)
    assert len(run["q1"]) == 3
    assert "q1" not in run["q1"]


def test_ranking_is_by_cosine_not_magnitude():
    # Vectors are normalised internally, so a long uninformative vector must not outrank a short aligned one.
    qids, qv = ["q"], np.array([[1.0, 0.0]], dtype=np.float32)
    cids = ["aligned", "long_but_off"]
    cv = np.array([[0.1, 0.0], [5.0, 5.0]], dtype=np.float32)
    run = scoring.build_run(qids, qv, cids, cv)
    assert max(run["q"], key=run["q"].get) == "aligned"


def test_evaluate_scopes_to_judged_queries_only():
    # Domain-partitioned qrel sets judge a subset of queries; unjudged ones must not be scored as zero.
    run = {"q1": {"d1": 0.9}, "q2": {"d2": 0.9}}
    qrels = {"q1": {"d1": 1}}
    result = scoring.evaluate(run, qrels, metric="ndcg_cut_100")
    assert result.num_queries == 1
    assert result.value == pytest.approx(1.0)


def test_evaluate_raises_when_no_query_is_judged():
    with pytest.raises(ValueError, match="no query in the run is judged"):
        scoring.evaluate({"q1": {"d1": 1.0}}, {"other": {"d1": 1}})


def test_unjudged_relevant_document_counts_against_you():
    # This is the out-of-domain protocol: a genuinely relevant in-domain document is absent from the Out
    # qrels and therefore scores as a false positive. Absolute Out values are comparable between systems,
    # not to the unpartitioned number.
    run = {"q": {"in_domain_rel": 0.99, "out_domain_rel": 0.10}}
    out_qrels = {"q": {"out_domain_rel": 1}}
    poor = scoring.evaluate(run, out_qrels)
    better = scoring.evaluate({"q": {"out_domain_rel": 0.99}}, out_qrels)
    assert poor.value < better.value


def test_bootstrap_interval_brackets_the_point_estimate():
    run = {f"q{i}": {f"d{i}": 1.0} for i in range(40)}
    qrels = {f"q{i}": {f"d{i}": 1 if i % 2 else 0} for i in range(40)}
    r = scoring.evaluate(run, qrels, bootstrap=200, seed=1)
    assert r.ci_low is not None and r.ci_low <= r.value <= r.ci_high


def test_reproduction_gate_accepts_a_close_match_and_rejects_a_far_one():
    ok, msg = scoring.check_reproduction(0.3424, 0.343)
    assert ok and "reproduces" in msg
    bad, msg = scoring.check_reproduction(0.28, 0.343)
    assert not bad and "DOES NOT REPRODUCE" in msg


def test_build_run_rejects_mismatched_ids_and_vectors():
    with pytest.raises(ValueError, match="same length"):
        scoring.build_run(["a", "b"], _unit([[1.0, 0.0]]), ["c"], _unit([[1.0, 0.0]]))
