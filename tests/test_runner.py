"""Tests for the end-to-end runner and the vector cache, with the deterministic hashing encoder."""

import numpy as np
import pytest

from pateval import runner
from pateval.tasks import FieldView, Reading, RetrievalTask
from pateval.testing import hashing_encoder


def _toy_task() -> RetrievalTask:
    # Relevant documents share vocabulary with their query; distractors share nothing.
    return RetrievalTask(
        name="toy",
        queries={"q1": "rotor blade pitch", "q2": "battery anode lithium"},
        corpus={
            "d1": "rotor blade pitch control",
            "d2": "lithium anode battery cell",
            "d3": "unrelated baking recipe",
            "q1": "rotor blade pitch",  # the query itself is in the corpus, as in family-level benchmarks
        },
        qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
        query_view=FieldView.FULL_TEXT,
        corpus_view=FieldView.FULL_TEXT,
        main_metric="ndcg_cut_10",
    )


def test_evaluate_encoder_scores_the_task_and_records_provenance():
    record = runner.evaluate_encoder(
        _toy_task(), hashing_encoder(dim=64), query_reading=Reading(1, 512), doc_reading=Reading(8, 512), bootstrap=0
    )
    score = record.main_score("main", "ndcg_cut_10")
    assert score.value == pytest.approx(1.0), "both relevant documents should rank first"
    assert score.num_queries == 2
    assert record.query_reading == "1x512" and record.doc_reading == "8x512"
    assert record.num_documents == 4 and record.encoder == "hashing"


def test_extra_qrels_are_scored_from_the_same_ranking():
    record = runner.evaluate_encoder(
        _toy_task(),
        hashing_encoder(),
        query_reading=Reading(1, 512),
        doc_reading=Reading(1, 512),
        bootstrap=0,
        extra_qrels={"Hard": {"q1": {"d3": 1}}},
    )
    assert record.scores["main"]["ndcg_cut_10"].value == pytest.approx(1.0)
    assert record.scores["Hard"]["ndcg_cut_10"].value < 1.0
    assert record.scores["Hard"]["ndcg_cut_10"].num_queries == 1


def test_record_round_trips_through_json(tmp_path):
    record = runner.evaluate_encoder(
        _toy_task(), hashing_encoder(), query_reading=Reading(1, 512), doc_reading=Reading(2, 512), bootstrap=50
    )
    path = tmp_path / "results.jsonl"
    runner.append_record(path, record)
    runner.append_record(path, record)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    restored = runner.EvaluationRecord.from_json(lines[0])
    assert restored.main_score("main", "ndcg_cut_10").value == record.main_score("main", "ndcg_cut_10").value
    assert restored.main_score("main", "ndcg_cut_10").ci_low is not None


def test_cache_serves_the_second_call_without_encoding(tmp_path, monkeypatch):
    cache = runner.VectorCache(tmp_path)
    encoder = hashing_encoder()
    task = _toy_task()
    first = runner.evaluate_encoder(
        task, encoder, query_reading=Reading(1, 512), doc_reading=Reading(8, 512), bootstrap=0, cache=cache
    )

    def explode(*args, **kwargs):
        raise AssertionError("encoder was called although vectors were cached")

    monkeypatch.setattr(encoder, "encode", explode)
    second = runner.evaluate_encoder(
        task, encoder, query_reading=Reading(1, 512), doc_reading=Reading(8, 512), bootstrap=0, cache=cache
    )
    assert second.main_score("main", "ndcg_cut_10").value == first.main_score("main", "ndcg_cut_10").value


def test_cache_key_depends_on_reading_and_texts(tmp_path):
    cache = runner.VectorCache(tmp_path)
    ids, texts = ["a"], ["some text"]
    base = cache.path_for("m", "document", Reading(8, 512), ids, texts)
    assert cache.path_for("m", "document", Reading(1, 4096), ids, texts) != base
    assert cache.path_for("m", "document", Reading(8, 512), ids, ["other text"]) != base
    assert cache.path_for("m", "query", Reading(8, 512), ids, texts) != base
    assert cache.path_for("other", "document", Reading(8, 512), ids, texts) != base


def test_encoder_returning_the_wrong_number_of_vectors_is_rejected():
    class Broken:
        name = "broken"

        def encode(self, texts, *, role, reading):
            return np.zeros((len(texts) + 1, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="returned 3 vectors for 2 texts"):
        runner.encode_side(Broken(), ["a", "b"], ["x", "y"], role="query", reading=Reading(1, 512))
