"""Tests for the end-to-end runner and the vector cache, with the deterministic hashing encoder."""

import numpy as np
import pytest

from pateval import runner
from pateval.tasks import FieldView, Reading, RetrievalTask
from pateval.testing import hashing_encoder


def _toy_task(qrels_name: str = "main") -> RetrievalTask:
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
        qrels_name=qrels_name,
        source={"dataset": "toy", "revision": "0" * 40},
    )


def test_evaluate_encoder_scores_the_task_and_records_provenance():
    record = runner.evaluate_encoder(
        _toy_task(), hashing_encoder(dim=64), query_reading=Reading(1, 512), doc_reading=Reading(8, 512), bootstrap=0
    )
    score = record.main_score("main", "ndcg_cut_10")
    assert score.value == pytest.approx(1.0), "both relevant documents should rank first"
    assert score.num_queries == 2
    assert record.num_documents == 4 and record.encoder == "hashing"


def test_record_carries_pooling_identity_source_settings_and_environment():
    record = runner.evaluate_encoder(
        _toy_task(),
        hashing_encoder(dim=64),
        query_reading=Reading(1, 512),
        doc_reading=Reading(8, 512, "normalized_mean"),
        bootstrap=0,
        encoder_spec="pateval.testing:hashing_encoder",
        encoder_arguments={"dim": 64, "batch_size": 8},
    )
    assert record.doc_reading == {"num_spans": 8, "span_length": 512, "across_span_pooling": "normalized_mean"}
    assert record.encoder_identity == {
        "name": "hashing",
        "spec": "pateval.testing:hashing_encoder",
        "arguments": {"dim": 64},  # batch size changes speed, not vectors
    }
    assert record.task_source == {"dataset": "toy", "revision": "0" * 40}
    assert record.settings == {"metrics": ["ndcg_cut_10"], "top_k": 100, "bootstrap": 0, "seed": 0}
    assert record.environment["pateval"] and record.environment["numpy"]


def test_scores_are_keyed_by_the_qrel_set_name():
    record = runner.evaluate_encoder(
        _toy_task(qrels_name="All"),
        hashing_encoder(),
        query_reading=Reading(1, 512),
        doc_reading=Reading(1, 512),
        bootstrap=0,
        extra_qrels={"Hard": {"q1": {"d3": 1}}},
    )
    assert set(record.scores) == {"All", "Hard"}
    assert record.scores["All"]["ndcg_cut_10"].value == pytest.approx(1.0)
    assert record.scores["Hard"]["ndcg_cut_10"].value < 1.0
    assert record.scores["Hard"]["ndcg_cut_10"].num_queries == 1


def test_extra_qrels_cannot_shadow_the_tasks_own():
    with pytest.raises(ValueError, match="same name"):
        runner.evaluate_encoder(
            _toy_task(qrels_name="All"),
            hashing_encoder(),
            query_reading=Reading(1, 512),
            doc_reading=Reading(1, 512),
            bootstrap=0,
            extra_qrels={"All": {"q1": {"d3": 1}}},
        )


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
    assert restored == record


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
    ids, texts, identity = ["a"], ["some text"], {"name": "m"}
    base = cache.path_for(identity, "document", Reading(8, 512), ids, texts)
    assert cache.path_for(identity, "document", Reading(1, 4096), ids, texts) != base
    assert cache.path_for(identity, "document", Reading(8, 512, "max"), ids, texts) != base
    assert cache.path_for(identity, "document", Reading(8, 512), ids, ["other text"]) != base
    assert cache.path_for(identity, "query", Reading(8, 512), ids, texts) != base


def test_cache_key_follows_the_encoder_identity(tmp_path):
    cache = runner.VectorCache(tmp_path)
    reading, ids, texts = Reading(8, 512), ["a"], ["some text"]
    pinned = {"name": "m", "revision": "1" * 40, "prompt": "MIXED", "dtype": "float32"}
    base = cache.path_for(pinned, "document", reading, ids, texts)
    for changed in ({"revision": "2" * 40}, {"prompt": "OUT"}, {"dtype": "bfloat16"}, {"name": "other"}):
        assert cache.path_for({**pinned, **changed}, "document", reading, ids, texts) != base, changed


def test_runtime_arguments_stay_out_of_the_identity():
    encoder = hashing_encoder()
    fast = runner.encoder_identity(encoder, "spec", {"dim": 64, "batch_size": 512, "device": "cuda"})
    slow = runner.encoder_identity(encoder, "spec", {"dim": 64, "batch_size": 8, "device": "cpu"})
    assert fast == slow
    assert runner.encoder_identity(encoder, "spec", {"dim": 32}) != slow


def test_cache_files_hold_no_pickles_and_reject_other_ids(tmp_path):
    cache = runner.VectorCache(tmp_path)
    path = cache.path_for({"name": "m"}, "document", Reading(1, 512), ["a", "b"], ["x", "y"])
    cache.store(path, ["a", "b"], np.ones((2, 3), dtype=np.float32))
    with np.load(path, allow_pickle=False) as stored:
        assert stored["ids"].tolist() == ["a", "b"]
    assert cache.load(path, ["a", "b"]).shape == (2, 3)
    with pytest.raises(ValueError, match="other ids"):
        cache.load(path, ["a", "c"])


def test_encoder_returning_the_wrong_number_of_vectors_is_rejected():
    class Broken:
        name = "broken"

        def encode(self, texts, *, role, reading):
            return np.zeros((len(texts) + 1, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="returned 3 vectors for 2 texts"):
        runner.encode_side(Broken(), ["a", "b"], ["x", "y"], role="query", reading=Reading(1, 512))
