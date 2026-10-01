"""Tests for the task interface and the field-view semantics benchmarks depend on."""

import pytest

from pateval.tasks import FieldView, Reading, RetrievalTask, dapfam


def test_reading_reports_its_budget():
    assert Reading(num_spans=8, span_length=512).budget == 4096
    assert str(Reading(num_spans=1, span_length=4096)) == "1x4096"


def test_equal_budgets_differ_in_span_structure():
    # 8x512 and 1x4096 read the same tokens; only the pooling differs. The harness must keep them distinct.
    assert Reading(8, 512).budget == Reading(1, 4096).budget
    assert str(Reading(8, 512)) != str(Reading(1, 4096))


def test_task_rejects_qrels_for_unknown_queries():
    with pytest.raises(ValueError, match="no query"):
        RetrievalTask(
            name="t",
            queries={"q1": "a"},
            corpus={"d1": "b"},
            qrels={"ghost": {"d1": 1}},
            query_view=FieldView.TITLE_ABSTRACT,
            corpus_view=FieldView.FULL_TEXT,
        )


def test_dapfam_render_puts_description_last():
    # Field order is why TAC and FULL coincide at small budgets: the description is appended, so the first
    # N tokens of both views are identical until the claims run out.
    row = {"title_en": "T", "abstract_en": "A", "claims_text": "C", "description_en": "D"}
    assert dapfam.render(row, FieldView.FULL_TEXT) == "T\nA\nC\nD"
    assert dapfam.render(row, FieldView.TITLE_ABSTRACT_CLAIMS) == "T\nA\nC"
    full = dapfam.render(row, FieldView.FULL_TEXT)
    assert full.startswith(dapfam.render(row, FieldView.TITLE_ABSTRACT_CLAIMS))


def test_dapfam_render_skips_empty_fields():
    row = {"title_en": "T", "abstract_en": "", "claims_text": None, "description_en": "D"}
    assert dapfam.render(row, FieldView.FULL_TEXT) == "T\nD"


def test_dapfam_in_paper_views_all_use_the_claims_corpus():
    # All six in-paper tasks take the TAC corpus; a full-text number is a different task, not a bigger budget.
    assert dapfam.IN_PAPER_CORPUS_VIEW is FieldView.TITLE_ABSTRACT_CLAIMS
    assert set(dapfam.QUERY_VIEWS) == {FieldView.TITLE_ABSTRACT, FieldView.TITLE_ABSTRACT_CLAIMS}
    assert len(dapfam.QUERY_VIEWS) * len(dapfam.SCOPES) == 6


def test_dapfam_load_rejects_an_unknown_scope():
    with pytest.raises(ValueError, match="scope must be one of"):
        dapfam.load(FieldView.TITLE_ABSTRACT, FieldView.TITLE_ABSTRACT_CLAIMS, scope="Middle")


def test_jsonl_task_loads_local_files(tmp_path):
    import json

    from pateval.tasks import jsonl

    (tmp_path / "q.jsonl").write_text(json.dumps({"id": 1, "text": "a"}) + "\n")
    (tmp_path / "c.jsonl").write_text(json.dumps({"id": "d", "text": "b"}) + "\n")
    (tmp_path / "qrels.json").write_text(json.dumps({"1": {"d": 1}}))
    task = jsonl.load(tmp_path / "q.jsonl", tmp_path / "c.jsonl", tmp_path / "qrels.json", name="local")
    assert task.queries == {"1": "a"} and task.corpus == {"d": "b"} and task.qrels == {"1": {"d": 1}}


def test_dapfam_reads_only_the_view_columns_at_the_pinned_commit(monkeypatch):
    rows = {
        "queries.parquet": [{"query_id": "q1", "title_en": "T", "abstract_en": "A", "claims_text": "C"}],
        "corpus.parquet": [{"relevant_id": "d1", "title_en": "U", "abstract_en": "B", "claims_text": "D"}],
        "qrels_all.parquet": [{"query_id": "q1", "relevant_id": "d1", "relevance_score": 1, "domain_rel": "IN"}],
    }
    calls = []

    def read(fname, revision, columns):
        calls.append((fname, revision, tuple(columns)))
        return rows[fname]

    monkeypatch.setattr(dapfam, "_read", read)
    task, extra = dapfam.load_with_scopes(FieldView.TITLE_ABSTRACT_CLAIMS, FieldView.TITLE_ABSTRACT_CLAIMS)
    assert task.queries == {"q1": "T\nA\nC"} and task.corpus == {"d1": "U\nB\nD"}
    assert task.qrels_name == "All" and set(extra) == {"In", "Out"} and extra["In"] == {"q1": {"d1": 1}}
    assert task.source["revision"] == dapfam.HF_REVISION and task.source["corpus_view"] == "TAC"
    assert {revision for _, revision, _ in calls} == {dapfam.HF_REVISION}
    assert ("corpus.parquet", dapfam.HF_REVISION, ("relevant_id", "title_en", "abstract_en", "claims_text")) in calls


def test_patenteb_local_file_is_recorded_by_digest(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from pateval.tasks import patenteb

    path = tmp_path / "data.parquet"
    rows = [{"q": "q1", "pos": "d1", "neg": "d2", "q_text": "a", "pos_text": "b", "neg_text": "c"}]
    pq.write_table(pa.Table.from_pylist(rows), path)
    task = patenteb.load("in", local_parquet=str(path))
    assert task.qrels == {"q1": {"d1": 1}} and set(task.corpus) == {"d1", "d2"}
    assert task.qrels_name == "IN" and len(task.source["sha256"]) == 64
