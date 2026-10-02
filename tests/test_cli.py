"""Smoke tests of the command line on local tasks with the hashing encoder."""

import json

import pytest

from pateval import cli, registry
from pateval.tasks import FieldView, RetrievalTask
from pateval.testing import hashing_encoder


def _write_task(tmp_path):
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"id": "q1", "text": "rotor blade pitch"})
        + "\n"
        + json.dumps({"id": "q2", "text": "battery anode"})
        + "\n"
    )
    (tmp_path / "corpus.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": "d1", "text": "rotor blade pitch control"},
                {"id": "d2", "text": "battery anode lithium"},
                {"id": "d3", "text": "baking recipe"},
            ]
        )
        + "\n"
    )
    (tmp_path / "qrels.json").write_text(json.dumps({"q1": {"d1": 1}, "q2": {"d2": 1}}))


def test_run_on_a_jsonl_task_appends_a_record(tmp_path, capsys):
    _write_task(tmp_path)
    out = tmp_path / "results.jsonl"
    code = cli.main(
        [
            "run",
            "--task",
            "jsonl",
            "--queries",
            str(tmp_path / "queries.jsonl"),
            "--corpus",
            str(tmp_path / "corpus.jsonl"),
            "--qrels",
            str(tmp_path / "qrels.json"),
            "--metric",
            "ndcg_cut_10",
            "--encoder",
            "pateval.testing:hashing_encoder",
            "--encoder-arg",
            "dim=32",
            "--query-reading",
            "1x512",
            "--doc-reading",
            "4x512",
            "--bootstrap",
            "0",
            "--cache",
            str(tmp_path / "cache"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    record = json.loads(out.read_text().splitlines()[0])
    assert record["scores"]["main"]["ndcg_cut_10"]["value"] == 1.0
    assert record["doc_reading"] == {"num_spans": 4, "span_length": 512, "across_span_pooling": "mean"}
    assert record["encoder_identity"]["spec"] == "pateval.testing:hashing_encoder"
    assert record["encoder_identity"]["arguments"] == {"dim": 32}
    assert len(record["task_source"]["corpus"]["sha256"]) == 64
    assert "ndcg_cut_10=1.0000" in capsys.readouterr().out


def test_parse_reading_and_encoder_args():
    reading = cli.parse_reading("8x512", "normalized_mean")
    assert (reading.num_spans, reading.span_length, reading.across_span_pooling) == (8, 512, "normalized_mean")
    assert cli.parse_encoder_args(["dim=16", "flag=true", "rate=0.5", "name=x"]) == {
        "dim": 16,
        "flag": True,
        "rate": 0.5,
        "name": "x",
    }


def test_registry_list_runs(capsys):
    assert cli.main(["registry", "list"]) == 0
    assert "paecter" in capsys.readouterr().out


def _fake_calibration(monkeypatch):
    """The calibration cell on a toy task, with every registry model replaced by the hashing encoder."""
    task = RetrievalTask(
        name="toy",
        queries={"q1": "rotor blade pitch", "q2": "battery anode"},
        corpus={"d1": "rotor blade pitch control", "d2": "battery anode lithium", "d3": "baking recipe"},
        qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
        query_view=FieldView.TITLE_ABSTRACT_CLAIMS,
        corpus_view=FieldView.TITLE_ABSTRACT_CLAIMS,
        qrels_name="All",
    )
    other = {"In": {"q1": {"d1": 1}}, "Out": {"q2": {"d2": 1}}}
    loaded = []
    monkeypatch.setattr(cli.dapfam, "load_with_scopes", lambda *args, **kwargs: (task, other))
    monkeypatch.setattr(cli, "load_encoder", lambda spec, **kwargs: loaded.append((spec, kwargs)) or hashing_encoder())
    return loaded


def test_reproduce_compares_each_baseline_with_its_own_values(monkeypatch, capsys):
    loaded = _fake_calibration(monkeypatch)
    code = cli.main(["reproduce", "--baseline", "paecter", "--baseline", "patembed-base", "--bootstrap", "0"])
    printed = capsys.readouterr().out
    assert code == 1  # the toy task scores 1.0, which reproduces nothing
    assert "published 0.3430" in printed and "published 0.3700" in printed and "published 0.3520" in printed
    assert [spec for spec, _ in loaded] == ["registry:paecter", "registry:patembed-base", "registry:patembed-base"]
    assert [kwargs["prompt"] for _, kwargs in loaded] == ["none", "default", "none"]


def test_reproduce_records_its_checks(tmp_path, monkeypatch):
    _fake_calibration(monkeypatch)
    ref = registry.Reference("paecter", "DAPFAM", "TAC->TAC", "1x512", "All", "ndcg_cut_100", "none", 1.0, "toy", True)
    monkeypatch.setattr(cli.registry, "references", lambda *args, **kwargs: [ref])
    out = tmp_path / "reproduce.jsonl"
    assert cli.main(["reproduce", "--bootstrap", "0", "--out", str(out)]) == 0
    (check,) = json.loads(out.read_text())["extra"]["reproduction"]
    assert check["ok"] and check["gated"] and check["published"] == 1.0 and check["measured"] == 1.0


def test_reproduce_refuses_a_baseline_without_a_published_value(monkeypatch):
    _fake_calibration(monkeypatch)
    with pytest.raises(SystemExit, match="no published DAPFAM"):
        cli.main(["reproduce", "--baseline", "bge-m3"])


def test_reproduce_refuses_arguments_that_change_the_configuration(monkeypatch):
    _fake_calibration(monkeypatch)
    with pytest.raises(SystemExit, match="published configuration"):
        cli.main(["reproduce", "--encoder-arg", "prompt=OUT"])


def test_all_means_every_model_with_a_gated_calibration_value():
    assert cli.calibration_baselines() == ["paecter", "patembed-base", "patembed-large", "bert-for-patents"]
