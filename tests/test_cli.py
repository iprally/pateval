"""Smoke test of the command line on a local task with the hashing encoder."""

import json

from pateval import cli


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
    assert record["doc_reading"] == "4x512"
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
