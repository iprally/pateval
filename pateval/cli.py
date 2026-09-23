"""Command-line interface.

    pateval run --task dapfam --query-view TAC --corpus-view TAC --encoder registry:paecter \\
        --query-reading 1x512 --doc-reading 8x512 --cache ./vectors --out results.jsonl
    pateval run --task patenteb --regime IN --local-parquet retrieval_IN/test/data.parquet --encoder ...
    pateval run --task jsonl --queries q.jsonl --corpus c.jsonl --qrels qrels.json --encoder my_pkg.eval:Encoder
    pateval matrix --encoder registry:paecter --cache ./vectors --out results.jsonl
    pateval reproduce --encoder registry:paecter
    pateval registry list | show paecter | verify paecter

An encoder is named by a spec (see `pateval.encoders.plugins`); `--encoder-arg key=value` passes
keyword arguments to its factory, so a model whose code lives in another package needs no change here.
"""

from __future__ import annotations

import argparse
import sys

from typing import Any, Sequence

from pateval import registry, scoring
from pateval.encoders.base import ACROSS_SPAN_POOLINGS
from pateval.encoders.plugins import load_encoder
from pateval.runner import EvaluationRecord, VectorCache, append_record, evaluate_encoder
from pateval.tasks import FieldView, Reading, RetrievalTask, dapfam, jsonl, patenteb

# The DAPFAM query->corpus field configurations `matrix` runs: the source paper's six tasks come from the first
# two rows crossed with the three scopes; the rest vary the fields, which the benchmark leaves free.
MATRIX_CONFIGS: tuple[tuple[FieldView, FieldView], ...] = (
    (FieldView.TITLE_ABSTRACT, FieldView.TITLE_ABSTRACT_CLAIMS),
    (FieldView.TITLE_ABSTRACT_CLAIMS, FieldView.TITLE_ABSTRACT_CLAIMS),
    (FieldView.TITLE_ABSTRACT_CLAIMS, FieldView.FULL_TEXT),
    (FieldView.CLAIMS, FieldView.FULL_TEXT),
    (FieldView.CLAIMS, FieldView.TITLE_ABSTRACT_CLAIMS),
    (FieldView.CLAIMS, FieldView.CLAIMS),
)


def parse_reading(text: str, across_span_pooling: str = "mean") -> Reading:
    """`8x512` -> Reading(num_spans=8, span_length=512)."""
    try:
        num_spans, span_length = (int(part) for part in text.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"reading must look like 8x512, got {text!r}") from error
    return Reading(num_spans=num_spans, span_length=span_length, across_span_pooling=across_span_pooling)


def parse_encoder_args(pairs: Sequence[str]) -> dict[str, Any]:
    """`key=value` pairs; values are cast to bool, int or float when they look like one."""
    parsed: dict[str, Any] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(f"encoder argument must be key=value, got {pair!r}")
        parsed[key] = _cast(value)
    return parsed


def _cast(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _add_encoder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoder", required=True, help="registry:<name>, <module>:<attribute>, or an entry point")
    parser.add_argument("--encoder-arg", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--query-reading", default="1x512", help="spans x span length for queries")
    parser.add_argument("--doc-reading", default="8x512", help="spans x span length for documents")
    parser.add_argument("--across-span-pooling", default="mean", choices=ACROSS_SPAN_POOLINGS)
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples over queries; 0 disables")
    parser.add_argument("--cache", help="directory for encoded vectors, reused across configurations")
    parser.add_argument("--out", help="JSON Lines file to append records to")


def _add_task_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True, choices=("dapfam", "patenteb", "jsonl"))
    parser.add_argument("--query-view", default="TAC", help="DAPFAM query fields: TA, TAC, CLM or FULL")
    parser.add_argument("--corpus-view", default="TAC", help="DAPFAM corpus fields: TA, TAC, CLM or FULL")
    parser.add_argument("--scopes", default="All,In,Out", help="DAPFAM qrel sets to score from one ranking")
    parser.add_argument("--max-chars", type=int, help="truncate rendered DAPFAM text; keep well above the token budget")
    parser.add_argument("--regime", default="IN", help="PatenTEB regime: IN, MIXED or OUT")
    parser.add_argument("--local-parquet", help="PatenTEB test/data.parquet, since the repositories are gated")
    parser.add_argument("--queries", help="jsonl task: queries file")
    parser.add_argument("--corpus", help="jsonl task: corpus file")
    parser.add_argument("--qrels", help="jsonl task: qrels JSON file")
    parser.add_argument("--metric", help="override the task's main metric, e.g. ndcg_cut_10")


def _load_task(args: argparse.Namespace) -> tuple[RetrievalTask, dict[str, dict[str, dict[str, int]]]]:
    if args.task == "dapfam":
        scopes = tuple(s.strip() for s in args.scopes.split(",") if s.strip())
        return dapfam.load_with_scopes(FieldView(args.query_view), FieldView(args.corpus_view), scopes, args.max_chars)
    if args.task == "patenteb":
        return patenteb.load(args.regime, args.local_parquet), {}
    if not (args.queries and args.corpus and args.qrels):
        raise SystemExit("--task jsonl needs --queries, --corpus and --qrels")
    return jsonl.load(args.queries, args.corpus, args.qrels), {}


def _evaluate(args: argparse.Namespace, task: RetrievalTask, extra_qrels: dict) -> EvaluationRecord:
    encoder = load_encoder(args.encoder, **parse_encoder_args(args.encoder_arg))
    cache = VectorCache(args.cache) if args.cache else None
    record = evaluate_encoder(
        task,
        encoder,
        query_reading=parse_reading(args.query_reading, args.across_span_pooling),
        doc_reading=parse_reading(args.doc_reading, args.across_span_pooling),
        metrics=[args.metric] if getattr(args, "metric", None) else None,
        bootstrap=args.bootstrap,
        cache=cache,
        extra_qrels=extra_qrels,
        extra={"encoder_spec": args.encoder},
    )
    if args.out:
        append_record(args.out, record)
    return record


def cmd_run(args: argparse.Namespace) -> int:
    task, extra_qrels = _load_task(args)
    print(task.summary())
    print(_evaluate(args, task, extra_qrels).summary())
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    for query_view, corpus_view in MATRIX_CONFIGS:
        args.task, args.query_view, args.corpus_view = "dapfam", query_view.value, corpus_view.value
        task, extra_qrels = _load_task(args)
        print(_evaluate(args, task, extra_qrels).summary())
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Run the calibration cell (PaECTER, TAC->TAC, 512 tokens per side) and gate on the published value."""
    args.task, args.query_view, args.corpus_view, args.scopes = "dapfam", "TAC", "TAC", "All"
    args.query_reading = args.doc_reading = "1x512"
    task, extra_qrels = _load_task(args)
    record = _evaluate(args, task, extra_qrels)
    print(record.summary())
    published = registry.PUBLISHED[("paecter", "DAPFAM", "TAC->TAC@512/All/ndcg@100")]
    ok, message = scoring.check_reproduction(record.main_score("main", task.main_metric).value, published)
    print(message)
    return 0 if ok else 1


def cmd_registry(args: argparse.Namespace) -> int:
    if args.action == "list":
        for name, config in registry.REGISTRY.items():
            print(f"{name:<18} {config.hf_id:<45} {config.token_pooling:<10} max {config.max_span_length}")
        return 0
    if args.action == "show":
        config = registry.get(args.name)
        for field, value in vars(config).items():
            print(f"{field}: {value}")
        for benchmark in ("DAPFAM", "PatenTEB"):
            for configuration, value in registry.published_for(args.name, benchmark).items():
                print(f"published {benchmark} {configuration}: {value}")
        return 0
    problems = registry.verify_against_hub(args.name)
    print("\n".join(problems) if problems else f"{args.name}: registry entry agrees with the model repository")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pateval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="evaluate one encoder on one task")
    _add_task_arguments(run)
    _add_encoder_arguments(run)
    run.set_defaults(func=cmd_run)

    matrix = subparsers.add_parser("matrix", help="evaluate one encoder on the DAPFAM field configurations")
    _add_encoder_arguments(matrix)
    matrix.add_argument("--scopes", default="All,In,Out")
    matrix.add_argument("--max-chars", type=int)
    matrix.set_defaults(func=cmd_matrix, metric=None)

    reproduce = subparsers.add_parser(
        "reproduce", help="run the PaECTER calibration cell and compare with its published value"
    )
    _add_encoder_arguments(reproduce)
    reproduce.add_argument("--max-chars", type=int)
    reproduce.set_defaults(func=cmd_reproduce, metric=None)

    reg = subparsers.add_parser("registry", help="inspect the published-configuration registry")
    reg.add_argument("action", choices=("list", "show", "verify"))
    reg.add_argument("name", nargs="?")
    reg.set_defaults(func=cmd_registry)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "registry" and args.action != "list" and not args.name:
        raise SystemExit("registry show/verify need a model name")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
