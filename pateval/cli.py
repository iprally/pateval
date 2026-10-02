"""Command-line interface.

    pateval run --task dapfam --query-view TAC --corpus-view TAC --encoder registry:paecter \\
        --query-reading 1x512 --doc-reading 8x512 --cache ./vectors --out results.jsonl
    pateval run --task patenteb --regime IN --local-parquet retrieval_IN/test/data.parquet --encoder ...
    pateval run --task jsonl --queries q.jsonl --corpus c.jsonl --qrels qrels.json --encoder my_pkg.eval:Encoder
    pateval matrix --encoder registry:paecter --cache ./vectors --out results.jsonl
    pateval reproduce --baseline paecter            # or --baseline all
    pateval registry list | show paecter | verify paecter

An encoder is named by a spec (see `pateval.encoders.plugins`); `--encoder-arg key=value` passes
keyword arguments to its factory, so a model whose code lives in another package needs no change here.
For registry models the arguments are `prompt` (a variant name, or `none`), `revision`, `dtype`, `device`
and `batch_size`.
"""

from __future__ import annotations

import argparse
import dataclasses
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

# The calibration cell: the configuration the published DAPFAM baselines were measured in.
CALIBRATION_TASK = "TAC->TAC"
CALIBRATION_READING = "1x512"
# Encoder arguments `reproduce` accepts: they change speed or numerical precision, not the configuration.
CALIBRATION_ARGUMENTS = frozenset({"batch_size", "device", "dtype"})


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


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoder-arg", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples over queries; 0 disables")
    parser.add_argument("--cache", help="directory for encoded vectors, reused across configurations")
    parser.add_argument("--out", help="JSON Lines file to append records to")


def _add_encoder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoder", required=True, help="registry:<name>, <module>:<attribute>, or an entry point")
    parser.add_argument("--query-reading", default="1x512", help="spans x span length for queries")
    parser.add_argument("--doc-reading", default="8x512", help="spans x span length for documents")
    parser.add_argument("--across-span-pooling", default="mean", choices=ACROSS_SPAN_POOLINGS)
    _add_common_arguments(parser)


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
    encoder_arguments = parse_encoder_args(args.encoder_arg)
    encoder = load_encoder(args.encoder, **encoder_arguments)
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
        encoder_spec=args.encoder,
        encoder_arguments=encoder_arguments,
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


def calibration_baselines() -> list[str]:
    """Registry models with a gated published value in the calibration cell, in registry order."""
    gated = {
        ref.model
        for ref in registry.references(benchmark="DAPFAM", task=CALIBRATION_TASK, reading=CALIBRATION_READING)
        if ref.gated
    }
    return [name for name in registry.REGISTRY if name in gated]


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Run published baselines in the calibration cell and compare each with its own published values.

    The cell is DAPFAM TAC->TAC at 512 tokens per side, scored under All, In and Out from one ranking. A
    baseline runs once per prompt variant its references name. Gated references decide the exit code;
    the others are printed next to the measurement.
    """
    names = calibration_baselines() if "all" in args.baseline else list(dict.fromkeys(args.baseline))
    encoder_arguments = parse_encoder_args(args.encoder_arg)
    changed = sorted(set(encoder_arguments) - CALIBRATION_ARGUMENTS)
    if changed:
        raise SystemExit(f"reproduce runs the published configuration; it does not take {changed}")
    plans: dict[str, list[registry.Reference]] = {}
    for name in names:
        refs = registry.references(name, "DAPFAM", task=CALIBRATION_TASK, reading=CALIBRATION_READING)
        if not refs:
            raise SystemExit(f"no published DAPFAM {CALIBRATION_TASK} value is recorded for {name!r}")
        plans[name] = refs

    task, extra_qrels = dapfam.load_with_scopes(FieldView.TITLE_ABSTRACT_CLAIMS, FieldView.TITLE_ABSTRACT_CLAIMS)
    print(task.summary())
    cache = VectorCache(args.cache) if args.cache else None
    reading = parse_reading(CALIBRATION_READING)
    failed: list[str] = []
    for name, refs in plans.items():
        for variant in dict.fromkeys(ref.prompt for ref in refs):
            spec = f"registry:{name}"
            arguments = dict(encoder_arguments, prompt=variant)
            record = evaluate_encoder(
                task,
                load_encoder(spec, **arguments),
                query_reading=reading,
                doc_reading=reading,
                bootstrap=args.bootstrap,
                cache=cache,
                extra_qrels=extra_qrels,
                encoder_spec=spec,
                encoder_arguments=arguments,
            )
            print(record.summary())
            checks = []
            for ref in (ref for ref in refs if ref.prompt == variant):
                measured = record.main_score(ref.scope, ref.metric).value
                ok, message = scoring.check_reproduction(measured, ref.value)
                label = f"{name} [prompt {record.encoder_identity.get('prompt', variant)}] {ref.scope}"
                print(f"  {label}: {message}{'' if ref.gated else ' (printed, not gated)'}")
                checks.append(
                    {
                        "scope": ref.scope,
                        "metric": ref.metric,
                        "published": ref.value,
                        "source": ref.source,
                        "measured": measured,
                        "relative_gap": (measured - ref.value) / ref.value,
                        "gated": ref.gated,
                        "ok": ok,
                    }
                )
                if ref.gated and not ok:
                    failed.append(label)
            record.extra["reproduction"] = checks
            if args.out:
                append_record(args.out, record)
    print("all gated values reproduce" if not failed else f"DOES NOT REPRODUCE: {', '.join(failed)}")
    return 1 if failed else 0


def cmd_registry(args: argparse.Namespace) -> int:
    if args.action == "list":
        for name, config in registry.REGISTRY.items():
            print(
                f"{name:<18} {config.hf_id:<45} {config.token_pooling:<10} max {config.max_span_length:<5} "
                f"prompt {config.default_prompt}"
            )
        return 0
    if args.action == "show":
        config = registry.get(args.name)
        for field in dataclasses.fields(config):
            if field.name == "prompts":
                for variant, prompt in config.prompts.items():
                    print(f"prompt {variant} ({prompt.origin}): query={prompt.query!r} document={prompt.document!r}")
                continue
            print(f"{field.name}: {getattr(config, field.name)}")
        for ref in registry.references(args.name):
            print(
                f"published {ref.benchmark} {ref.task} {ref.reading} {ref.scope} {ref.metric} prompt={ref.prompt}: "
                f"{ref.value}{' (gated)' if ref.gated else ''} -- {ref.source}"
            )
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
        "reproduce", help="run published baselines in their published configuration and compare with their values"
    )
    reproduce.add_argument(
        "--baseline",
        action="append",
        default=None,
        help="registry model with a published DAPFAM value (repeatable), or 'all'; default paecter",
    )
    _add_common_arguments(reproduce)
    reproduce.set_defaults(func=cmd_reproduce)

    reg = subparsers.add_parser("registry", help="inspect the published-configuration registry")
    reg.add_argument("action", choices=("list", "show", "verify"))
    reg.add_argument("name", nargs="?")
    reg.set_defaults(func=cmd_registry)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "registry" and args.action != "list" and not args.name:
        raise SystemExit("registry show/verify need a model name")
    if args.command == "reproduce" and not args.baseline:
        args.baseline = ["paecter"]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
