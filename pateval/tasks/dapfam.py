"""DAPFAM: family-level patent retrieval with citation-based relevance and domain partitions.

Ayaou et al., arXiv:2506.22141. 1,247 query families, 45,336 target families, full text available.

Two properties are easy to get wrong and are handled explicitly here.

*Only six of the 18 MTEB tasks are the source paper's*, and all six take the title+abstract+claims
corpus: the two query views crossed with the three qrel sets. No headline configuration reads the
description, so a full-text number is a different task, not a larger budget.

*Fields concatenate with the description last.* Title+abstract+claims alone fills a 512-token window for
roughly 92% of documents, so at that budget the title+abstract+claims and full-text views are the same
text; above it they diverge sharply. Any comparison across views must state the budget.
"""

from __future__ import annotations

from pateval.tasks.base import FieldView, RetrievalTask

HF_DATASET = "datalyes/DAPFAM_patent"

# Field order matters: description last, which is what makes the views coincide at small budgets.
_FIELDS: dict[FieldView, list[str]] = {
    FieldView.TITLE_ABSTRACT: ["title_en", "abstract_en"],
    FieldView.TITLE_ABSTRACT_CLAIMS: ["title_en", "abstract_en", "claims_text"],
    FieldView.CLAIMS: ["claims_text"],
    FieldView.FULL_TEXT: ["title_en", "abstract_en", "claims_text", "description_en"],
}
QUERY_VIEWS = (FieldView.TITLE_ABSTRACT, FieldView.TITLE_ABSTRACT_CLAIMS)
SCOPES = ("All", "In", "Out")
# The six tasks the source paper reports: both query views, always the TAC corpus.
IN_PAPER_CORPUS_VIEW = FieldView.TITLE_ABSTRACT_CLAIMS


def render(row: dict, view: FieldView) -> str:
    """Join a row's fields for `view` the way the MTEB task definition does: newline, skipping empties."""
    return "\n".join(str(row[f]) for f in _FIELDS[view] if row.get(f))


def _read(fname: str) -> list[dict]:
    import pyarrow.parquet as pq

    from huggingface_hub import hf_hub_download

    return pq.read_table(hf_hub_download(HF_DATASET, fname, repo_type="dataset")).to_pylist()


def load_qrels(scope: str = "All") -> dict[str, dict[str, int]]:
    """Relevant pairs for one scope. `domain_rel` is IN / OUT / NC; a pair counts when its label matches."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    qrels: dict[str, dict[str, int]] = {}
    for r in _read("qrels_all.parquet"):
        if scope != "All" and str(r["domain_rel"]).upper() != scope.upper():
            continue
        if float(r["relevance_score"]) <= 0:
            continue
        qrels.setdefault(str(r["query_id"]), {})[str(r["relevant_id"])] = 1
    return qrels


def load(
    query_view: FieldView,
    corpus_view: FieldView,
    scope: str = "All",
    max_chars: int | None = None,
) -> RetrievalTask:
    """Build a DAPFAM task from the HuggingFace dataset.

    `max_chars` truncates rendered text. Keep it comfortably above the intended token budget: cutting at
    fewer characters than the model will read silently under-feeds it, which is a measurement error rather
    than a protocol choice.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")

    def _clip(text: str) -> str:
        return text[:max_chars] if max_chars else text

    queries = {str(r["query_id"]): _clip(render(r, query_view)) for r in _read("queries.parquet")}
    corpus = {str(r["relevant_id"]): _clip(render(r, corpus_view)) for r in _read("corpus.parquet")}

    return RetrievalTask(
        name=f"DAPFAM{scope}{query_view.value}To{corpus_view.value}",
        queries=queries,
        corpus=corpus,
        qrels=load_qrels(scope),
        query_view=query_view,
        corpus_view=corpus_view,
        main_metric="ndcg_cut_100",
    )


def load_with_scopes(
    query_view: FieldView,
    corpus_view: FieldView,
    scopes: tuple[str, ...] = SCOPES,
    max_chars: int | None = None,
) -> tuple[RetrievalTask, dict[str, dict[str, dict[str, int]]]]:
    """One task (first scope) plus the other scopes' qrels, so a single ranking is scored under all of them.

    All scopes rank the same corpus and only the judgments differ, which is why the out-of-domain
    numbers are comparable between systems but an order of magnitude below the unpartitioned ones.
    """
    if not scopes:
        raise ValueError("at least one scope is required")
    task = load(query_view, corpus_view, scopes[0], max_chars)
    task.name = f"DAPFAM{query_view.value}To{corpus_view.value}"
    extra = {scope: load_qrels(scope) for scope in scopes[1:]}
    return task, extra


def in_paper_tasks(max_chars: int | None = None) -> list[RetrievalTask]:
    """The six tasks the DAPFAM paper reports, which is what published numbers are comparable to."""
    return [load(qv, IN_PAPER_CORPUS_VIEW, scope, max_chars) for qv in QUERY_VIEWS for scope in SCOPES]
