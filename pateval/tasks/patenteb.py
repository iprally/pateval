"""PatenTEB retrieval tasks, reconstructed from triplets into a retrieval corpus.

Ayaou et al., arXiv:2510.22264. The three retrieval sets are partitioned by how far the IPC3 codes of a
negative overlap the query's: `retrieval_IN` (full overlap), `retrieval_MIXED` (partial) and
`retrieval_OUT` (none). They are disjoint difficulty regimes --- MIXED is *not* a union of the other two,
despite the name; every row of MIXED carries a single `neg_computed_domain` of `PART_MIX`.

Two caveats belong with any number produced here.

*The corpus is our reconstruction.* The released files are `(query, positive, negative)` triplets, not a
corpus with qrels. We take the corpus to be the union of positives and negatives and the qrels to be the
positive links. That is a defensible reading but it is ours, so a published value should be reproduced
through it before new numbers are quoted.

*The texts are short.* Queries average about 390 tokens and documents about 362, so a single 512-token
span already reads roughly 92% of either side. These tasks measure the quality of a short patent
embedding; they say nothing about how a system reads a full-length document.
"""

from __future__ import annotations

from pateval.tasks.base import FieldView, RetrievalTask

REGIMES = {"IN": "datalyes/retrieval_IN", "MIXED": "datalyes/retrieval_MIXED", "OUT": "datalyes/retrieval_OUT"}


def load(regime: str, local_parquet: str | None = None) -> RetrievalTask:
    """Build a PatenTEB retrieval task.

    The HuggingFace repositories are gated; pass `local_parquet` to read an already-downloaded
    `test/data.parquet` instead of fetching (`git lfs pull` is needed after cloning, or the file is a
    133-byte pointer).
    """
    import pyarrow.parquet as pq

    regime = regime.upper()
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {sorted(REGIMES)}, got {regime!r}")

    if local_parquet:
        path = local_parquet
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(REGIMES[regime], "test/data.parquet", repo_type="dataset")
    rows = pq.read_table(path).to_pylist()

    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    for r in rows:
        qid, pid, nid = str(r["q"]), str(r["pos"]), str(r["neg"])
        queries.setdefault(qid, r["q_text"])
        corpus.setdefault(pid, r["pos_text"])
        corpus.setdefault(nid, r["neg_text"])
        # A query recurs across rows with different positives; every positive link is relevant.
        qrels.setdefault(qid, {})[pid] = 1

    return RetrievalTask(
        name=f"PatenTEB_retrieval_{regime}",
        queries=queries,
        corpus=corpus,
        qrels=qrels,
        # Both sides are title [SEP] abstract.
        query_view=FieldView.TITLE_ABSTRACT,
        corpus_view=FieldView.TITLE_ABSTRACT,
        main_metric="ndcg_cut_10",
    )
