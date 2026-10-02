"""PatenTEB retrieval tasks, reconstructed from triplets into a retrieval corpus.

Ayaou et al., arXiv:2510.22264. The three retrieval sets are partitioned by how far the IPC3 codes of a
negative overlap the query's: `retrieval_IN` (full overlap), `retrieval_MIXED` (partial) and
`retrieval_OUT` (none). They are disjoint difficulty regimes --- MIXED is *not* a union of the other two,
despite the name; every row of MIXED carries a single `neg_computed_domain` of `PART_MIX`.

Two caveats belong with any number produced here.

*The corpus is our reconstruction.* The released files are `(query, positive, negative)` triplets, not a
corpus with qrels. We take the corpus to be the union of positives and negatives and the qrels to be the
positive links. PaECTER's published values reproduce through it to within 2% on `retrieval_IN` and
`retrieval_MIXED`, but `retrieval_OUT` comes out 14% low (0.1028 against 0.120), so OUT numbers are
indicative only until that gap is explained.

*The texts are short.* Both sides are title, abstract and claim text, a few hundred tokens each, so a single
512-token span already reads nearly all of either side. These tasks measure the quality of a short patent
embedding; they say nothing about how a system reads a full-length document.
"""

from __future__ import annotations

import hashlib

from pateval.tasks.base import FieldView, RetrievalTask

# Repository and pinned commit of each regime; `revision` selects another commit.
REGIMES = {
    "IN": ("datalyes/retrieval_IN", "161c119f15561d9b6adcde2a759cdf76a17b0a62"),
    "MIXED": ("datalyes/retrieval_MIXED", "cabb81aa342a192678062832dad877e82c1b47ed"),
    "OUT": ("datalyes/retrieval_OUT", "d156b1ec2d307af32a366c9c44fedc56808ed49b"),
}


def load(regime: str, local_parquet: str | None = None, revision: str | None = None) -> RetrievalTask:
    """Build a PatenTEB retrieval task.

    The Hugging Face repositories are gated: accept their conditions on the dataset pages, or pass
    `local_parquet` to read an already-downloaded `test/data.parquet` instead of fetching (`git lfs pull` is
    needed after cloning, or the file is a 133-byte pointer). A local file is recorded by its SHA-256.
    """
    import pyarrow.parquet as pq

    regime = regime.upper()
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {sorted(REGIMES)}, got {regime!r}")

    repository, pinned = REGIMES[regime]
    if local_parquet:
        path = local_parquet
        source = {"file": str(local_parquet), "sha256": _sha256(local_parquet)}
    else:
        from huggingface_hub import hf_hub_download

        commit = revision or pinned
        path = hf_hub_download(repository, "test/data.parquet", repo_type="dataset", revision=commit)
        source = {"dataset": repository, "revision": commit}
    rows = pq.read_table(path, columns=["q", "pos", "neg", "q_text", "pos_text", "neg_text"]).to_pylist()

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
        # Both sides are "title [SEP] abstract [SEP] claim".
        query_view=FieldView.TITLE_ABSTRACT_CLAIMS,
        corpus_view=FieldView.TITLE_ABSTRACT_CLAIMS,
        main_metric="ndcg_cut_10",
        qrels_name=regime,
        source=source,
    )


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
