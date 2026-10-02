"""Tests for the BM25 baseline: the formula on a hand-sized corpus, and its place in the harness."""

import math

import numpy as np
import pytest

from pateval import runner
from pateval.encoders import Ranker, load_encoder
from pateval.encoders.bm25 import BM25, tokenize
from pateval.tasks import FieldView, Reading, RetrievalTask

CORPUS = {
    "d1": "rotor blade pitch control",
    "d2": "lithium anode battery cell",
    "d3": "rotor rotor rotor assembly",
    "q1": "rotor blade pitch",  # the query itself, as in family-level benchmarks
}


def _reference(query: str, document: str, k1: float, b: float) -> float:
    """BM25 written out longhand, independently of the index."""
    docs = [tokenize(text) for text in CORPUS.values()]
    avgdl = sum(len(d) for d in docs) / len(docs)
    terms = tokenize(document)
    score = 0.0
    for term in tokenize(query):
        tf = terms.count(term)
        if not tf:
            continue
        df = sum(1 for d in docs if term in d)
        idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
        score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(terms) / avgdl))
    return score


def test_tokenizer_lowercases_and_drops_single_characters():
    assert tokenize("A Rotor-Blade, pitch 2 x") == ["rotor", "blade", "pitch"]
    assert tokenize("the rotor of the blade", frozenset({"the", "of"})) == ["rotor", "blade"]


@pytest.mark.parametrize("k1,b", [(1.2, 0.75), (0.9, 0.4)])
def test_scores_match_the_formula_and_exclude_the_query_itself(k1, b):
    run = BM25(k1=k1, b=b).rank({"q1": "rotor blade pitch"}, CORPUS, top_k=10)
    assert "q1" not in run["q1"], "the query's own document is dropped from its ranking"
    for doc_id, score in run["q1"].items():
        assert score == pytest.approx(_reference("rotor blade pitch", CORPUS[doc_id], k1, b), rel=1e-5)
    assert list(run["q1"]) == ["d1", "d3"], "d1 shares three terms, d3 one; d2 scores zero and is left out"


def test_stop_words_are_removed_on_both_sides():
    run = BM25(stopwords="lucene").rank({"q": "the rotor"}, {"a": "the the the rotor", "b": "the the"}, top_k=5)
    assert list(run["q"]) == ["a"]
    with pytest.raises(ValueError, match="stopwords"):
        BM25(stopwords="nltk")


def test_bm25_is_a_ranker_the_harness_evaluates_at_full_text():
    task = RetrievalTask(
        name="toy",
        queries={"q1": "rotor blade pitch", "q2": "battery anode lithium"},
        corpus=CORPUS | {"d2": "lithium anode battery cell"},
        qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
        query_view=FieldView.FULL_TEXT,
        corpus_view=FieldView.FULL_TEXT,
        main_metric="ndcg_cut_10",
    )
    encoder = load_encoder("pateval.encoders.bm25:BM25", k1=1.5)
    assert isinstance(encoder, Ranker) and encoder.k1 == 1.5
    record = runner.evaluate_encoder(
        task, encoder, query_reading=Reading(1, 512), doc_reading=Reading(8, 512), bootstrap=0, encoder_spec="bm25"
    )
    assert record.main_score("main", "ndcg_cut_10").value == pytest.approx(1.0)
    assert record.query_reading["span_length"] == 0 and record.doc_reading["span_length"] == 0
    assert "full text" in record.summary()
    assert record.encoder_identity["k1"] == 1.5 and record.encoder_identity["stopwords"] == "none"


def test_empty_corpus_and_unknown_query_terms_do_not_break_ranking():
    assert BM25().rank({"q": "anything"}, {"d": ""}, top_k=5) == {"q": {}}
    assert BM25().rank({"q": "unseen words"}, CORPUS, top_k=5) == {"q": {}}
    scores = BM25().rank({"q": "rotor"}, CORPUS, top_k=1)["q"]
    assert len(scores) == 1 and isinstance(next(iter(scores.values())), float)
    np.testing.assert_array_less(0, list(scores.values()))
