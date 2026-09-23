"""Tests for the published-configuration registry."""

import pytest

from pateval import registry


def test_cls_native_models_are_recorded_as_cls():
    # Running either with mean pooling understated them by 106% and 28%.
    assert registry.get("bge-m3").token_pooling == "cls"
    assert registry.get("gte-modernbert").token_pooling == "cls"


def test_patembed_base_keeps_its_dense_head_and_prompts():
    cfg = registry.get("patembed-base")
    assert cfg.dense_head is True
    assert cfg.query_prompt and cfg.corpus_prompt


def test_qwen_models_use_last_token_pooling():
    assert registry.get("qwen3-0.6b").token_pooling == "last_token"


def test_unknown_model_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match="no published configuration"):
        registry.get("some-new-model")


def test_published_values_are_recorded_for_the_reproduction_gate():
    assert registry.PUBLISHED[("paecter", "DAPFAM", "TAC->TAC@512/All/ndcg@100")] == pytest.approx(0.343)
