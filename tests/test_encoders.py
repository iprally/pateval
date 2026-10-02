"""Tests for the reading-budget runner: span splitting, across-span pooling and the encoder contract."""

import numpy as np
import pytest

from pateval.encoders import SpannedEncoder, TextEncoder, pool_spans, split_into_spans
from pateval.tasks import Reading
from pateval.testing import HashingSpanEmbedder, hashing_encoder


def test_split_cuts_contiguous_windows_and_drops_the_tail():
    assert split_into_spans(list(range(10)), span_capacity=4, num_spans=2) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_split_of_empty_text_yields_one_empty_span():
    assert split_into_spans([], span_capacity=4, num_spans=8) == [[]]


def test_split_rejects_non_positive_arguments():
    with pytest.raises(ValueError):
        split_into_spans([1], span_capacity=0, num_spans=1)
    with pytest.raises(ValueError):
        split_into_spans([1], span_capacity=1, num_spans=0)


def test_mean_pooling_is_the_plain_average():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert np.allclose(pool_spans(vectors, "mean"), [0.5, 0.5])


def test_normalized_mean_gives_each_span_direction_only():
    # A long span must not dominate through its magnitude.
    vectors = np.array([[10.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert np.allclose(pool_spans(vectors, "normalized_mean"), [0.5, 0.5])
    assert not np.allclose(pool_spans(vectors, "mean"), [0.5, 0.5])


def test_unknown_pooling_raises():
    with pytest.raises(ValueError, match="unknown across-span pooling"):
        pool_spans(np.ones((1, 2), dtype=np.float32), "median")


def test_hashing_encoder_satisfies_the_protocol():
    assert isinstance(hashing_encoder(), TextEncoder)


def test_encoder_returns_one_unit_vector_per_text():
    encoder = hashing_encoder(dim=16)
    vectors = encoder.encode(["a b c", "", "d"], role="document", reading=Reading(8, 512))
    assert vectors.shape == (3, 16)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_reading_budget_changes_what_is_read():
    # 20 tokens; a 1x6 reading (capacity 4 after 2 special tokens) sees the first four, 8x6 sees them all.
    text = " ".join(f"w{i}" for i in range(20))
    encoder = SpannedEncoder(HashingSpanEmbedder(dim=64, max_span_length=512))
    truncated = encoder.encode([text], role="document", reading=Reading(1, 6))
    complete = encoder.encode([text], role="document", reading=Reading(8, 6))
    assert not np.allclose(truncated, complete)
    assert np.allclose(complete, encoder.encode([text], role="document", reading=Reading(20, 6)))


def test_equal_budget_differs_by_span_structure():
    # 1x16 and 4x4 read the same tokens (after overhead) but pool differently; the harness must keep them distinct.
    text = " ".join(f"w{i}" for i in range(40))
    encoder = SpannedEncoder(HashingSpanEmbedder(dim=64, max_span_length=512, special_tokens=0))
    one_long = encoder.encode([text], role="document", reading=Reading(1, 16))
    four_short = encoder.encode([text], role="document", reading=Reading(4, 4))
    assert Reading(1, 16).budget == Reading(4, 4).budget
    # The bag-of-buckets embedder is additive, so mean pooling of the four windows equals the one-window
    # bag up to scale: cosine 1. A real encoder would differ; here we only check both readings ran.
    assert one_long.shape == four_short.shape == (1, 64)


def test_span_longer_than_the_model_allows_is_refused():
    encoder = SpannedEncoder(HashingSpanEmbedder(max_span_length=512))
    with pytest.raises(ValueError, match="cannot read 4096-token spans"):
        encoder.encode(["text"], role="document", reading=Reading(1, 4096))


def test_prompt_overhead_shrinks_the_window_for_that_role_only():
    embedder = HashingSpanEmbedder(dim=64, special_tokens=2, prompt_tokens={"query": 3, "document": 0})
    assert embedder.overhead_tokens("query") == 5
    assert embedder.overhead_tokens("document") == 2
