"""Tests for resolving encoders from specification strings."""

import pytest

from pateval.encoders import SpannedEncoder, TextEncoder, load_encoder


def test_module_attribute_spec_calls_the_factory_with_arguments():
    encoder = load_encoder("pateval.testing:hashing_encoder", dim=16)
    assert isinstance(encoder, TextEncoder)
    assert encoder.embedder.dim == 16


def test_a_span_embedder_is_wrapped_so_the_harness_does_the_reading():
    encoder = load_encoder("pateval.testing:HashingSpanEmbedder", dim=8, batch_size=4)
    assert isinstance(encoder, SpannedEncoder)
    assert encoder.embedder.dim == 8 and encoder.batch_size == 4


def test_unknown_spec_raises():
    with pytest.raises(ValueError, match="cannot resolve encoder"):
        load_encoder("no-such-entry-point")


def test_attribute_that_is_not_an_encoder_raises_type_error():
    with pytest.raises(TypeError, match="implements neither"):
        load_encoder("pateval.testing:_bucket", token="x", num_buckets=3)
