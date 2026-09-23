"""Encoders: how texts become vectors.

The harness separates two decisions that published evaluations conflate. The *within-span readout* (CLS,
mean or last token; a projection head; a prompt) belongs to the model and is captured by `SpanEmbedder`.
The *reading* --- how many spans of which length are encoded and how their vectors are combined --- is an
experimental variable and belongs to the harness (`SpannedEncoder`). A model that does its own reading,
such as a span-trained encoder with a learned across-span pooling, implements `TextEncoder` directly and
is injected without the harness knowing anything about its internals.
"""

from pateval.encoders.base import (
    Role,
    SpanEmbedder,
    SpannedEncoder,
    TextEncoder,
    pool_spans,
    split_into_spans,
)
from pateval.encoders.plugins import load_encoder

__all__ = [
    "Role",
    "SpanEmbedder",
    "SpannedEncoder",
    "TextEncoder",
    "load_encoder",
    "pool_spans",
    "split_into_spans",
]
