"""Loading encoders from a specification string.

Three forms are accepted:

* `registry:<name>` --- a public model in its verified published configuration, via the Hugging Face
  backend (needs the `hf` extra).
* `<module.path>:<attribute>` --- anything importable. If the attribute is a class or function it is
  called with the keyword arguments given on the command line; the result must satisfy `TextEncoder`, or
  `SpanEmbedder`, in which case the harness wraps it in `SpannedEncoder` and does the reading itself.
  This is how a model whose code lives elsewhere is evaluated without the harness importing it by name.
* `<entry point name>` --- an installed package advertising an encoder under the entry-point group
  `pateval.encoders`.
"""

from __future__ import annotations

import importlib
import importlib.metadata

from typing import Any

from pateval.encoders.base import SpanEmbedder, SpannedEncoder, TextEncoder

ENTRY_POINT_GROUP = "pateval.encoders"
REGISTRY_PREFIX = "registry:"


def load_encoder(spec: str, **kwargs: Any) -> TextEncoder:
    """Resolve `spec` to an encoder instance; see the module docstring for the accepted forms."""
    if spec.startswith(REGISTRY_PREFIX):
        return _load_registry_encoder(spec[len(REGISTRY_PREFIX) :], **kwargs)
    if ":" in spec:
        module_name, _, attribute = spec.partition(":")
        target = getattr(importlib.import_module(module_name), attribute)
        return _instantiate(target, spec, **kwargs)
    for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        if entry_point.name == spec:
            return _instantiate(entry_point.load(), spec, **kwargs)
    raise ValueError(
        f"cannot resolve encoder {spec!r}: expected 'registry:<name>', '<module>:<attribute>' or an "
        f"entry point in group {ENTRY_POINT_GROUP!r}"
    )


def _load_registry_encoder(name: str, **kwargs: Any) -> TextEncoder:
    from pateval import registry
    from pateval.encoders.hf import HFSpanEmbedder

    batch_size = int(kwargs.pop("batch_size", 32))
    embedder = HFSpanEmbedder.from_config(name, registry.get(name), **kwargs)
    return SpannedEncoder(embedder, batch_size=batch_size)


def _instantiate(target: Any, spec: str, **kwargs: Any) -> TextEncoder:
    """Call `target` with the given arguments; a bare `SpanEmbedder` is wrapped so the harness does the reading."""
    batch_size = int(kwargs.pop("batch_size", 32))
    produced = target(**kwargs) if callable(target) else target
    if isinstance(produced, TextEncoder):
        return produced
    if isinstance(produced, SpanEmbedder):
        return SpannedEncoder(produced, batch_size=batch_size)
    raise TypeError(
        f"{spec!r} produced {type(produced).__name__}, which implements neither TextEncoder nor SpanEmbedder"
    )
