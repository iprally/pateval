"""Evaluation harness for patent retrieval benchmarks.

Encodes and scores a benchmark while keeping the *reading budget* --- how much of each document a system
encoded --- as an explicit variable rather than a property of the model. Models plug in through the
`TextEncoder` protocol in `pateval.encoders`; the scoring functions exported here also work on precomputed
vectors.
"""

__version__ = "0.3.0"

from pateval.scoring import Result, build_run, check_reproduction, evaluate  # noqa: E402
from pateval.tasks import FieldView, Reading, RetrievalTask  # noqa: E402

__all__ = [
    "FieldView",
    "Reading",
    "Result",
    "RetrievalTask",
    "__version__",
    "build_run",
    "check_reproduction",
    "evaluate",
]
