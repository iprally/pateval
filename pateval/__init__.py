"""Evaluation harness for patent retrieval benchmarks.

Encodes and scores a benchmark while keeping the *reading budget* --- how much of each document a system
encoded --- as an explicit variable rather than a property of the model. Models plug in through the
`TextEncoder` protocol in `pateval.encoders`; the scoring functions exported here also work on precomputed
vectors.
"""

from pateval.scoring import Result, build_run, check_reproduction, evaluate
from pateval.tasks import FieldView, Reading, RetrievalTask

__all__ = ["FieldView", "Reading", "Result", "RetrievalTask", "build_run", "check_reproduction", "evaluate"]
