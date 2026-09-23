# pateval

An evaluation harness for patent retrieval benchmarks that treats **how much of a document a system
reads** as an experimental variable rather than a property of the model.

Published patent-retrieval numbers almost all sit at 512 tokens per side. That is not a rule of any
benchmark: it is `max_position_embeddings` of BERT-derived encoders. The same weights read as eight
independently encoded 512-token spans, averaged, score several percent higher on DAPFAM, and a model
trained through such an average behaves differently again. This package makes those configurations
first-class, pins the published configuration of every public baseline it knows (readout, projection head,
prompt), and refuses to report a new number before a published one has been reproduced through the same
code path.

It scores vectors and orchestrates encoders. It contains no model training code and ships no data;
benchmarks are fetched from their public sources under their own licences (DAPFAM and PatenTEB are
CC-BY-NC-SA 4.0; the PatenTEB retrieval sets are gated and must be requested from their authors).

## Install

From a checkout:

```bash
pip install -e .            # scoring, tasks, runner, CLI (numpy, pyarrow, pytrec_eval, huggingface_hub)
pip install -e '.[hf]'      # + torch/transformers backend for public Hugging Face models
pip install -e '.[dev]'     # + pytest
# or, without installing the package: pip install -r requirements.txt
```

## Quick start

```bash
# 1. Calibrate: PaECTER, DAPFAM TAC->TAC, 512 tokens per side must reproduce the published 0.343.
pateval reproduce --encoder registry:paecter --cache ./vectors

# 2. The same weights reading more: one 512-token query span, eight 512-token document spans.
pateval run --task dapfam --query-view TAC --corpus-view TAC \
    --encoder registry:paecter --query-reading 1x512 --doc-reading 8x512 \
    --cache ./vectors --out results.jsonl

# 3. All DAPFAM field configurations, all three qrel sets from one ranking each.
pateval matrix --encoder registry:paecter --cache ./vectors --out results.jsonl

# 4. PatenTEB (gated: point at the downloaded test/data.parquet).
pateval run --task patenteb --regime IN --local-parquet retrieval_IN/test/data.parquet \
    --encoder registry:paecter --query-reading 1x512 --doc-reading 1x512 --out results.jsonl

# 5. A model whose code lives somewhere else.
pateval run --task dapfam --encoder my_models.eval:build_encoder --encoder-arg checkpoint=/path/best.ckpt \
    --query-reading 4x512 --doc-reading 8x512 --out results.jsonl
```

Every record appended to `results.jsonl` carries the task, the encoder name, both readings, the
across-span pooling, every metric with a bootstrap confidence interval over queries, and the harness
version, so a table cell can always be traced back to the configuration that produced it.

`registry list` prints the known public models; `registry show paecter` prints the configuration and the
published values the reproduction gate uses; `registry verify bge-m3` checks the entry against the model
repository's own `1_Pooling/config.json` and `modules.json`.

## How it is put together

```
pateval/
  tasks/        benchmark adapters -> RetrievalTask(queries, corpus, qrels, field views, main metric)
                dapfam.py (18 MTEB tasks, field views TA/TAC/CLM/FULL, three qrel scopes)
                patenteb.py (IN/MIXED/OUT reconstructed from triplets), jsonl.py (local files)
  encoders/     the encoder contract and the reading-budget runner
                base.py: TextEncoder, SpanEmbedder, SpannedEncoder, split_into_spans, pool_spans
                hf.py: transformers backend in the registry's published configuration
                plugins.py: load_encoder("registry:paecter" | "module:attr" | entry point)
  registry.py   verified published configurations + published reference values
  runner.py     evaluate_encoder -> EvaluationRecord; VectorCache keyed by encoder, role, reading and texts
  scoring.py    pytrec_eval metrics, self-exclusion, bootstrap CIs, reproduction gate
  cli.py        run / matrix / reproduce / registry
  testing.py    deterministic hashing encoder for tests and dry runs
```

Two decisions are kept apart on purpose. The **within-span readout** (CLS, mean or last token, a
projection head, a prompt) belongs to the model and is captured by `SpanEmbedder`. The **reading**
(`Reading(num_spans, span_length, across_span_pooling)`) belongs to the experiment and is applied by
`SpannedEncoder`, identically for every public model, so that only the within-span readout is the model's
own. The prompt is prepended to every span, since each span is an independent encoder input.

## Evaluating your own model

Implement one protocol and point the CLI at it. Nothing in the harness needs to import your package by name.

```python
# my_models/eval.py
import numpy as np
from pateval.encoders import Role
from pateval.tasks import Reading


class MyEncoder:
    name = "my-model@2026-09"  # part of every record and every cache key: change it when the weights change

    def __init__(self, checkpoint: str):
        self.model = load_my_model(checkpoint)

    def encode(self, texts: list[str], *, role: Role, reading: Reading) -> np.ndarray:
        # Do your own reading here (spanning, learned across-span pooling, prompts...). `reading` says what
        # the experiment intends; if your model fixes its own reading, validate the request and raise on a
        # mismatch rather than silently reading something else, because the reading is recorded with the score.
        return self.model.embed(texts, side=role, num_spans=reading.num_spans, span_length=reading.span_length)


def build_encoder(checkpoint: str) -> MyEncoder:
    return MyEncoder(checkpoint)
```

```bash
pateval run --task dapfam --encoder my_models.eval:build_encoder --encoder-arg checkpoint=best.ckpt ...
```

If your model embeds one span at a time and you want the harness to do the reading, implement
`SpanEmbedder` instead (`tokenize`, `overhead_tokens`, `embed_spans`, `max_span_length`) and wrap it in
`SpannedEncoder`; `pateval/testing.py` is a complete, dependency-free example. An installed package can
also advertise encoders through the entry-point group `pateval.encoders` and be addressed by name.

Python API:

```python
from pateval import runner
from pateval.encoders import load_encoder
from pateval.tasks import FieldView, Reading, dapfam

task, other_scopes = dapfam.load_with_scopes(FieldView.TITLE_ABSTRACT_CLAIMS, FieldView.TITLE_ABSTRACT_CLAIMS)
record = runner.evaluate_encoder(
    task,
    load_encoder("registry:paecter"),
    query_reading=Reading(1, 512),
    doc_reading=Reading(8, 512),
    extra_qrels=other_scopes,
)
print(record.summary())
```

## Things that silently ruin a number

All caught in our own runs, and pinned by the registry for every model it lists: a CLS-native model run
with mean pooling (BGE-M3, gte-modernbert: 106% and 28% too low), a dropped sentence-transformers `Dense`
head (Stella, patembed-base), the wrong prompt variant (patembed's `retrieval_IN/OUT/MIXED` are domain
labels, 11%), and conflating the within-span readout with the across-span pooling. A model missing from the registry raises rather than
getting a default; add it after reading its model card, `1_Pooling/config.json` and `modules.json`.

## Development

```bash
pip install -e '.[dev]'
python -m pytest
```

## License

MIT, see [LICENSE](LICENSE). The benchmarks the harness downloads keep their own licences (see above).
