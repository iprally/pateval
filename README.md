# pateval

An evaluation harness for patent retrieval benchmarks that treats **how much of a document a system
reads** as an experimental variable rather than a property of the model.

Published patent-retrieval numbers almost all sit at 512 tokens per side. That is not a rule of any
benchmark: it is `max_position_embeddings` of BERT-derived encoders. The same weights read as eight
independently encoded 512-token spans, averaged, score several percent higher on DAPFAM, and a model
trained through such an average behaves differently again. This package makes those configurations
first-class and pins the published configuration of every public baseline it knows: readout, projection
head, prompt and repository commit. It also ships a calibration command, `pateval reproduce`, which runs
published baselines through the same code path and compares them with their published values. Run it
before trusting new numbers: without that anchor a harness bug is indistinguishable from a finding.

It scores vectors and orchestrates encoders. It contains no model training code and ships no data or
weights: benchmarks and models are downloaded from their public sources at run time and keep their own
licences. DAPFAM, the PatenTEB retrieval sets and the patembed models are CC BY-NC-SA 4.0
(non-commercial, share-alike). The PatenTEB retrieval sets are gated on the Hugging Face Hub: accept the
conditions on each dataset page before downloading.

## Install

From a checkout:

```bash
pip install -e .            # scoring, tasks, runner, CLI (numpy, pyarrow, pytrec_eval, huggingface_hub)
pip install -e '.[hf]'      # + torch/transformers backend, needed for every registry: model
pip install -e '.[dev]'     # + pytest
# or, without installing the package: pip install -r requirements.txt (the core only)
```

`requirements-paper.txt` pins the environment the reproduction values below were measured in.

## Quick start

```bash
# 1. Calibrate: PaECTER on DAPFAM TAC->TAC at 512 tokens per side must reproduce its published 0.343.
pateval reproduce --baseline paecter --cache ./vectors
pateval reproduce --baseline all --cache ./vectors   # every baseline with a published DAPFAM value

# 2. The same weights reading more: one 512-token query span, eight 512-token document spans.
pateval run --task dapfam --query-view TAC --corpus-view TAC \
    --encoder registry:paecter --query-reading 1x512 --doc-reading 8x512 \
    --cache ./vectors --out results.jsonl

# 3. All DAPFAM field configurations, all three qrel sets from one ranking each.
pateval matrix --encoder registry:paecter --cache ./vectors --out results.jsonl

# 4. A prompt variant other than the registry default, or none at all.
pateval run --task dapfam --encoder registry:patembed-large --encoder-arg prompt=OUT \
    --query-reading 1x512 --doc-reading 1x512 --out results.jsonl

# 5. PatenTEB (gated: accept the conditions on the Hub, or point at a downloaded test/data.parquet).
pateval run --task patenteb --regime IN --local-parquet retrieval_IN/test/data.parquet \
    --encoder registry:paecter --query-reading 1x512 --doc-reading 1x512 --out results.jsonl

# 6. A model whose code lives somewhere else.
pateval run --task dapfam --encoder my_models.eval:build_encoder --encoder-arg checkpoint=/path/best.ckpt \
    --query-reading 4x512 --doc-reading 8x512 --out results.jsonl
```

Every record appended to `results.jsonl` carries what is needed to trace a table cell back to the
configuration that produced it:

* the task and its source: dataset and commit, or file and SHA-256, plus the field views and any truncation;
* the encoder's identity. For a registry model this is the repository, commit, readout, projection head,
  prompt variant and strings, dtype and library versions. For your own encoder it is its spec, its
  arguments and whatever its `identity()` returns;
* both readings, with their across-span pooling;
* every metric with a bootstrap confidence interval over queries, for every qrel set;
* the scoring settings, and the environment (Python, libraries, harness version and commit).

`--cache` reuses encoded vectors across field views, qrel sets and configurations. Its key is that
encoder identity plus the role, the reading and a digest of the exact ids and texts. A new model commit,
prompt, dtype, library version or text rendering therefore never reuses stale vectors.

The `registry` command:

* `registry list` prints the known public models.
* `registry show paecter` prints the configuration, the pinned commit and the published values the
  calibration uses.
* `registry verify bge-m3` checks the entry, at its pinned commit, against the model repository's own
  `1_Pooling/config.json`, `modules.json` and `config_sentence_transformers.json`.

## Reproduction

`pateval reproduce --baseline all` runs each baseline that has a published DAPFAM value in the
configuration it was published in: TAC->TAC, 512 tokens per side, nDCG@100. It compares All with the
published value and fails if any gated value is more than 2% off. In, Out and values below 0.1 are
printed but not gated. Measured on one GPU in float32 with `requirements-paper.txt` (brackets: bootstrap
95% confidence intervals over queries):

| Baseline | Prompt | All | In | Out | Published |
|---|---|---|---|---|---|
| PaECTER | none | 0.3424 [0.3278, 0.3548] | 0.3860 | 0.0598 | 0.343 / 0.387 / 0.060 |
| patembed-base | MIXED | 0.3654 [0.3506, 0.3776] | 0.4134 | 0.0609 | 0.370 |
| patembed-base | none | 0.3512 [0.3365, 0.3633] | 0.3970 | 0.0602 | 0.352 |
| patembed-large | MIXED | 0.3721 [0.3571, 0.3841] | 0.4210 | 0.0615 | 0.377 |
| patembed-large | none | 0.0407 [0.0368, 0.0446] | 0.0457 | 0.0053 | 0.044 (not gated) |
| BERT-for-Patents | none | 0.2260 [0.2159, 0.2359] | 0.2532 | 0.0411 | 0.228 |

Every gated value reproduces to within 1.3%.

Published values are from the PatenTEB paper (arXiv:2510.22264, Table 16), whose "with prompt" column
does not name the variant; MIXED is the best of the three on this task.

## How it is put together

```
pateval/
  tasks/        benchmark adapters -> RetrievalTask(queries, corpus, qrels, field views, main metric, source)
                dapfam.py (18 MTEB tasks, field views TA/TAC/CLM/FULL, three qrel scopes, pinned commit)
                patenteb.py (IN/MIXED/OUT reconstructed from triplets), jsonl.py (local files)
  encoders/     the encoder contract and the reading-budget runner
                base.py: TextEncoder, SpanEmbedder, SpannedEncoder, split_into_spans, pool_spans
                hf.py: transformers backend in the registry's published configuration
                plugins.py: load_encoder("registry:paecter" | "module:attr" | entry point)
  registry.py   pinned published configurations + published reference values
  runner.py     evaluate_encoder -> EvaluationRecord; VectorCache keyed by encoder identity, role, reading, texts
  scoring.py    pytrec_eval metrics, self-exclusion, bootstrap CIs, reproduction check
  cli.py        run / matrix / reproduce / registry
  testing.py    deterministic hashing encoder for tests and dry runs
```

Two decisions are kept apart on purpose. The **within-span readout** (CLS, mean or last token, a
projection head, a prompt) belongs to the model and is captured by `SpanEmbedder`. The **reading**
(`Reading(num_spans, span_length, across_span_pooling)`) belongs to the experiment and is applied by
`SpannedEncoder`, identically for every public model, so that only the within-span readout is the model's
own. The prompt is prepended to every span, since each span is an independent encoder input. Prompts come
in named variants per model (`--encoder-arg prompt=OUT`), and `none` is always available, so an
unprompted run is a deliberate, recorded choice.

## Evaluating your own model

Implement one protocol and point the CLI at it. Nothing in the harness needs to import your package by name.

```python
# my_models/eval.py
import numpy as np
from pateval.encoders import Role
from pateval.tasks import Reading


class MyEncoder:
    name = "my-model"  # a display name, part of every record

    def __init__(self, checkpoint: str):
        self.checkpoint_sha256 = sha256_of_file(checkpoint)
        self.model = load_my_model(checkpoint)

    def identity(self) -> dict:
        # Optional, but recorded with every score and part of the cache key: return whatever determines the
        # vectors that the spec and its arguments do not show, such as the weights behind a path.
        return {"checkpoint_sha256": self.checkpoint_sha256}

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
    encoder_spec="registry:paecter",
)
print(record.summary())
```

## Things that silently ruin a number

Each of these yields a plausible-looking number that is simply wrong. The registry pins all of them for
every model it lists, and `registry verify` checks the first three against the model repository:

* a CLS-native model such as BGE-M3 or gte-modernbert run with mean pooling, or the reverse
  (`1_Pooling/config.json`);
* a dropped sentence-transformers `Dense` projection head, as patembed-base ships (`modules.json`);
* a missing or wrong prompt (`config_sentence_transformers.json`). patembed's `retrieval_IN/OUT/MIXED` are
  domain labels rather than task instructions, and patembed-large without its prompt scores a fraction
  of its prompted value. With spans the prompt belongs in every span, not only the first;
* conflating the within-span readout with the across-span pooling.

A model missing from the registry raises rather than getting a default. To add it, read its model card,
`1_Pooling/config.json`, `modules.json` and `config_sentence_transformers.json`, and pin the commit you read.

## Development

```bash
pip install -e '.[dev]'
python -m pytest
```

## License

MIT, see [LICENSE](LICENSE). The benchmarks and models the harness downloads keep their own licences (see above).
