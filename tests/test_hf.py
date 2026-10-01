"""Tests for the Hugging Face backend on a tiny random BERT built locally; skipped without the `hf` extra."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pateval.encoders import SpannedEncoder
from pateval.encoders.hf import HFSpanEmbedder, special_token_template
from pateval.tasks import Reading

WORDS = ["rotor", "blade", "pitch", "control", "battery", "anode", "query", "document", ":"]


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {token: i for i, token in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", *WORDS])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])]
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]"
    )


def _embedder(token_pooling: str, query_prompt: str | None = None) -> HFSpanEmbedder:
    torch.manual_seed(0)
    config = transformers.BertConfig(
        vocab_size=4 + len(WORDS),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    return HFSpanEmbedder(
        name="tiny-bert",
        tokenizer=_tokenizer(),
        model=transformers.BertModel(config).eval(),
        token_pooling=token_pooling,
        max_span_length=64,
        query_prompt=query_prompt,
    )


def test_special_token_template_is_read_from_the_tokenizer_output():
    tokenizer = _tokenizer()
    cls, sep = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
    assert special_token_template(tokenizer) == ([cls], [sep])


def test_overhead_counts_special_tokens_and_the_role_prompt():
    embedder = _embedder("mean", query_prompt="query :")
    assert embedder.overhead_tokens("query") == 4
    assert embedder.overhead_tokens("document") == 2


@pytest.mark.parametrize("token_pooling", ["cls", "mean", "last_token"])
def test_padding_never_changes_a_span_vector(token_pooling):
    embedder = _embedder(token_pooling)
    short, long = embedder.tokenize(["rotor blade", "battery anode pitch control rotor blade"])
    alone = embedder.embed_spans([short], role="document")
    padded = embedder.embed_spans([short, long], role="document")
    assert np.allclose(alone[0], padded[0], atol=1e-5)


def test_spanned_hf_encoder_returns_unit_vectors_at_any_reading():
    encoder = SpannedEncoder(_embedder("mean"), batch_size=2)
    texts = ["rotor blade pitch control " * 10, "battery anode"]
    for reading in (Reading(1, 8), Reading(4, 8)):
        vectors = encoder.encode(texts, role="document", reading=reading)
        assert vectors.shape == (2, 16)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_identity_records_the_prompt_and_the_commit():
    embedder = _embedder("mean", query_prompt="query :")
    embedder.revision, embedder.prompt_variant = "1" * 40, "q"
    identity = embedder.identity()
    assert identity["query_prompt"] == "query :" and identity["corpus_prompt"] is None
    assert identity["revision"] == "1" * 40 and identity["prompt"] == "q"
    assert identity["transformers"] == transformers.__version__


def _fake_hub(monkeypatch, tmp_path, files):
    """`hf_hub_download` serving `files` (name -> JSON-able content) and raising EntryNotFoundError otherwise."""
    import huggingface_hub

    from huggingface_hub.errors import EntryNotFoundError

    def download(repo_id, filename, **kwargs):
        if filename not in files:
            raise EntryNotFoundError(f"{filename} not in {repo_id}")
        path = tmp_path / filename.replace("/", "_")
        path.write_text(json.dumps(files[filename]))
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)


def test_a_dense_head_shipped_only_as_a_pickle_is_refused(monkeypatch, tmp_path):
    from pateval.encoders.hf import load_dense_head

    modules = [{"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"}]
    modules.append({"idx": 2, "path": "2_Dense", "type": "sentence_transformers.models.Dense"})
    _fake_hub(monkeypatch, tmp_path, {"modules.json": modules, "2_Dense/config.json": {"bias": True}})
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("a pickle was loaded"))
    with pytest.raises(ValueError, match="refusing to unpickle"):
        load_dense_head("someone/model", "1" * 40)


def test_repository_code_needs_a_pinned_code_revision(monkeypatch):
    import transformers as hf_transformers

    from pateval.registry import BaselineConfig

    monkeypatch.setattr(
        hf_transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: pytest.fail("downloaded before the check")
    )
    config = BaselineConfig("someone/model", "1" * 40, "mean", 512, trust_remote_code=True)
    with pytest.raises(ValueError, match="pinned code_revision"):
        HFSpanEmbedder.from_config("model", config)


def test_an_unknown_prompt_variant_fails_before_any_download(monkeypatch):
    import transformers as hf_transformers

    from pateval.registry import get

    monkeypatch.setattr(
        hf_transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: pytest.fail("downloaded before the check")
    )
    with pytest.raises(KeyError, match="no prompt variant"):
        HFSpanEmbedder.from_config("patembed-base", get("patembed-base"), prompt="SAME")
