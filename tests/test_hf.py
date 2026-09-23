"""Tests for the Hugging Face backend on a tiny random BERT built locally; skipped without the `hf` extra."""

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
