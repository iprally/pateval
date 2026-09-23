"""Hugging Face backend: a `SpanEmbedder` over `transformers` models in their published configuration.

Requires the `hf` extra (`pip install pateval[hf]`). Nothing else in the package imports torch.

The three things that silently produce a wrong number for a public model are handled here from the
registry entry: the within-span readout (CLS, mean or last token), a projection head shipped as a
sentence-transformers `Dense` module, and the prompt prefix. The prompt is prepended to every span, since
each span is an independent encoder input and the prompt conditions the embedding.
"""

from __future__ import annotations

import dataclasses
import json

from typing import Any, Sequence

import numpy as np

from pateval.encoders.base import Role
from pateval.registry import BaselineConfig

POOLINGS = ("cls", "mean", "last_token")


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only without the extra installed
        raise ImportError("the Hugging Face backend needs the `hf` extra: pip install 'pateval[hf]'") from error
    return torch


@dataclasses.dataclass
class DenseHead:
    """A sentence-transformers `Dense` module: linear projection plus activation, applied after pooling."""

    weight: Any  # torch.Tensor (out_features, in_features)
    bias: Any | None
    activation: str

    def __call__(self, pooled: Any) -> Any:
        torch = _require_torch()
        projected = torch.nn.functional.linear(pooled, self.weight, self.bias)
        if self.activation.endswith("Tanh"):
            return torch.tanh(projected)
        if self.activation.endswith("Identity"):
            return projected
        raise ValueError(f"unsupported Dense activation {self.activation!r}")


def load_dense_head(hf_id: str, revision: str | None = None) -> DenseHead | None:
    """Load the `Dense` module a sentence-transformers repository declares in `modules.json`, if any."""
    from huggingface_hub import hf_hub_download

    torch = _require_torch()
    with open(hf_hub_download(hf_id, "modules.json", revision=revision)) as handle:
        modules = json.load(handle)
    dense_modules = [m for m in modules if m["type"].endswith("Dense")]
    if not dense_modules:
        return None
    if len(dense_modules) > 1:
        raise ValueError(f"{hf_id} declares {len(dense_modules)} Dense modules; only one is supported")
    path = dense_modules[0]["path"]
    with open(hf_hub_download(hf_id, f"{path}/config.json", revision=revision)) as handle:
        config = json.load(handle)
    try:
        from safetensors.torch import load_file

        state = load_file(hf_hub_download(hf_id, f"{path}/model.safetensors", revision=revision))
    except Exception:  # older repositories ship a pickle instead
        state = torch.load(hf_hub_download(hf_id, f"{path}/pytorch_model.bin", revision=revision), map_location="cpu")
    return DenseHead(
        weight=state["linear.weight"],
        bias=state.get("linear.bias") if config.get("bias", True) else None,
        activation=config.get("activation_function", "torch.nn.modules.linear.Identity"),
    )


def special_token_template(tokenizer: Any) -> tuple[list[int], list[int]]:
    """The special tokens `tokenizer` puts around a single sequence, as (prefix, suffix).

    Probed from the tokenizer's own output instead of asked of an API: `build_inputs_with_special_tokens` and
    `num_special_tokens_to_add` behave differently across transformers versions and tokenizer backends (and
    are gone from transformers 5), while the post-processed output is what the model was trained on.
    """
    probe = "patent"
    bare = list(tokenizer(probe, add_special_tokens=False)["input_ids"])
    full = list(tokenizer(probe, add_special_tokens=True)["input_ids"])
    for start in range(len(full) - len(bare) + 1):
        if full[start : start + len(bare)] == bare:
            return full[:start], full[start + len(bare) :]
    raise ValueError(f"cannot locate the text inside the special-token template of {type(tokenizer).__name__}")


def _dtype_argument() -> str:
    """`from_pretrained` took `torch_dtype` until transformers 4.56 renamed it to `dtype`."""
    import transformers

    from packaging.version import Version

    return "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"


@dataclasses.dataclass
class HFSpanEmbedder:
    """Embeds one span at a time with a `transformers` model, in the model's own readout."""

    name: str
    tokenizer: Any
    model: Any
    token_pooling: str
    max_span_length: int
    query_prompt: str | None = None
    corpus_prompt: str | None = None
    dense: DenseHead | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.token_pooling not in POOLINGS:
            raise ValueError(f"token_pooling must be one of {POOLINGS}, got {self.token_pooling!r}")
        self._prefix, self._suffix = special_token_template(self.tokenizer)
        pad_id = self.tokenizer.pad_token_id
        self._pad_id = 0 if pad_id is None else pad_id  # always masked, so its value never reaches the readout
        self._prompt_ids: dict[str, list[int]] = {
            "query": self._encode_prompt(self.query_prompt),
            "document": self._encode_prompt(self.corpus_prompt),
        }

    @classmethod
    def from_config(
        cls,
        name: str,
        config: BaselineConfig,
        *,
        device: str | None = None,
        dtype: str = "float32",
        revision: str | None = None,
        trust_remote_code: bool | None = None,
    ) -> "HFSpanEmbedder":
        """Load a registry entry: weights, tokenizer, readout, optional Dense head and prompts.

        Repository code is executed only for entries that need it (`config.trust_remote_code`), unless
        `trust_remote_code` overrides that.
        """
        torch = _require_torch()
        from transformers import AutoModel, AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        trust = config.trust_remote_code if trust_remote_code is None else trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(config.hf_id, revision=revision, trust_remote_code=trust)
        model = AutoModel.from_pretrained(
            config.hf_id, revision=revision, trust_remote_code=trust, **{_dtype_argument(): getattr(torch, dtype)}
        )
        model = model.to(device).eval()
        dense = load_dense_head(config.hf_id, revision) if config.dense_head else None
        if dense is not None:
            dense = DenseHead(
                dense.weight.to(device, model.dtype),
                None if dense.bias is None else dense.bias.to(device, model.dtype),
                dense.activation,
            )
        return cls(
            name=name,
            tokenizer=tokenizer,
            model=model,
            token_pooling=config.token_pooling,
            max_span_length=config.max_span_length,
            query_prompt=config.query_prompt,
            corpus_prompt=config.corpus_prompt,
            dense=dense,
            device=device,
        )

    def _encode_prompt(self, prompt: str | None) -> list[int]:
        if not prompt:
            return []
        return list(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def tokenize(self, texts: Sequence[str]) -> list[list[int]]:
        encoded = self.tokenizer(list(texts), add_special_tokens=False, truncation=False, return_attention_mask=False)
        return [list(ids) for ids in encoded["input_ids"]]

    def overhead_tokens(self, role: Role) -> int:
        return len(self._prefix) + len(self._suffix) + len(self._prompt_ids[role])

    def embed_spans(self, spans: Sequence[list[int]], *, role: Role) -> np.ndarray:
        torch = _require_torch()
        prompt = self._prompt_ids[role]
        sequences = [self._prefix + prompt + list(span) + self._suffix for span in spans]
        # Right padding, whatever the tokenizer's default, so that the last real token is a mask sum.
        width = max(len(sequence) for sequence in sequences)
        input_ids = torch.full((len(sequences), width), self._pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, : len(sequence)] = 1
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        with torch.inference_mode():
            hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            pooled = _readout(hidden, attention_mask, self.token_pooling)
            if self.dense is not None:
                pooled = self.dense(pooled)
        return pooled.float().cpu().numpy()


def _readout(hidden: Any, attention_mask: Any, token_pooling: str) -> Any:
    """Within-span readout over the final hidden states; padding is excluded from the mean."""
    if token_pooling == "cls":
        return hidden[:, 0]
    if token_pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    last = attention_mask.sum(dim=1) - 1  # right padding: the last real token of each row
    return hidden[torch_arange(hidden), last]


def torch_arange(hidden: Any) -> Any:
    torch = _require_torch()
    return torch.arange(hidden.shape[0], device=hidden.device)
