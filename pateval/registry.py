"""Published configurations for public patent and general-purpose encoders.

A public model run with the wrong readout, without its projection head or with the wrong prompt still produces a
plausible-looking number, and nothing downstream flags it. Every entry below was checked against the model's own
`1_Pooling/config.json`, `modules.json`, `config_sentence_transformers.json` and model card, and is pinned to the
repository commit it was checked at. Treat a missing entry as a reason to read the model card, not as a default.
"""

from __future__ import annotations

import dataclasses
import json
import re

from typing import Any

FULL_SHA = re.compile(r"[0-9a-f]{40}")
NO_PROMPT = "none"
DEFAULT_PROMPT = "default"

# Where a prompt's wording comes from; `registry verify` checks only the first against the repository.
FROM_REPOSITORY = "config_sentence_transformers.json"
FROM_MODEL_CARD = "model card"
OURS = "ours"


@dataclasses.dataclass(frozen=True)
class Prompt:
    """Prefixes prepended to every span of a query and of a document; `None` leaves that side unprompted."""

    query: str | None = None
    document: str | None = None
    origin: str = FROM_REPOSITORY


@dataclasses.dataclass(frozen=True)
class BaselineConfig:
    """How a published model must be run for its number to mean anything."""

    hf_id: str
    revision: str  # full commit SHA of the model repository: weights, tokenizer and configuration
    token_pooling: str
    max_span_length: int
    dense_head: bool = False
    prompts: dict[str, Prompt] = dataclasses.field(default_factory=dict)
    default_prompt: str = NO_PROMPT
    trust_remote_code: bool = False  # the repository ships its own modelling code, which then runs locally
    code_revision: str | None = None  # full commit SHA of the repository that holds that code
    note: str = ""

    def prompt(self, variant: str | None = None) -> tuple[str, Prompt]:
        """The named prompt variant, the default one when `variant` is None; `none` always means no prompt."""
        name = self.default_prompt if variant in (None, DEFAULT_PROMPT) else variant
        if name == NO_PROMPT:
            return name, Prompt()
        if name not in self.prompts:
            known = sorted([NO_PROMPT, *self.prompts])
            raise KeyError(f"{self.hf_id} has no prompt variant {name!r}; known variants: {known}")
        return name, self.prompts[name]


# The authors' own `retrieval_*` prompts (q_text for queries, pos_text for documents). IN, OUT and MIXED name the
# IPC-domain relation between query and document; they are domain labels, not task instructions.
_PATEMBED_PROMPTS = {
    "IN": Prompt("encode query for same document retrieval: ", "encode document for same retrieval: "),
    "MIXED": Prompt("encode query for mixed document retrieval: ", "encode document for mixed retrieval: "),
    "OUT": Prompt("encode query for different document retrieval: ", "encode document for different retrieval: "),
}

REGISTRY: dict[str, BaselineConfig] = {
    "paecter": BaselineConfig(
        "mpi-inno-comp/paecter",
        "1f355d9e9a34dc4859e22d05ce27e970eeacfbd2",
        "mean",
        512,
        note="BERT-large; max_position_embeddings=512 caps a single pass, not the total text it can read via spans.",
    ),
    "patembed-base": BaselineConfig(
        "datalyes/patembed-base",
        "244ba452908a3093be11efb8a44396652317fb14",
        "mean",
        512,
        dense_head=True,
        prompts=_PATEMBED_PROMPTS,
        default_prompt="MIXED",
        note="Ships a Dense module (modules.json). MIXED scores best on DAPFAM TAC->TAC.",
    ),
    "patembed-large": BaselineConfig(
        "datalyes/patembed-large",
        "2d5c0f92a3e5dc3d5415c08e612c57543c0e03ad",
        "mean",
        512,
        prompts=_PATEMBED_PROMPTS,
        default_prompt="MIXED",
        note="Prompt-conditioned: its authors report 0.044 on DAPFAM TAC->TAC without a prompt (PatenTEB paper, "
        "Table 16). MIXED scores best there with one.",
    ),
    "patentsberta": BaselineConfig(
        "AI-Growth-Lab/PatentSBERTa",
        "3ff1d553c861d8f5bfd902333d97fc95eb6b4c8f",
        "cls",
        512,
        note="pooling_mode_cls_token=true, mean explicitly false.",
    ),
    "bert-for-patents": BaselineConfig(
        "anferico/bert-for-patents",
        "148e559789f59c2660cb7e7aaa0812041660c607",
        "mean",
        512,
        note="Raw MLM checkpoint with no retrieval training and no sentence-transformers config.",
    ),
    "scincl": BaselineConfig(
        "malteos/scincl",
        "ebc5348d184ba2fc9beee69b4e394263fce57b2e",
        "cls",
        512,
        note="Scientific citations, not patents; a control for whether citation training transfers domains.",
    ),
    "bge-m3": BaselineConfig(
        "BAAI/bge-m3",
        "5617a9f61b028005a4858fdac845db406aefb181",
        "cls",
        8192,
        note="CLS-native (1_Pooling/config.json).",
    ),
    "gte-modernbert": BaselineConfig(
        "Alibaba-NLP/gte-modernbert-base",
        "e7f32e3c00f91d699e8c43b53106206bcc72bb22",
        "cls",
        8192,
        note="CLS-native (1_Pooling/config.json).",
    ),
    "granite-r2": BaselineConfig(
        "ibm-granite/granite-embedding-english-r2",
        "47ea694b257b703fee9253d75c2b1f2985180498",
        "cls",
        8192,
    ),
    "nomic-v1": BaselineConfig(
        "nomic-ai/nomic-embed-text-v1",
        "3ac47f125a41961d13b397d0332866be2f9152e1",
        "mean",
        2048,
        prompts={"card": Prompt("search_query: ", "search_document: ", origin=FROM_MODEL_CARD)},
        default_prompt="card",
        trust_remote_code=True,
        # config.json's auto_map loads the modelling code from nomic-ai/nomic-bert-2048, a second repository.
        code_revision="7710840340a098cfb869c4f65e87cf2b1b70caca",
        note="The model card requires the search_query/search_document task prefixes.",
    ),
    "qwen3-0.6b": BaselineConfig(
        "Qwen/Qwen3-Embedding-0.6B",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "last_token",
        8192,
        prompts={
            "patent": Prompt(
                "Instruct: Given a patent claim, retrieve prior art that discloses the same invention.\nQuery: ",
                None,
                origin=OURS,
            )
        },
        default_prompt="patent",
        note="Causal: only the final non-padding position has seen the whole span. The model card asks for a task "
        "instruction on queries and leaves its wording to the user; the `patent` wording is ours.",
    ),
    "qwen3-4b": BaselineConfig(
        "Qwen/Qwen3-Embedding-4B",
        "5cf2132abc99cad020ac570b19d031efec650f2b",
        "last_token",
        8192,
        prompts={
            "patent": Prompt(
                "Instruct: Given a patent claim, retrieve prior art that discloses the same invention.\nQuery: ",
                None,
                origin=OURS,
            )
        },
        default_prompt="patent",
        note="As qwen3-0.6b.",
    ),
}


@dataclasses.dataclass(frozen=True)
class Reference:
    """A published value and the exact cell it was measured on."""

    model: str
    benchmark: str  # "DAPFAM" or "PatenTEB"
    task: str  # DAPFAM: "<query view>-><corpus view>"; PatenTEB: the regime, "IN", "MIXED" or "OUT"
    reading: str  # per side, as the CLI writes it
    scope: str  # the qrel set: DAPFAM's All / In / Out, PatenTEB's regime
    metric: str  # pytrec_eval key
    # "none"; a variant name; "default" when the source used a prompt without naming it (the entry's default is
    # run); or "as published" when the source's prompt is not one the registry holds
    prompt: str
    value: float
    source: str
    gated: bool = False  # `pateval reproduce` fails when a gated value is missed; others are printed only


_TABLE16 = "PatenTEB paper (arXiv:2510.22264), Table 16"
_TABLE15 = "PatenTEB paper (arXiv:2510.22264), Table 15"

# DAPFAM values are single TAC->TAC cells at 512 tokens per side, the configuration the PatenTEB paper evaluated.
# The In/Out scopes and values below 0.1 are printed next to the measurement but not gated: 2% of a small value is
# within the noise of the benchmark.
REFERENCES: tuple[Reference, ...] = (
    Reference("paecter", "DAPFAM", "TAC->TAC", "1x512", "All", "ndcg_cut_100", "none", 0.343, _TABLE16, gated=True),
    Reference("paecter", "DAPFAM", "TAC->TAC", "1x512", "In", "ndcg_cut_100", "none", 0.387, _TABLE16),
    Reference("paecter", "DAPFAM", "TAC->TAC", "1x512", "Out", "ndcg_cut_100", "none", 0.060, _TABLE16),
    Reference(
        "patembed-base", "DAPFAM", "TAC->TAC", "1x512", "All", "ndcg_cut_100", "default", 0.370, _TABLE16, gated=True
    ),
    Reference(
        "patembed-base", "DAPFAM", "TAC->TAC", "1x512", "All", "ndcg_cut_100", "none", 0.352, _TABLE16, gated=True
    ),
    Reference(
        "patembed-large", "DAPFAM", "TAC->TAC", "1x512", "All", "ndcg_cut_100", "default", 0.377, _TABLE16, gated=True
    ),
    Reference("patembed-large", "DAPFAM", "TAC->TAC", "1x512", "All", "ndcg_cut_100", "none", 0.044, _TABLE16),
    Reference(
        "bert-for-patents", "DAPFAM", "TAC->TAC", "1x512", "All", "ndcg_cut_100", "none", 0.228, _TABLE16, gated=True
    ),
    # PatenTEB nDCG@10; the patembed rows there are in-domain, and the regime prompts are the models' own.
    Reference("paecter", "PatenTEB", "IN", "1x512", "IN", "ndcg_cut_10", "none", 0.421, _TABLE15),
    Reference("paecter", "PatenTEB", "MIXED", "1x512", "MIXED", "ndcg_cut_10", "none", 0.356, _TABLE15),
    Reference("paecter", "PatenTEB", "OUT", "1x512", "OUT", "ndcg_cut_10", "none", 0.120, _TABLE15),
    Reference("patembed-large", "PatenTEB", "IN", "1x512", "IN", "ndcg_cut_10", "IN", 0.512, _TABLE15),
    Reference("patembed-large", "PatenTEB", "MIXED", "1x512", "MIXED", "ndcg_cut_10", "MIXED", 0.443, _TABLE15),
    Reference("patembed-large", "PatenTEB", "OUT", "1x512", "OUT", "ndcg_cut_10", "OUT", 0.172, _TABLE15),
    Reference("patembed-base", "PatenTEB", "IN", "1x512", "IN", "ndcg_cut_10", "IN", 0.501, _TABLE15),
    Reference("patembed-base", "PatenTEB", "MIXED", "1x512", "MIXED", "ndcg_cut_10", "MIXED", 0.434, _TABLE15),
    Reference("patembed-base", "PatenTEB", "OUT", "1x512", "OUT", "ndcg_cut_10", "OUT", 0.168, _TABLE15),
    Reference("qwen3-0.6b", "PatenTEB", "IN", "1x512", "IN", "ndcg_cut_10", "as published", 0.395, _TABLE15),
    Reference("qwen3-0.6b", "PatenTEB", "MIXED", "1x512", "MIXED", "ndcg_cut_10", "as published", 0.324, _TABLE15),
    Reference("qwen3-0.6b", "PatenTEB", "OUT", "1x512", "OUT", "ndcg_cut_10", "as published", 0.109, _TABLE15),
    Reference("gte-modernbert", "PatenTEB", "IN", "1x512", "IN", "ndcg_cut_10", "none", 0.363, _TABLE15),
    Reference("gte-modernbert", "PatenTEB", "MIXED", "1x512", "MIXED", "ndcg_cut_10", "none", 0.299, _TABLE15),
    Reference("gte-modernbert", "PatenTEB", "OUT", "1x512", "OUT", "ndcg_cut_10", "none", 0.096, _TABLE15),
    Reference("bert-for-patents", "PatenTEB", "IN", "1x512", "IN", "ndcg_cut_10", "none", 0.286, _TABLE15),
    Reference("bert-for-patents", "PatenTEB", "MIXED", "1x512", "MIXED", "ndcg_cut_10", "none", 0.239, _TABLE15),
    Reference("bert-for-patents", "PatenTEB", "OUT", "1x512", "OUT", "ndcg_cut_10", "none", 0.074, _TABLE15),
)


def get(name: str) -> BaselineConfig:
    if name not in REGISTRY:
        raise KeyError(f"no published configuration recorded for {name!r}; check its model card before running it")
    return REGISTRY[name]


def references(model: str | None = None, benchmark: str | None = None, **fields: str) -> list[Reference]:
    """Published references, filtered by model, benchmark and any other `Reference` field given by keyword."""
    return [
        ref
        for ref in REFERENCES
        if (model is None or ref.model == model)
        and (benchmark is None or ref.benchmark == benchmark)
        and all(getattr(ref, key) == value for key, value in fields.items())
    ]


def verify_against_hub(name: str) -> list[str]:
    """Compare a registry entry with what the model repository declares at the pinned commit; returns discrepancies.

    Reads `1_Pooling/config.json` (readout), `modules.json` (Dense head) and `config_sentence_transformers.json`
    (prompts) through the Hugging Face hub. Files a repository does not have yield no check rather than a
    discrepancy.
    """
    config = get(name)
    problems: list[str] = []

    pooling = _hub_json(config, "1_Pooling/config.json")
    if pooling is not None:
        declared = {
            "cls": pooling.get("pooling_mode_cls_token", False),
            "mean": pooling.get("pooling_mode_mean_tokens", False),
            "last_token": pooling.get("pooling_mode_lasttoken", False),
        }
        if not declared.get(config.token_pooling, False):
            active = [mode for mode, on in declared.items() if on]
            problems.append(
                f"{name}: registry says {config.token_pooling!r} but 1_Pooling/config.json declares {active}"
            )

    modules = _hub_json(config, "modules.json")
    if modules is not None:
        has_dense = any(module["type"].endswith("Dense") for module in modules)
        if has_dense != config.dense_head:
            problems.append(
                f"{name}: registry dense_head={config.dense_head} but modules.json "
                f"{'declares' if has_dense else 'has no'} Dense module"
            )

    declared_prompts = _strings(((_hub_json(config, "config_sentence_transformers.json") or {}).get("prompts")) or {})
    declared_prompts.discard("")
    if declared_prompts and not config.prompts:
        problems.append(f"{name}: the repository declares prompts but the registry entry has none")
    for variant, prompt in config.prompts.items():
        if prompt.origin != FROM_REPOSITORY:
            continue
        for side, text in (("query", prompt.query), ("document", prompt.document)):
            if text and text not in declared_prompts:
                problems.append(f"{name}: the {side} prompt of variant {variant!r} is not one the repository declares")
    return problems


def _hub_json(config: BaselineConfig, filename: str) -> Any:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        path = hf_hub_download(config.hf_id, filename, revision=config.revision)
    except EntryNotFoundError:
        return None
    with open(path) as handle:
        return json.load(handle)


def _strings(value: Any) -> set[str]:
    """Every string inside a JSON value; prompts are plain strings or, for patembed, {q_text, pos_text} objects."""
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(*(_strings(item) for item in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_strings(item) for item in value)) if value else set()
    return set()
