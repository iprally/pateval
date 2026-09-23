"""Published configurations for public patent and general-purpose encoders.

Running a public model with the wrong readout, a dropped projection head or the wrong prompt produces a
plausible-looking number that is 25--100% too low. Every entry below was verified against the model's own
`1_Pooling/config.json`, `modules.json` and model card, and the `note` field records what went wrong when
it was got wrong. Treat a missing entry as a reason to check the model card, not as a default.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class BaselineConfig:
    """How a published model must be run for its number to mean anything."""

    hf_id: str
    token_pooling: str
    max_span_length: int
    dense_head: bool = False
    query_prompt: str | None = None
    corpus_prompt: str | None = None
    trust_remote_code: bool = False  # the repository ships its own modelling code, which then runs locally
    note: str = ""


REGISTRY: dict[str, BaselineConfig] = {
    "paecter": BaselineConfig(
        "mpi-inno-comp/paecter",
        "mean",
        512,
        note="BERT-large; max_position_embeddings=512 caps a single pass, not the total text it can read via spans.",
    ),
    "patembed-base": BaselineConfig(
        "datalyes/patembed-base",
        "mean",
        512,
        dense_head=True,
        query_prompt="encode query for mixed document retrieval: ",
        corpus_prompt="encode document for mixed retrieval: ",
        note="Ships a Dense module (modules.json). retrieval_IN/OUT/MIXED are domain labels, not task "
        "instructions; the wrong variant costs ~11%, and MIXED suits a mixed-domain corpus.",
    ),
    "patembed-large": BaselineConfig(
        "datalyes/patembed-large",
        "mean",
        512,
        note="The prompt-conditioned checkpoint scores a factor of three below its own base model under "
        "every prompt tried; the no_prompts sibling is the usable artefact.",
    ),
    "patentsberta": BaselineConfig(
        "AI-Growth-Lab/PatentSBERTa",
        "cls",
        512,
        note="pooling_mode_cls_token=true, mean explicitly false.",
    ),
    "bert-for-patents": BaselineConfig(
        "anferico/bert-for-patents",
        "mean",
        512,
        note="Raw MLM checkpoint with no retrieval training and no sentence-transformers config.",
    ),
    "scincl": BaselineConfig(
        "malteos/scincl",
        "cls",
        512,
        note="Scientific citations, not patents; a control for whether citation training transfers domains.",
    ),
    "bge-m3": BaselineConfig(
        "BAAI/bge-m3",
        "cls",
        8192,
        note="CLS-native. Running it with mean pooling understated it by 106%.",
    ),
    "gte-modernbert": BaselineConfig(
        "Alibaba-NLP/gte-modernbert-base",
        "cls",
        8192,
        note="CLS-native. Running it with mean pooling understated it by 28%.",
    ),
    "granite-r2": BaselineConfig(
        "ibm-granite/granite-embedding-english-r2",
        "cls",
        8192,
    ),
    "nomic-v1": BaselineConfig(
        "nomic-ai/nomic-embed-text-v1",
        "mean",
        2048,
        trust_remote_code=True,
    ),
    "qwen3-0.6b": BaselineConfig(
        "Qwen/Qwen3-Embedding-0.6B",
        "last_token",
        8192,
        note="Causal: only the final non-padding position has seen the whole span.",
    ),
    "qwen3-4b": BaselineConfig(
        "Qwen/Qwen3-Embedding-4B",
        "last_token",
        8192,
    ),
}

# Published reference values, for check_reproduction. Keyed by (model, benchmark, configuration).
# DAPFAM values are nDCG@100 on All, TAC->TAC at 512 tokens per side, from the QaECTER paper's Table 5
# (arXiv:2604.22897), which copies the PatenTEB paper's baselines. PatenTEB values are nDCG@10 from the
# PatenTEB paper's Table 15 (arXiv:2510.22264); the patembed rows there are in-domain.
PUBLISHED: dict[tuple[str, str, str], float] = {
    ("paecter", "DAPFAM", "TAC->TAC@512/All/ndcg@100"): 0.343,
    ("patembed-base", "DAPFAM", "best-prompt/All/ndcg@100"): 0.370,
    ("patembed-large", "DAPFAM", "best-prompt/All/ndcg@100"): 0.377,
    ("bert-for-patents", "DAPFAM", "All/ndcg@100"): 0.228,
    ("paecter", "PatenTEB", "retrieval_IN/ndcg@10"): 0.421,
    ("paecter", "PatenTEB", "retrieval_MIXED/ndcg@10"): 0.356,
    ("paecter", "PatenTEB", "retrieval_OUT/ndcg@10"): 0.120,
    ("patembed-large", "PatenTEB", "retrieval_IN/ndcg@10"): 0.512,
    ("patembed-large", "PatenTEB", "retrieval_MIXED/ndcg@10"): 0.443,
    ("patembed-large", "PatenTEB", "retrieval_OUT/ndcg@10"): 0.172,
    ("patembed-base", "PatenTEB", "retrieval_IN/ndcg@10"): 0.501,
    ("patembed-base", "PatenTEB", "retrieval_MIXED/ndcg@10"): 0.434,
    ("patembed-base", "PatenTEB", "retrieval_OUT/ndcg@10"): 0.168,
    ("qwen3-0.6b", "PatenTEB", "retrieval_IN/ndcg@10"): 0.395,
    ("qwen3-0.6b", "PatenTEB", "retrieval_MIXED/ndcg@10"): 0.324,
    ("qwen3-0.6b", "PatenTEB", "retrieval_OUT/ndcg@10"): 0.109,
    ("gte-modernbert", "PatenTEB", "retrieval_IN/ndcg@10"): 0.363,
    ("gte-modernbert", "PatenTEB", "retrieval_MIXED/ndcg@10"): 0.299,
    ("gte-modernbert", "PatenTEB", "retrieval_OUT/ndcg@10"): 0.096,
    ("bert-for-patents", "PatenTEB", "retrieval_IN/ndcg@10"): 0.286,
    ("bert-for-patents", "PatenTEB", "retrieval_MIXED/ndcg@10"): 0.239,
    ("bert-for-patents", "PatenTEB", "retrieval_OUT/ndcg@10"): 0.074,
}


def get(name: str) -> BaselineConfig:
    if name not in REGISTRY:
        raise KeyError(f"no published configuration recorded for {name!r}; check its model card before running it")
    return REGISTRY[name]


def published_for(name: str, benchmark: str) -> dict[str, float]:
    """Published reference values of one model on one benchmark, keyed by configuration."""
    return {
        config: value for (model, bench, config), value in PUBLISHED.items() if model == name and bench == benchmark
    }


def verify_against_hub(name: str) -> list[str]:
    """Compare a registry entry with what the model repository itself declares; returns discrepancies.

    Reads `1_Pooling/config.json` (readout) and `modules.json` (Dense head) through the Hugging Face hub.
    Repositories without a sentence-transformers layout yield no check rather than a discrepancy.
    """
    import json

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    config = get(name)
    problems: list[str] = []
    try:
        with open(hf_hub_download(config.hf_id, "1_Pooling/config.json")) as handle:
            pooling = json.load(handle)
    except EntryNotFoundError:
        pooling = None
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
    try:
        with open(hf_hub_download(config.hf_id, "modules.json")) as handle:
            modules = json.load(handle)
    except EntryNotFoundError:
        modules = None
    if modules is not None:
        has_dense = any(module["type"].endswith("Dense") for module in modules)
        if has_dense != config.dense_head:
            problems.append(
                f"{name}: registry dense_head={config.dense_head} but modules.json {'declares' if has_dense else 'has no'} Dense module"
            )
    return problems
