"""Tests for the published-configuration registry."""

import pytest

from pateval import registry


def test_every_entry_is_pinned_to_a_full_commit():
    for name, config in registry.REGISTRY.items():
        assert registry.FULL_SHA.fullmatch(config.revision), name


def test_entries_that_run_repository_code_pin_that_code():
    for name, config in registry.REGISTRY.items():
        if config.trust_remote_code:
            assert config.code_revision and registry.FULL_SHA.fullmatch(config.code_revision), name


def test_every_default_prompt_is_a_known_variant():
    for name, config in registry.REGISTRY.items():
        variant, _ = config.prompt()
        assert variant == config.default_prompt, name


def test_cls_native_models_are_recorded_as_cls():
    assert registry.get("bge-m3").token_pooling == "cls"
    assert registry.get("gte-modernbert").token_pooling == "cls"


def test_patembed_models_default_to_their_mixed_prompt():
    for name in ("patembed-base", "patembed-large"):
        variant, prompt = registry.get(name).prompt()
        assert variant == "MIXED"
        assert prompt.query == "encode query for mixed document retrieval: "
        assert prompt.document == "encode document for mixed retrieval: "
    assert registry.get("patembed-base").dense_head is True


def test_none_is_always_an_unprompted_variant():
    variant, prompt = registry.get("patembed-large").prompt("none")
    assert variant == "none" and prompt.query is None and prompt.document is None


def test_unknown_prompt_variant_raises():
    with pytest.raises(KeyError, match="no prompt variant"):
        registry.get("patembed-base").prompt("SAME")


def test_qwen_models_use_last_token_pooling_and_a_query_instruction():
    config = registry.get("qwen3-0.6b")
    assert config.token_pooling == "last_token"
    _, prompt = config.prompt()
    assert prompt.query.startswith("Instruct: ") and prompt.document is None


def test_unknown_model_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match="no published configuration"):
        registry.get("some-new-model")


def test_references_name_registry_entries_and_known_prompts():
    for ref in registry.REFERENCES:
        config = registry.get(ref.model)
        if ref.prompt != "as published":
            config.prompt(ref.prompt)  # raises for a variant the entry does not have


def test_calibration_references_are_recorded():
    refs = registry.references("paecter", "DAPFAM", task="TAC->TAC", reading="1x512", scope="All")
    assert [ref.value for ref in refs] == [pytest.approx(0.343)]
    assert refs[0].gated


def test_verify_flags_an_entry_without_the_prompts_its_repository_declares(monkeypatch, tmp_path):
    import json

    import huggingface_hub

    from huggingface_hub.errors import EntryNotFoundError

    files = {
        "1_Pooling/config.json": {"pooling_mode_mean_tokens": True},
        "modules.json": [{"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"}],
        "config_sentence_transformers.json": {"prompts": {"retrieval_MIXED": {"q_text": "q: ", "pos_text": "d: "}}},
    }

    def download(repo_id, filename, **kwargs):
        if filename not in files:
            raise EntryNotFoundError(filename)
        path = tmp_path / filename.replace("/", "_")
        path.write_text(json.dumps(files[filename]))
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    unprompted = registry.BaselineConfig("someone/model", "1" * 40, "mean", 512)
    monkeypatch.setitem(registry.REGISTRY, "unprompted", unprompted)
    assert registry.verify_against_hub("unprompted") == [
        "unprompted: the repository declares prompts but the registry entry has none"
    ]
    prompted = registry.BaselineConfig(
        "someone/model", "1" * 40, "mean", 512, prompts={"MIXED": registry.Prompt("q: ", "d: ")}, default_prompt="MIXED"
    )
    monkeypatch.setitem(registry.REGISTRY, "prompted", prompted)
    assert registry.verify_against_hub("prompted") == []
