import pytest

from cayu_tau3_banking.condition_preservation import (
    CONDITION_PRESERVATION,
    LEGACY_CONDITION_PRESERVATION,
    replace_condition_preservation,
)
from cayu_tau3_banking.inventory_decision_probe import variant


def test_decision_contrast_changes_only_instructions_and_preserves_source():
    source = {
        "instructions": "Original instructions",
        "input": [{"text": "Observed records"}],
        "tools": [{"name": "example"}],
        "model": "example-model",
    }
    original = variant(source, False)
    candidate = variant(source, True)
    assert original == source
    assert {k: v for k, v in candidate.items() if k != "instructions"} == {
        k: v for k, v in source.items() if k != "instructions"
    }
    assert candidate["instructions"].startswith(source["instructions"])
    assert "unspecified-scope" in candidate["instructions"]
    assert "O1" not in candidate["instructions"]
    assert "Evergreen" not in candidate["instructions"]


def test_document_contrast_preserves_prior_context_and_instructions():
    source = {
        "instructions": "Original instructions",
        "input": [{"role": "user", "content": "State"}],
        "tools": [],
    }
    document = {"id": "scope", "title": "Coverage", "content": "All records, including revoked."}
    assert variant(source, False, document=document) == source
    result = variant(source, True, document=document)
    assert result["instructions"] == source["instructions"]
    assert result["tools"] == source["tools"]
    assert result["input"][:-1] == source["input"]
    assert document["content"] in result["input"][-1]["content"][0]["text"]


def test_condition_contrast_changes_only_one_instruction_block():
    source = {
        "instructions": "Before\n" + LEGACY_CONDITION_PRESERVATION + "\nAfter",
        "input": [{"type": "reasoning", "encrypted_content": "unchanged"}],
        "tools": [{"name": "some_tool", "parameters": {"type": "object"}}],
        "model": "example-model",
    }
    baseline = variant(source, False, condition_preservation=True)
    candidate = variant(source, True, condition_preservation=True)
    assert baseline == source
    assert candidate.pop("instructions") == "Before\n" + CONDITION_PRESERVATION + "\nAfter"
    assert candidate == {k: v for k, v in source.items() if k != "instructions"}
    candidate["input"].clear()
    assert source["input"]
    assert "14" not in CONDITION_PRESERVATION
    assert "checking" not in CONDITION_PRESERVATION.lower()


@pytest.mark.parametrize("text", ["", LEGACY_CONDITION_PRESERVATION * 2])
def test_condition_contrast_rejects_uncontrolled_baseline(text):
    with pytest.raises(ValueError, match="exactly one"):
        replace_condition_preservation(text)


def test_condition_and_document_contrasts_cannot_be_mixed():
    with pytest.raises(ValueError, match="mutually exclusive"):
        variant({}, True, document={}, condition_preservation=True)
