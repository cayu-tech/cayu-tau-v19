import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "probe_selection_context", Path(__file__).parents[1] / "scripts/probe_selection_context.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_contrast_changes_instructions_only_and_preserves_original():
    original = {
        "instructions": "<instructions>prior</instructions><policy>public</policy>",
        "input": [{"type": "reasoning", "encrypted_content": "fixture"}],
        "tools": [{"name": "fixture", "parameters": {}}],
        "model": "fixture",
    }
    before = copy.deepcopy(original)
    result = probe.add_guidance(original)
    assert original == before
    assert result["instructions"].count(probe.GUIDANCE) == 1
    assert result["instructions"].endswith("<policy>public</policy>")
    assert {k: v for k, v in result.items() if k != "instructions"} == {
        k: v for k, v in before.items() if k != "instructions"
    }
    with pytest.raises(ValueError):
        probe.add_guidance(result)


@pytest.mark.parametrize("instructions", ["none", "</instructions></instructions>"])
def test_ambiguous_instruction_target_is_rejected(instructions):
    with pytest.raises(ValueError):
        probe.add_guidance({"instructions": instructions})


def test_document_contrast_does_not_add_guidance_or_alter_conversation():
    original = {
        "instructions": "Original",
        "input": [{"content": "User message"}],
        "tools": ["fixture"],
    }
    document = {"id": "fixture", "title": "Public fixture", "content": "Public conditions"}
    result = probe.add_document(original, document)
    assert original["instructions"] == "Original"
    assert result["instructions"].startswith("Original\n")
    assert probe.GUIDANCE not in result["instructions"]
    assert result["input"] == original["input"] and result["tools"] == original["tools"]
    assert document["content"] in result["instructions"]


@pytest.mark.parametrize("document", [None, {}, {"id": "x", "title": "t", "content": ""}])
def test_reject_incomplete_document_treatment(document):
    with pytest.raises(ValueError):
        probe.add_document({"instructions": "Original"}, document)
