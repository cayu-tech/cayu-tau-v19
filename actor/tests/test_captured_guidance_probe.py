from copy import deepcopy

import pytest
from cayu import Message
from cayu.providers.base import ModelRequest

from cayu_tau3_banking.captured_guidance_probe import (
    checked_guidance_payload,
    replace_captured_instructions,
)


def request(messages):
    return ModelRequest(model="gpt-5.6-sol", messages=messages, tools=[])


def test_replaces_in_place_preserves_context_and_compiles_exactly_one_prompt():
    source = [
        Message.text("system", "Original instructions"),
        Message.text("user", "Quoted text: Original instructions; public evidence is unchanged."),
        Message.text("assistant", "Prior recommendation."),
        Message.text("user", "Continue with the same facts."),
    ]
    before = deepcopy(source)
    baseline = replace_captured_instructions(
        source, original="Original instructions", replacement="Original instructions"
    )
    candidate = replace_captured_instructions(
        source, original="Original instructions", replacement="New instructions"
    )
    assert source == before == baseline
    assert len(candidate) == len(source) and candidate[1:] == source[1:]
    a = checked_guidance_payload(request(baseline), expected="Original instructions")
    b = checked_guidance_payload(request(candidate), expected="New instructions")
    assert a["input"] == b["input"]
    assert {k: v for k, v in a.items() if k != "instructions"} == {
        k: v for k, v in b.items() if k != "instructions"
    }


@pytest.mark.parametrize("variant", ["missing", "duplicate", "late", "mismatch", "empty"])
def test_rejects_ambiguous_or_mismatched_captures(variant):
    source = [Message.text("system", "Original"), Message.text("user", "Facts")]
    replacement = "New"
    if variant == "missing":
        source.pop(0)
    elif variant == "duplicate":
        source.append(Message.text("system", "Original"))
    elif variant == "late":
        source.reverse()
    elif variant == "mismatch":
        source[0] = Message.text("system", "Different")
    elif variant == "empty":
        replacement = " "
    with pytest.raises(ValueError):
        replace_captured_instructions(source, original="Original", replacement=replacement)


@pytest.mark.parametrize("second", ["Original", "New"])
def test_payload_validation_catches_the_observed_duplicate_instruction_bug(second):
    broken = request(
        [
            Message.text("system", "Original"),
            Message.text("system", second),
            Message.text("user", "Facts"),
        ]
    )
    with pytest.raises(ValueError, match="exactly once"):
        checked_guidance_payload(broken, expected="Original")
