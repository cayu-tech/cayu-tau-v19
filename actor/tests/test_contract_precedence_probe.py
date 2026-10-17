import dataclasses

import pytest

from cayu_tau3_banking.argument_contract import (
    ARGUMENT_GROUNDING,
    CONTRACT_PRECEDENCE,
    LEGACY_PRODUCT_NAMING,
)
from cayu_tau3_banking.argument_grounding_probe import build_request
from cayu_tau3_banking.captured_argument_probe import variant
from cayu_tau3_banking.contract_precedence_probe import CASES


def test_contrast_uses_current_baseline_and_candidate_with_unchanged_input():
    source = {
        "instructions": LEGACY_PRODUCT_NAMING,
        "input": [{"type": "reasoning", "encrypted_content": "opaque"}],
        "tools": [],
    }
    baseline = variant(source, False, contract_precedence=True)
    candidate = variant(source, True, contract_precedence=True)
    assert baseline["instructions"] == ARGUMENT_GROUNDING
    assert candidate["instructions"] == CONTRACT_PRECEDENCE
    assert baseline["input"] == candidate["input"] == source["input"]
    assert source["instructions"] == LEGACY_PRODUCT_NAMING


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_safety_expectations_are_not_given_to_actor(case):
    changed = dataclasses.replace(case, expected={"hidden": "Not for actor"})
    assert build_request(case, "prompt") == build_request(changed, "prompt")


def test_rule_explicitly_preserves_business_constraints():
    for word in ("eligibility", "consent", "prerequisites", "disclosures", "timing"):
        assert word in CONTRACT_PRECEDENCE
    assert "does not authorize" in CONTRACT_PRECEDENCE
    assert "scope='" not in CONTRACT_PRECEDENCE
