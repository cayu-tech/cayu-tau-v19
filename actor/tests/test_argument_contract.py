import pytest

from cayu_tau3_banking.argument_contract import (
    ARGUMENT_GROUNDING,
    LEGACY_PRODUCT_NAMING,
    replace_legacy_naming,
)
from cayu_tau3_banking.captured_argument_probe import variant


def test_only_instruction_paragraph_changes_and_source_is_not_mutated():
    source = {
        "instructions": "before\n" + LEGACY_PRODUCT_NAMING + "\nafter",
        "input": [{"x": "y"}],
        "tools": [{"name": "t"}],
    }
    baseline = variant(source, False)
    treated = variant(source, True)
    assert treated["instructions"] == "before\n" + ARGUMENT_GROUNDING + "\nafter"
    assert {k: v for k, v in treated.items() if k != "instructions"} == {
        k: v for k, v in source.items() if k != "instructions"
    }
    assert baseline == source
    treated["input"][0]["x"] = "changed"
    assert source["input"][0]["x"] == "y"


@pytest.mark.parametrize("prompt", ["missing", LEGACY_PRODUCT_NAMING * 2])
def test_replacement_rejects_wrong_prompt_version(prompt):
    with pytest.raises(ValueError):
        replace_legacy_naming(prompt)


def test_reasoning_ablation_preserves_all_visible_conversation_and_tools():
    visible = [{"role": "user", "content": "Evidence"}, {"type": "function_call", "name": "read"}]
    source = {
        "instructions": LEGACY_PRODUCT_NAMING,
        "input": [visible[0], {"type": "reasoning", "encrypted_content": "opaque"}, visible[1]],
        "tools": [],
    }
    changed = variant(source, False, drop_reasoning=True)
    assert changed["input"] == visible
    assert changed["instructions"] == source["instructions"]
    assert len(source["input"]) == 3


@pytest.mark.parametrize("mode", ["exact", "outcomes"])
def test_production_prompt_separates_names_from_argument_contracts(mode):
    from cayu_tau3_banking.agent import banking_system_prompt
    from cayu_tau3_banking.argument_contract import ACTIVE_ARGUMENT_GROUNDING

    prompt = banking_system_prompt("Synthetic policy", planning_mode=mode)
    assert prompt.count(ACTIVE_ARGUMENT_GROUNDING) == 1
    assert LEGACY_PRODUCT_NAMING not in prompt
    assert "{argument_grounding}" not in prompt
    assert "Synthetic policy" in prompt


def test_recovery_is_generic_and_keeps_the_prior_contract_rules():
    from cayu_tau3_banking.argument_contract import (
        ACTIVE_ARGUMENT_GROUNDING,
        BASE_ARGUMENT_GROUNDING,
        SOURCE_BINDING_RECOVERY,
    )

    assert ACTIVE_ARGUMENT_GROUNDING == BASE_ARGUMENT_GROUNDING + "\n\n" + SOURCE_BINDING_RECOVERY
    assert "Green" not in SOURCE_BINDING_RECOVERY
    assert "account_class" not in SOURCE_BINDING_RECOVERY
    assert "explicit API mapping wins" in SOURCE_BINDING_RECOVERY
    assert "do not guess undocumented codes" in SOURCE_BINDING_RECOVERY
