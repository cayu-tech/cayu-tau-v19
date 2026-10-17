import importlib.util
from pathlib import Path

import pytest
from cayu import ScriptedModelProvider
from test_runtime_banking import text


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/qualify_comparison_workspace.py"
    spec = importlib.util.spec_from_file_location("comparison_workspace_qualification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("case", ["helpful", "costly", "ineligible", "declined", "unknown_fee"])
def test_terms_are_in_public_documents_not_the_prompt_or_customer_message(case):
    module = runner()
    cfg = module.configuration(case)
    prompts = cfg["policy"] + " ".join(cfg["customers"])
    assert "60 GB" not in prompts and "130 GB" not in prompts
    assert module.expectation(case) not in repr(cfg)
    assert "Ember companion terms" in repr(cfg["documents"])
    assert len(cfg["customers"]) == 2
    assert ("will not subscribe" in cfg["customers"][-1]) == (case == "declined")


def test_qualification_really_compacts_between_customer_turns(tmp_path):
    module = runner()
    provider = ScriptedModelProvider([text("Provisional advice."), text("Current advice.")])
    compactor = module.LossyCompactor()
    module.run_case(
        tmp_path,
        mismatch=False,
        provider_factory=lambda: provider,
        tools=[],
        advice_mode="lean",
        compaction=module.CheckpointCompactionContextPolicy(
            compactor=compactor,
            compact_after_messages=2,
            max_user_turns=1,
        ),
        assessment=lambda host, plans: {"no_effects": not host.calls},
        **module.configuration("helpful"),
    )
    assert len(provider.requests) == 2
    assert compactor.calls > 0
    assert "Earlier advice was provisional" in str(provider.requests[-1].messages)
