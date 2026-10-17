import importlib.util
from pathlib import Path

import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/qualify_recall_recovery.py"
    spec = importlib.util.spec_from_file_location("recall_recovery_qualification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("case", ["dependent", "immediate", "blocked"])
def test_procedure_is_retrievable_but_not_in_persistent_system_instructions(case):
    module = runner()
    cfg = module.configuration(case)
    assert module.POLICY not in cfg["policy"]
    assert cfg["documents"][0][2] == module.POLICY
    assert len(cfg["customers"]) == 2
    assert "expected" not in repr(cfg)
    assert "Do not make changes yet" in cfg["customers"][0]


def test_retirement_only_control_does_not_request_an_export():
    request = runner().configuration("immediate")["customers"][-1]
    assert "I do not want an export" in request


def test_stress_inspection_padding_is_one_time_and_does_not_change_state():
    module = runner()
    host = module.QualificationHost("blocked")
    first = host.execute("inspect_archive", {"archive_id": "A7"})
    second = host.execute("inspect_archive", {"archive_id": "A7"})
    assert len(first.pop("inspection_history")) > 80_000
    assert first == second
    assert first["active"] and not first["owner_approval"] and not first["exported"]


def test_initial_native_prompt_and_schemas_fit_stress_compaction_limit(tmp_path):
    from cayu import ModelStreamEvent, ScriptedModelProvider
    from tau2.environment.tool import Tool

    module = runner()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("No change made."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
            for _ in range(2)
        ]
    )
    module.run_case(
        tmp_path,
        mismatch=False,
        provider_factory=lambda: provider,
        host=module.QualificationHost("dependent"),
        assessment=module.assess,
        tools=[
            Tool(module.inspect_archive),
            Tool(module.export_archive),
            Tool(module.retire_credential),
        ],
        advice_mode="lean",
        compaction=module.compaction_policy(),
        **module.configuration("dependent"),
    )
    # Setup qualification, not an operation pass: the fake actor does no work.
    assert len(provider.requests) == 2
