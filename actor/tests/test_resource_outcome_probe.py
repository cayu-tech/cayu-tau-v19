import pytest
from cayu import ScriptedModelProvider
from test_outcome_controller import outcome
from test_runtime_banking import call, text

from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
from cayu_tau3_banking.resource_outcome_probe import SyntheticWorkspace, assess, run_case


@pytest.mark.parametrize("mismatch", [False, True])
def test_host_preserves_authoritative_result_and_blocks_duplicate(mismatch):
    host = SyntheticWorkspace(mismatch=mismatch)
    result = host.execute("create_workspace", {"sku": "arc_team_7", "scope": "organization"})
    assert result["success"]
    assert result["scope"] == ("personal" if mismatch else "organization")
    assert host.execute("read_workspace", {"workspace_id": "ws_81"})["scope"] == result["scope"]
    with pytest.raises(ValueError, match="Duplicate"):
        host.execute("create_workspace", {"sku": "arc_team_7", "scope": "organization"})


def test_assessment_rejects_false_success_empty_plan_and_duplicate():
    host = SyntheticWorkspace(mismatch=True)
    host.execute("create_workspace", {"sku": "arc_team_7", "scope": "organization"})
    plan = {"items": [outcome(status="completed")]}
    assert not assess(host, [plan, plan])["structural_pass"]
    assert not assess(host, [{"items": []}, {"items": []}])["structural_pass"]
    pending = {"items": [outcome()]}
    assert assess(host, [pending, pending])["structural_pass"]
    host.calls.append(host.calls[0])
    assert not assess(host, [pending, pending])["structural_pass"]


@pytest.mark.parametrize("mismatch", [False, True])
def test_native_runtime_records_but_does_not_semantically_validate_resource_scope(
    tmp_path, mismatch
):
    completed = outcome(
        status="completed",
        evidence="Created organization workspace",
        evidence_refs=[{"kind": "tool_result", "reference_id": "created"}],
    )
    provider = ScriptedModelProvider(
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}),
            call("create_workspace", {"sku": "arc_team_7", "scope": "organization"}, "created"),
            call(UPDATE_WORK_PLAN_TOOL, {"items": [completed]}),
            text("Created your organization workspace."),
            call("read_workspace", {"workspace_id": "ws_81"}, "read"),
            text("Confirmed organization scope on the existing workspace."),
        ]
    )
    result = run_case(
        tmp_path, mismatch=mismatch, provider_factory=lambda: provider, model="scripted"
    )
    # Successful tool evidence alone does not prove semantic intent. The
    # unchanged controller accepts the scripted claim in either condition.
    assert result["structural_pass"] is (not mismatch)
    assert result["read_back_observed"]
    assert len(result["plans"]) == 2
    assert (tmp_path / "transcript.json").exists()
