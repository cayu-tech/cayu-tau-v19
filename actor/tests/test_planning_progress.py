import copy
import json

import pytest

from cayu_tau3_banking.planning_progress import (
    PlanningProgress,
    PlanningStallError,
    plan_fingerprint,
)


def plan():
    return [
        {
            "id": "request",
            "goal": "Resolve request",
            "status": "pending",
            "evidence": "original",
            "evidence_refs": [],
            "obligations": [
                {
                    "id": "lookup",
                    "goal": "Read policy",
                    "kind": "knowledge_search",
                    "status": "ready",
                    "evidence": "original",
                    "evidence_refs": [],
                }
            ],
        }
    ]


def test_wording_is_not_progress_but_new_evidence_is():
    original = plan()
    changed = copy.deepcopy(original)
    changed[0]["evidence"] = "different words"
    changed[0]["obligations"][0]["evidence"] = "still looking"
    assert plan_fingerprint(original) == plan_fingerprint(changed)
    changed[0]["evidence_refs"] = [{"kind": "tool_result", "reference_id": "real-result"}]
    assert plan_fingerprint(original) != plan_fingerprint(changed)
    assert original == plan()  # Computing identity never mutates the controller state.


def test_rewrites_warn_then_stop_without_claiming_success():
    progress = PlanningProgress()
    for _ in range(2):
        progress.observe_plan("same", "same", failed=False)
    assert progress.warning
    progress.check()
    for _ in range(3):
        progress.observe_plan("same", "same", failed=False)
    with pytest.raises(PlanningStallError, match="planning_no_progress"):
        progress.check()


def test_new_structured_state_and_observation_allow_continuation():
    progress = PlanningProgress()
    for _ in range(4):
        progress.observe_plan("same", "same", failed=False)
    progress.observe_plan("same", "new-evidence", failed=False)
    assert progress.stalls == 0
    progress.check()
    progress.reset()
    assert (progress.updates, progress.stalls, progress.seen) == (0, 0, set())


def test_oscillation_and_unbounded_novel_plans_are_bounded():
    progress = PlanningProgress()
    for i in range(12):
        progress.observe_plan(str(i), str(i + 1), failed=False)
    with pytest.raises(PlanningStallError):
        progress.check()
    progress.reset()
    for i in range(6):
        progress.observe_plan(str(i % 2), str((i + 1) % 2), failed=False)
    with pytest.raises(PlanningStallError):
        progress.check()


def test_failed_updates_never_reset_stall_count():
    progress = PlanningProgress()
    for i in range(5):
        progress.observe_plan(str(i), str(i + 1), failed=True)
    with pytest.raises(PlanningStallError):
        progress.check()


def test_repeated_private_result_cannot_launder_plan_stalls_through_budget_changes():
    progress = PlanningProgress()
    for index in range(5):
        progress.observe_private_result(
            "check",
            json.dumps({"facts": ["same"], "remaining_calls": 6 - index}),
            failed=False,
        )
        progress.observe_plan("saved", "saved", failed=False)
    with pytest.raises(PlanningStallError):
        progress.check()
    assert "already saved" in progress.warning


def test_new_private_evidence_and_external_observations_reset_stalls():
    progress = PlanningProgress()
    progress.observe_private_result("read", '{"revision":1}', failed=False)
    for _ in range(4):
        progress.observe_plan("same", "same", failed=False)
    progress.observe_private_result("read", '{"revision":2}', failed=False)
    assert (progress.updates, progress.stalls) == (0, 0)
    progress.observe_plan("same", "same", failed=False)
    progress.observe_private_result("read", '{"revision":3}', failed=True)
    assert progress.stalls == 1
    progress.reset()
    progress.observe_private_result("read", '{"revision":2}', failed=False)
    assert progress.stalls == 0


def test_private_observation_identity_preserves_tool_name_and_nested_fields():
    progress = PlanningProgress()
    for name, content in [
        ("a", "plain text"),
        ("b", "plain text"),
        ("b", '{"record":{"remaining_calls":1}}'),
        ("b", '{"record":{"remaining_calls":2}}'),
    ]:
        progress.observe_plan("same", "same", failed=False)
        progress.observe_private_result(name, content, failed=False)
        assert progress.stalls == 0
