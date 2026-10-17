import pytest
from cayu import ScriptedModelProvider
from test_runtime_banking import call, text

from cayu_tau3_banking import advice
from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL
from cayu_tau3_banking.shared_quota_probe import CASES, Service, assess, run_probe


def completed_plan():
    return [{"items": [{"completion_mode": "operation", "outcome": "succeeded"}]}]


@pytest.mark.parametrize("case", CASES)
def test_service_scope_and_expected_allocation_are_order_independent(case):
    host = Service(case)
    list(host.customers())
    for device in ("Birch", "Aster"):
        mode = "priority" if host.case["priority"] in (device, None) else "standard"
        assert host.execute("book_service", {"device_id": device, "processing": mode})["success"]
    assert assess(host, completed_plan())["structural_pass"]
    assert not host.execute("book_service", {"device_id": "Aster", "processing": "standard"})[
        "success"
    ]


def test_premature_effect_really_consumes_shared_capacity_not_a_guard_veto():
    host = Service("missing_priority")
    host.turn = 1
    assert host.execute("book_service", {"device_id": "Aster", "processing": "priority"})["success"]
    host.turn = 2
    assert not host.execute("book_service", {"device_id": "Birch", "processing": "priority"})[
        "success"
    ]
    assert not assess(host, completed_plan())["structural_pass"]


def test_native_checklist_can_ask_for_shared_priority_without_new_primitive(tmp_path):
    plan = {
        "id": "repairs",
        "goal": "Complete both requested repairs",
        "completion_mode": "operation",
        "status": "in_progress",
        "outcome": "unresolved",
        "depends_on": [],
        "obligations": [
            {
                "id": "priority",
                "goal": "Obtain priority before allocating the shared slot",
                "kind": "communication",
                "status": "ready",
            }
        ],
    }
    provider = ScriptedModelProvider(
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [plan]}, "plan"),
            text("Which device should receive priority processing?"),
            text("I have your preference, but the bookings are still pending."),
        ]
    )
    original = advice.LEAN_INSTRUCTIONS
    # One input only: test the real customer-handoff path without inventing results.
    host = Service("missing_priority")
    initial = next(host.customers())
    import unittest.mock

    with unittest.mock.patch.object(Service, "customers", lambda self: iter([initial])):
        result = run_probe(
            tmp_path, "missing_priority", "shared_scope", lambda: provider, model="scripted"
        )
    assert result["calls"] == []
    assert not result["structural_pass"]
    assert original == advice.LEAN_INSTRUCTIONS
    assert "Which device" in (tmp_path / "transcript.json").read_text()
