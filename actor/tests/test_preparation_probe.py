import json

import pytest
from cayu import ScriptedModelProvider
from test_outcome_controller import obligation, outcome
from test_runtime_banking import call, text

from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
from cayu_tau3_banking.preparation_probe import SyntheticDelivery, run_case
from cayu_tau3_banking.runtime_agent import RuntimeAgentError


@pytest.mark.parametrize("present", [True, False])
@pytest.mark.parametrize("initial_advisory", [False, True])
def test_probe_executes_host_tools_and_records_both_phases(tmp_path, present, initial_advisory):
    args = {"record_id": "D17"}
    steps = [
        call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}),
        call("read_delivery_record", args, "read-first"),
    ]
    if initial_advisory:
        steps.insert(
            0,
            call(
                SYNC_WORK_PLAN_TOOL,
                {
                    "items": [
                        {**outcome(), "obligations": [obligation("waiting", evidence_refs=[])]}
                    ]
                },
            ),
        )
    if present:
        steps.append(call("validate_delivery_address", args, "validate"))
    steps.extend(
        [
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "request",
                            "status": "awaiting_customer",
                            "evidence": "Need customer input",
                            "obligations": [obligation("waiting")],
                        }
                    ]
                },
            ),
            text("Please supply the missing details and postage."),
            call("read_delivery_record", args, "read-second"),
        ]
    )
    if not present:
        steps.append(call("validate_delivery_address", args, "validate"))
    steps.extend(
        [
            call("dispatch_documents", args, "dispatch"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "request",
                            "status": "completed",
                            "outcome": "succeeded",
                            "evidence": "Dispatched",
                            "evidence_refs": [{"kind": "tool_result", "reference_id": "dispatch"}],
                            "obligations": [
                                obligation(
                                    "resolved",
                                    evidence_refs=[
                                        {"kind": "tool_result", "reference_id": "validate"}
                                    ],
                                )
                            ],
                        }
                    ]
                },
            ),
            text("Dispatched."),
        ]
    )
    result = run_case(
        tmp_path,
        address_present=present,
        provider_factory=lambda: ScriptedModelProvider(steps),
        model="scripted",
    )
    assert result["passed"]
    assert result["context_snapshots"] == len(steps)
    assert json.loads((tmp_path / "transcript.json").read_text())


def test_synthetic_host_enforces_dependencies_and_duplicate_effects():
    host = SyntheticDelivery(address_present=False)
    args = {"record_id": "D17"}
    with pytest.raises(ValueError, match="missing address"):
        host.execute("validate_delivery_address", args)
    with pytest.raises(ValueError, match="unauthorized"):
        host.execute("dispatch_documents", args)
    host.customer_update()
    host.execute("validate_delivery_address", args)
    host.execute("dispatch_documents", args)
    with pytest.raises(ValueError, match="duplicate"):
        host.execute("dispatch_documents", args)


def test_probe_preserves_artifacts_and_stops_on_invalid_execution(tmp_path):
    steps = [
        call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}),
        call("dispatch_documents", {"record_id": "D17"}, "invalid-dispatch"),
    ]
    with pytest.raises(ValueError, match="unauthorized"):
        run_case(
            tmp_path,
            address_present=True,
            provider_factory=lambda: ScriptedModelProvider(steps),
            model="scripted",
        )
    assert (tmp_path / "error.json").exists()
    assert (tmp_path / "transcript.json").exists()


def test_fabricated_evidence_still_stops_before_model_repair(tmp_path):
    provider = ScriptedModelProvider(
        [
            call(
                SYNC_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            **outcome(),
                            "obligations": [
                                obligation(
                                    "waiting",
                                    evidence_refs=[
                                        {"kind": "customer_message", "reference_id": "invented"}
                                    ],
                                )
                            ],
                        }
                    ]
                },
            ),
            text("This repair must never be requested."),
        ]
    )
    with pytest.raises(RuntimeAgentError):
        run_case(
            tmp_path, address_present=True, provider_factory=lambda: provider, model="scripted"
        )
    assert len(provider.requests) == 1
    assert json.loads((tmp_path / "error.json").read_text())["calls"] == []


def test_customer_authorization_precedes_all_record_tools(tmp_path):
    args = {"record_id": "D17"}
    provider = ScriptedModelProvider(
        [
            call(
                SYNC_WORK_PLAN_TOOL,
                {"items": [outcome(status="awaiting_customer", evidence="Need access code")]},
            ),
            text("Please provide your delivery access code and pay postage."),
            call("verify_delivery_access", {"access_code": "D17-ACCESS"}, "verify"),
            call("read_delivery_record", args, "read"),
            call("validate_delivery_address", args, "validate"),
            call("dispatch_documents", args, "dispatch"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "request",
                            "status": "completed",
                            "outcome": "succeeded",
                            "evidence": "Dispatch confirmed",
                            "evidence_refs": [{"kind": "tool_result", "reference_id": "dispatch"}],
                        }
                    ]
                },
            ),
            text("Dispatched."),
        ]
    )
    result = run_case(
        tmp_path,
        address_present=True,
        require_access=True,
        provider_factory=lambda: provider,
        model="scripted",
    )
    assert result["passed"]
    assert result["first_phase_calls"] == []
    assert "not authorized yet" in provider.requests[0].messages[0].content[0].text
    host = SyntheticDelivery(address_present=True, require_access=True)
    with pytest.raises(ValueError, match="verification"):
        host.execute("read_delivery_record", args)
