import pytest
from cayu import ScriptedModelProvider
from test_runtime_banking import call, text, work

from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL
from cayu_tau3_banking.navigation_probe import run_case
from cayu_tau3_banking.product_navigation import BROWSE_TOOL, READ_TOOL
from cayu_tau3_banking.research_recovery_probe import CASES, ExhaustedResearchMemory


@pytest.mark.parametrize("missing", [False, True])
def test_exhausted_fixture_preserves_evidence_and_unresolved_work(tmp_path, missing):
    case = CASES[int(missing)]
    pending = work()
    pending.pop("next_operation")
    pending["goal"] = "Compare the products"
    steps = [call(SYNC_WORK_PLAN_TOOL, {"items": [pending]}, "plan")]
    if not missing:
        completed = work(done=True, reference="doc_savings_accounts_birch_001")
        completed.pop("next_operation")
        completed["goal"] = pending["goal"]
        steps.append(call(UPDATE_WORK_PLAN_TOOL, {"items": [completed]}, "complete"))
    else:
        pending["obligations"] = [
            {
                "id": "missing_fee",
                "goal": "Find the unpublished fee",
                "kind": "knowledge_search",
                "status": "ready",
            },
            {
                "id": "explain_gap",
                "goal": "Explain the missing fee and conditional result without claiming completion",
                "kind": "communication",
                "status": "ready",
            },
        ]
        steps = [call(SYNC_WORK_PLAN_TOOL, {"items": [pending]}, "plan")]
    reply = (
        "The annual fee is unknown; no unconditional winner." if missing else "Birch earns $350."
    )
    provider = ScriptedModelProvider([*steps, *[text(reply)]])
    result = run_case(
        tmp_path,
        {"id": case["id"], "customer": case["customer"]},
        documents=case["documents"],
        provider_factory=lambda: provider,
        model="scripted",
        memory_runtime_factory=ExhaustedResearchMemory,
    )
    assert result["reply"] == reply
    assert result["plan"]["items"][0]["status"] == ("pending" if missing else "completed")
    for request in provider.requests:
        assert not {READ_TOOL, BROWSE_TOOL, KNOWLEDGE_SEARCH_TOOL} & {
            t["name"] for t in request.tools
        }
        assert "research_exhausted" in request.model_dump_json()
        assert case["expected_evaluator_only"] not in request.model_dump_json()
