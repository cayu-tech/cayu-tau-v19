from cayu import ScriptedModelProvider
from tau2.environment.tool import Tool
from test_outcome_controller import outcome
from test_runtime_banking import call, text

from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
from cayu_tau3_banking.operational_recovery_probe import (
    CUSTOMER,
    POLICY,
    ReportHost,
    archive_report,
    assess,
    publish_report,
)
from cayu_tau3_banking.research_recovery_probe import ExhaustedResearchMemory
from cayu_tau3_banking.resource_outcome_probe import run_case


def test_execution_and_honest_partial_reply_with_research_exhausted(tmp_path):
    archive = outcome("archive", goal="Archive the report")
    archive["obligations"] = [
        {
            "id": "procedure",
            "goal": "Find retention procedure",
            "kind": "knowledge_search",
            "status": "ready",
        },
        {
            "id": "explain",
            "goal": "Report unavailable procedure honestly",
            "kind": "communication",
            "status": "ready",
        },
    ]
    provider = ScriptedModelProvider(
        [
            call(
                SYNC_WORK_PLAN_TOOL,
                {"items": [outcome("publish", goal="Publish the report"), archive]},
            ),
            call("publish_report", {"report_id": "R1"}, "published"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "publish",
                            "status": "completed",
                            "outcome": "succeeded",
                            "evidence": "The host confirms publication",
                            "evidence_refs": [{"kind": "tool_result", "reference_id": "published"}],
                        }
                    ]
                },
                "update",
            ),
            text("Report R1 is published. I could not archive it without the retention procedure."),
        ]
    )
    result = run_case(
        tmp_path,
        mismatch=False,
        policy=POLICY,
        customers=(CUSTOMER,),
        host=ReportHost(),
        tools=[Tool(publish_report), Tool(archive_report)],
        assessment=assess,
        memory_runtime_factory=ExhaustedResearchMemory,
        model="scripted",
        provider_factory=lambda: provider,
    )
    assert result["authorized_action_only"]
    statuses = {item["id"]: item["status"] for item in result["plans"][-1]["items"]}
    assert statuses == {"publish": "completed", "archive": "pending"}
