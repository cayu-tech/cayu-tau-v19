"""Compact observed banking receipts without mandatory model-authored packets."""

import json

from cayu_tau3_banking.controller import UPDATE_WORK_PLAN_TOOL
from cayu_tau3_banking.decision_packets import normalized_operation
from cayu_tau3_banking.ledger import ToolActionStatus


class ExecutionReceipts:
    def __init__(self, ledger, controller):
        self.ledger, self.controller = ledger, controller
        self.receipts = []
        self.seen = set()

    def observe_result(self, call_id):
        if call_id in self.seen:
            return
        status = self.ledger.tool_action_status(call_id)
        if status not in (ToolActionStatus.SUCCEEDED, ToolActionStatus.FAILED):
            return
        self.seen.add(call_id)
        name = self.ledger.tool_action_name(call_id)
        arguments = self.ledger.tool_action_arguments(call_id)
        try:
            normalized = normalized_operation(name, arguments)
        except (ValueError, TypeError, RecursionError):
            normalized = (name, "{}")
        settled = []
        if status is ToolActionStatus.SUCCEEDED:
            for item in self.controller.export_state()["items"]:
                for step in item.get("obligations", []):
                    planned = step.get("planned_tool")
                    if not planned or step["status"] != "ready":
                        continue
                    try:
                        matches = (
                            normalized_operation(planned["tool_name"], planned["arguments"])
                            == normalized
                        )
                    except (KeyError, ValueError, TypeError):
                        continue
                    if matches:
                        result = self.controller.invoke(
                            UPDATE_WORK_PLAN_TOOL,
                            {
                                "items": [
                                    {
                                        "id": item["id"],
                                        "obligations": [
                                            {
                                                "id": step["id"],
                                                "status": "resolved",
                                                "evidence": "Observed successful execution of the exact existing action.",
                                                "evidence_refs": [
                                                    {"kind": "tool_result", "reference_id": call_id}
                                                ],
                                            }
                                        ],
                                    }
                                ]
                            },
                        )
                        if not result.is_error:
                            settled.append({"outcome": item["id"], "step": step["id"]})
        self.receipts = [
            *self.receipts[-11:],
            {
                "tool_result": call_id,
                "tool": normalized[0],
                "execution_status": status.value,
                "settled_steps": settled,
            },
        ]

    def context(self):
        return (
            '<observed_bank_receipts outcome_completion_not_inferred="true">'
            + json.dumps(self.receipts)
            + "</observed_bank_receipts>"
        )
