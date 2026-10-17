"""Synthetic forced-loss compaction and customer-owned action regression."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from cayu import ModelStreamEvent, ScriptedModelProvider
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
from tau2.environment.tool import Tool

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.navigation_probe import run_case as run_advice
from cayu_tau3_banking.needs_notice_probe import ADVICE_POLICY, DOCUMENTS
from cayu_tau3_banking.resource_outcome_probe import run_case as run_effect


def enable_customer_action(action_name: str) -> str:
    """Enable one documented customer-owned action; does not execute it.

    Args:
        action_name: Exact public action identifier from the procedure.
    """
    raise AssertionError("Synthetic host only")


def get_upload_status(workspace_id: str) -> str:
    """Read authoritative upload status for a workspace.

    Args:
        workspace_id: Exact workspace identifier.
    """
    raise AssertionError("Synthetic host only")


class UploadHost:
    def __init__(self):
        self.calls = []
        self.enabled = False
        self.uploaded = False
        self.first_turn_enabled = False

    def execute(self, name, arguments):
        if name == "enable_customer_action" and arguments == {"action_name": "upload_document_v2"}:
            if self.enabled:
                raise ValueError("Action already enabled")
            self.calls.append({"name": name, "arguments": dict(arguments)})
            self.enabled = True
            return {
                "enabled": True,
                "action_name": "upload_document_v2",
                "required_customer_arguments": {
                    "workspace_id": "ws_81",
                    "filename": "statement.pdf",
                },
                "upload_performed": False,
            }
        if name == "get_upload_status" and arguments == {"workspace_id": "ws_81"}:
            self.calls.append(
                {"name": name, "arguments": dict(arguments), "uploaded": self.uploaded}
            )
            return {"workspace_id": "ws_81", "uploaded": self.uploaded}
        raise ValueError("Invalid synthetic host operation")

    def customers(self):
        yield "Please help me upload statement.pdf into my workspace ws_81."
        self.first_turn_enabled = self.enabled
        if self.enabled:
            # This effect is executed by the simulated customer, never the actor.
            self.uploaded = True
            yield "I used the enabled upload_document_v2 action with workspace_id ws_81 and filename statement.pdf. Please verify that my upload is complete."
        else:
            yield "I cannot upload because the customer action has not been enabled. Please enable it."


UPLOAD_POLICY = """Fictional workspace support service. Customer is authenticated and
authorized for ws_81. Procedures are in the public knowledge base; discover the
relevant procedure before helping the customer execute it. Do not invent operations
or claim an upload succeeded without authoritative status. Only the customer may
upload a document. All available agent tools are exposed; no banking verification
or personal financial information is relevant.
"""
UPLOAD_DOCUMENTS = (
    (
        "workspace_upload",
        "Uploading a workspace document",
        "To help upload a document, first call enable_customer_action with action_name='upload_document_v2'. Then instruct the customer to run that action with workspace_id and filename. Enabling does not upload anything. After the customer reports submitting the document, use get_upload_status(workspace_id) to verify completion.",
    ),
)


def assess_upload(host, plans):
    verified = any(c["name"] == "get_upload_status" and c["uploaded"] for c in host.calls)
    return {
        "structural_pass": host.first_turn_enabled and host.uploaded and verified,
        "enabled_before_first_reply": host.first_turn_enabled,
        "customer_uploaded": host.uploaded,
        "verified_after_upload": verified,
        "limits": "Manually review instructions and final claims; no tau score.",
    }


def main():
    from cayu_tau3_banking.cli import build_provider

    repo = Path(__file__).resolve().parents[2]
    root = Path(tempfile.mkdtemp(prefix="locator-handoff-", dir=repo / "artifacts"))
    shutil.copytree(repo / "src", root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    case = {
        "id": "forced_locator_loss",
        "customer": "Compare Cedar and Harbor for $2000 left untouched for one year. I bank entirely online, never deposit cash, and want the highest interest. Advice only. Which one?",
        "followups": [
            "Please recheck that comparison with the comparison tool and confirm your recommendation."
        ],
        "expected": "Cedar, $84 versus $72. A successful source comparison with exact IDs despite deliberately lossy compaction; no invented IDs.",
    }
    save_private_json(
        root / "protocol.json",
        {
            "cases": [
                case,
                {"id": "customer_upload", "policy": UPLOAD_POLICY, "documents": UPLOAD_DOCUMENTS},
            ],
            "scope": "Synthetic integration regressions, not tau; one trial each",
            "model": "gpt-5.6-sol",
            "provider": "subscription",
            "effort": "xhigh",
            "compaction": "First case uses scripted deliberately lossy summaries; actor remains live Sol.",
            "customer": "Deterministic host executes upload only after agent enabled it; no language-regex routing.",
            "source_hashes": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((root / "source").rglob("*.py"))
            },
        },
    )
    print(f"Output: {root}", flush=True)

    def provider():
        return build_provider(
            provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
        )

    summary_provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta(
                    "Earlier lookup details omitted. Use current metadata and authoritative evidence."
                ),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
            for _ in range(64)
        ]
    )
    compaction = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(provider=summary_provider, model="lossy-script"),
        compact_after_messages=2,
        max_user_turns=1,
    )
    output = root / case["id"]
    output.mkdir(mode=0o700)
    result = run_advice(
        output,
        case,
        provider_factory=provider,
        documents=DOCUMENTS,
        policy=ADVICE_POLICY,
        compaction=compaction,
    )
    print(
        json.dumps(
            {
                "case": case["id"],
                "seconds": result["seconds"],
                "compactor_calls": len(summary_provider.requests),
            }
        ),
        flush=True,
    )
    output = root / "customer_upload"
    output.mkdir(mode=0o700)
    host = UploadHost()
    result = run_effect(
        output,
        mismatch=False,
        provider_factory=provider,
        policy=UPLOAD_POLICY,
        documents=UPLOAD_DOCUMENTS,
        host=host,
        tools=[Tool(enable_customer_action), Tool(get_upload_status)],
        customers=host.customers(),
        assessment=assess_upload,
    )
    print(
        json.dumps(
            {
                "case": "customer_upload",
                "seconds": result["seconds"],
                "structural_pass": result["structural_pass"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
