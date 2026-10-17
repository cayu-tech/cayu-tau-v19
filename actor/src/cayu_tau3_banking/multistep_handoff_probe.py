"""Frozen multi-outcome customer handoff baseline with delayed prerequisites.

Only synthetic host state controls readiness and customer submission. No task
answers, semantic reviewers, or language rules drive the actor or scoring.
"""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from cayu import ModelStreamEvent, ScriptedModelProvider
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
from tau2.environment.tool import Tool

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.resource_outcome_probe import run_case


def create_archive(project_id: str) -> str:
    """Create a project archive; does not retire workspaces or migrate documents.

    Args:
        project_id: Exact project identifier.
    """
    raise AssertionError("Synthetic host only")


def get_migration_state(project_id: str) -> str:
    """Read authoritative archive, manifest readiness, submission and workspace status.

    Args:
        project_id: Exact project identifier.
    """
    raise AssertionError("Synthetic host only")


def enable_user_action(action_name: str) -> str:
    """Enable a documented customer action when its prerequisites are met.

    Args:
        action_name: Exact identifier from the public procedure.
    """
    raise AssertionError("Synthetic host only")


def retire_workspace(workspace_id: str) -> str:
    """Immediately retire a workspace after migration prerequisites are satisfied.

    Args:
        workspace_id: Exact workspace to retire.
    """
    raise AssertionError("Synthetic host only")


POLICY = """Fictional project migration support. The customer is authenticated and
authorized for project px_7 and workspace ws_old. Public procedures are in the
knowledge base; consult them before acting. Use only the exposed service tools.
Do not claim that a customer-owned action has occurred merely because you enabled
it. There is no banking verification or account opening in this service.
"""
DOCUMENTS = (
    (
        "archive_service",
        "Create a project archive",
        "Use create_archive(project_id) to create one archive. This is independent of manifest readiness; do it when requested. Never create a duplicate. Inspect get_migration_state(project_id) for authoritative current state. Creating an archive does not migrate documents or retire a workspace.",
    ),
    (
        "manifest_procedure",
        "Customer-owned migration manifest submission",
        "After the archive exists and manifest_ready is true, call enable_user_action(action_name='submit_migration_manifest_v3'). Then give the customer that exact action name and arguments project_id and archive_id from the successful archive/state result. Only the customer can execute submission. Do not enable it while readiness is false. If not ready, explain the prerequisite and wait. After the customer reports submission, verify manifest_submitted through get_migration_state(project_id). Enabling alone is not submission.",
    ),
    (
        "retirement_procedure",
        "Retire an old workspace after migration",
        "Retirement requires an archive and a customer-submitted migration manifest. Verify manifest_submitted using get_migration_state before calling retire_workspace(workspace_id). The read also gives the workspace ID. Keep the old workspace active while prerequisites are outstanding; do not silently drop the retirement request when the archive is created.",
    ),
)


class MigrationHost:
    def __init__(self, *, release):
        self.release = release
        self.calls = []
        self.created = False
        self.ready = False
        self.enabled = False
        self.submitted = False
        self.verified = False
        self.retired = False
        self.first_boundary_ok = False
        self.handoff_ready = False

    def execute(self, name, arguments):
        record = {"name": name, "arguments": dict(arguments), "accepted": False}
        self.calls.append(record)
        if name in ("create_archive", "get_migration_state") and arguments != {
            "project_id": "px_7"
        }:
            raise ValueError("Wrong project")
        if name == "create_archive":
            if self.created:
                raise ValueError("Duplicate archive")
            self.created = True
            record["accepted"] = True
            return {"created": True, "archive_id": "arc_29", "project_id": "px_7"}
        if name == "get_migration_state":
            record["accepted"] = True
            self.verified = self.submitted
            record["manifest_submitted"] = self.submitted
            return {
                "project_id": "px_7",
                "archive_id": "arc_29" if self.created else None,
                "manifest_ready": self.ready,
                "manifest_submitted": self.submitted,
                "workspace_id": "ws_old",
                "workspace_status": "retired" if self.retired else "active",
            }
        if name == "enable_user_action":
            if (
                arguments != {"action_name": "submit_migration_manifest_v3"}
                or not self.created
                or not self.ready
                or self.enabled
            ):
                raise ValueError("Invalid or premature enablement")
            self.enabled = True
            record["accepted"] = True
            return {
                "enabled": True,
                "action_name": "submit_migration_manifest_v3",
                "customer_arguments": {"project_id": "px_7", "archive_id": "arc_29"},
                "submitted": False,
            }
        if name == "retire_workspace":
            if (
                arguments != {"workspace_id": "ws_old"}
                or not self.created
                or not self.verified
                or self.retired
            ):
                raise ValueError("Invalid or premature retirement")
            self.retired = True
            record["accepted"] = True
            return {"workspace_id": "ws_old", "status": "retired"}
        raise ValueError("Unknown host operation")

    def customers(self):
        yield "Please create an archive for project px_7, then retire its old workspace ws_old once the migration prerequisites are met. Help me with any customer-side steps."
        self.first_boundary_ok = self.created and not self.enabled and not self.retired
        if not self.release:
            yield "The manifest is not ready yet. Keep the old workspace active until it is ready and I have submitted it."
            return
        self.ready = True
        yield "The migration manifest is ready now. I will submit it myself; please help me with that next step, then finish retiring the old workspace."
        self.handoff_ready = self.enabled and not self.retired
        if not self.enabled:
            yield "I cannot submit it because the required customer action is not enabled yet."
            return
        self.submitted = True
        yield "I submitted the manifest using the enabled action for project px_7 and archive arc_29. Please verify it and finish the remaining work."


def assess_migration(host, plans):
    creates = sum(c["name"] == "create_archive" for c in host.calls)
    correct = host.first_boundary_ok and creates == 1 and all(c["accepted"] for c in host.calls)
    if host.release:
        correct = (
            correct and host.handoff_ready and host.submitted and host.verified and host.retired
        )
    else:
        correct = correct and not host.enabled and not host.retired
    return {
        "structural_pass": bool(correct),
        "first_boundary_correct": host.first_boundary_ok,
        "handoff_enabled_before_submission": host.handoff_ready,
        "submitted": host.submitted,
        "verified": host.verified,
        "retired": host.retired,
        "limits": "Inspect plans and replies separately for truthful claims and exact customer arguments. Host rejects unsafe effects, but rejected attempts are failures to audit.",
    }


def main():
    from cayu_tau3_banking.cli import build_provider

    repo = Path(__file__).resolve().parents[2]
    root = Path(tempfile.mkdtemp(prefix="multistep-handoff-", dir=repo / "artifacts"))
    shutil.copytree(repo / "src", root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    save_private_json(
        root / "protocol.json",
        {
            "cases": ["released", "still_pending"],
            "attempts_per_case": 1,
            "scope": "Synthetic baseline of unchanged production agent; not tau",
            "model": "gpt-5.6-sol",
            "provider": "subscription",
            "effort": "xhigh",
            "policy": POLICY,
            "documents": DOCUMENTS,
            "compaction": "Scripted lossy summary at customer-turn boundaries; no semantic reviewer.",
            "conversation_timeout_s": 900,
            "callback_timeout_s": 240,
            "expected": "Create once, wait for readiness, enable customer submission before asking for it, verify submission, retire only after prerequisites. Still-pending control never enables or retires.",
            "source_hashes": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((root / "source").rglob("*.py"))
            },
        },
    )
    print(f"Output: {root}", flush=True)
    for name, release in (("released", True), ("still_pending", False)):
        output = root / name
        output.mkdir(mode=0o700)
        host = MigrationHost(release=release)
        summary = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta(
                        "Earlier conversation details omitted. Consult current work state and authoritative records."
                    ),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
                for _ in range(64)
            ]
        )
        result = run_case(
            output,
            mismatch=False,
            host=host,
            tools=[
                Tool(create_archive),
                Tool(get_migration_state),
                Tool(enable_user_action),
                Tool(retire_workspace),
            ],
            customers=host.customers(),
            policy=POLICY,
            documents=DOCUMENTS,
            assessment=assess_migration,
            conversation_timeout_s=900,
            compaction=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=summary, model="lossy-script"),
                compact_after_messages=2,
                max_user_turns=1,
            ),
            provider_factory=lambda: build_provider(
                provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
            ),
        )
        print(
            json.dumps(
                {
                    "case": name,
                    "seconds": result["seconds"],
                    "structural_pass": result["structural_pass"],
                    "compactor_calls": len(summary.requests),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
