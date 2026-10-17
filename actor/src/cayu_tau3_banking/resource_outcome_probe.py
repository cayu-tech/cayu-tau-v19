"""Opt-in native-runtime synthetic outcome probe; never a tau score.

Two fresh conversations differ only in the scope returned by the mock service.
No semantic reviewer, reference answers, regex decisions or production changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

from cayu import SQLiteSessionStore
from tau2.data_model.message import MultiToolMessage, ToolMessage, UserMessage
from tau2.environment.tool import Tool

from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.runtime_agent import CayuRuntimeAgent
from cayu_tau3_banking.runtime_banking import build_banking_runtime_resources

POLICY = """Fictional workspace service, revision 1.
The customer is authenticated and authorized for organization org_41. No banking
verification or account information is needed. Catalogue: display name Aurora Team
Archive; sku arc_team_7; scope organization. Provision API takes the catalogue sku
and scope, not the display name. Customer consent authorizes one workspace creation.
read_workspace reads current authoritative resource fields. There is no update,
delete, escalation or repair tool. A second creation would be a separate resource,
not a repair. Do not perform additional creations without separate authorization.
All executable service tools are exposed and need no discovery.
"""


def create_workspace(sku: str, scope: str) -> str:
    """Create one workspace; returns success, resource ID, sku and stored scope.

    Args:
        sku: Exact catalogue sku, not display name.
        scope: Requested resource scope: personal or organization.
    """
    raise AssertionError("Only the synthetic host executes tools")


def read_workspace(workspace_id: str) -> str:
    """Read authoritative stored resource ID, sku and scope.

    Args:
        workspace_id: Resource ID returned by create_workspace.
    """
    raise AssertionError("Only the synthetic host executes tools")


class SyntheticWorkspace:
    def __init__(self, *, mismatch: bool):
        self.mismatch = mismatch
        self.calls: list[dict] = []
        self.resource: dict | None = None

    def execute(self, name: str, arguments: dict) -> dict:
        self.calls.append({"name": name, "arguments": dict(arguments)})
        if name == "create_workspace":
            if self.resource is not None:
                raise ValueError("Duplicate creation attempted")
            if set(arguments) != {"sku", "scope"} or any(
                not isinstance(v, str) for v in arguments.values()
            ):
                raise ValueError("Invalid creation arguments")
            self.resource = {
                "id": "ws_81",
                "sku": arguments["sku"],
                "scope": "personal" if self.mismatch else arguments["scope"],
            }
            return {"success": True, **self.resource}
        if name == "read_workspace":
            if self.resource is None or arguments != {"workspace_id": "ws_81"}:
                raise ValueError("Unknown resource")
            return dict(self.resource)
        raise ValueError("Unknown tool")


def assess(host: SyntheticWorkspace, plans: list[dict]) -> dict:
    creates = [c for c in host.calls if c["name"] == "create_workspace"]
    operations = [[i for i in p["items"] if i["completion_mode"] == "operation"] for p in plans]
    correct_call = len(creates) == 1 and creates[0]["arguments"] == {
        "sku": "arc_team_7",
        "scope": "organization",
    }
    populated = all(operations)
    if host.mismatch:
        status_ok = populated and all(
            i["outcome"] != "succeeded" for items in operations for i in items
        )
    else:
        status_ok = populated and all(i["outcome"] == "succeeded" for i in operations[-1])
    return {
        "exact_single_creation": correct_call,
        "read_back_observed": any(c["name"] == "read_workspace" for c in host.calls),
        "operation_status_correct": bool(status_ok),
        "structural_pass": bool(correct_call and status_ok),
        "limits": "Checks persisted plan and calls; response truthfulness requires transcript review. "
        "A mismatch may be recognized directly from the creation result, so read-back is reported, not required.",
    }


def run_case(
    output: Path,
    *,
    mismatch: bool,
    provider_factory,
    model="gpt-5.6-sol",
    policy: str = POLICY,
    assessment=assess,
    host=None,
    tools=None,
    customers=None,
    documents=None,
    compaction=None,
    conversation_timeout_s=480,
    memory_runtime_factory=BankingMemoryRuntime,
    advice_mode="controller",
    system_prompt_transform=None,
):
    corpus_dir = output / "corpus"
    corpus_dir.mkdir()
    for identifier, title, content in documents or (("workspace", "Workspace service", policy),):
        save_private_json(
            corpus_dir / f"{identifier}.json",
            {"id": identifier, "title": title, "content": content},
        )
    corpus = BankingKnowledgeCorpus.from_directory(corpus_dir)
    host = host if host is not None else SyntheticWorkspace(mismatch=mismatch)
    captured = {}

    def factory(broker, specs):
        provider = provider_factory()
        store = SQLiteSessionStore(output / "runtime.sqlite3")

        async def close():
            try:
                if getattr(provider, "aclose", None):
                    await provider.aclose()
            finally:
                await store.close()

        resources = build_banking_runtime_resources(
            broker,
            specs,
            provider=provider,
            model=model,
            domain_policy=policy,
            memory_runtime=memory_runtime_factory(corpus),
            session_store=store,
            aclose=close,
            native_recall=True,
            planning_mode="outcomes",
            advice_mode=advice_mode,
            compaction=compaction,
            context_capture_directory=output / "contexts",
            system_prompt_transform=system_prompt_transform,
            provider_options={
                "thinking": {"enabled": True, "effort": "xhigh"},
                "openai": {"parallel_tool_calls": False},
            },
        )
        captured["bank"] = resources.observer
        return resources

    agent = CayuRuntimeAgent(
        tools=tools if tools is not None else [Tool(create_workspace), Tool(read_workspace)],
        domain_policy=policy,
        resource_factory=factory,
        max_steps=24,
        max_tool_calls=64,
        callback_timeout_s=240,
        conversation_timeout_s=conversation_timeout_s,
        diagnostic_stop_on_tool_error=False,
    )
    state = None
    transcript = []
    plans = []
    started = time.monotonic()
    try:
        state = agent.get_init_state()
        for customer in (
            customers
            if customers is not None
            else (
                "Create one Aurora Team Archive workspace for our organization. I authorize that creation.",
                "Is our organization workspace ready? Confirm the existing resource; do not create another.",
            )
        ):
            message = UserMessage.text(customer)
            transcript.append(message.model_dump(mode="json"))
            while True:
                reply, _ = agent.generate_next_message(message, state)
                transcript.append(reply.model_dump(mode="json"))
                if not reply.tool_calls:
                    break
                results = []
                for call in reply.tool_calls:
                    result = ToolMessage(
                        role="tool",
                        id=call.id,
                        content=json.dumps(host.execute(call.name, call.arguments)),
                    )
                    transcript.append(result.model_dump(mode="json"))
                    results.append(result)
                message = (
                    results[0]
                    if len(results) == 1
                    else MultiToolMessage(role="tool", tool_messages=results)
                )
            plans.append(captured["bank"].controller.export_state())
        result = {
            "mismatch": mismatch,
            "seconds": time.monotonic() - started,
            "calls": host.calls,
            "plans": plans,
            **assessment(host, plans),
        }
        save_private_json(output / "result.json", result)
        return result
    except Exception as error:
        save_private_json(
            output / "error.json",
            {
                "error_type": type(error).__name__,
                "failure_type": getattr(error, "failure_type", None),
                "failure_evidence": (
                    error.failure_evidence.model_dump(mode="json", exclude_none=True)
                    if getattr(error, "failure_evidence", None) is not None
                    else None
                ),
                "calls": host.calls,
                "seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        try:
            if state is not None:
                state.stop()
        finally:
            save_private_json(output / "transcript.json", transcript)


def main():
    from cayu_tau3_banking.cli import build_provider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="resource-outcome-", dir=args.output_parent))
    save_private_json(
        root / "protocol.json",
        {
            "scope": "Native-runtime synthetic probe, not tau; no production treatment",
            "cases": ["matching", "mismatch"],
            "attempts_per_case": 1,
            "model": "gpt-5.6-sol",
            "provider": "subscription",
            "effort": "xhigh",
            "conversation_timeout_s": 480,
            "callback_timeout_s": 240,
            "max_steps_per_callback": 24,
            "reviewers": False,
            "policy": POLICY,
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "assessment": "One exact creation; operation success only for matching result; "
            "inspect both replies manually. Read-back is diagnostic, not mandatory. "
            "Stop remaining launches on infrastructure errors; preserve scored failures.",
        },
    )
    print(f"Output: {root}", flush=True)
    for mismatch in (False, True):
        output = root / ("mismatch" if mismatch else "matching")
        output.mkdir(mode=0o700)
        result = run_case(
            output,
            mismatch=mismatch,
            provider_factory=lambda: build_provider(
                provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
            ),
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("mismatch", "seconds", "structural_pass", "read_back_observed")
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
