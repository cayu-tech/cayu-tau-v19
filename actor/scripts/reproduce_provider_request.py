"""One diagnostic model request from a captured context; never executes tool calls.

The historical tool catalogue was not captured. Reconstruct it from the frozen
source and public environment, and disclose that limitation rather than claim
byte-identical replay. Defaults to offline preparation; --live explicitly dispatches.
"""

import argparse
import asyncio
import importlib.util
import json
import tempfile
import time
from pathlib import Path

from cayu import Message
from cayu.providers.base import ModelRequest, ModelStreamEventType
from tau2.domains.banking_knowledge.environment import get_environment

from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.cli import build_provider
from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.runtime_banking import BankingRuntimeSession


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    frozen = args.frozen_run.resolve()
    assert (
        Path(__import__("cayu_tau3_banking.controller", fromlist=["__file__"]).__file__)
        .resolve()
        .is_relative_to(frozen / "src")
    )
    snapshots = [json.loads(p.read_text()) for p in (frozen / "task_047/contexts").glob("*/*.json")]
    context = next(
        s for s in snapshots if s["model_step_id"] == "mstep_5a44b0016dfa460a9b89d1a6636d99d0"
    )
    corpus = BankingKnowledgeCorpus.from_directory(
        Path(".vendor/tau2/data/tau2/domains/banking_knowledge/documents")
    )
    memory = BankingRuntimeSession(
        BankingMemoryRuntime(corpus), (), planning_mode="outcomes"
    ).memory
    schemas = {s["name"]: s for s in memory.tool_schemas()}
    schemas.update({s["name"]: s for s in memory.controller.registration_tool_schemas()})
    schemas.pop(SYNC_WORK_PLAN_TOOL)
    tools = list(schemas.values())
    for tool in get_environment(retrieval_variant="no_knowledge").get_tools():
        schema = tool.openai_schema["function"]
        tools.append(
            {
                "name": tool.name,
                "description": schema.get("description", ""),
                "input_schema": schema["parameters"],
            }
        )
    request = ModelRequest(
        model=context["model"],
        messages=[Message.model_validate(m) for m in context["messages"]],
        tools=tools,
        options={
            "thinking": {"enabled": True, "effort": "xhigh"},
            "openai": {"parallel_tool_calls": False},
        },
    )
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="request-reproduction-", dir=args.output_parent))
    preparation = {
        "scope": "one request diagnostic, no tau execution or score",
        "historical_step": context["model_step_id"],
        "messages": len(request.messages),
        "tools": len(tools),
        "limitation": "Historical context retained; tool schemas reconstructed from frozen registration, not captured original provider envelope. No tool proposals will execute.",
        "live": args.live,
    }
    save_private_json(output / "preparation.json", preparation)
    print(json.dumps({"output": str(output), **preparation}), flush=True)
    if not args.live:
        return
    spec = importlib.util.spec_from_file_location(
        "diagnostics",
        Path(__file__).resolve().parents[1] / "src/cayu_tau3_banking/provider_diagnostics.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider = build_provider(
        provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=180
    )
    provider.transport = module.DiagnosticOpenAITransport(provider.transport, output)
    started = time.monotonic()
    result = {"completed": 0, "errors": 0, "proposed_tool_calls": 0}
    try:
        async with asyncio.timeout(300):
            async for event in provider.runtime_stream(request):
                if event.type is ModelStreamEventType.COMPLETED:
                    result["completed"] += 1
                elif event.type is ModelStreamEventType.ERROR:
                    result["errors"] += 1
                elif event.type is ModelStreamEventType.TOOL_CALL:
                    result["proposed_tool_calls"] += 1
    except Exception:
        result["local_exception"] = True
    finally:
        await provider.aclose()
    result["seconds"] = time.monotonic() - started
    save_private_json(output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
