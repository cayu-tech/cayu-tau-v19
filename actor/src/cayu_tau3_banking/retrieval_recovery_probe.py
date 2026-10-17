"""Opt-in native-runtime evidence-gap probe, not a tau score or reviewer.

Controlled fault injection: automatic recall is empty; the first explicit search
returns a fixed real corpus result. Subsequent searches use the real Cayu engine.
The actor never receives case names, expected values, or the injected-fault rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

from cayu import SQLiteSessionStore
from tau2.data_model.message import ToolMessage, UserMessage
from tau2.environment.tool import Tool

from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus, BankingRecall
from cayu_tau3_banking.runtime_agent import CayuRuntimeAgent
from cayu_tau3_banking.runtime_banking import build_banking_runtime_resources

POLICY = """You answer public specification questions for a fictional parcel service.
Use the provided knowledge tools for public facts. No customer authentication,
bank-account lookup, product recommendation or financial operation is needed.
record_research_result records your answer in this local research exercise. It does
not dispatch a parcel or modify a customer account. Its tools are already exposed.
Record a supported maximum_kg with the source document IDs, or null if you cannot
establish it, explaining the limitation in unresolved_reason. Do not use another
service's specification. Then explain the result to the customer.
"""


def record_research_result(
    maximum_kg: float | None, evidence_entry_ids: list[str], unresolved_reason: str
) -> str:
    """Record the researched Harbor parcel mass limit or an unresolved finding.

    Args:
        maximum_kg: Supported maximum in kilograms, or null when not established.
        evidence_entry_ids: IDs of the documents supporting the stated limit.
        unresolved_reason: Why the limit could not be established; empty when supported.
    """
    raise AssertionError("Only the synthetic host executes this tool")


def make_corpus(directory: Path, case: str) -> BankingKnowledgeCorpus:
    directory.mkdir()
    docs = [
        {
            "id": "reception",
            "title": "Harbor reception schedule",
            "content": "Harbor reception opens at 09:00 and closes at 17:00. Parcels may be dropped off during these hours.",
        },
        {
            "id": "other-service",
            "title": "Summit parcel specification",
            "content": "Summit service accepts parcels weighing no more than 38 kilograms.",
        },
    ]
    if case != "unavailable":
        docs.append(
            {
                "id": "specification",
                "title": "Harbor parcel specification",
                "content": "Harbor service accepts parcels weighing no more than 17 kilograms. This is the maximum parcel mass.",
            }
        )
    for doc in docs:
        save_private_json(directory / (doc["id"] + ".json"), doc)
    return BankingKnowledgeCorpus.from_directory(directory)


def assess(case: str, reports: list[dict], searches: list[dict]) -> dict:
    delivered = {d["entry_id"] for search in searches for d in search["response"]["documents"]}
    if len(reports) != 1:
        return {"passed": False, "reason": "Expected exactly one recorded finding."}
    report = reports[0]
    ids = report.get("evidence_entry_ids", [])
    supported_ids = isinstance(ids, list) and all(
        isinstance(i, str) and i in delivered for i in ids
    )
    if case == "unavailable":
        passed = (
            report.get("maximum_kg") is None
            and bool(report.get("unresolved_reason"))
            and len(searches) >= 2
            and supported_ids
        )
    else:
        passed = (
            type(report.get("maximum_kg")) in (int, float)
            and report["maximum_kg"] == 17
            and supported_ids
            and "specification" in ids
            and len(searches) >= (2 if case == "recoverable" else 1)
        )
    return {
        "passed": passed,
        "reason": "Supported finding or explicit limitation required.",
        "delivered_ids": sorted(delivered),
        "search_count": len(searches),
    }


def run_case(output: Path, *, case: str, provider_factory, model="gpt-5.6-sol"):
    if case not in {"complete", "recoverable", "unavailable"}:
        raise ValueError("Unknown recovery case")
    corpus = make_corpus(output / "corpus", case)
    searches, reports, transcript = [], [], []
    engine_recall = corpus.discovery_engine.recall
    original_search = corpus.search_with_evidence
    first = True

    async def injected_recall(situation):
        nonlocal first
        if first:
            first = False
            seed = (
                "Harbor parcel specification" if case == "complete" else "Harbor reception schedule"
            )
            result = await engine_recall(corpus._situation(seed))
            chosen = "specification" if case == "complete" else "reception"
            # Keep a genuine candidate and its provenance, but deliberately omit
            # other candidates. Record the artificial cap honestly in diagnostics.
            candidates = tuple(
                c for c in result.candidates if c.record.locator.get("entry_id") == chosen
            )[:1]
            return result.model_copy(update={"candidates": candidates, "truncated": True})
        return await engine_recall(situation)

    async def capture_search(query, **kwargs):
        started = time.monotonic()
        content, refs = await original_search(query, **kwargs)
        row = {
            "query": query,
            "mode": kwargs.get("mode", "default"),
            "cursor": kwargs.get("cursor"),
            "response": json.loads(content),
            "seconds": time.monotonic() - started,
        }
        searches.append(row)
        save_private_json(output / f"search-{len(searches):02d}.json", row)
        return content, refs

    async def no_automatic_recall(*args, **kwargs):
        return BankingRecall(
            context=None,
            evidence_refs=(),
            record_ids=(),
            result_sha256="0" * 64,
            candidate_count=0,
            injected_count=0,
            source_coverage=(),
        )

    corpus.discovery_engine.recall = injected_recall
    corpus.search_with_evidence = capture_search
    corpus.recall = no_automatic_recall
    contexts = output / "contexts"
    contexts.mkdir()

    def factory(broker, specs):
        provider = provider_factory()
        store = SQLiteSessionStore(output / "runtime.sqlite3")

        async def close():
            try:
                if getattr(provider, "aclose", None) is not None:
                    await provider.aclose()
            finally:
                await store.close()

        return build_banking_runtime_resources(
            broker,
            specs,
            provider=provider,
            model=model,
            domain_policy=POLICY,
            memory_runtime=BankingMemoryRuntime(corpus),
            session_store=store,
            aclose=close,
            native_recall=False,
            planning_mode="outcomes",
            context_capture_directory=contexts,
            provider_options={
                "thinking": {"enabled": True, "effort": "xhigh"},
                "openai": {"parallel_tool_calls": False},
            },
        )

    agent = CayuRuntimeAgent(
        tools=[Tool(record_research_result)],
        domain_policy=POLICY,
        resource_factory=factory,
        max_steps=20,
        max_tool_calls=48,
        callback_timeout_s=180,
        conversation_timeout_s=360,
        diagnostic_stop_on_tool_error=False,
    )
    state = None
    started = time.monotonic()
    try:
        state = agent.get_init_state()
        message = UserMessage.text(
            "What is the maximum parcel mass accepted by Harbor service? Record the researched finding, then tell me the result."
        )
        transcript.append(message.model_dump(mode="json"))
        while True:
            reply, _ = agent.generate_next_message(message, state)
            transcript.append(reply.model_dump(mode="json"))
            if not reply.tool_calls:
                final = reply.content
                break
            if len(reply.tool_calls) != 1 or reply.tool_calls[0].name != "record_research_result":
                raise ValueError("Unexpected external probe call")
            call = reply.tool_calls[0]
            reports.append(call.arguments)
            # No semantic feedback or corrections: assess only after completion.
            message = ToolMessage(role="tool", id=call.id, content=json.dumps({"recorded": True}))
            transcript.append(message.model_dump(mode="json"))
        result = {
            "case": case,
            **assess(case, reports, searches),
            "reports": reports,
            "final_reply": final,
            "seconds": time.monotonic() - started,
            "context_snapshots": len(list(contexts.rglob("*.json"))),
        }
        save_private_json(output / "result.json", result)
        return result
    except Exception as error:
        save_private_json(
            output / "error.json",
            {
                "error_type": type(error).__name__,
                "seconds": time.monotonic() - started,
                "reports": reports,
                "search_count": len(searches),
            },
        )
        raise
    finally:
        if state is not None:
            state.stop()
        save_private_json(output / "transcript.json", transcript)


def main():
    from cayu_tau3_banking.cli import build_provider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("complete", "recoverable", "unavailable"), required=True)
    args = parser.parse_args()
    parent = Path("artifacts/retrieval-recovery-validation")
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=args.case + "-", dir=parent))
    save_private_json(
        output / "manifest.json",
        {
            "case": args.case,
            "model": "gpt-5.6-sol",
            "provider": "subscription",
            "native_runtime": True,
            "automatic_recall": "disabled by probe",
            "first_search": "controlled candidate omission",
            "limit_seconds": 360,
            "scope": "Synthetic diagnostic, not tau, not a causal old/new prompt comparison.",
            "source_sha256": {
                str(path.relative_to(Path(__file__).parent)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(Path(__file__).parent.glob("*.py"))
            },
        },
    )
    print(output, flush=True)
    result = run_case(
        output,
        case=args.case,
        provider_factory=lambda: build_provider(
            provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
