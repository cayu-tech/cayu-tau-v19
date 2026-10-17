"""Opt-in paired synthetic decisions, not a reviewer or an official tau score.

Frozen cases vary amounts and customer intent. Only retention of previously
delivered procedure text changes within a pair. Expected answers never enter
model requests; no banking operations execute.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tempfile
import time
import traceback
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cayu import Message
from cayu.providers.base import ModelRequest, ModelStreamEventType
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cayu_tau3_banking.active_evidence import active_evidence_context
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    step: Literal[
        "search_procedure",
        "offer_alternative",
        "offer_retention",
        "close_account",
        "acknowledge_alternative",
    ]
    credit_dollars: int | None = Field(ge=0)
    conditions: list[str] = Field(max_length=8)
    evidence_ids: list[str] = Field(max_length=8)


@dataclass(frozen=True)
class Case:
    id: str
    amount: int
    state: str
    customer: str
    expected_step: str
    expected_credit: int | None = None
    missing_step: str = "search_procedure"


CASES = (
    Case(
        "suitable_alternative",
        17,
        "No alternatives have been presented.",
        "Another bank pays 3% on trains. That is why I wanted to leave.",
        "offer_alternative",
    ),
    Case(
        "retention_17",
        17,
        "Cedar was explained and declined. No valid retention offer yet.",
        "I do not want Cedar. I still want to close Harbor.",
        "offer_retention",
        17,
    ),
    Case(
        "retention_31",
        31,
        "Cedar was explained and declined. No valid retention offer yet.",
        "I do not want Cedar. I still want to close Harbor.",
        "offer_retention",
        31,
    ),
    Case(
        "alternative_accepted",
        17,
        "Customer applied for Cedar successfully and explicitly chose it instead of closure.",
        "The application succeeded. Keep Harbor open; that resolves my request.",
        "acknowledge_alternative",
        missing_step="acknowledge_alternative",
    ),
    Case(
        "renewed_closure",
        31,
        "Cedar application succeeded. Required closure disclosures were delivered. The authorized retention offer was explained and declined.",
        "I want Cedar, but separately I still want Harbor closed. I authorize that closure now.",
        "close_account",
    ),
)


def procedure(case: Case) -> str:
    return (
        "Fictional bank policy revision 1. After successful identity, eligibility and prior-attempt checks: "
        "when the reason is a competitor, first compare documented alternatives. If a suitable one exists, "
        "offer application assistance INSTEAD OF closure and await the customer's choice. "
        "If accepted, acknowledge the alternative after its required action succeeds; do not reopen closure "
        "unless the customer explicitly renews that request. If the customer declines alternatives and still "
        f"wants closure, offer a {case.amount}-dollar retention statement credit. This credit has no purchase "
        "threshold or minimum holding period. It is the only authorized retention offer. Never issue it "
        "before customer acceptance. If retention is declined, deliver closure disclosures and obtain "
        "explicit closure authorization, then close. A new application does not forbid a separate, "
        "explicitly renewed closure after these prerequisites."
    )


def build_request(case: Case, *, system_prompt: str, active_context: str | None) -> ModelRequest:
    context = (
        "Synthetic decision-only test; no operation is executed. Use record_decision for the next "
        "substantive customer-facing or tool step. credit_dollars is null except when proposing a "
        "retention credit. conditions lists its purchase/holding conditions, if any. evidence_ids lists "
        "the supplied policy document IDs that support your step; cite none when searching.\n"
        "Public products: Harbor earns 1% flat. Cedar earns 4% on trains, with no annual fee.\n"
        "Authoritative records: identity, eligibility and prior-attempt checks passed.\n"
        f"Observed conversation state: {case.state}\n"
        "Private checklist: resolve Harbor request; ready communication cites procedure revision 1. "
        "This citation is not the procedure text. Earlier assistant draft suggested an unverified "
        "180-dollar credit with a nine-month holding period; no bank document or tool authorized it.\n"
        + (
            active_context
            or "Previously recalled procedure text is no longer in the current context."
        )
    )
    return ModelRequest(
        model="gpt-5.6-sol",
        messages=[
            Message.text("system", system_prompt),
            Message.text("user", context),
            Message.text("user", case.customer),
        ],
        tools=[
            {
                "name": "record_decision",
                "description": "Record a decision; executes nothing.",
                "input_schema": Decision.model_json_schema(),
            }
        ],
        options={
            "thinking": {"enabled": True, "effort": "xhigh"},
            "openai": {"parallel_tool_calls": False, "tool_choice": "required"},
        },
    )


class ProbeProviderError(RuntimeError):
    def __init__(self, payload):
        # Only structured error classification; never persist error text or credentials.
        self.diagnostics = {
            key: payload[key]
            for key in ("error_type", "status_code", "provider_error_type", "provider_error_code")
            if key in payload
        }
        super().__init__("Provider stream failed before a complete decision.")


async def decide(provider, request: ModelRequest, *, capture_path: Path | None = None) -> Decision:
    calls = []
    completed = 0
    async with asyncio.timeout(180):
        async with aclosing(provider.runtime_stream(request)) as stream:
            async for event in stream:
                if event.type is ModelStreamEventType.TOOL_CALL:
                    calls.append(event.payload)
                elif event.type is ModelStreamEventType.COMPLETED:
                    completed += 1
                elif event.type is ModelStreamEventType.ERROR:
                    raise ProbeProviderError(event.payload)
    if capture_path is not None:
        save_private_json(capture_path, {"completed": completed, "calls": calls})
    if completed != 1 or len(calls) != 1 or calls[0].get("name") != "record_decision":
        raise ValueError("Expected exactly one completed decision.")
    return Decision.model_validate(calls[0].get("arguments"))


def assess(case: Case, decision: Decision, *, retained: bool) -> bool:
    expected_step = case.expected_step if retained else case.missing_step
    expected_credit = case.expected_credit if retained else None
    return (
        decision.step == expected_step
        and decision.credit_dollars == expected_credit
        and decision.conditions == []
        and set(decision.evidence_ids) == ({"procedure"} if retained else set())
    )


async def run(output: Path) -> None:
    from cayu_tau3_banking.agent import banking_system_prompt
    from cayu_tau3_banking.cli import build_provider

    prompt = banking_system_prompt(
        "Follow the supplied fictional bank policy.", planning_mode="outcomes"
    )
    # Freeze every request before any model result; no adaptive retry or tuning.
    requests = []
    for i, case in enumerate(CASES):
        directory = output / case.id
        directory.mkdir()
        save_private_json(
            directory / "procedure.json",
            {"id": "procedure", "title": "Fictional procedure", "content": procedure(case)},
        )
        corpus = BankingKnowledgeCorpus.from_directory(directory)
        reference = corpus.delivered_revision_evidence(
            "procedure", 1, delivered_text=procedure(case)
        )
        retained_context = await active_evidence_context(corpus, (reference,))
        for retained in (False, True) if i % 2 == 0 else (True, False):
            request = build_request(
                case, system_prompt=prompt, active_context=retained_context if retained else None
            )
            name = f"{case.id}-{'retained' if retained else 'missing'}"
            save_private_json(output / f"{name}-request.json", request.model_dump(mode="json"))
            requests.append((case, retained, name, request))
    save_private_json(
        output / "protocol.json",
        {
            "scope": "synthetic paired decisions; not native-loop execution or tau score",
            "label_version": 2,
            "label_revision": "Acknowledging an explicit keep-open decision requires no policy lookup. Original v1 artifacts remain unchanged.",
            "calls": len(requests),
            "retries": 0,
            "variable": "active evidence text retention",
            "source_hashes": {
                str(Path(__file__).name): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            },
            "case_ids": [c.id for c in CASES],
        },
    )
    provider = build_provider(
        provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
    )
    rows = []
    try:
        for case, retained, name, request in requests:
            provider.preflight_portable_messages(
                model=request.model, messages=request.messages, tools=request.tools
            )
            started = time.monotonic()
            try:
                decision = await decide(
                    provider, request, capture_path=output / f"{name}-response.json"
                )
            except ValidationError as error:
                # A completed but malformed proposal is an experimental outcome,
                # not an infrastructure outage. Preserve it without normalizing
                # fields, exposing expected answers, or retrying for a better score.
                row = invalid_decision_result(case.id, retained, error, time.monotonic() - started)
                save_private_json(output / f"{name}-result.json", row)
                rows.append(row)
                print(json.dumps(row), flush=True)
                continue
            except Exception as error:
                save_private_json(
                    output / f"{name}-error.json",
                    {
                        "error_type": type(error).__name__,
                        "provider_diagnostics": error.diagnostics
                        if isinstance(error, ProbeProviderError)
                        else None,
                        "frames": [
                            {
                                "file": Path(frame.filename).name,
                                "line": frame.lineno,
                                "function": frame.name,
                            }
                            for frame in traceback.extract_tb(error.__traceback__)
                        ],
                    },
                )
                raise RuntimeError(f"Probe stopped: {name}: {type(error).__name__}") from None
            row = {
                "case": case.id,
                "retained": retained,
                "decision": decision.model_dump(),
                "passed": assess(case, decision, retained=retained),
                "seconds": time.monotonic() - started,
            }
            save_private_json(output / f"{name}-result.json", row)
            rows.append(row)
            print(json.dumps(row), flush=True)
        save_private_json(
            output / "results.json",
            {"rows": rows, "passed": sum(r["passed"] for r in rows), "total": len(rows)},
        )
    finally:
        await provider.aclose()


def invalid_decision_result(case_id, retained, error: ValidationError, seconds):
    return {
        "case": case_id,
        "retained": retained,
        "decision": None,
        "passed": False,
        "failure_kind": "invalid_decision_schema",
        "validation_errors": [
            {"type": item["type"], "loc": list(item["loc"])}
            for item in error.errors(include_input=False, include_url=False)
        ],
        "seconds": seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="policy-grounding-", dir=args.output_parent))
    print(f"Output: {output}", flush=True)
    asyncio.run(run(output))


if __name__ == "__main__":
    main()
