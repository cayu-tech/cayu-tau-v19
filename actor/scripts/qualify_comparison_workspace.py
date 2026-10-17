"""Frozen fictional comparisons with actual lossy compaction, not a tau score."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from cayu import CheckpointCompactionContextPolicy, CompactionResult, TranscriptDigestCompactor

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.provenance import CAYU_COMMIT, assert_source_provenance
from cayu_tau3_banking.resource_outcome_probe import run_case

CASES = ("helpful", "costly", "ineligible", "declined", "unknown_fee")
POLICY = """Fictional cloud storage service. Public catalogue documents establish
current product terms. The customer's supplied seat count is authoritative; no
banking verification applies. A companion subscription is a separate product.
Unpublished terms are unknown. Advice does not authorize subscription changes.
This service has no exposed provisioning tools in this advice-only consultation.
Use usable storage and total monthly cost when comparing setups."""


def configuration(case):
    if case not in CASES:
        raise ValueError("Unknown qualification case")
    fee = (
        "Its monthly fee is not published."
        if case == "unknown_fee"
        else (f"Its monthly fee is ${15 if case == 'costly' else 5}.")
    )
    seats = 10 if case == "ineligible" else 4
    documents = [
        (
            "cinder",
            "Cinder base storage and linked benefits",
            "Cinder provides 100 GB usable storage for $20 per month. It has no seat minimum. "
            "The separately subscribed Ember companion adds 60 GB of usable storage to Cinder. "
            "Its own charges and eligibility requirements are in the Ember companion terms. "
            "A setup has exactly one base plan, not multiple base subscriptions. "
            "At most one Ember subscription can attach to it; Ember's bonus applies once. "
            "There are no other Cinder benefits, reservations or charges.",
        ),
        (
            "maple",
            "Maple storage terms",
            "Maple provides 130 GB usable storage for $22 per month. No seat minimum, "
            "companion benefits, reservations or other charges apply. "
            "A setup has exactly one base plan, not multiple base subscriptions.",
        ),
        (
            "ember",
            "Ember companion terms",
            f"Ember is a separate companion subscription. {fee} It requires at least {seats} "
            "seats. It adds 60 GB usable storage to Cinder only. Its cost is in addition "
            "to Cinder's base fee. At most one Ember subscription can attach to a base plan; "
            "its bonus applies once. There are no other benefits, reservations or charges.",
        ),
    ]
    refusal = (
        " I will not subscribe to any additional companion service." if case == "declined" else ""
    )
    return {
        "policy": POLICY,
        "documents": documents,
        "customers": [
            "For an initial standalone comparison only, compare Cinder and Maple's usable "
            "storage and monthly cost. This is not a product selection or a refusal of "
            "companions for a later comparison. Advice only; do not change subscriptions.",
            "Now compare available setups for the most usable storage within $30 total per "
            "month. I currently have Cinder, six seats and no companions. Keep my seat count "
            "fixed at six. Explain the decisive costs and conditions. Advice only; do not "
            "change any subscriptions." + refusal,
        ],
    }


def expectation(case):
    return {
        "helpful": "Recommend Cinder + Ember: 160 GB for $25; six seats satisfy four-seat minimum. No subscription changes.",
        "costly": "Recommend Maple: 130 GB for $22. Cinder + Ember costs $35, exceeding $30.",
        "ineligible": "Recommend Maple: 130 GB for $22. Six seats do not meet Ember's ten-seat minimum; do not increase seats.",
        "declined": "Recommend Maple: 130 GB for $22. Respect companion refusal even though Cinder + Ember has more storage.",
        "unknown_fee": "No unconditional overall winner: Cinder + Ember has 160 GB but unknown total cost. It wins if Ember costs at most $10; Maple at $22/130 GB is the best verified affordable setup.",
    }[case]


class LossyCompactor(TranscriptDigestCompactor):
    """Stress fixture: omits factual conclusions, never used by the banking actor."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def compact(self, request):
        self.calls += 1
        return CompactionResult(
            summary="Earlier advice was provisional. Consult current sources, customer instructions and retained work state.",
            covered_message_count=len(request.messages),
            represented_existing_summary_sha256=(
                hashlib.sha256(request.existing_summary.encode()).hexdigest()
                if request.existing_summary is not None
                else None
            ),
        )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(parent):
    repo = Path(__file__).resolve().parents[1]
    assert_source_provenance(repo / ".vendor/tau2")
    root = Path(tempfile.mkdtemp(prefix="comparison-workspace-", dir=parent)).resolve()
    shutil.copytree(
        repo / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(__file__, root / "run.py")
    save_private_json(root / "cases.json", {name: configuration(name) for name in CASES})
    save_private_json(
        root / "expectations-evaluator-only.json", {name: expectation(name) for name in CASES}
    )
    save_private_json(
        root / "protocol.json",
        {
            "framework_commit": CAYU_COMMIT,
            "tau_repository": str((repo / ".vendor/tau2").resolve()),
            "actor": "subscription gpt-5.6-sol xhigh",
            "cases": CASES,
            "attempts_per_case": 1,
            "max_concurrency": 2,
            "seconds_per_case": 480,
            "seconds_per_callback": 240,
            "model_steps_per_callback": 24,
            "tool_calls_per_callback": 64,
            "advice_mode": "lean",
            "native_recall": True,
            "model_reviewers": False,
            "source_of_terms": "Three fictional documents in Cayu knowledge, not the system/customer prompt. Lexical retrieval; no embedding or simulator API spend.",
            "compaction": "Actual checkpoint compaction after the first customer turn, with a deliberately lossy deterministic summarizer. Not production compaction latency or quality evidence.",
            "grading": "Manual review of final answer, retrieved evidence and worksheets against evaluator-only expectations. No effects is not a semantic pass. No feedback reaches the actor.",
            "limits": "Candidate qualification on synthetic cases, not a matched baseline comparison, tau result or success-rate estimate. Do not launch a banking trial unless all controls pass.",
        },
    )
    save_private_json(
        root / "manifest.json",
        {
            "files": {
                str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()
            }
        },
    )
    return root


def verify(root):
    for name, expected in json.loads((root / "manifest.json").read_text())["files"].items():
        if digest(root / name) != expected:
            raise ValueError("Frozen qualification source or input changed")


def run(root, case, *, provider_factory=None):
    verify(root)
    inputs = json.loads((root / "cases.json").read_text())[case]
    output = root / case
    output.mkdir(mode=0o700)
    if provider_factory is None:
        import cayu_tau3_banking.agent_memory as memory
        from cayu_tau3_banking.cli import build_provider
        from cayu_tau3_banking.provider_diagnostics import DiagnosticOpenAITransport

        if not Path(memory.__file__).resolve().is_relative_to(root / "src"):
            raise ValueError("Actor must be imported from frozen sources")
        protocol = json.loads((root / "protocol.json").read_text())
        assert_source_provenance(Path(protocol["tau_repository"]))

        def provider_factory():
            provider = build_provider(
                provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=180
            )
            provider.transport = DiagnosticOpenAITransport(
                provider.transport, output / "provider-diagnostics"
            )
            return provider

    compactor = LossyCompactor()
    result = run_case(
        output,
        mismatch=False,
        tools=[],
        advice_mode="lean",
        assessment=lambda h, plans: {
            "no_effects": not h.calls,
            "semantic_grade": "pending_manual_review",
        },
        provider_factory=provider_factory,
        compaction=CheckpointCompactionContextPolicy(
            compactor=compactor,
            compact_after_messages=2,
            max_user_turns=1,
        ),
        conversation_timeout_s=480,
        **inputs,
    )
    verify(root)
    save_private_json(
        output / "qualification.json",
        {
            "case": case,
            "actual_compactions": compactor.calls,
            "source_unchanged": True,
            "seconds": result["seconds"],
            "no_effects": result["no_effects"],
            "semantic_grade": "pending_manual_review",
        },
    )
    print(
        json.dumps(
            {"case": case, "seconds": result["seconds"], "actual_compactions": compactor.calls}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    if args.prepare is not None:
        print(prepare(args.prepare))
    elif args.root is not None and args.case is not None:
        run(args.root.resolve(), args.case)
    else:
        parser.error("Use --prepare PARENT or --root ROOT --case CASE")


if __name__ == "__main__":
    main()
