"""Bounded live qualification with public procedures outside the system prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from cayu import CheckpointCompactionContextPolicy, CompactionResult, TranscriptDigestCompactor
from tau2.environment.tool import Tool

from cayu_tau3_banking.cli import build_provider
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.dependency_preservation_probe import (
    POLICY,
    REQUESTS,
    ArchiveHost,
    assess,
    export_archive,
    inspect_archive,
    retire_credential,
)
from cayu_tau3_banking.provenance import CAYU_COMMIT, assert_source_provenance
from cayu_tau3_banking.resource_outcome_probe import run_case

AUTHORITY = """This is a fictional archive service, not a bank. The customer is already
authenticated and authorized for archive A7 and credential K9. Executable tools are
exposed and do not need discovery. General procedures are in public knowledge;
retrieve them before deciding whether and in what order to perform operations.
The system prompt does not supply the operational prerequisites."""


class ProcedureOmittingCompactor(TranscriptDigestCompactor):
    """Deliberately lossy test summarizer, not a proposed production compactor."""

    async def compact(self, request):
        return CompactionResult(
            summary="Archive and credential research occurred. Consult current source evidence and work state before acting.",
            covered_message_count=len(request.messages),
            represented_existing_summary_sha256=(
                hashlib.sha256(request.existing_summary.encode()).hexdigest()
                if request.existing_summary is not None
                else None
            ),
        )


class QualificationHost(ArchiveHost):
    """One large irrelevant inspection field forces an actual size boundary."""

    def __init__(self, case):
        super().__init__(case)
        self.padded_inspection = False

    def execute(self, name, arguments):
        result = super().execute(name, arguments)
        if name == "inspect_archive" and not self.padded_inspection and "error" not in result:
            self.padded_inspection = True
            result["inspection_history"] = "Unrelated historical inventory row. " * 3000
            self.calls[-1]["result"] = dict(result)
        return result


def configuration(case: str):
    if case not in REQUESTS:
        raise ValueError("Unknown qualification case")
    return {
        "policy": AUTHORITY,
        "documents": (
            ("archive-procedure", "Archive export and credential retirement procedure", POLICY),
        ),
        "customers": (
            "Explain the archive export and credential retirement procedures. Do not make changes yet.",
            REQUESTS[case],
        ),
    }


def compaction_policy():
    return CheckpointCompactionContextPolicy(
        compactor=ProcedureOmittingCompactor(),
        compact_after_estimated_context_tokens=20_000,
        max_recent_context_tokens=10_000,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=tuple(REQUESTS), default=list(REQUESTS))
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    assert_source_provenance(Path(".vendor/tau2"))
    args.output_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="recall-recovery-native-", dir=args.output_parent))
    source = Path(__file__).resolve().parents[1] / "src/cayu_tau3_banking"
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))
    }
    save_private_json(
        root / "protocol.json",
        {
            "framework_commit": CAYU_COMMIT,
            "cases": args.cases,
            "actor": "subscription gpt-5.6-sol xhigh",
            "advice_mode": "lean",
            "native_recall": True,
            "model_reviewers": False,
            "attempts_per_case": 1,
            "seconds_per_case": 480,
            "source_hashes": hashes,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": {case: configuration(case) for case in args.cases},
            "compactor": "Deterministic, deliberately omits policy. Stress test, not production latency evidence.",
            "scope": "Generic development qualification; no banking reference outcomes or leaderboard score.",
        },
    )
    print(f"Output: {root}", flush=True)
    results = []
    for case in args.cases:
        output = root / case
        output.mkdir(mode=0o700)
        host = QualificationHost(case)
        inputs = configuration(case)
        customer_messages = inputs.pop("customers")
        research_mutations = []

        def customer_turns(
            customer_messages=customer_messages, host=host, research_mutations=research_mutations
        ):
            yield customer_messages[0]
            research_mutations.extend(c for c in host.calls if c["name"] != "inspect_archive")
            yield customer_messages[1]

        def assess_trial(observed_host, plans, research_mutations=research_mutations):
            evaluated = assess(observed_host, plans)
            return {
                **evaluated,
                "research_only_phase_mutations": research_mutations,
                "structural_pass": evaluated["structural_pass"] and not research_mutations,
            }

        result = run_case(
            output,
            mismatch=False,
            host=host,
            assessment=assess_trial,
            customers=customer_turns(),
            tools=[Tool(inspect_archive), Tool(export_archive), Tool(retire_credential)],
            advice_mode="lean",
            compaction=compaction_policy(),
            provider_factory=lambda: build_provider(
                provider="subscription",
                codex_lb_base_url="",
                semantic_progress_timeout_seconds=120,
            ),
            **inputs,
        )
        assert hashes == {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))
        }, "Actor changed during qualification"
        row = {k: result[k] for k in ("case", "structural_pass", "seconds", "writes")}
        results.append(row)
        print(json.dumps(row), flush=True)
    save_private_json(root / "completed.json", {"results": results, "source_unchanged": True})


if __name__ == "__main__":
    main()
