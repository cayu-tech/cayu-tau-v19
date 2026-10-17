"""Single-factor synthetic four-versus-eight document read comparison."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from cayu_tau3_banking import product_navigation
from cayu_tau3_banking.comparison_workflow_probe import catalogue_case
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.navigation_probe import POLICY, run_case
from cayu_tau3_banking.provenance import CAYU_COMMIT
from cayu_tau3_banking.runtime_profile import profile_session


def main():
    from cayu_tau3_banking.cli import build_provider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-limit", type=int, choices=(4, 8), required=True)
    args = parser.parse_args()
    product_navigation.MAX_READ_DOCUMENTS = args.read_limit
    documents, case, expected = catalogue_case()
    root = Path(tempfile.mkdtemp(prefix=f"read-batch-{args.read_limit}-", dir="artifacts"))
    source = Path(__file__).resolve().parent
    shutil.copytree(source, root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")}
    save_private_json(
        root / "protocol.json",
        {
            "read_limit": args.read_limit,
            "source_hashes": hashes,
            "framework_commit": CAYU_COMMIT,
            "case": case,
            "documents": documents,
            "policy": POLICY,
            "expected_evaluator_only": expected,
            "model": "subscription gpt-5.6-sol xhigh",
            "attempts": 1,
            "max_result_bytes": product_navigation.MAX_READ_RESULT_BYTES,
            "limits": "Synthetic catalogue comparison, native compaction policy. No tau simulator "
            "or score. Same source for both arms; only the read-schema count differs. "
            "One pair is exploratory, not statistical proof. No adaptive retries.",
        },
    )
    print(json.dumps({"artifacts": str(root)}), flush=True)
    result = run_case(
        root,
        case,
        documents=documents,
        policy=POLICY,
        provider_factory=lambda: build_provider(
            provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
        ),
    )
    assert hashes == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")
    }
    profile = profile_session(root / "runtime.sqlite3")
    save_private_json(root / "profile.json", profile)
    print(
        json.dumps(
            {
                "read_limit": args.read_limit,
                "seconds": result["seconds"],
                "turns": result["turns"],
                "profile": profile,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
