"""Opt-in regression for the promoted needs instruction; historical trials frozen."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from cayu_tau3_banking.agent import PRODUCT_NEEDS_INSTRUCTIONS
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.navigation_probe import run_case
from cayu_tau3_banking.needs_notice_probe import ADVICE_POLICY, DOCUMENTS, NEEDS

COMPLETE = {
    "id": "already_complete",
    "customer": "Compare Cedar and Harbor for $2000 left untouched for one year. I bank entirely online, never deposit cash, and want the highest interest. Advice only. Which one?",
    "expected": "Direct Cedar recommendation, $84 versus $72; no unnecessary clarification prerequisite.",
}


def main():
    from cayu_tau3_banking.cli import build_provider

    repo = Path(__file__).resolve().parents[2]
    root = Path(tempfile.mkdtemp(prefix="needs-treatment-", dir=repo / "artifacts"))
    shutil.copytree(repo / "src", root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    save_private_json(
        root / "protocol.json",
        {
            "scope": "Integrated agent needs-instruction regression; not tau",
            "cases": [*NEEDS, COMPLETE],
            "policy": ADVICE_POLICY,
            "agent_needs_instruction": PRODUCT_NEEDS_INSTRUCTIONS,
            "documents": DOCUMENTS,
            "model": "gpt-5.6-sol",
            "provider": "subscription",
            "effort": "xhigh",
            "attempts_per_case": 1,
            "baseline": "needs-notice-ar5sybtk",
            "expectation": "Initial incomplete requests elicit needs; final requests choose the appropriate product; complete input is answered without required clarification.",
            "limits": "Small unblinded development comparison; no statistical or benchmark improvement claim. Followups are scripted, not a simulator responsiveness test.",
            "source_hashes": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((root / "source").rglob("*.py"))
            },
        },
    )
    print(f"Output: {root}", flush=True)
    for case in (*NEEDS, COMPLETE):
        output = root / case["id"]
        output.mkdir(mode=0o700)
        result = run_case(
            output,
            case,
            documents=DOCUMENTS,
            policy=ADVICE_POLICY,
            provider_factory=lambda: build_provider(
                provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
            ),
        )
        print(json.dumps({"case": case["id"], "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
