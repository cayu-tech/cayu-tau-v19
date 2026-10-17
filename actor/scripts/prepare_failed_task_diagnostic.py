"""Prepare an isolated current-source run for selected historical τ failures."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import argparse
from pathlib import Path


# Selected after the full97 replay audit. Excluded are known reference/simulator
# disputes (017, 038, 046, 053, 070, 077, 078, 079, 084, 090, 102) and terminal
# infrastructure failures. These ten are diagnostic candidates where the replay
# shows a concrete missing, wrong, or incomplete business outcome.
TASKS = (
    "task_002", "task_029", "task_039", "task_044", "task_045",
    "task_051", "task_054", "task_059", "task_062", "task_073",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated", action="store_true")
    options = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    census = project / "artifacts/inventory-full97-7b478p50"
    prior = json.loads((census / "protocol.json").read_text())
    root = Path(tempfile.mkdtemp(prefix="failed-task-diagnostic-", dir=project / "artifacts"))
    for name in ("src", "tests", "scripts"):
        shutil.copytree(project / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(project / name, root / name)
    shutil.copy2(project / "scripts/run_lean_advice_task.py", root / "run_pilot.py")
    if options.curated:
        shutil.copy2(project / "artifacts/curated-task-020-context.txt", root / "curated-task-020-context.txt")
    (root / ".vendor").symlink_to((project / ".vendor").resolve(), target_is_directory=True)
    args = list(prior["cli_arguments"])
    provider_index = args.index("--provider")
    args[provider_index + 1] = "subscription"
    if options.curated:
        args.extend(["--curated-context-file", "curated-task-020-context.txt"])
    protocol = {
        "task_ids": list(TASKS),
        "cli_arguments": args,
        "tau_repository": prior["tau_repository"],
        "corpus_embedding_cache": prior["corpus_embedding_cache"],
        "comparison": (
            "Historical failures only; curated-context ablation, not census scoring."
            if options.curated
            else "Historical failures only; exploratory subscription-provider diagnostic, not census scoring."
        ),
        "source_sha256": {str(p.relative_to(root)): digest(p) for p in root.rglob("*.py") if ".vendor" not in p.parts},
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    files = {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file() and ".vendor" not in p.parts}
    (root / "source-manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    print(json.dumps({"root": str(root), "tasks": TASKS, "provider": "subscription", "curated": options.curated, "source_manifest": digest(root / "source-manifest.json")}))


if __name__ == "__main__":
    main()
