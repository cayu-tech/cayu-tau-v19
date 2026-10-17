"""Offline SQLite/Cayu microbenchmark: fixed scripted calls, no model/API access.

Run each mode in a separate process. This measures local runtime overhead only,
not subscription service latency or benchmark score. Artifacts are preserved.
"""

import argparse
import concurrent.futures
import json
import multiprocessing
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))


def worker(args):
    from tau2.data_model.message import UserMessage
    from test_runtime_banking import assembly, call, text

    root, index, rounds = args
    folder = Path(root) / f"task-{index:03d}"
    folder.mkdir()
    steps = [call("cayu_browse_product_documents", {}, f"browse-{i}") for i in range(rounds)]
    steps.append(text("Diagnostic finished."))
    start = time.perf_counter()
    agent, provider, _ = assembly(
        folder,
        steps,
        persistent=True,
        planning_mode="outcomes",
        advice_mode="lean",
        controller_mode="advisory",
        callback_timeout_s=600,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("List the public categories."), state
        )
        assert reply.content == "Diagnostic finished."
        return {
            "index": index,
            "elapsed_seconds": time.perf_counter() - start,
            "model_calls": len(provider.requests),
        }
    finally:
        state.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["threads", "processes"], required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.workers <= 8 and 1 <= args.tasks <= 32 and 1 <= args.rounds <= 24):
        parser.error("Bounds: workers 1-8, tasks 1-32, rounds 1-24")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    pool = (
        concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
        if args.mode == "threads"
        else concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        )
    )
    with pool:
        results = list(
            pool.map(worker, [(str(args.output), i, args.rounds) for i in range(args.tasks)])
        )
    report = {
        "mode": args.mode,
        "workers": args.workers,
        "tasks": args.tasks,
        "rounds": args.rounds,
        "wall_seconds_including_startup": time.perf_counter() - started,
        "results": results,
        "scope": "scripted local runtime, no live model or benchmark grade",
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
