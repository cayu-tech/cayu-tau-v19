"""Pause dispatch after the first complete trial, without changing upstream results.

The upstream batch still owns task loading, four-trial metadata, seeds, execution,
grading, and checkpoints. This process-local executor adapter only delays trial 2+
before the upstream task function creates an agent. It is never actor-visible.
"""

import json
import random
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import contextmanager

from run_full_benchmark import digest, write_json


def first_trial_status(root):
    try:
        protocol = json.loads((root / "protocol.json").read_text())
        result = json.loads((root / "official/results.json").read_text())
        tasks = [task["id"] for task in result["tasks"]]
        rows = [sim for sim in result["simulations"] if sim["trial"] == 0]
        if (len(tasks) != 97 or len(set(tasks)) != 97 or len(rows) != 97
                or {row["task_id"] for row in rows} != set(tasks)
                or any(row["seed"] != protocol["trial_seeds"][0] for row in rows)):
            return None
        rewards = [(row.get("reward_info") or {}).get("reward") for row in rows]
        if any(value not in (0, 1, None) for value in rewards):
            return None
        return {"trial": 1, "recorded": 97, "passed": rewards.count(1),
                "scored_failures": rewards.count(0), "unscored": rewards.count(None)}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def continuation_allowed(root, manifest=None):
    try:
        receipt = json.loads((root / "continue-after-trial-1.json").read_text())
        return (
            isinstance(receipt, dict)
            and type(receipt.get("continue_after_trial")) is int
            and receipt["continue_after_trial"] == 1
            and receipt.get("source_manifest_sha256")
            == (manifest or digest(root / "source-manifest.json"))
        )
    except (OSError, ValueError, TypeError):
        return False


class TrialBoundaryExecutor(ThreadPoolExecutor):
    def __init__(self, root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.manifest = digest(root / "source-manifest.json")
        self.cancel_boundary = threading.Event()
        self.boundary_lock = threading.Lock()
        self.boundary = None
        self.last_check = float("-inf")

    def submit(self, fn, /, *args, **kwargs):
        # This is the pinned batch's (task, trial, seed, progress) contract.
        if len(args) != 4 or type(args[1]) is not int or args[1] not in range(4):
            raise ValueError("Unexpected upstream trial dispatch contract")

        def dispatch():
            if args[1] > 0:
                while not self.cancel_boundary.is_set():
                    with self.boundary_lock:
                        if self.boundary is None and time.monotonic() - self.last_check >= 1:
                            self.last_check = time.monotonic()
                            self.boundary = first_trial_status(self.root)
                            if self.boundary is not None:
                                path = self.root / "trial-1-boundary.json"
                                if not path.exists():
                                    write_json(path, {**self.boundary,
                                        "source_manifest_sha256": self.manifest,
                                        "checked_unix": time.time(),
                                        "scope": "Dispatch pause only; all outcomes preserved"})
                    if self.boundary is not None and continuation_allowed(self.root, self.manifest):
                        break
                    self.cancel_boundary.wait(1)
                if self.cancel_boundary.is_set():
                    raise CancelledError("Trial-boundary dispatch cancelled before task creation")
            return fn(*args, **kwargs)

        return super().submit(dispatch)

    def shutdown(self, wait=True, *, cancel_futures=False):
        # Upstream interrupt handling uses cancel_futures=True. Release waiting
        # workers without constructing agents or inventing simulation outcomes.
        if cancel_futures or not wait:
            self.cancel_boundary.set()
        return super().shutdown(wait=wait, cancel_futures=cancel_futures)


@contextmanager
def staged_trial_dispatch(root, protocol):
    if protocol.get("pause_after_trial") != 1:
        yield
        return
    if protocol.get("trials") != 4 or len(protocol.get("trial_seeds", [])) != 4:
        raise ValueError("Staged evaluation requires four predeclared trials and seeds")
    rng = random.Random(protocol["seed"])
    if protocol["trial_seeds"] != [rng.randint(0, 1000000) for _ in range(4)]:
        raise ValueError("Registered seeds do not match the pinned upstream trial schedule")
    from tau2.runner import batch

    original = batch.ThreadPoolExecutor
    batch.ThreadPoolExecutor = lambda *a, **kw: TrialBoundaryExecutor(root, *a, **kw)
    try:
        yield
    finally:
        batch.ThreadPoolExecutor = original
