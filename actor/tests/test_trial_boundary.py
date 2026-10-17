"""A staged official campaign preserves upstream tasks, seeds and scored zeros."""

import importlib
import json
import random
import threading
import time
from concurrent.futures import CancelledError
from pathlib import Path

import pytest


@pytest.fixture
def stage(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    module = importlib.import_module("trial_boundary")
    rng = random.Random(300)
    protocol = {"trials": 4, "pause_after_trial": 1, "seed": 300,
                "trial_seeds": [rng.randint(0, 1000000) for _ in range(4)]}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "source-manifest.json").write_text('{}')
    (tmp_path / "official").mkdir()
    tasks = [{"id": f"synthetic_{i}"} for i in range(97)]
    rows = [{"task_id": t["id"], "trial": 0, "seed": protocol["trial_seeds"][0],
             "reward_info": {"reward": i % 2}} for i, t in enumerate(tasks)]
    result = {"tasks": tasks, "simulations": rows}
    (tmp_path / "official/results.json").write_text(json.dumps(result))
    return module, protocol, result


@pytest.mark.parametrize("damage", ["missing", "duplicate", "wrong_seed", "partial_write"])
def test_incomplete_trial_does_not_open_boundary(stage, tmp_path, damage):
    module, _, result = stage
    assert module.first_trial_status(tmp_path) == {
        "trial": 1, "recorded": 97, "passed": 48, "scored_failures": 49, "unscored": 0}
    if damage == "missing":
        result["simulations"].pop()
    elif damage == "duplicate":
        result["simulations"][-1] = result["simulations"][0]
    elif damage == "wrong_seed":
        result["simulations"][0]["seed"] = -1
    path = tmp_path / "official/results.json"
    path.write_text('{"simulations":' if damage == "partial_write" else json.dumps(result))
    assert module.first_trial_status(tmp_path) is None


def test_pause_retains_infrastructure_and_scored_failures(stage, tmp_path):
    module, _, result = stage
    result["simulations"][0]["reward_info"] = None
    (tmp_path / "official/results.json").write_text(json.dumps(result))
    assert module.first_trial_status(tmp_path) == {
        "trial": 1, "recorded": 97, "passed": 48, "scored_failures": 48, "unscored": 1}


def test_later_trial_waits_without_constructing_agent_or_changing_args(stage, tmp_path):
    module, protocol, _ = stage
    received = []
    original = (tmp_path / "official/results.json").read_bytes()
    def execute(*args):
        received.append(args)
        return args
    pool = module.TrialBoundaryExecutor(tmp_path, max_workers=2)
    later = (object(), 1, protocol["trial_seeds"][1], "trial 2")
    try:
        queued = pool.submit(execute, *later)
        first = pool.submit(execute, object(), 0, protocol["trial_seeds"][0], "trial 1")
        assert first.result(timeout=2)[1] == 0
        assert len(received) == 1 and not queued.done()
        (tmp_path / "continue-after-trial-1.json").write_text(json.dumps({
            "continue_after_trial": 1, "source_manifest_sha256": "wrong-candidate"}))
        assert not module.continuation_allowed(tmp_path)
        (tmp_path / "continue-after-trial-1.json").write_text(json.dumps({
            "continue_after_trial": 1, "source_manifest_sha256": pool.manifest}))
        assert queued.result(timeout=3) == later
        assert received[-1][0] is later[0]
        assert (tmp_path / "official/results.json").read_bytes() == original
    finally:
        pool.shutdown(cancel_futures=True)


def test_interrupt_releases_waiters_without_running_task(stage, tmp_path):
    module, protocol, _ = stage
    called = threading.Event()
    pool = module.TrialBoundaryExecutor(tmp_path, max_workers=1)
    queued = pool.submit(lambda *args: called.set(), object(), 1, protocol["trial_seeds"][1], "later")
    pool.shutdown(wait=True, cancel_futures=True)
    with pytest.raises(CancelledError):
        queued.result()
    assert not called.is_set()


def test_changed_seed_schedule_is_rejected_before_dispatch(stage, tmp_path):
    module, protocol, _ = stage
    protocol["trial_seeds"][1] += 1
    with pytest.raises(ValueError, match="seeds"), module.staged_trial_dispatch(tmp_path, protocol):
        pytest.fail("Invalid schedule admitted")


def test_actual_upstream_batch_stages_all_388_and_resume_skips_scored_zeros(stage, tmp_path, monkeypatch):
    from uuid import uuid4

    from tau2.data_model.simulation import RewardInfo, SimulationRun, TextRunConfig
    from tau2.data_model.tasks import Task
    from tau2.environment.environment import EnvironmentInfo
    from tau2.runner import batch, helpers

    module, protocol, _ = stage
    # Start from an empty actual upstream checkpoint, not the unit-test fixture.
    path = tmp_path / "official/results.json"
    path.unlink()
    tasks = [Task(id=f"synthetic_{i}", user_scenario={"instructions": "Synthetic test."}) for i in range(97)]
    calls = []
    def simulate(config, task, *, seed, **kwargs):
        calls.append((task.id, seed))
        return SimulationRun(id=uuid4().hex, task_id=task.id, seed=seed,
            start_time="2026-01-01T00:00:00", end_time="2026-01-01T00:00:01", duration=1,
            messages=[], termination_reason="user_stop", reward_info=RewardInfo(reward=0))
    monkeypatch.setattr(batch, "run_single_task", simulate)
    monkeypatch.setattr(helpers, "get_environment_info", lambda *a, **kw: EnvironmentInfo(domain_name="mock", policy="Synthetic policy."))
    config = TextRunConfig(domain="mock", num_trials=4, seed=300, max_concurrency=8,
                           max_retries=0, auto_resume=True)
    observed = []
    def inspect_and_continue():
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if module.first_trial_status(tmp_path) is not None:
                observed.append((len(calls), json.loads(path.read_text())))
                (tmp_path / "continue-after-trial-1.json").write_text(json.dumps({
                    "continue_after_trial": 1,
                    "source_manifest_sha256": module.digest(tmp_path / "source-manifest.json")}))
                return
            time.sleep(0.05)
        observed.append("boundary_not_reached")
        # Release on test failure so pytest does not hang indefinitely.
        (tmp_path / "continue-after-trial-1.json").write_text(json.dumps({
            "continue_after_trial": 1,
            "source_manifest_sha256": module.digest(tmp_path / "source-manifest.json")}))
    inspector = threading.Thread(target=inspect_and_continue, daemon=True)
    inspector.start()
    original_executor = batch.ThreadPoolExecutor
    with module.staged_trial_dispatch(tmp_path, protocol):
        result = batch.run_tasks(config, tasks, save_path=path, console_display=False)
    inspector.join(timeout=2)
    assert batch.ThreadPoolExecutor is original_executor
    assert observed and observed[0][0] == 97
    assert observed[0][1]["info"]["num_trials"] == 4
    assert {s["trial"] for s in observed[0][1]["simulations"]} == {0}
    assert len(result.simulations) == 388 and len(calls) == 388
    expected = {(task.id, trial, seed) for task in tasks for trial, seed in enumerate(protocol["trial_seeds"])}
    assert {(s.task_id, s.trial, s.seed) for s in result.simulations} == expected
    with module.staged_trial_dispatch(tmp_path, protocol):
        resumed = batch.run_tasks(config, tasks, save_path=path, console_display=False)
    assert len(calls) == 388 and len(resumed.simulations) == 388
    assert all(s.reward_info.reward == 0 for s in resumed.simulations)
