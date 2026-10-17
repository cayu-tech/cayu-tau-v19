import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "full_benchmark", Path(__file__).parents[1] / "scripts/run_full_benchmark.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_partial_success_is_not_a_full_score():
    result = module.summarize(["a", "b", "c"], {"a": {"reward": 1}})
    assert result["official_pass1_percent"] is None
    assert result["operational_pass_percent"] is None
    assert result["confirmed_pass_lower_bound_percent"] == pytest.approx(100 / 3)


def test_unscored_failure_stays_in_denominator():
    result = module.summarize(
        ["a", "b", "c"], {"a": {"reward": 1}, "b": {"reward": 0}, "c": {"reward": None}}
    )
    assert result["official_pass1_percent"] is None
    assert result["operational_pass_percent"] == pytest.approx(100 / 3)
    assert result["unscored"] == 1


def test_complete_official_score():
    result = module.summarize(["a", "b"], {"a": {"reward": 1.0}, "b": {"reward": 0.0}})
    assert result["official_pass1_percent"] == 50


@pytest.mark.parametrize("reward", [True, "1", 0.5, float("nan")])
def test_invalid_reward_rejected(reward):
    with pytest.raises(ValueError):
        module.summarize(["a"], {"a": {"reward": reward}})


def test_unregistered_result_rejected():
    with pytest.raises(ValueError):
        module.summarize(["a"], {"b": {"reward": 1}})


def test_archive_preserves_exact_bytes_and_leaves_official_results(tmp_path):
    (tmp_path / "contexts/one").mkdir(parents=True)
    data = b"fixture\x00\xff" * 100
    (tmp_path / "contexts/one/request.json").write_bytes(data)
    (tmp_path / "official").mkdir()
    (tmp_path / "official/results.json").write_text("untouched")
    result = module.archive_diagnostics(tmp_path)
    assert result["files"] == 1
    assert not (tmp_path / "contexts").exists()
    with zipfile.ZipFile(tmp_path / result["archive"]) as archive:
        assert archive.read("contexts/one/request.json") == data
    assert (tmp_path / "official/results.json").read_text() == "untouched"
    assert json.loads((tmp_path / "archive-manifest.json").read_text())["verified"]


def test_archive_refuses_symlink_without_deletion(tmp_path):
    (tmp_path / "outside").mkdir()
    (tmp_path / "contexts").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module.archive_diagnostics(tmp_path)
    assert (tmp_path / "outside").is_dir()


def test_failed_archive_verification_keeps_original(tmp_path, monkeypatch):
    (tmp_path / "contexts").mkdir()
    original = tmp_path / "contexts/request.json"
    original.write_text("original")
    monkeypatch.setattr(module, "digest", lambda path: "wrong")
    with pytest.raises(ValueError, match="content mismatch"):
        module.archive_diagnostics(tmp_path)
    assert original.read_text() == "original"


def test_shared_error_returns_only_classification(tmp_path):
    (tmp_path / "error.json").write_text(
        json.dumps({"type": "AuthenticationError", "private": "DO NOT OUTPUT"})
    )
    result = module.shared_failure(tmp_path)
    assert "authenticationerror" in result
    assert "DO NOT OUTPUT" not in result


def test_customer_words_cannot_trip_shared_failure_circuit(tmp_path):
    (tmp_path / "official").mkdir()
    module.write_json(
        tmp_path / "official/results.json",
        {
            "simulations": [
                {
                    "termination_reason": "user_stop",
                    "info": {},
                    "messages": [{"content": "invalid_api_key insufficient_quota"}],
                }
            ]
        },
    )
    assert module.shared_failure(tmp_path) is None


def test_official_infrastructure_error_trips_shared_failure_circuit(tmp_path):
    (tmp_path / "official").mkdir()
    module.write_json(
        tmp_path / "official/results.json",
        {
            "simulations": [
                {
                    "termination_reason": "infrastructure_error",
                    "info": {"error_type": "RateLimitError", "error": {"code": "insufficient_quota"}},
                }
            ]
        },
    )
    assert "insufficient_quota" in module.shared_failure(tmp_path)


def test_source_change_fails_verification(tmp_path):
    source = tmp_path / "actor.py"
    source.write_text("original")
    module.write_json(
        tmp_path / "source-manifest.json", {"files": {"actor.py": module.digest(source)}}
    )
    module.verify(tmp_path)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Frozen"):
        module.verify(tmp_path)


def test_finished_task_requires_matching_official_reward(tmp_path):
    task = tmp_path / "task_001"
    (task / "official").mkdir(parents=True)
    module.write_json(
        task / "assessment.json",
        {
            "source_unchanged": True,
            "rows": [
                {
                    "task": "task_001",
                    "reward": 1,
                    "termination": "user_stop",
                    "duration_seconds": 10,
                }
            ],
        },
    )
    module.write_json(
        task / "official/results.json",
        {"simulations": [{"task_id": "task_001", "reward_info": {"reward": 0}}]},
    )
    with pytest.raises(ValueError, match="mismatch"):
        module.finish_task(tmp_path, "task_001", 0, 0)


def test_launcher_failure_is_not_lost(tmp_path):
    row, reason = module.finish_task(tmp_path, "task_001", 1, module.time.time())
    assert row["reward"] is None
    assert row["termination"] == "launcher_error"
    assert reason is None
    assert (tmp_path / "task_001-outcome.json").exists()


def test_active_quota_pauses_before_task_finishes(tmp_path):
    folder = tmp_path / "task_001/provider-diagnostics/provider"
    folder.mkdir(parents=True)
    (folder / "a.error.json").write_text(json.dumps({"error": {"type": "usage_limit_reached"}}))
    assert "usage_limit_reached" in module.active_shared_failure(tmp_path, ["task_001"])
    assert not (tmp_path / "task_001/assessment.json").exists()


def test_live_partial_diagnostic_is_rechecked(tmp_path):
    folder = tmp_path / "task_001/provider-diagnostics/provider"
    folder.mkdir(parents=True)
    path = folder / "a.error.json"
    path.write_text('{"error":')
    assert module.active_shared_failure(tmp_path, ["task_001"]) is None
    path.write_text(json.dumps({"error": {"code": "insufficient_quota"}, "http_status_code": 429}))
    assert "insufficient_quota" in module.active_shared_failure(tmp_path, ["task_001"])


def test_overload_and_untyped_prose_do_not_pause_shared_dispatch(tmp_path):
    path = tmp_path / "error.json"
    for error in [
        {"error": {"type": "service_unavailable_error", "code": "server_is_overloaded"}, "http_status_code": 503},
        {"type": "ValueError", "error": "customer said usage_limit_reached"},
    ]:
        path.write_text(json.dumps(error))
        assert module.shared_failure(tmp_path) is None


def test_supervisor_does_not_launch_next_task_after_live_quota(tmp_path, monkeypatch):
    module.write_json(tmp_path / "protocol.json", {
        "task_ids": ["task_001", "task_002"], "minimum_free_disk_bytes": 0,
        "worker_watchdog_seconds": 100, "max_parallel_task_processes": 2,
    })
    module.write_json(tmp_path / "progress.json", {
        "status": "prepared", "pause_reason": None, "running": {}, "rows": {},
    })
    key = tmp_path / "key"
    key.write_text("synthetic")
    launched = []
    monkeypatch.setattr(module, "verify", lambda root: None)
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    def publish(root, protocol, state):
        state["summary"] = module.summarize(protocol["task_ids"], state["rows"])
        module.write_json(root / "progress.json", state)
    monkeypatch.setattr(module, "publish", publish)
    class Process:
        pid = 12345
        def poll(self):
            return 1
    def launch(args, **kwargs):
        launched.append(args[args.index("--task") + 1])
        folder = tmp_path / "task_001/provider-diagnostics/provider"
        folder.mkdir(parents=True)
        (folder / "quota.error.json").write_text(json.dumps({"error": {"type": "usage_limit_reached"}}))
        return Process()
    monkeypatch.setattr(module.subprocess, "Popen", launch)
    module.run(tmp_path, key)
    state = json.loads((tmp_path / "progress.json").read_text())
    assert launched == ["task_001"]
    assert state["status"] == "paused"
    assert state["rows"]["task_001"]["reward"] is None
    assert (tmp_path / "task_001/diagnostics.zip").exists()


@pytest.mark.parametrize("error", [
    {"error": {}, "http_status_code": 429},
    {"error": {"type": "rate_limit_error"}, "http_status_code": 429},
    {"error_type": "RateLimitError"},
    {"error": {"code": "rate_limit_exceeded"}, "retry_after_s": 13},
])
def test_recoverable_throttle_cannot_latch_dispatch_pause(tmp_path, error):
    folder = tmp_path / "task_001/provider-diagnostics/provider"
    folder.mkdir(parents=True)
    (folder / "old.error.json").write_text(json.dumps(error))
    assert module.active_shared_failure(tmp_path, ["task_001"]) is None
    assert module.shared_failure(tmp_path / "task_001") is None


def test_supervisor_keeps_launching_after_recovered_429(tmp_path, monkeypatch):
    module.write_json(tmp_path / "protocol.json", {
        "task_ids": ["task_001", "task_002"], "minimum_free_disk_bytes": 0,
        "worker_watchdog_seconds": 100, "max_parallel_task_processes": 1,
        "pause_after_consecutive_unscored": 1,
    })
    module.write_json(tmp_path / "progress.json", {
        "status": "prepared", "pause_reason": None, "running": {}, "rows": {},
    })
    key = tmp_path / "key"
    key.write_text("synthetic")
    launched = []
    monkeypatch.setattr(module, "verify", lambda root: None)
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    def publish(root, protocol, state):
        state["summary"] = module.summarize(protocol["task_ids"], state["rows"])
        module.write_json(root / "progress.json", state)
    monkeypatch.setattr(module, "publish", publish)
    class Process:
        pid = 12345
        def poll(self):
            return 0
    def launch(args, **kwargs):
        task = args[args.index("--task") + 1]
        launched.append(task)
        folder = tmp_path / task / "provider-diagnostics/provider"
        folder.mkdir(parents=True)
        (folder / "throttle.error.json").write_text(json.dumps({"http_status_code": 429}))
        return Process()
    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(module, "finish_task", lambda root, task, *args: (
        {"reward": 1, "termination": "user_stop"}, module.shared_failure(root / task)))
    module.run(tmp_path, key)
    state = json.loads((tmp_path / "progress.json").read_text())
    assert launched == ["task_001", "task_002"]
    assert state["pause_reason"] is None
    assert len(state["rows"]) == 2
