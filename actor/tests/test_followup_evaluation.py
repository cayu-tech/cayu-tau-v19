"""No cloud, credentials or model requests: fail-closed follow-up launch gates."""

import hashlib
import importlib.util
import json
import signal
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "followup_runner", Path(__file__).resolve().parents[1] / "scripts/run_followup_evaluation.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def ready(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "source.py").write_text("immutable")
    manifest = {"files": {"source.py": hashlib.sha256(b"immutable").hexdigest()}}
    (root / "source-manifest.json").write_text(json.dumps(manifest))
    digest = hashlib.sha256((root / "source-manifest.json").read_bytes()).hexdigest()
    (root / "linux-qualified.json").write_text(
        json.dumps({"passed": True, "manifest_sha256": digest})
    )
    (root / "connectivity.json").write_text(json.dumps({"passed": True}))
    census = tmp_path / "census.json"
    census.write_text(
        json.dumps({"status": "completed", "running": {}, "summary": {"total": 97, "finished": 97}})
    )
    return root, census


def test_requires_complete_census_not_only_inactive_workers(tmp_path):
    root, census = ready(tmp_path)
    runner.require_ready(root, census)
    data = json.loads(census.read_text())
    data["summary"]["finished"] = 96
    census.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="census"):
        runner.require_ready(root, census)


@pytest.mark.parametrize("field,value", [("status", "running"), ("running", {"task": {}})])
def test_active_census_cannot_launch_followup(tmp_path, field, value):
    root, census = ready(tmp_path)
    data = json.loads(census.read_text())
    data[field] = value
    census.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="census"):
        runner.require_ready(root, census)


@pytest.mark.parametrize("change", ["source", "receipt", "connectivity", "symlink"])
def test_source_or_qualification_mismatch_fails_closed(tmp_path, change):
    root, census = ready(tmp_path)
    if change == "source":
        (root / "source.py").write_text("changed")
    elif change == "receipt":
        (root / "linux-qualified.json").write_text('{"passed":true,"manifest_sha256":"old"}')
    elif change == "connectivity":
        (root / "connectivity.json").write_text('{"passed":false}')
    else:
        outside = tmp_path / "outside.py"
        (root / "source.py").rename(outside)
        (root / "source.py").symlink_to(outside)
    with pytest.raises(ValueError):
        runner.require_ready(root, census)


def test_timeout_is_not_retried_and_only_its_process_group_is_stopped(tmp_path, monkeypatch):
    class Worker:
        pid = 12345
        waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits <= 2:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return -signal.SIGKILL

    worker = Worker()
    calls, signals = [], []

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        return worker

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    assert runner.execute(["fixture"], tmp_path / "run.log", cwd=tmp_path, timeout=1) == 124
    assert len(calls) == 1
    assert calls[0][1]["start_new_session"] is True
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    with pytest.raises(FileExistsError):
        runner.execute(["fixture"], tmp_path / "run.log", cwd=tmp_path, timeout=1)
    assert len(calls) == 1


def test_progress_replaces_only_the_selected_candidate_file(tmp_path):
    path = tmp_path / "progress.json"
    runner.write_progress(path, {"status": "running"})
    runner.write_progress(path, {"status": "completed"})
    assert json.loads(path.read_text()) == {"status": "completed"}
    assert not path.with_suffix(".pending").exists()
