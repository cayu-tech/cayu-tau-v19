"""Offline fail-closed checks for the new census deployment entry points."""

import importlib.util
import sys
from pathlib import Path

import pytest


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts" / f"{name}.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_installer_rejects_unregistered_archive_before_network(monkeypatch, tmp_path):
    installer = module("install_inventory_census")
    monkeypatch.setattr(
        sys,
        "argv",
        ["install", "--archive", str(tmp_path / "other.tar.gz"), "--archive-sha256", "unused"],
    )
    monkeypatch.setattr(
        installer.urllib.request,
        "urlopen",
        lambda *_a, **_k: pytest.fail("Unexpected network request"),
    )
    with pytest.raises(ValueError, match="Unexpected candidate"):
        installer.main()


def test_installer_rejects_another_vm_before_source_mutation(monkeypatch, tmp_path):
    installer = module("install_inventory_census")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install",
            "--archive",
            str(tmp_path / "inventory-full97-test.tar.gz"),
            "--archive-sha256",
            "unused",
        ],
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self):
            return b"different-instance"

    monkeypatch.setattr(installer.urllib.request, "urlopen", lambda *_a, **_k: Response())
    with pytest.raises(ValueError, match="Not the authorized VM"):
        installer.main()
    assert not list(tmp_path.iterdir())


def test_start_refuses_missing_frozen_source_manifest(monkeypatch, tmp_path):
    starter = module("start_inventory_census")
    runner = module("run_full_benchmark")
    monkeypatch.setitem(sys.modules, "run_benchmark", runner)
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules,
        "archive_support",
        SimpleNamespace(archive_preserving=lambda *_: pytest.fail("Unexpected archive")),
    )
    monkeypatch.setattr(sys, "argv", ["start", "--root", str(tmp_path)])
    with pytest.raises(FileNotFoundError):
        starter.main()


def test_dashboard_uses_frozen_provider_and_limits(tmp_path):
    runner = module("run_full_benchmark")
    report = tmp_path / "report.html"
    protocol = {
        "task_ids": ["task_001"],
        "report": str(report),
        "actor": "Codex LB <test>",
        "simulator": "Platform GPT-5.2",
        "max_parallel_task_processes": 4,
        "worker_watchdog_seconds": 7500,
        "limits": {"simulation_seconds": 7200},
    }
    runner.publish(tmp_path, protocol, {"status": "prepared", "rows": {}, "running": {}})
    text = report.read_text()
    assert "Codex LB &lt;test&gt;" in text
    assert "4 isolated workers" in text
    assert "7200 seconds" in text and "7500 seconds" in text
    assert "subscription" not in text and "8 GB laptop" not in text
