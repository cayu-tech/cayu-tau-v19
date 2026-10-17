import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "batched_launcher", Path(__file__).parents[1] / "scripts/run_batched_pilot.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_batching_changes_only_the_existing_cli_option():
    original = ["pilot", "--seed", "300", "--model", "gpt-5.6-sol"]
    changed = launcher.enable_batching(original)
    assert changed == [*original, "--runtime-parallel-tools"]
    assert original[-1] == "gpt-5.6-sol"


def test_launcher_adds_flag_after_pilot_builds_actual_cli_arguments(monkeypatch):
    observed = []
    pilot = SimpleNamespace(cli=SimpleNamespace(main=lambda: observed.append(list(sys.argv))))

    def run():
        sys.argv = ["pilot", "--task-ids", "task_064", "--seed", "300"]
        pilot.cli.main()

    pilot.main = run
    monkeypatch.setitem(sys.modules, "run_pilot", pilot)
    monkeypatch.setattr(sys, "argv", ["launcher", "--task", "task_064"])
    launcher.main()
    assert observed == [
        ["pilot", "--task-ids", "task_064", "--seed", "300", "--runtime-parallel-tools"]
    ]
