import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_model_task_evaluation",
    Path(__file__).parents[1] / "scripts/prepare_model_task_evaluation.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("effort", ["medium", "high", "xhigh"])
def test_only_model_configuration_changes(effort):
    original = ["--model", "gpt-5.6-sol", "--seed", "300", "--reasoning-effort", "xhigh"]
    result = module.replace_model(original, "gpt-6-astra", effort)
    assert result == ["--model", "gpt-6-astra", "--seed", "300", "--reasoning-effort", effort]
    assert original[1] == "gpt-5.6-sol" and original[-1] == "xhigh"


@pytest.mark.parametrize("args", [[], ["--model"], ["--model", "unexpected"], ["--model"] * 2])
def test_ambiguous_baseline_is_rejected(args):
    with pytest.raises(ValueError):
        module.replace_model(args, "gpt-6-astra", "high")
