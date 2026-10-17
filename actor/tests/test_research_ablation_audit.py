import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "research_audit", Path(__file__).parents[1] / "scripts/audit_research_ablation.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(reward=1, seconds=100, termination="user_stop"):
    return {
        "assessment": {"reward": reward, "duration_seconds": seconds, "termination": termination}
    }


@pytest.mark.parametrize("seconds,expected", [(75, True), (76, False), (50, True)])
def test_threshold_requires_full_pass_and_at_least_quarter_reduction(seconds, expected):
    assert audit.compare(row(), row(seconds=seconds))["meets_exploratory_threshold"] is expected


@pytest.mark.parametrize(
    "baseline,candidate",
    [
        (row(), row(reward=None, seconds=0, termination="infrastructure_error")),
        (row(reward=None, seconds=0, termination="infrastructure_error"), row(seconds=10)),
        (row(), row(reward=0, seconds=10)),
        (row(reward=0), row(seconds=10)),
    ],
)
def test_failure_cannot_count_as_speed_success(baseline, candidate):
    assert not audit.compare(baseline, candidate)["meets_exploratory_threshold"]
