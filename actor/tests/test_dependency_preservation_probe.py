import pytest

from cayu_tau3_banking.agent import banking_system_prompt
from cayu_tau3_banking.dependency_preservation_probe import ArchiveHost, assess


def plan():
    return [{"items": [{"id": "retire", "depends_on": []}]}]


def test_outcome_prompt_preserves_cross_outcome_constraints_without_task_identifiers():
    prompt = banking_system_prompt("Synthetic policy", planning_mode="outcomes")
    assert "Before an irreversible change" in prompt
    assert "Do not invent dependent work" in prompt
    assert "task_065" not in prompt
    assert "credential K9" not in prompt


def test_retirement_is_locally_valid_but_destroys_other_requested_outcome():
    host = ArchiveHost("dependent")
    assert host.execute("retire_credential", {"credential_id": "K9"})["success"]
    assert not host.execute("export_archive", {"archive_id": "A7", "credential_id": "K9"})[
        "success"
    ]
    assert not assess(host, plan())["structural_pass"]


@pytest.mark.parametrize("case", ["dependent", "immediate", "blocked"])
def test_evaluator_accepts_correct_causal_sequence(case):
    host = ArchiveHost(case)
    if case == "dependent":
        host.execute("export_archive", {"archive_id": "A7", "credential_id": "K9"})
    if case != "blocked":
        host.execute("retire_credential", {"credential_id": "K9"})
    assert assess(host, plan())["structural_pass"]
    assert not assess(host, [])["structural_pass"]


def test_blocked_case_cannot_gain_approval_by_asserting_consent():
    host = ArchiveHost("blocked")
    assert not host.execute("export_archive", {"archive_id": "A7", "credential_id": "K9"})[
        "success"
    ]
    assert host.active and not host.exported
    assert not assess(host, plan())["structural_pass"]
