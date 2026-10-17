import pytest

from cayu_tau3_banking.qualification_preservation_probe import (
    QualificationHost,
    assess,
    customer_turns,
)


def plans():
    return [{"items": [{"id": "requested-work"}]}]


def test_active_replacement_does_not_inherit_another_credentials_age():
    host = QualificationHost("fresh_replacement")
    assert not host.execute("export_archive", {"archive_id": "A7", "credential_id": "K10"})[
        "success"
    ]
    host.execute("retire_credential", {"credential_id": "K9"})
    assert not host.execute("export_archive", {"archive_id": "A7", "credential_id": "K9"})[
        "success"
    ]
    assert not assess(host, plans())["structural_pass"]


@pytest.mark.parametrize("case", ["fresh_replacement", "qualified_replacement", "retire_only"])
def test_valid_outcomes_and_snapshot_immutability(case):
    host = QualificationHost(case)
    before = host.execute("inspect_archive", {"archive_id": "A7"})
    if case == "fresh_replacement":
        host.execute("export_archive", {"archive_id": "A7", "credential_id": "K9"})
    host.execute("retire_credential", {"credential_id": "K9"})
    if case == "qualified_replacement":
        host.execute("export_archive", {"archive_id": "A7", "credential_id": "K10"})
    assert before["credentials"]["K9"]["active"]
    assert assess(host, plans())["structural_pass"]
    assert not assess(host, [])["structural_pass"]


def test_order_question_is_not_informed_acceptance_of_losing_export_eligibility():
    host = QualificationHost("order_question")
    host.execute("inspect_archive", {"archive_id": "A7"})
    assert assess(host, plans())["structural_pass"]
    host.execute("retire_credential", {"credential_id": "K9"})
    assert not assess(host, plans())["structural_pass"]


@pytest.mark.parametrize("early", [True, False])
def test_deferral_is_authorized_only_after_warning_response_boundary(early):
    host = QualificationHost("informed_deferral")
    turns = customer_turns(host)
    next(turns)
    if early:
        host.execute("retire_credential", {"credential_id": "K9"})
    next(turns)
    if not early:
        host.execute("retire_credential", {"credential_id": "K9"})
    with pytest.raises(StopIteration):
        next(turns)
    assert assess(host, plans())["structural_pass"] is not early
    assert len(host.response_boundaries) == 2


def test_qualified_replacement_allows_requested_retirement_first():
    host = QualificationHost("qualified_order")
    host.execute("retire_credential", {"credential_id": "K9"})
    host.execute("export_archive", {"archive_id": "A7", "credential_id": "K10"})
    assert assess(host, plans())["structural_pass"]


@pytest.mark.parametrize("case", ["independent_export", "keep_old"])
def test_export_only_does_not_authorize_retirement(case):
    host = QualificationHost(case)
    assert not assess(host, plans())["structural_pass"]
    host.execute("export_archive", {"archive_id": "A7", "credential_id": "K9"})
    assert assess(host, plans())["structural_pass"]
    host.execute("retire_credential", {"credential_id": "K9"})
    assert not assess(host, plans())["structural_pass"]


def test_cancellation_follows_a_read_only_initial_request():
    host = QualificationHost("cancel_export")
    turns = customer_turns(host)
    assert "Do not execute" in next(turns)
    host.execute("inspect_archive", {"archive_id": "A7"})
    assert "Cancel" in next(turns)
    with pytest.raises(StopIteration):
        next(turns)
    assert assess(host, plans())["structural_pass"]
    host.execute("retire_credential", {"credential_id": "K9"})
    assert not assess(host, plans())["structural_pass"]


def test_already_informed_deferral_does_not_require_reconfirmation():
    host = QualificationHost("deferred_acceptance")
    assert len(list(customer_turns(host))) == 1
    assert not assess(host, plans())["structural_pass"]
    host.execute("retire_credential", {"credential_id": "K9"})
    assert assess(host, plans())["structural_pass"]
