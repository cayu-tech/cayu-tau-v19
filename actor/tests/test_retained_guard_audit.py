import json

import pytest
from test_admission import confirm_effect, confirm_verification, proposed

from cayu_tau3_banking.admission import (
    BankingToolAdmissionGate,
    _lookup_contains_one_identity_record,
)
from cayu_tau3_banking.ledger import ConversationLedger


def test_identity_does_not_match_a_longer_id_or_other_field():
    row = {"user_id": "user-70", "note": "user-7"}
    assert not _lookup_contains_one_identity_record(json.dumps(row), ["user-7"])
    assert _lookup_contains_one_identity_record(json.dumps(row), ["user-70"])


@pytest.mark.parametrize(
    "changes",
    [
        {"address": "11 Main Street"},
        {"email": "otherexample@example.com"},
        {"phone_number": "1555010099"},
        {"date_of_birth": "01/02/19901"},
    ],
)
def test_identity_requires_complete_field_values(changes):
    row = dict(
        user_id="user-7",
        address="1 Main Street",
        date_of_birth="01/02/1990",
        email="example@example.com",
        phone_number="555-0100",
    )
    expected = [row[k] for k in ("user_id", "address", "date_of_birth", "email", "phone_number")]
    assert _lookup_contains_one_identity_record(json.dumps(row), expected)
    row.update(changes)
    assert not _lookup_contains_one_identity_record(json.dumps(row), expected)


def test_duplicate_identity_fields_are_not_authoritative():
    assert not _lookup_contains_one_identity_record(
        '{"user_id":"other","user_id":"user-7"}', ["user-7"]
    )
    assert not _lookup_contains_one_identity_record(
        "1. Record ID: user-7\n user_id: other\n user_id: user-7", ["user-7"]
    )
    assert not _lookup_contains_one_identity_record("[" * 2000 + "]" * 2000, ["user-7"])


def test_email_can_be_restored_after_an_intervening_change():
    ledger = ConversationLedger()
    confirm_verification(ledger)
    for index, email in enumerate(("a@example.com", "b@example.com")):
        confirm_effect(
            ledger, f"change-{index}", "change_user_email", user_id="user-7", new_email=email
        )
    gate = BankingToolAdmissionGate(ledger, authoritative_only=True)
    restore = proposed("restore", "change_user_email", user_id="user-7", new_email="a@example.com")
    redundant = proposed("repeat", "change_user_email", user_id="user-7", new_email="b@example.com")
    assert gate.review((restore,))[0].allowed
    assert not gate.review((redundant,))[0].allowed
    batch = tuple(
        proposed(str(i), "change_user_email", user_id="user-7", new_email=email)
        for i, email in enumerate(
            ("a@example.com", "b@example.com", "a@example.com", "a@example.com")
        )
    )
    assert [d.allowed for d in gate.review(batch)] == [True, True, True, False]
