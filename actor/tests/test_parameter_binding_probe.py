import pytest

from cayu_tau3_banking.parameter_binding_probe import CASES, Host, assess


def plan(outcome):
    return [{"items": [{"completion_mode": "operation", "outcome": outcome}]}]


@pytest.mark.parametrize(
    "case",
    ["sourced_official_name", "display_label", "explicit_optional_note", "literal_parentheses"],
)
def test_assessor_uses_actual_stored_fields_and_rejects_wrong_value(case):
    expected = CASES[case]["expected"]
    host = Host()
    host.execute("provision_asset", expected)
    assert assess(host, plan("succeeded"), expected)["structural_pass"]
    assert not assess(host, plan("unresolved"), expected)["structural_pass"]
    wrong = Host()
    wrong.execute("provision_asset", {"sku": "wrong"})
    assert not assess(wrong, plan("succeeded"), expected)["structural_pass"]
    with pytest.raises(ValueError, match="duplicate"):
        host.execute("provision_asset", expected)


def test_optional_omission_and_ambiguity_are_distinct():
    host = Host()
    assert assess(host, [], None)["structural_pass"]
    assert not assess(host, [], CASES["display_label"]["expected"])["structural_pass"]
    assert assess(host, plan("unresolved"), None)["structural_pass"]
    assert not assess(host, plan("succeeded"), None)["structural_pass"]
    host.execute("provision_asset", {"sku": "nv_team"})
    assert assess(host, plan("succeeded"), CASES["display_label"]["expected"])["structural_pass"]
    assert not assess(host, plan("unresolved"), None)["structural_pass"]
