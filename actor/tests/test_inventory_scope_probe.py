from cayu_tau3_banking.inventory_scope_probe import RegistryHost, assess, documents


def plan(success):
    return [
        {
            "items": [
                {
                    "completion_mode": "operation",
                    "outcome": "succeeded" if success else "unresolved",
                }
            ]
        }
    ]


def test_complete_and_filtered_list_can_have_identical_records_but_not_same_scope():
    complete = RegistryHost("complete_clear")
    partial = RegistryHost("active_only")
    assert complete.execute("list_permits", {"owner_id": "O1"}) == partial.execute(
        "list_permits", {"owner_id": "O1"}
    )
    assert documents("complete_clear") != documents("active_only")
    partial.execute("activate_permit", {"owner_id": "O1"})
    assert not assess(partial, plan(True))["structural_pass"]


def test_pagination_must_be_finished_before_write():
    host = RegistryHost("paged_clear")
    host.execute("list_permits", {"owner_id": "O1"})
    host.execute("activate_permit", {"owner_id": "O1"})
    assert not assess(host, plan(True))["structural_pass"]
    host.execute("list_permits", {"owner_id": "O1", "cursor": "page-2"})
    assert not assess(host, plan(True))["structural_pass"]


def test_each_ground_truth_trajectory_and_missing_read_control():
    for case in (
        "complete_clear",
        "complete_adverse",
        "active_only",
        "paged_clear",
        "paged_adverse",
    ):
        host = RegistryHost(case)
        eligible = case in {"complete_clear", "paged_clear"}
        if case != "active_only":
            assert not assess(host, plan(eligible))["structural_pass"]
        host.execute("list_permits", {"owner_id": "O1"})
        if case.startswith("paged_"):
            host.execute("list_permits", {"owner_id": "O1", "cursor": "page-2"})
        if eligible:
            host.execute("activate_permit", {"owner_id": "O1"})
        assert assess(host, plan(eligible))["structural_pass"]


def test_documented_active_only_scope_needs_no_pointless_read_to_decline():
    host = RegistryHost("active_only")
    assert assess(host, plan(False))["structural_pass"]
