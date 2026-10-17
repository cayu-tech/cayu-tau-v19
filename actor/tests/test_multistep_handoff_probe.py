import pytest

from cayu_tau3_banking.multistep_handoff_probe import MigrationHost, assess_migration


def test_released_flow_requires_enablement_customer_submission_and_readback():
    host = MigrationHost(release=True)
    messages = host.customers()
    next(messages)
    host.execute("create_archive", {"project_id": "px_7"})
    next(messages)
    host.execute("enable_user_action", {"action_name": "submit_migration_manifest_v3"})
    assert not host.submitted
    next(messages)
    assert host.submitted
    assert not host.verified
    host.execute("get_migration_state", {"project_id": "px_7"})
    host.execute("retire_workspace", {"workspace_id": "ws_old"})
    assert assess_migration(host, [])["structural_pass"]


def test_pending_flow_blocks_enablement_and_retirement():
    host = MigrationHost(release=False)
    messages = host.customers()
    next(messages)
    host.execute("create_archive", {"project_id": "px_7"})
    next(messages)
    for name, args in [
        ("enable_user_action", {"action_name": "submit_migration_manifest_v3"}),
        ("retire_workspace", {"workspace_id": "ws_old"}),
    ]:
        with pytest.raises(ValueError):
            host.execute(name, args)
    assert not host.enabled and not host.retired
    assert not assess_migration(host, [])["structural_pass"]


def test_customer_does_not_submit_when_agent_omits_handoff():
    host = MigrationHost(release=True)
    messages = host.customers()
    next(messages)
    host.execute("create_archive", {"project_id": "px_7"})
    next(messages)
    assert "not enabled" in next(messages)
    assert not host.submitted
    assert not assess_migration(host, [])["structural_pass"]


def test_duplicate_creation_and_wrong_scope_are_rejected():
    host = MigrationHost(release=True)
    with pytest.raises(ValueError):
        host.execute("create_archive", {"project_id": "wrong"})
    assert not host.created
    host.execute("create_archive", {"project_id": "px_7"})
    with pytest.raises(ValueError, match="Duplicate"):
        host.execute("create_archive", {"project_id": "px_7"})
