import pytest

from cayu_tau3_banking.locator_handoff_probe import UploadHost, assess_upload


def test_customer_cannot_upload_before_enabled():
    host = UploadHost()
    messages = host.customers()
    next(messages)
    assert "not been enabled" in next(messages)
    assert not host.uploaded
    assert not assess_upload(host, [])["structural_pass"]


def test_enabling_is_not_uploading_and_status_is_verified_after_customer_action():
    host = UploadHost()
    messages = host.customers()
    next(messages)
    result = host.execute("enable_customer_action", {"action_name": "upload_document_v2"})
    assert result["enabled"] and not result["upload_performed"]
    assert not host.execute("get_upload_status", {"workspace_id": "ws_81"})["uploaded"]
    next(messages)
    assert host.uploaded
    assert not assess_upload(host, [])["structural_pass"]
    host.execute("get_upload_status", {"workspace_id": "ws_81"})
    assert assess_upload(host, [])["structural_pass"]
    with pytest.raises(ValueError, match="already enabled"):
        host.execute("enable_customer_action", {"action_name": "upload_document_v2"})


def test_unknown_action_or_workspace_has_no_effect():
    host = UploadHost()
    for name, args in [
        ("upload_document_v2", {}),
        ("get_upload_status", {"workspace_id": "wrong"}),
    ]:
        with pytest.raises(ValueError):
            host.execute(name, args)
    assert not host.calls and not host.uploaded
