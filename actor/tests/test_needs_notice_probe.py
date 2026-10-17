import pytest

from cayu_tau3_banking.agent import PRODUCT_NEEDS_INSTRUCTIONS, banking_system_prompt
from cayu_tau3_banking.needs_notice_probe import NEEDS, NoticeHost, assess_closed


@pytest.mark.parametrize("memory_enabled", [False, True])
@pytest.mark.parametrize("planning_mode", ["exact", "outcomes"])
def test_needs_instruction_is_shared_without_requiring_memory(memory_enabled, planning_mode):
    prompt = banking_system_prompt(
        "Public test policy", memory_enabled=memory_enabled, planning_mode=planning_mode
    )
    assert prompt.count(PRODUCT_NEEDS_INSTRUCTIONS) == 1
    assert "already supplied sufficient needs, answer directly" in prompt
    assert "BEFORE broad catalogue research" in prompt
    assert "Public information requests do not require identification" in prompt
    assert "never relabel executable work as waiting" in prompt
    assert "A clarification is a next step, not task completion" in prompt
    assert "Silver Plus" not in PRODUCT_NEEDS_INSTRUCTIONS
    assert prompt.index(PRODUCT_NEEDS_INSTRUCTIONS) < prompt.index("<policy>")
    if memory_enabled:
        assert "OPTIONAL arithmetic and retained-evidence aid" in prompt
        assert "Do not create an obligation\nto fill a worksheet" in prompt
        assert "its source, numeric and quote validation remain mandatory" in prompt
        assert prompt.index("OPTIONAL arithmetic") < prompt.index(PRODUCT_NEEDS_INSTRUCTIONS)
    else:
        assert "OPTIONAL arithmetic" not in prompt


def test_needs_cases_share_incomplete_initial_request_but_not_followups():
    assert NEEDS[0]["customer"] == NEEDS[1]["customer"]
    assert NEEDS[0]["followups"] != NEEDS[1]["followups"]


def test_notice_host_rejects_wrong_target_and_duplicate_effect():
    host = NoticeHost()
    with pytest.raises(ValueError):
        host.execute("close_workspace", {"workspace_id": "wrong"})
    assert not host.calls
    assert host.execute("close_workspace", {"workspace_id": "ws_81"})["effective"] == "immediate"
    with pytest.raises(ValueError):
        host.execute("close_workspace", {"workspace_id": "ws_81"})
    assert len(host.calls) == 1


def test_notice_success_needs_effect_and_completed_operation():
    host = NoticeHost()
    plans = [{"items": [{"completion_mode": "operation", "outcome": "succeeded"}]}]
    assert not assess_closed(host, plans)["structural_pass"]
    host.execute("close_workspace", {"workspace_id": "ws_81"})
    assert assess_closed(host, plans)["structural_pass"]
    assert not assess_closed(host, [{"items": []}])["structural_pass"]
