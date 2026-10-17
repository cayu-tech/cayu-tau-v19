import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from cayu_tau3_banking.agent import banking_system_prompt
from cayu_tau3_banking.argument_contract import SOURCE_BINDING_RECOVERY


def module():
    spec = importlib.util.spec_from_file_location(
        "probe_source_binding", Path("scripts/probe_source_binding.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("index", range(6))
def test_requests_change_only_recovery_guidance_and_never_include_expectations(index):
    mod = module()
    case = mod.CASES[index]
    prompt = banking_system_prompt("Fictional service", planning_mode="outcomes")
    pair = mod.requests(case, prompt)
    a, b = pair["baseline"], pair["candidate"]
    assert a.model == b.model == "gpt-5.6-sol"
    assert a.messages[1:] == b.messages[1:]
    assert a.tools == b.tools
    assert a.options == b.options
    assert SOURCE_BINDING_RECOVERY not in a.messages[0].content[0].text
    assert b.messages[0].content[0].text.count(SOURCE_BINDING_RECOVERY) == 1
    assert case.evidence in a.messages[1].content[0].text
    poisoned = replace(case, expected={"secret_expected_answer": "MUST_NOT_ENTER_REQUEST"})
    assert mod.requests(poisoned, prompt) == pair


@pytest.mark.parametrize(
    "prompt", ["absent", "\n\n" + SOURCE_BINDING_RECOVERY + "\n\n" + SOURCE_BINDING_RECOVERY]
)
def test_fail_closed_on_prompt_version_mismatch(prompt):
    mod = module()
    with pytest.raises(ValueError, match="exactly one"):
        mod.requests(mod.CASES[0], prompt)


def test_frozen_protocol_and_requests_cannot_be_silently_changed(tmp_path):
    mod = module()
    root = mod.prepare(tmp_path)
    mod.verify(root)
    request = root / "sourced_name_and_type-candidate-request.json"
    request.write_text("{}")
    with pytest.raises(ValueError, match="Frozen source changed"):
        mod.verify(root)


def test_resume_never_retries_uncertain_dispatch(tmp_path):
    mod = module()
    row = {"name": "case", "arm": "candidate", "expected": {"decision": "pause", "arguments": {}}}
    assert mod.completed_result(tmp_path, row) is None
    (tmp_path / "case").mkdir()
    with pytest.raises(ValueError, match="Uncertain prior dispatch"):
        mod.completed_result(tmp_path, row)


def test_resume_requires_completed_audited_payload(tmp_path):
    mod = module()
    row = {"name": "case", "arm": "candidate", "expected": {"decision": "pause", "arguments": {}}}
    data = {**row, "decision": row["expected"], "passed": True, "actual_payload_matches": True}
    (tmp_path / "case-result.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="capture is incomplete"):
        mod.completed_result(tmp_path, row)
