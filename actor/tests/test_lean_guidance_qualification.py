import importlib.util
import json
from pathlib import Path

import pytest
from cayu import ScriptedModelProvider
from test_outcome_controller import outcome
from test_runtime_banking import call, text

from cayu_tau3_banking.advice import COMPARISON_COVERAGE
from cayu_tau3_banking.argument_contract import OPTIONAL_ARGUMENT_DEFAULTS
from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL

spec = importlib.util.spec_from_file_location(
    "qualify_lean_guidance", Path(__file__).parents[1] / "scripts/qualify_lean_guidance.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def freeze(tmp_path):
    baseline = tmp_path / "old_advice.py"
    baseline.write_text("# synthetic baseline source\n")
    return probe.prepare(tmp_path, baseline)


@pytest.mark.parametrize("name", probe.COMPARISON_CASES)
def test_comparison_controls_are_long_source_complete_and_have_no_answer_field(name):
    case, expected = probe.comparison_case(name)
    assert set(case) == {"id", "kind", "customer", "policy", "catalogue"}
    assert case["catalogue"] in case["customer"]
    assert len(case["customer"]) > 25000
    assert "task_066" not in case["customer"] and "Silver Rewards" not in case["customer"]
    assert "Advice only" in case["customer"]
    assert "six seats" in case["customer"]
    assert expected["usable_gb"] == (160 if name in {"helpful", "non_stacking"} else 130)
    if name == "deductions":
        assert "30%" in case["customer"] and 180 * 70 / 100 == 126
    if name == "non_stacking":
        assert "do not stack" in case["customer"]
    if name == "ineligible":
        assert "Keep my seat count fixed at six" in case["customer"]


def test_freeze_separates_expectations_and_changes_only_advice_between_arms(tmp_path):
    root = freeze(tmp_path)
    probe.verify(root)
    baseline, candidate = root / "baseline/src", root / "candidate/src"
    changed = [
        p.relative_to(baseline)
        for p in baseline.rglob("*.py")
        if p.read_bytes() != (candidate / p.relative_to(baseline)).read_bytes()
    ]
    assert changed == [Path("cayu_tau3_banking/advice.py")]
    cases = json.loads((root / "cases.json").read_text())
    assert len(cases) == 9 and all("expected" not in c for c in cases)
    assert len(json.loads((root / "protocol.json").read_text())["schedule"]) == 18
    (root / "cases.json").write_text("[]")
    with pytest.raises(ValueError, match="Frozen"):
        probe.verify(root)


def test_native_comparison_captures_actual_prompt_but_does_not_self_grade_prose(tmp_path):
    root = freeze(tmp_path)
    provider = ScriptedModelProvider([text("Deliberately wrong answer.")])
    probe.run(root, "candidate", "helpful", provider_factory=lambda: provider, model="scripted")
    output = root / "results/helpful-candidate"
    result = json.loads((output / "result.json").read_text())
    assert result["no_effects"] and result["semantic_grade"] == "pending_evaluator_review"
    assert "passed" not in result and not (output / "assessment.json").exists()
    stored = json.loads((output / "corpus/service_catalogue.json").read_text())
    assert "Ember" in stored["content"]
    instructions = "\n".join(p.text for p in provider.requests[0].messages[0].content)
    assert COMPARISON_COVERAGE in instructions and OPTIONAL_ARGUMENT_DEFAULTS in instructions
    assert "provision_asset" not in {t["name"] for t in provider.requests[0].tools}


def test_subset_has_exactly_two_registered_runs_and_no_other_cases(tmp_path):
    baseline = tmp_path / "old_advice.py"
    baseline.write_text("# synthetic baseline source\n")
    root = probe.prepare(tmp_path, baseline, ["ineligible"])
    assert [c["id"] for c in json.loads((root / "cases.json").read_text())] == ["ineligible"]
    schedule = json.loads((root / "protocol.json").read_text())["schedule"]
    assert schedule == [
        {"case": "ineligible", "arm": "baseline"},
        {"case": "ineligible", "arm": "candidate"},
    ]
    probe.verify(root)


@pytest.mark.parametrize(
    "name,note",
    [
        ("ordinary_note", None),
        ("requested_note", "Migration request C47"),
        ("policy_note", "Migration control M12"),
    ],
)
def test_native_optional_controls_execute_and_record_exact_fields(tmp_path, name, note):
    root = freeze(tmp_path)
    arguments = {"sku": "nv_team"}
    if note is not None:
        arguments["change_note"] = note
    completed = outcome(
        status="completed",
        evidence="Resource returned by service",
        evidence_refs=[{"kind": "tool_result", "reference_id": "created"}],
    )
    provider = ScriptedModelProvider(
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}),
            call("provision_asset", arguments, "created"),
            call(UPDATE_WORK_PLAN_TOOL, {"items": [completed]}),
            text("Created the requested resource."),
        ]
    )
    probe.run(root, "candidate", name, provider_factory=lambda: provider, model="scripted")
    assessment = json.loads((root / f"results/{name}-candidate/assessment.json").read_text())
    assert assessment["structural_pass"]
    result = json.loads((root / f"results/{name}-candidate/result.json").read_text())
    assert result["calls"] == [{"name": "provision_asset", "arguments": arguments}]


def test_repeated_case_does_not_overwrite_first_attempt(tmp_path):
    root = freeze(tmp_path)
    provider = ScriptedModelProvider([text("Advice.")])
    probe.run(root, "candidate", "helpful", provider_factory=lambda: provider, model="scripted")
    with pytest.raises(FileExistsError):
        probe.run(root, "candidate", "helpful", provider_factory=lambda: provider, model="scripted")
    assert len(provider.requests) == 1
