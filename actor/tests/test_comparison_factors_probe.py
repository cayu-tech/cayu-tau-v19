import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
from cayu import ScriptedModelProvider
from test_runtime_banking import call, text

from cayu_tau3_banking.controller import UPDATE_WORK_PLAN_TOOL

spec = importlib.util.spec_from_file_location(
    "probe_comparison_factors", Path(__file__).parents[1] / "scripts/probe_comparison_factors.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_plan_conditions_differ_in_one_status_only():
    ready = probe.seed_plan("ready")
    resolved = probe.seed_plan("resolved")
    changed = deepcopy(ready)
    changed["items"][0]["obligations"][0]["status"] = "resolved"
    assert changed == resolved


def test_confirmation_only_adds_the_missing_single_base_constraint(tmp_path):
    root = probe.prepare(tmp_path, confirmation=True)
    protocol = json.loads((root / "protocol.json").read_text())
    assert protocol["single_base_confirmation"] is True
    assert len(protocol["schedule"]) == 2
    for row in protocol["schedule"]:
        assert (row["history"], row["research"]) == ("dense", "resolved")
        case = json.loads((root / f"{row['name']}-input.json").read_text())
        assert case["customer"] == probe.customer(row["case"], "dense") + probe.SINGLE_BASE
        assert case["catalogue"] == probe.catalogue(row["case"])
        assert case["plan"] == probe.seed_plan("resolved")


def test_history_treatment_preserves_current_facts_and_request():
    for case in ("case_a", "case_b"):
        compact = probe.customer(case, "compact")
        dense = probe.customer(case, "dense")
        assert (
            compact.split("Current complete catalogue:")[1]
            == dense.split("Current complete catalogue:")[1]
        )
        assert compact.split("\n")[0] == dense.split("\n")[0]
        assert "Closed support ticket 239" in dense and "Closed support ticket" not in compact


@pytest.mark.parametrize("research", ["ready", "resolved"])
def test_native_episode_updates_real_plan_and_finishes(tmp_path, research):
    root = probe.prepare(tmp_path)
    steps = []
    if research == "ready":
        steps.append(
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "compare",
                            "obligations": [
                                {
                                    "id": "research",
                                    "status": "resolved",
                                    "evidence": "Catalogue compared.",
                                    "evidence_refs": [probe.REFERENCE],
                                }
                            ],
                        }
                    ]
                },
                "resolve",
            )
        )
    steps.append(text("Sable plus Moss provides 175 GB for $22; five seats meet the requirement."))
    provider = ScriptedModelProvider(steps)
    name = f"case_a-compact-{research}"
    result = probe.run(root, name, provider_factory=lambda: provider, model="scripted", timeout=90)
    assert result["status"] == "answered", result
    assert len(provider.requests) == len(steps)
    request = provider.requests[0]
    rendered = "\n".join(p.text for m in request.messages for p in m.content if p.type == "text")
    assert probe.catalogue("case_a") in rendered
    assert rendered.count("<workflow-plan>") == 1
    plan = json.loads(rendered.split("<workflow-plan>")[1].split("</workflow-plan>")[0])
    assert plan["items"][0]["obligations"][0]["id"] == "research"
    assert plan["items"][0]["obligations"][0]["status"] == research
    assert sum(m.role == "system" for m in request.messages) == 1
    assert request.model == "scripted"
    assert result["external_tools_executed"] == 0
    boot = json.loads((root / "results" / name / "bootstrap.json").read_text())
    assert boot["plan"]["items"][0]["obligations"][0]["status"] == research


def test_frozen_input_changes_rejected_before_provider(tmp_path):
    root = probe.prepare(tmp_path)
    path = root / "case_a-compact-ready-input.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Frozen input"):
        probe.run(root, "case_a-compact-ready", provider_factory=lambda: pytest.fail("dispatch"))


@pytest.mark.parametrize("research", ["ready", "resolved"])
def test_readiness_gate_does_not_claim_semantic_correctness(tmp_path, research):
    root = probe.prepare(tmp_path)
    wrong = "Birch gives the highest usable capacity: 150 GB for $23."
    steps = [text(wrong)]
    if research == "ready":
        steps += [
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "compare",
                            "obligations": [
                                {
                                    "id": "research",
                                    "status": "resolved",
                                    "evidence": "Catalogue compared.",
                                    "evidence_refs": [probe.REFERENCE],
                                }
                            ],
                        }
                    ]
                },
                "resolve",
            ),
            text(wrong),
        ]
    provider = ScriptedModelProvider(steps)
    result = probe.run(
        root,
        f"case_a-compact-{research}",
        provider_factory=lambda: provider,
        model="scripted",
        timeout=90,
    )
    assert result["status"] == "answered", result
    assert result["reply"] == wrong
    assert len(provider.requests) == (3 if research == "ready" else 1)
