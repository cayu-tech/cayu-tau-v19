import importlib.util
import json
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path

import pytest
from cayu import ScriptedModelProvider
from test_runtime_banking import text

SCRIPTS = Path(__file__).parents[1] / "scripts"
for name in ("probe_advice_paths", "probe_advice_compaction"):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
probe = sys.modules["probe_advice_compaction"]


def test_freezes_larger_catalogue_and_isolates_expectations(tmp_path):
    root = probe.prepare(tmp_path)
    protocol = probe.verify(root)
    assert len(list((root / "corpus").glob("*.json"))) == 32
    cases = json.loads((root / "cases.json").read_text())
    assert len(cases) == 2
    assert all(set(case) == {"id", "customer", "followups"} for case in cases)
    assert all(len(case["followups"]) == 2 for case in cases)
    assert protocol["compaction"] == {"compact_after_messages": 2, "max_user_turns": 1}
    assert str(SCRIPTS / "probe_advice_paths.py") in protocol["source_hashes"]
    assert str(SCRIPTS / "probe_advice_compaction.py") in protocol["source_hashes"]
    (root / "cases.json").write_text("[]")
    with pytest.raises(ValueError, match="Frozen input"):
        probe.verify(root)


def test_declared_decisions_are_consistent_with_all_catalogue_alternatives():
    # Independent evaluator arithmetic, never a runtime tool or actor input.
    terms = {
        "Aspen": ("3.6", 0, 100),
        "Beacon": ("4.4", 60, 100),
        "Clover": ("3", 0, 100),
        "Delta": ("5", 0, 50000),
        "Elm": ("5.8", 0, 100),
        "Fjord": ("4", 0, 100),
        "Grove": ("4.45", 80, 100),
        "Heath": ("4.1", 40, 100),
        "Indigo": ("3.3", 0, 100),
        "Juniper": ("4.2", 65, 100),
        "Kestrel": ("3.8", 0, 100),
        "Linden": ("4.25", 50, 100),
    }
    settings = [
        (10000, 600, False, True, "Clover", 440),
        (10000, 300, False, True, "Fjord", 400),
        (50000, 300, False, True, "Delta", 2500),
        (20000, 0, True, False, "Elm", 928),
        (20000, 0, False, False, "Beacon", 820),
        (20000, 600, False, True, "Clover", 920),
    ]
    for balance, spend, donation, cards, winner, net in settings:
        values = {}
        for name, (rate, fee, minimum) in terms.items():
            if balance < minimum or (name == "Elm" and not donation):
                continue
            value = balance * Decimal(rate) / 100
            if name == "Elm":
                value *= Decimal("0.8")
            if name == "Clover" and cards and spend >= 500:
                value += balance * Decimal("0.018") - 40
            values[name] = value - fee
        assert max(values, key=values.get) == winner
        assert values[winner] == net
        assert sum(value == net for value in values.values()) == 1
    assert [e["net"] for group in probe.expectations().values() for e in group] == [
        row[-1] for row in settings
    ]


def test_multiturn_runtime_really_compacts_and_keeps_same_session(tmp_path):
    root = probe.prepare(tmp_path)
    output = root / "test-run"
    output.mkdir()
    actor = ScriptedModelProvider(
        [text("First answer."), text("Second answer."), text("Third answer.")]
    )
    summarizer = ScriptedModelProvider(
        [
            text("First customer constraints preserved."),
            text("Updated customer constraints preserved."),
        ]
    )
    case = {
        "id": "test",
        "customer": "Compare savings. No donations.",
        "followups": ["Change balance to 20000.", "Change monthly spend to 300."],
    }
    result = probe.run_case(
        output,
        case,
        arm="minimal",
        corpus_directory=root / "corpus",
        policy=probe.POLICY,
        provider_factory=lambda: actor,
        compactor_provider_factory=lambda: summarizer,
        model="scripted",
        timeout=90,
    )
    assert result["status"] == "answered", result
    assert len(result["turns"]) == len(actor.requests) == 3
    assert len(summarizer.requests) == 2
    # The first follow-up compacts the old turn; the latest customer message stays live.
    last = json.dumps([m.model_dump(mode="json") for m in actor.requests[-1].messages])
    assert "Updated customer constraints preserved." in last
    assert "Change monthly spend to 300." in last
    summaries = [
        json.loads(p.read_text())["compaction"] for p in (output / "contexts").rglob("*.json")
    ]
    assert sum(bool(summary.get("summary")) for summary in summaries) == 2
    metrics = json.loads((output / "compaction-metrics.json").read_text())
    assert len(metrics["calls"]["actor"]) == 3
    assert len(metrics["calls"]["compactor"]) == 2
    assert len(metrics["compactions"]) == 2
    assert all(c["event"] == "context.compaction.completed" for c in metrics["compactions"])
    with sqlite3.connect(output / "runtime.sqlite3") as db:
        assert db.execute("SELECT COUNT(DISTINCT session_id) FROM cayu_events").fetchone()[0] == 1
    assert all(turn["plan"]["items"] == [] for turn in result["turns"])
    assert len(json.loads((output / "transcript.json").read_text())) == 6


def test_rejects_evaluator_fields_before_provider_creation(tmp_path):
    with pytest.raises(ValueError, match="actor-visible"):
        probe.run_case(
            tmp_path,
            {"id": "bad", "customer": "Hi", "followups": [], "expected": "X"},
            arm="minimal",
            corpus_directory=tmp_path,
            policy="",
            provider_factory=None,
        )


def test_assembly_error_is_preserved_even_if_profile_cannot_be_built(tmp_path):
    def broken_provider():
        raise ValueError("Synthetic provider assembly failure")

    result = probe.run_case(
        tmp_path,
        {"id": "bad", "customer": "Hi", "followups": []},
        arm="minimal",
        corpus_directory=tmp_path,
        policy="",
        provider_factory=broken_provider,
        timeout=10,
    )
    assert result["status"] == "error"
    assert result["error_type"]
    assert result["diagnostic_error_type"]
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "error"
