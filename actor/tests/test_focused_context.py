import asyncio
import hashlib
import json

import pytest
from tau2.data_model.message import UserMessage
from test_runtime_banking import assembly, call, text, work

from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL
from cayu_tau3_banking.focused_context import FocusedPolicyBundle
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus

SOURCE = "The purchase limit is $4,500. host_action(value: int) performs the requested action."


def manifest(tmp_path, digest=None):
    path = tmp_path / "bundle-manifest.txt"
    path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "entry_id": "limits",
                        "revision": 1,
                        "content_sha256": digest or hashlib.sha256(SOURCE.encode()).hexdigest(),
                    }
                ]
            }
        )
    )
    return path


def completed_outcome():
    item = work(done=True)
    item.pop("next_operation")
    return item


def test_focused_runtime_delivers_citable_sources_without_recall(tmp_path):
    agent, _provider, captured = assembly(
        tmp_path,
        [call(SYNC_WORK_PLAN_TOOL, {"items": [completed_outcome()]}), text("The limit is $4,500.")],
        controller_mode="advisory",
        planning_mode="outcomes",
        advice_mode="lean",
        native_recall=True,
        lexical_navigation=True,
        focused_policy_file=manifest(tmp_path),
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("What is the purchase limit?"), state
        )
        assert "$4,500" in reply.content
        observer = captured["state"]
        assert observer.focused_bundle.references[0].reference_id == "limits"
        assert not observer.observed_recall_exposures
        assert SOURCE in observer.focused_bundle.text
        assert not observer.controller.invoke(
            SYNC_WORK_PLAN_TOOL, {"items": [completed_outcome()]}
        ).is_error
        assert "cayu_search_banking_knowledge" in {
            s["name"] for s in observer.memory.tool_schemas()
        }
    finally:
        agent.stop(state)


def test_source_hash_mismatch_cannot_register_evidence(tmp_path):
    (tmp_path / "limits.json").write_text(
        json.dumps({"id": "limits", "title": "Limits", "content": SOURCE})
    )
    bundle = FocusedPolicyBundle(manifest(tmp_path, "0" * 64))
    corpus = BankingKnowledgeCorpus.from_directory(tmp_path)
    with pytest.raises(ValueError, match="hash"):
        asyncio.run(bundle.prepare(corpus))
    assert bundle.references == ()
    assert bundle.text is None


def test_manifest_cannot_supply_answers_or_arbitrary_text(tmp_path):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    data["expected_action"] = "do this"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="only"):
        FocusedPolicyBundle(path)
