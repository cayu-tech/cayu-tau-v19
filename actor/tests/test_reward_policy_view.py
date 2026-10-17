import asyncio
import json

from test_private_helper_availability import session
from test_reward_audit import ledger_with_transactions

from cayu_tau3_banking.reward_policy_view import MAX_OVERVIEW_BYTES, progressive_reward_policy_view


def document(entry_id, text, complete=True):
    return {
        "entry_id": entry_id,
        "entry_revision": 1,
        "title": "Apply for a premium card",
        "text": text,
        "text_complete": complete,
        "content_sha256": "source-digest",
    }


def payload(*docs):
    return {"policy_dossier_id": "handle", "card_types": ["Example"], "documents": docs}


def test_application_title_does_not_hide_primary_terms_or_deliver_directory_only_sources():
    primary = document("doc_example_001", "## Key terms\nCash back on all purchases: 2.5%")
    secondary = document("doc_example_002", "## Exclusions\nIneligible purchases earn zero.")
    result, delivered = progressive_reward_policy_view(payload(primary, secondary))
    assert result["documents"][0]["text"] == primary["text"]
    assert delivered == {primary["entry_id"]}
    assert result["documents"][1]["section_headings"] == ["Exclusions"]
    assert "text" not in result["documents"][1]
    assert not result["documents"][1]["text_complete"]


def test_oversized_or_incomplete_overviews_are_never_misrepresented_as_complete():
    oversized = document("doc_large_001", "é" * (MAX_OVERVIEW_BYTES // 2 + 1))
    partial = document("doc_partial_001", "Partial terms", complete=False)
    small = document("doc_small_001", "## Terms\nWhole article")
    result, delivered = progressive_reward_policy_view(payload(oversized, partial, small))
    assert delivered == {small["entry_id"]}
    assert all("text" not in d for d in result["documents"][:2])
    assert result["documents"][2]["text"] == small["text"]


def test_total_full_text_budget_applies_across_products():
    docs = [document(f"doc_product{i}_001", "x" * 9000) for i in range(4)]
    result, delivered = progressive_reward_policy_view(payload(*docs))
    assert len(delivered) == 2
    assert sum(len(d.get("text", "").encode()) for d in result["documents"]) <= MAX_OVERVIEW_BYTES
    assert len(result["documents"]) == 4


def test_advisory_helper_records_only_actually_delivered_evidence(tmp_path):
    for suffix, content in [
        ("001", "## Key terms\nCash back: 2%."),
        ("002", "## Exclusions\nNo gift cards."),
    ]:
        entry = f"doc_credit_cards_silver_rewards_card_{suffix}"
        (tmp_path / f"{entry}.json").write_text(
            json.dumps(
                {
                    "id": entry,
                    "title": f"Silver Rewards Card: {suffix}",
                    "content": content,
                }
            )
        )
    memory = session(tmp_path)
    memory.ledger = ledger_with_transactions()
    memory.advisory_workflow = True
    result = asyncio.run(memory._prepare_reward_policy_dossier({"user_id": "user-7"}))
    assert not result.is_error, result.content
    view = json.loads(result.content.splitlines()[-1])
    full = {d["entry_id"] for d in view["documents"] if d["text_complete"]}
    assert full == {"doc_credit_cards_silver_rewards_card_001"}
    assert memory._turn_delivered_entry_ids == full
    assert memory.controller.has_delivered_knowledge_revision(
        "doc_credit_cards_silver_rewards_card_001", "1"
    )
    assert not memory.controller.has_delivered_knowledge_revision(
        "doc_credit_cards_silver_rewards_card_002", "1"
    )
    assert "Cash back: 2%." in result.content
