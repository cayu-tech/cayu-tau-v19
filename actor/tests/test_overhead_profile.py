from datetime import UTC, datetime, timedelta

import pytest

from cayu_tau3_banking.overhead_profile import returned_document_revisions, summarize, union_seconds


def event(kind, second, *, tool=None, **payload):
    return {
        "session_id": "one",
        "event_type": kind,
        "timestamp": (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=second)).isoformat(),
        "tool_name": tool,
        "payload": payload,
    }


def test_intervals_do_not_double_count_parallel_work():
    assert union_seconds([(0, 10), (1, 2), (5, 12), (15, 16)]) == 13
    with pytest.raises(ValueError):
        union_seconds([(4, 2)])


def test_packet_reads_use_delivered_documents_not_only_explicit_request_ids():
    import json

    payload = {
        "arguments": {"product_id": "service/blue"},
        "result": {
            "is_error": False,
            "content": json.dumps(
                {
                    "documents": [
                        {"entry_id": "blue_terms", "entry_revision": 1},
                        {"entry_id": "blue_costs", "entry_revision": 2},
                    ]
                }
            ),
        },
    }
    assert returned_document_revisions(payload) == [("blue_terms", 1), ("blue_costs", 2)]
    with pytest.raises(ValueError):
        returned_document_revisions(
            {"arguments": {"documents": [{"entry_id": "never_returned", "revision": 1}]}}
        )


def test_phase_overlap_and_post_effect_boundary_are_not_checkpoint_attribution():
    rows = [
        event("model.started", 0, model_attempt_id="m"),
        event("model.completed", 5, model_attempt_id="m"),
        event(
            "tool.call.started",
            5,
            tool="controller_update_work_plan",
            model_attempt_id="m",
            tool_call_id="t",
        ),
        event(
            "tool.call.completed",
            9,
            tool="controller_update_work_plan",
            tool_call_id="t",
            tool_effect_completed_at=event("ignored", 6)["timestamp"],
            tool_terminal_publication_started_at=event("ignored", 9)["timestamp"],
        ),
        event("session.completed", 12),
    ]
    result = summarize(rows)
    assert result["phase_union_seconds"]["model"] == 5
    assert result["phase_union_seconds"]["tool"] == 4
    assert result["post_effect_to_publication_start"]["sum_seconds_nonexclusive"] == 3
    assert result["uncovered_event_span_seconds"] == 3
    assert result["model_seconds_by_proposed_work_nonexclusive"]["plan_proposals"] == 5


def test_partial_compaction_is_one_interval_and_unfinished_work_stays_unknown():
    rows = [
        event("context.compaction.started", 0, model_step_id="s"),
        event("context.compaction.completed", 2, model_step_id="s", coverage_mode="partial_prefix"),
        event("context.compaction.completed", 5, model_step_id="s", coverage_mode="full"),
        event("model.started", 7, model_attempt_id="unfinished"),
        event("session.interrupted", 10),
    ]
    result = summarize(rows)
    assert result["phase_union_seconds"]["compaction"] == 5
    assert result["unmatched_starts"] == {"model": 1}
    assert result["uncovered_event_span_seconds"] == 5


def test_private_payloads_are_not_exported_and_wrong_clocks_fail():
    rows = [event("session.started", 0, secret="NEVER_PRINT"), event("session.completed", 1)]
    assert "NEVER_PRINT" not in str(summarize(rows))
    assert summarize(list(reversed(rows)))["timestamp_reversals_in_publication_order"] == 1
    with pytest.raises(ValueError, match="Negative"):
        summarize(
            [
                event("model.started", 5, model_attempt_id="m"),
                event("model.completed", 1, model_attempt_id="m"),
            ]
        )
