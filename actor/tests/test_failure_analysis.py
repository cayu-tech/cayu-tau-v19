from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cayu_tau3_banking.failure_analysis import (
    FailureAnalysisError,
    _classify_db_diff,
    _deep_replay,
    _diff_database,
    _state_profile,
    analyze_results,
)


class _ReplayEnvironment:
    def __init__(self) -> None:
        self.tools = SimpleNamespace(db=SimpleNamespace(model_dump=lambda: {}))

    def set_state(self, **_kwargs) -> None:
        pass

    def make_tool_call(self, **_kwargs) -> None:
        pass

    def get_db_hash(self) -> str:
        return "agent-match"

    def get_user_db_hash(self) -> str:
        return "user-match"


def simulation(
    *,
    task_id: str,
    reward: float,
    basis: list[str],
    db_match: bool,
    action_match: bool,
    actual_argument: str = "observed",
    expected_tool: str = "mutate_record",
    workflow_items: list[dict[str, object]] | None = None,
    workflow_handoffs: dict[str, int] | None = None,
    product_selection_transition_rejections: int = 0,
    tool_error: bool = False,
) -> dict[str, object]:
    items = [] if workflow_items is None else workflow_items
    return {
        "task_id": task_id,
        "termination_reason": "user_stop",
        "reward_info": {
            "reward": reward,
            "reward_basis": basis,
            "db_check": {"db_match": db_match, "db_reward": float(db_match)},
            "action_checks": [
                {
                    "action": {
                        "name": expected_tool,
                        "arguments": {"secret": "gold-value"},
                        "compare_args": ["secret"],
                    },
                    "action_match": action_match,
                    "action_reward": float(action_match),
                    "tool_type": "write",
                }
            ],
            "env_assertions": None,
            "nl_assertions": None,
            "communicate_checks": None,
        },
        "messages": [
            {
                "role": "assistant",
                "content": "private transcript content",
                "tool_calls": [
                    {
                        "id": f"call-{task_id}",
                        "name": "mutate_record",
                        "arguments": {"secret": actual_argument},
                        "requestor": "assistant",
                    }
                ],
                "raw_data": {
                    "cayu_tau3_workflow": {
                        "revision": 1,
                        "items": items,
                        "customer_response_handoffs": workflow_handoffs or {},
                        "product_selection_transition_rejections": (
                            product_selection_transition_rejections
                        ),
                    },
                    "cayu_tau3_memory": {
                        "knowledge_retrieval": {
                            "embedding_dimensions": 3072,
                            "embedding_model": "text-embedding-3-large",
                            "embedding_representation": "cayu.tau3.banking.title-body.v1",
                            "fusion_configuration": (
                                "cayu.tau3.banking.bm25-semantic-offer-facets.v1"
                            ),
                            "mode": "hybrid",
                        },
                        "semantic_plan_verifier": {
                            "enabled": True,
                            "provider_calls": 1,
                            "acceptances": 1,
                            "rejections": 0,
                            "cache_hits": 0,
                            "prompt_tokens": 20,
                            "completion_tokens": 4,
                        },
                        "semantic_action_verifier": {
                            "enabled": True,
                            "provider_calls": 2,
                            "acceptances": 1,
                            "rejections": 1,
                            "cache_hits": 1,
                            "prompt_tokens": 30,
                            "completion_tokens": 5,
                        },
                        "semantic_response_verifier": {
                            "enabled": True,
                            "provider_calls": 3,
                            "acceptances": 2,
                            "rejections": 1,
                            "cache_hits": 1,
                            "prompt_tokens": 40,
                            "completion_tokens": 6,
                        },
                    },
                    "cayu_tau3_bridge": {
                        "internal_calls_by_tool": {
                            "cayu_search_banking_knowledge": 1,
                            "controller_sync_work_plan": 2,
                        },
                        "internal_continuations": 2,
                        "internal_tool_rounds": 1,
                        "internal_context_refreshes": 1,
                        "staged_external_batches": 1,
                        "staged_external_calls": 2,
                        "provider_call_count": 4,
                        "provider_usages": [
                            {
                                "input_tokens": 100,
                                "output_tokens": 20,
                                "reasoning_output_tokens": 5,
                                "cached_input_tokens": 10,
                                "attribution": {"secret": "private-provider-detail"},
                            },
                            {
                                "input_tokens": 120,
                                "output_tokens": 30,
                                "output_tokens_details": {"reasoning_tokens": 7},
                                "input_tokens_details": {"cached_tokens": 15},
                            },
                        ],
                        "provider_call_trace": {
                            "schema_version": 1,
                            "calls": [
                                {
                                    "forced_tool": False,
                                    "message_bytes": 1000,
                                    "message_count": 4,
                                    "phase": "initial",
                                    "result_kind": "tool_calls",
                                    "system_message_bytes": 600,
                                    "system_message_count": 2,
                                    "tool_schema_bytes": 400,
                                    "tool_schema_count": 3,
                                },
                                {
                                    "forced_tool": True,
                                    "message_bytes": 1400,
                                    "message_count": 6,
                                    "phase": "internal_tool_result",
                                    "result_kind": "terminal_text",
                                    "system_message_bytes": 700,
                                    "system_message_count": 2,
                                    "tool_schema_bytes": 450,
                                    "tool_schema_count": 4,
                                },
                            ],
                        },
                        "empty_response_retries": 0,
                        "internal_remediation": {
                            "schema_version": 1,
                            "issued": 1,
                            "repeated": 0,
                            "resolved": 1,
                            "superseded": 0,
                            "unresolved": 0,
                            "exhausted": 0,
                            "repair_rounds_total": 1,
                            "repair_rounds_max": 1,
                            "rejection_codes": {"selected_product_requires_advancement": 1},
                            "accepted_repair_classes": {"customer_tool_handoff": 1},
                        },
                    },
                },
            },
            {
                "role": "tool",
                "id": f"call-{task_id}",
                "content": "private result content",
                "error": tool_error,
            },
        ],
    }


def test_analysis_separates_scoring_failures_from_reference_action_diagnostics() -> None:
    report = analyze_results(
        {
            "simulations": [
                simulation(
                    task_id="db-failure",
                    reward=0,
                    basis=["DB"],
                    db_match=False,
                    action_match=False,
                ),
                simulation(
                    task_id="action-failure",
                    reward=0,
                    basis=["ACTION"],
                    db_match=True,
                    action_match=False,
                ),
            ]
        }
    )

    assert report["aggregate"]["scoring_failure_components"] == {
        "ACTION_MISMATCH": 1,
        "DB_MISMATCH": 1,
    }
    audit = report["aggregate"]["reference_action_audit"]
    assert audit["total"] == 2
    assert audit["missed"] == 2
    assert audit["by_tool_type"]["write"] == {"total": 2, "matched": 0, "missed": 2}
    assert audit["scoring_action_tasks"] == 1
    assert audit["scoring_action_checks"] == {"total": 1, "matched": 0, "missed": 1}
    assert report["aggregate"]["bridge"]["internal_calls_by_tool"] == {
        "cayu_search_banking_knowledge": 2,
        "controller_sync_work_plan": 4,
    }
    assert report["aggregate"]["bridge"]["staged_external_batches"] == 2
    assert report["aggregate"]["bridge"]["staged_external_calls"] == 4
    assert report["aggregate"]["bridge"]["internal_context_refreshes"] == 2
    assert report["aggregate"]["bridge"]["acting_prompt_tokens"] == 440
    assert report["aggregate"]["bridge"]["acting_completion_tokens"] == 100
    assert report["aggregate"]["bridge"]["acting_reasoning_tokens"] == 24
    assert report["aggregate"]["bridge"]["acting_cached_input_tokens"] == 50
    assert report["aggregate"]["bridge"]["provider_calls_by_phase"] == {
        "initial": 2,
        "internal_tool_result": 2,
    }
    assert report["aggregate"]["bridge"]["provider_results_by_kind"] == {
        "terminal_text": 2,
        "tool_calls": 2,
    }
    assert report["aggregate"]["bridge"]["provider_calls_without_trace"] == 4
    assert report["aggregate"]["bridge"]["provider_calls_without_usage"] == 4
    assert report["aggregate"]["bridge"]["provider_request_message_bytes"] == 4800
    assert report["aggregate"]["bridge"]["maximum_provider_request_message_bytes"] == 1400
    assert (
        report["simulations"][0]["bridge"]["provider_turns"][0]["provider_forced_tool_calls"] == 1
    )
    assert report["aggregate"]["bridge"]["internal_remediation"] == {
        "accepted_repair_classes": {"customer_tool_handoff": 2},
        "exhausted": 0,
        "issued": 2,
        "rejection_codes": {"selected_product_requires_advancement": 2},
        "repair_rounds_max": 1,
        "repair_rounds_total": 2,
        "repeated": 0,
        "resolved": 2,
        "superseded": 0,
        "unresolved": 0,
    }
    assert report["aggregate"]["semantic_plan_verifier"] == {
        "enabled_simulations": 2,
        "provider_calls": 2,
        "acceptances": 2,
        "rejections": 0,
        "cache_hits": 0,
        "prompt_tokens": 40,
        "completion_tokens": 8,
        "explicit_operation_floor_rejections": 0,
    }
    assert report["aggregate"]["semantic_action_verifier"] == {
        "enabled_simulations": 2,
        "provider_calls": 4,
        "acceptances": 2,
        "rejections": 2,
        "cache_hits": 2,
        "prompt_tokens": 60,
        "completion_tokens": 10,
    }
    assert report["aggregate"]["semantic_response_verifier"] == {
        "enabled_simulations": 2,
        "provider_calls": 6,
        "acceptances": 4,
        "rejections": 2,
        "cache_hits": 2,
        "prompt_tokens": 80,
        "completion_tokens": 12,
    }
    assert report["aggregate"]["knowledge_retrieval"] == {
        "embedding_dimensions": {"3072": 2},
        "embedding_models": {"text-embedding-3-large": 2},
        "embedding_representations": {"cayu.tau3.banking.title-body.v1": 2},
        "fusion_configurations": {"cayu.tau3.banking.bm25-semantic-offer-facets.v1": 2},
        "modes": {"hybrid": 2},
    }
    assert report["simulations"][0]["memory"]["knowledge_retrieval"] == {
        "embedding_dimensions": 3072,
        "embedding_model": "text-embedding-3-large",
        "embedding_representation": "cayu.tau3.banking.title-body.v1",
        "fusion_configuration": "cayu.tau3.banking.bm25-semantic-offer-facets.v1",
        "mode": "hybrid",
    }


def test_analysis_classifies_absent_tool_and_same_tool_argument_mismatch() -> None:
    mismatched = simulation(
        task_id="wrong-arguments",
        reward=0,
        basis=["ACTION"],
        db_match=True,
        action_match=False,
    )
    absent = simulation(
        task_id="not-called",
        reward=0,
        basis=["ACTION"],
        db_match=True,
        action_match=False,
        expected_tool="different_tool",
    )

    report = analyze_results({"simulations": [mismatched, absent]})

    assert report["aggregate"]["reference_action_audit"]["miss_reasons"] == {
        "SAME_TOOL_NONMATCHING_ARGUMENTS": 1,
        "TOOL_NOT_CALLED": 1,
    }


def test_analysis_reports_workflow_false_positives_before_terminal_gate() -> None:
    actionable = [
        {
            "id": "work-1",
            "goal": "complete work",
            "completion_mode": "operation",
            "status": "pending",
            "depends_on": [],
            "evidence": "",
        }
    ]
    report = analyze_results(
        {
            "simulations": [
                simulation(
                    task_id="failed",
                    reward=0,
                    basis=["DB"],
                    db_match=False,
                    action_match=False,
                    workflow_items=actionable,
                    workflow_handoffs={"work-1": 2},
                    product_selection_transition_rejections=2,
                ),
                simulation(
                    task_id="successful",
                    reward=1,
                    basis=["DB"],
                    db_match=True,
                    action_match=True,
                    workflow_items=actionable,
                    workflow_handoffs={"work-1": 2},
                ),
            ]
        }
    )

    workflow = report["aggregate"]["workflow"]
    assert workflow["failures_actionable_at_last_response"] == 1
    assert workflow["successes_actionable_at_last_response"] == 1
    assert workflow["text_responses_while_actionable"] == 2
    assert workflow["operation_customer_response_handoffs"] == 4
    assert workflow["tasks_with_operation_customer_response_handoffs"] == 2
    assert workflow["maximum_operation_customer_response_handoffs"] == 2
    assert workflow["product_selection_transition_rejections"] == 2
    assert workflow["tasks_with_product_selection_transition_rejections"] == 1
    simulation_workflow = report["simulations"][0]["workflow"]
    assert simulation_workflow["product_selection_transition_rejections"] == 2
    assert simulation_workflow["completion_mode_counts"] == {"operation": 1}
    assert simulation_workflow["status_counts_by_completion_mode"] == {"operation": {"pending": 1}}
    serialized = json.dumps(report)
    assert "work-1" not in serialized
    assert "complete work" not in serialized


def test_analysis_counts_tool_errors_and_redacts_oracle_payloads() -> None:
    report = analyze_results(
        {
            "simulations": [
                simulation(
                    task_id="errored",
                    reward=0,
                    basis=["ACTION"],
                    db_match=True,
                    action_match=False,
                    tool_error=True,
                )
            ]
        }
    )

    execution = report["aggregate"]["tool_execution"]
    assert execution["error_results"] == 1
    assert execution["errors_by_tool"] == {"mutate_record": 1}
    serialized = json.dumps(report)
    assert "gold-value" not in serialized
    assert "private transcript content" not in serialized
    assert "private result content" not in serialized
    assert "private-provider-detail" not in serialized


def test_analysis_collapses_unrecognized_remediation_labels() -> None:
    item = simulation(
        task_id="remediation-labels",
        reward=1,
        basis=["DB"],
        db_match=True,
        action_match=True,
    )
    lifecycle = item["messages"][0]["raw_data"]["cayu_tau3_bridge"]["internal_remediation"]
    lifecycle["rejection_codes"] = {"customer_account_secret": 2}
    lifecycle["accepted_repair_classes"] = {"private_customer_instruction": 1}

    report = analyze_results({"simulations": [item]})

    remediation = report["aggregate"]["bridge"]["internal_remediation"]
    assert remediation["rejection_codes"] == {"other": 2}
    assert remediation["accepted_repair_classes"] == {"other": 1}
    serialized = json.dumps(report)
    assert "customer_account_secret" not in serialized
    assert "private_customer_instruction" not in serialized


def test_analysis_rejects_duplicate_task_coordinates() -> None:
    item = simulation(
        task_id="duplicate",
        reward=1,
        basis=["DB"],
        db_match=True,
        action_match=True,
    )

    with pytest.raises(FailureAnalysisError, match="duplicate task/trial coordinates"):
        analyze_results({"simulations": [item, item]})


def test_analysis_supports_repeated_trials_and_computes_pass_hat_k() -> None:
    trials = []
    for task_id, rewards in {"a": (1, 1), "b": (1, 0)}.items():
        for trial, reward in enumerate(rewards):
            item = simulation(
                task_id=task_id,
                reward=reward,
                basis=["DB"],
                db_match=bool(reward),
                action_match=bool(reward),
            )
            item["trial"] = trial
            trials.append(item)

    report = analyze_results({"simulations": trials})

    assert report["aggregate"]["tasks"] == 2
    assert report["aggregate"]["simulations"] == 4
    assert report["aggregate"]["pass_hat_k"] == {"1": 0.75, "2": 0.5}


def test_analysis_retains_infrastructure_outcome_without_reward_info() -> None:
    report = analyze_results(
        {
            "simulations": [
                {
                    "task_id": "infra",
                    "trial": 0,
                    "termination_reason": "infrastructure_error",
                    "reward_info": None,
                    "messages": [],
                }
            ]
        }
    )

    item = report["simulations"][0]
    assert report["schema_version"] == 5
    assert item["reward"] is None
    assert item["scoring_failure_components"] == [
        "PREMATURE_TERMINATION",
        "MISSING_REWARD_INFO",
    ]
    assert item["bridge"]["internal_remediation"] == {
        "accepted_repair_classes": {},
        "exhausted": 0,
        "issued": 0,
        "rejection_codes": {},
        "repair_rounds_max": 0,
        "repair_rounds_total": 0,
        "repeated": 0,
        "resolved": 0,
        "superseded": 0,
        "unresolved": 0,
    }


def test_deep_replay_does_not_treat_missing_infrastructure_db_check_as_false() -> None:
    task = SimpleNamespace(
        id="infra",
        evaluation_criteria=SimpleNamespace(actions=[], reward_basis=[]),
        initial_state=None,
    )
    simulation = SimpleNamespace(task_id="infra", trial=0, messages=[], reward_info=None)
    results = SimpleNamespace(
        tasks=[task],
        simulations=[simulation],
        info=SimpleNamespace(
            retrieval_config="no_knowledge",
            retrieval_config_kwargs={},
        ),
    )
    environments = iter((_ReplayEnvironment(), _ReplayEnvironment()))

    replay = _deep_replay(
        results,
        get_environment=lambda **_kwargs: next(environments),
        repository=Path("/pinned/tau2"),
    )

    assert replay["stored_result_consistent"] is True
    assert replay["inconsistent_task_ids"] == []
    assert replay["stored_results_compared"] == 0
    assert replay["replays_without_stored_db_check"] == 1
    assert replay["tasks"][0]["stored_db_match"] is None
    assert replay["tasks"][0]["replayed_db_match"] is True


def test_deep_replay_still_reports_a_real_stored_db_disagreement() -> None:
    task = SimpleNamespace(
        id="scored",
        evaluation_criteria=SimpleNamespace(
            actions=[],
            reward_basis=[SimpleNamespace(value="DB")],
        ),
        initial_state=None,
    )
    simulation = SimpleNamespace(
        task_id="scored",
        trial=0,
        messages=[],
        reward_info=SimpleNamespace(db_check=SimpleNamespace(db_match=False)),
    )
    results = SimpleNamespace(
        tasks=[task],
        simulations=[simulation],
        info=SimpleNamespace(retrieval_config="no_knowledge", retrieval_config_kwargs={}),
    )
    environments = iter((_ReplayEnvironment(), _ReplayEnvironment()))

    replay = _deep_replay(
        results,
        get_environment=lambda **_kwargs: next(environments),
        repository=Path("/pinned/tau2"),
    )

    assert replay["stored_result_consistent"] is False
    assert replay["inconsistent_task_ids"] == ["scored"]
    assert replay["stored_results_compared"] == 1
    assert replay["replays_without_stored_db_check"] == 0
    assert replay["tasks"][0]["stored_db_match"] is False
    assert replay["tasks"][0]["replayed_db_match"] is True


def test_database_diff_reports_only_counts_and_classifies_shape() -> None:
    expected = {
        "accounts": {
            "data": {
                "missing-id": {"state": "expected"},
                "changed-id": {"state": "expected", "balance": 1},
            },
            "notes": "",
        }
    }
    predicted = {
        "accounts": {
            "data": {
                "changed-id": {"state": "actual", "balance": 1},
                "extra-id": {"state": "extra"},
            },
            "notes": "",
        }
    }

    diff = _diff_database(expected, predicted)

    assert diff == [
        {
            "table": "accounts",
            "counts": {
                "missing_records": 1,
                "unexpected_records": 1,
                "changed_records": 1,
                "changed_fields": 1,
                "metadata_changed": 0,
            },
        }
    ]
    assert _classify_db_diff(diff) == "MIXED_STATE_DIFFERENCE"
    assert "missing-id" not in json.dumps(diff)
    assert "extra-id" not in json.dumps(diff)


@pytest.mark.parametrize(
    ("expected", "predicted", "classification"),
    [
        ({"t": {"data": {"a": {}}}}, {"t": {"data": {}}}, "MISSING_EXPECTED_STATE"),
        ({"t": {"data": {}}}, {"t": {"data": {"a": {}}}}, "UNEXPECTED_STATE"),
        (
            {"t": {"data": {"a": {"value": 1}}}},
            {"t": {"data": {"a": {"value": 2}}}},
            "CHANGED_STATE",
        ),
        ({"t": {"data": {}}}, {"t": {"data": {}}}, "MATCH"),
    ],
)
def test_database_diff_classifications(
    expected: dict[str, object], predicted: dict[str, object], classification: str
) -> None:
    assert _classify_db_diff(_diff_database(expected, predicted)) == classification


def test_database_state_profile_separates_controller_and_business_tables() -> None:
    control = [{"table": "verification_history", "counts": {}}]
    business = [{"table": "accounts", "counts": {}}]

    assert _state_profile(control) == "CONTROL_ONLY"
    assert _state_profile(business) == "BUSINESS_ONLY"
    assert _state_profile(control + business) == "CONTROL_AND_BUSINESS"
    assert _state_profile([]) == "MATCH"
