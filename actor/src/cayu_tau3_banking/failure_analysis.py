"""Evaluation-only diagnostics for saved tau banking trajectories.

This module consumes completed ``tau2`` result artifacts. It is deliberately not
imported by the acting agent and never emits task scenarios, expected arguments,
database record identifiers, message content, or tool-result content. The official
tau reward remains authoritative; these diagnostics only explain correlations and
replay-visible state differences after a run.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cayu_tau3_banking.attempt_diagnostics import (
    SAFE_INTERNAL_REMEDIATION_CODES,
    SAFE_INTERNAL_REPAIR_CLASSES,
)

_SCHEMA_VERSION = 5
_NORMAL_TERMINATIONS = {"agent_stop", "user_stop"}
_ACTIONABLE_STATUSES = {"pending", "in_progress"}
_UNRESOLVED_STATUSES = _ACTIONABLE_STATUSES | {"awaiting_customer", "blocked"}
_CONTROL_STATE_TABLES = {
    "agent_discoverable_tools",
    "user_discoverable_tool_calls",
    "user_discoverable_tools",
    "verification_history",
}
_DEFAULT_TAU2_REPOSITORY = Path(__file__).resolve().parents[2] / ".vendor" / "tau2"
_REMEDIATION_TOTAL_FIELDS = (
    "issued",
    "repeated",
    "resolved",
    "superseded",
    "unresolved",
    "exhausted",
    "repair_rounds_total",
    "repair_rounds_max",
)
_SAFE_EMBEDDING_MODELS = {
    "text-embedding-3-large",
    "text-embedding-3-small",
}
_SAFE_FUSION_CONFIGURATIONS = {
    "cayu.tau3.banking.bm25-lexical.v1",
    "cayu.tau3.banking.bm25-semantic-offer-facets.v1",
}
_SAFE_EMBEDDING_REPRESENTATIONS = {"cayu.tau3.banking.title-body.v1"}
_SAFE_PROVIDER_CALL_PHASES = {
    "empty_response_retry",
    "external_admission_correction",
    "initial",
    "internal_tool_result",
    "provider_continuation",
    "terminal_readiness_correction",
}
_SAFE_PROVIDER_RESULT_KINDS = {
    "continuation_text",
    "empty",
    "mixed_tool_calls",
    "terminal_text",
    "tool_calls",
}
_PROVIDER_MAX_FIELDS = (
    "maximum_provider_request_message_bytes",
    "maximum_provider_request_message_count",
    "maximum_provider_request_system_message_bytes",
    "maximum_provider_request_tool_schema_bytes",
    "maximum_provider_request_tool_schema_count",
)
_MAX_PROVIDER_CALL_TRACE_RECORDS = 128


class FailureAnalysisError(ValueError):
    """A saved result artifact is malformed or cannot be analyzed safely."""


def analyze_results(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return oracle-redacted aggregate and per-task diagnostics.

    The analyzer distinguishes scoring failures from reference-trajectory action
    mismatches. In tau, action checks are authoritative only when ``ACTION`` appears
    in the task's reward basis; for DB-scored tasks they are diagnostic correlations.
    """

    root = _mapping(payload, "results")
    raw_simulations = _sequence(root.get("simulations"), "results.simulations")
    simulations = [
        _analyze_simulation(_mapping(raw, f"results.simulations[{index}]"))
        for index, raw in enumerate(raw_simulations)
    ]
    coordinates = [(item["task_id"], item["trial"]) for item in simulations]
    if len(coordinates) != len(set(coordinates)):
        raise FailureAnalysisError("results contains duplicate task/trial coordinates")

    reward_values = [item["reward"] for item in simulations if item["reward"] is not None]
    successes = sum(item["success"] for item in simulations)
    failures = len(simulations) - successes
    scoring_failures: Counter[str] = Counter()
    termination_reasons: Counter[str] = Counter()
    actual_tool_calls: Counter[str] = Counter()
    tool_errors: Counter[str] = Counter()
    reference_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    reference_miss_reasons: Counter[str] = Counter()
    reference_misses_by_tool: Counter[str] = Counter()
    scoring_action_totals = Counter()
    bridge_totals = Counter()
    bridge_internal_calls: Counter[str] = Counter()
    bridge_provider_phases: Counter[str] = Counter()
    bridge_provider_results: Counter[str] = Counter()
    bridge_provider_maxima = Counter()
    bridge_remediation_totals = Counter()
    bridge_remediation_codes: Counter[str] = Counter()
    bridge_repair_classes: Counter[str] = Counter()
    plan_verifier_totals = Counter()
    plan_verifier_enabled_simulations = 0
    action_verifier_totals = Counter()
    action_verifier_enabled_simulations = 0
    response_verifier_totals = Counter()
    response_verifier_enabled_simulations = 0
    retrieval_modes: Counter[str] = Counter()
    embedding_models: Counter[str] = Counter()
    embedding_dimensions: Counter[str] = Counter()
    embedding_representations: Counter[str] = Counter()
    fusion_configurations: Counter[str] = Counter()

    for item in simulations:
        scoring_failures.update(item["scoring_failure_components"])
        termination_reasons[item["termination_reason"]] += 1
        actual_tool_calls.update(item["tool_execution"]["calls_by_tool"])
        tool_errors.update(item["tool_execution"]["errors_by_tool"])
        for tool_type, counts in item["reference_actions"]["by_tool_type"].items():
            reference_by_type[tool_type].update(counts)
        reference_miss_reasons.update(item["reference_actions"]["miss_reasons"])
        reference_misses_by_tool.update(item["reference_actions"]["misses_by_tool"])
        if "ACTION" in item["reward_basis"]:
            scoring_action_totals.update(
                {
                    "total": item["reference_actions"]["total"],
                    "matched": item["reference_actions"]["matched"],
                    "missed": item["reference_actions"]["missed"],
                }
            )
        bridge = dict(item["bridge"])
        bridge.pop("provider_turns", None)
        bridge_internal_calls.update(bridge.pop("internal_calls_by_tool", {}))
        bridge_provider_phases.update(bridge.pop("provider_calls_by_phase", {}))
        bridge_provider_results.update(bridge.pop("provider_results_by_kind", {}))
        for field in _PROVIDER_MAX_FIELDS:
            bridge_provider_maxima[field] = max(
                bridge_provider_maxima[field],
                bridge.pop(field, 0),
            )
        remediation = dict(bridge.pop("internal_remediation", {}))
        bridge_remediation_codes.update(remediation.pop("rejection_codes", {}))
        bridge_repair_classes.update(remediation.pop("accepted_repair_classes", {}))
        bridge_remediation_totals["repair_rounds_max"] = max(
            bridge_remediation_totals["repair_rounds_max"],
            remediation.pop("repair_rounds_max", 0),
        )
        bridge_remediation_totals.update(remediation)
        bridge_totals.update(bridge)
        plan_verifier = item["memory"]["semantic_plan_verifier"]
        plan_verifier_enabled_simulations += plan_verifier["enabled"]
        plan_verifier_totals.update(
            {key: value for key, value in plan_verifier.items() if key != "enabled"}
        )
        action_verifier = item["memory"]["semantic_action_verifier"]
        action_verifier_enabled_simulations += action_verifier["enabled"]
        action_verifier_totals.update(
            {key: value for key, value in action_verifier.items() if key != "enabled"}
        )
        response_verifier = item["memory"]["semantic_response_verifier"]
        response_verifier_enabled_simulations += response_verifier["enabled"]
        response_verifier_totals.update(
            {key: value for key, value in response_verifier.items() if key != "enabled"}
        )
        retrieval = item["memory"]["knowledge_retrieval"]
        retrieval_modes[retrieval["mode"]] += 1
        embedding_models[retrieval["embedding_model"] or "none"] += 1
        embedding_dimensions[str(retrieval["embedding_dimensions"] or "none")] += 1
        embedding_representations[retrieval["embedding_representation"] or "none"] += 1
        fusion_configurations[retrieval["fusion_configuration"] or "none"] += 1

    reference_totals = Counter()
    for counts in reference_by_type.values():
        reference_totals.update(counts)

    aggregate = {
        "tasks": len({item["task_id"] for item in simulations}),
        "simulations": len(simulations),
        "successes": successes,
        "failures": failures,
        "mean_reward": sum(reward_values) / len(reward_values) if reward_values else 0.0,
        "pass_hat_k": _pass_hat_k(simulations),
        "scoring_failure_components": _ordered_counts(scoring_failures),
        "termination_reasons": _ordered_counts(termination_reasons),
        "tool_execution": {
            "calls": sum(actual_tool_calls.values()),
            "calls_by_tool": _ordered_counts(actual_tool_calls),
            "error_results": sum(tool_errors.values()),
            "errors_by_tool": _ordered_counts(tool_errors),
        },
        "reference_action_audit": {
            "scope_warning": (
                "Reference actions determine DB target state but are scoring requirements only "
                "for tasks whose reward basis contains ACTION."
            ),
            "total": reference_totals["total"],
            "matched": reference_totals["matched"],
            "missed": reference_totals["missed"],
            "by_tool_type": {
                key: _action_counts(reference_by_type[key]) for key in sorted(reference_by_type)
            },
            "miss_reasons": _ordered_counts(reference_miss_reasons),
            "misses_by_tool": _ordered_counts(reference_misses_by_tool),
            "scoring_action_tasks": sum("ACTION" in item["reward_basis"] for item in simulations),
            "scoring_action_checks": _action_counts(scoring_action_totals),
        },
        "workflow": _aggregate_workflow(simulations),
        "bridge": {
            **dict(bridge_totals),
            **dict(bridge_provider_maxima),
            "internal_calls_by_tool": _ordered_counts(bridge_internal_calls),
            "provider_calls_by_phase": _ordered_counts(bridge_provider_phases),
            "provider_results_by_kind": _ordered_counts(bridge_provider_results),
            "internal_remediation": {
                **{field: bridge_remediation_totals[field] for field in _REMEDIATION_TOTAL_FIELDS},
                "rejection_codes": _ordered_counts(bridge_remediation_codes),
                "accepted_repair_classes": _ordered_counts(bridge_repair_classes),
            },
        },
        "semantic_plan_verifier": {
            "enabled_simulations": plan_verifier_enabled_simulations,
            **dict(plan_verifier_totals),
        },
        "semantic_action_verifier": {
            "enabled_simulations": action_verifier_enabled_simulations,
            **dict(action_verifier_totals),
        },
        "semantic_response_verifier": {
            "enabled_simulations": response_verifier_enabled_simulations,
            **dict(response_verifier_totals),
        },
        "knowledge_retrieval": {
            "modes": _ordered_counts(retrieval_modes),
            "embedding_models": _ordered_counts(embedding_models),
            "embedding_dimensions": _ordered_counts(embedding_dimensions),
            "embedding_representations": _ordered_counts(embedding_representations),
            "fusion_configurations": _ordered_counts(fusion_configurations),
        },
        "correlations": _aggregate_correlations(simulations),
    }
    return {
        "schema_version": _SCHEMA_VERSION,
        "integrity": {
            "evaluation_only": True,
            "official_tau_reward_is_authoritative": True,
            "raw_oracle_payloads_emitted": False,
            "prohibited_runtime_inputs": [
                "task scenario",
                "expected arguments",
                "database record identifiers",
                "message content",
                "tool-result content",
            ],
        },
        "aggregate": aggregate,
        "simulations": simulations,
    }


def attach_deep_replay(
    report: dict[str, Any],
    *,
    results_path: Path,
    tau2_repository: Path,
) -> None:
    """Attach redacted database diffs produced by the pinned tau environment replay."""

    from loguru import logger

    repository = tau2_repository.resolve()
    source_root = (repository / "src").resolve()
    logger.disable("tau2")
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            sys.path.insert(0, str(source_root))
            try:
                import tau2
                from tau2.data_model.simulation import Results
                from tau2.domains.banking_knowledge.environment import get_environment

                from cayu_tau3_banking.provenance import assert_source_provenance

                assert_source_provenance(repository)
            finally:
                sys.path.pop(0)
            module_path = Path(tau2.__file__).resolve()
            if not module_path.is_relative_to(source_root):
                raise FailureAnalysisError(
                    f"loaded tau2 from {module_path}, outside pinned source root {source_root}"
                )
            replay = _deep_replay(
                Results.load(results_path),
                get_environment=get_environment,
                repository=repository,
            )
    finally:
        logger.enable("tau2")
    report["deep_replay"] = replay


def _deep_replay(results: Any, *, get_environment: Any, repository: Path) -> dict[str, Any]:
    tasks_by_id = {str(task.id): task for task in results.tasks}
    per_task: list[dict[str, Any]] = []
    table_impacts: dict[str, Counter[str]] = defaultdict(Counter)
    scoring_table_impacts: dict[str, Counter[str]] = defaultdict(Counter)
    classifications: Counter[str] = Counter()
    scoring_db_classifications: Counter[str] = Counter()
    scoring_db_profiles: Counter[str] = Counter()
    replay_inconsistencies: list[str] = []
    stored_results_compared = 0
    replays_without_stored_db_check = 0

    for simulation in results.simulations:
        task_id = str(simulation.task_id)
        task = tasks_by_id.get(task_id)
        if task is None:
            raise FailureAnalysisError(f"simulation {task_id!r} has no corresponding task")
        evaluation = task.evaluation_criteria
        actions = [] if evaluation is None else (evaluation.actions or [])
        allowlist = {
            action.arguments.get("agent_tool_name")
            for action in actions
            if action.name == "call_discoverable_agent_tool"
            and action.arguments.get("agent_tool_name")
        }
        env_kwargs = {
            "retrieval_variant": results.info.retrieval_config,
            "retrieval_kwargs": results.info.retrieval_config_kwargs,
            "task": task,
            "read_log_allowlist": allowlist,
        }
        initial = task.initial_state
        initialization_data = None if initial is None else initial.initialization_data
        initialization_actions = None if initial is None else initial.initialization_actions
        initial_messages = [] if initial is None else (initial.message_history or [])

        predicted = get_environment(**env_kwargs)
        predicted.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=simulation.messages or [],
            strict=False,
        )
        expected = get_environment(**env_kwargs)
        expected.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=initial_messages,
            strict=False,
        )
        golden_action_errors = 0
        for action in actions:
            try:
                expected.make_tool_call(
                    tool_name=action.name,
                    requestor=action.requestor,
                    **action.arguments,
                )
            except Exception:
                # Match the official environment evaluator, which warns and continues.
                golden_action_errors += 1

        agent_match = predicted.get_db_hash() == expected.get_db_hash()
        user_match = predicted.get_user_db_hash() == expected.get_user_db_hash()
        replay_match = agent_match and user_match
        stored_match = (
            None
            if simulation.reward_info is None or simulation.reward_info.db_check is None
            else bool(simulation.reward_info.db_check.db_match)
        )
        if stored_match is None:
            replays_without_stored_db_check += 1
        else:
            stored_results_compared += 1
        if stored_match is not None and replay_match != stored_match:
            replay_inconsistencies.append(task_id)

        predicted_db = predicted.tools.db.model_dump()
        expected_db = expected.tools.db.model_dump()
        table_diffs = _diff_database(expected_db, predicted_db)
        classification = _classify_db_diff(table_diffs)
        classifications[classification] += 1
        reward_basis = (
            [] if evaluation is None else [item.value for item in evaluation.reward_basis]
        )
        scoring_db_failure = "DB" in reward_basis and not replay_match
        if scoring_db_failure:
            scoring_db_classifications[classification] += 1
        for table in table_diffs:
            counts = table["counts"]
            table_impacts[table["table"]].update(
                {
                    "tasks": 1,
                    "missing_records": counts["missing_records"],
                    "unexpected_records": counts["unexpected_records"],
                    "changed_records": counts["changed_records"],
                    "changed_fields": counts["changed_fields"],
                }
            )
            if scoring_db_failure:
                scoring_table_impacts[table["table"]].update(
                    {
                        "tasks": 1,
                        "missing_records": counts["missing_records"],
                        "unexpected_records": counts["unexpected_records"],
                        "changed_records": counts["changed_records"],
                        "changed_fields": counts["changed_fields"],
                    }
                )
        if scoring_db_failure:
            scoring_db_profiles[_state_profile(table_diffs)] += 1
        per_task.append(
            {
                "task_id": task_id,
                "trial": simulation.trial,
                "stored_db_match": stored_match,
                "replayed_db_match": replay_match,
                "agent_db_match": agent_match,
                "user_db_match": user_match,
                "classification": classification,
                "golden_action_errors": golden_action_errors,
                "table_diffs": table_diffs,
            }
        )

    return {
        "pinned_tau2_repository": str(repository),
        "simulations_replayed": len(per_task),
        "tasks_replayed": len({item["task_id"] for item in per_task}),
        "stored_results_compared": stored_results_compared,
        "replays_without_stored_db_check": replays_without_stored_db_check,
        "stored_result_consistent": not replay_inconsistencies,
        "inconsistent_task_ids": sorted(replay_inconsistencies),
        "db_diff_classifications": _ordered_counts(classifications),
        "scoring_db_failure_classifications": _ordered_counts(scoring_db_classifications),
        "scoring_db_failure_profiles": _ordered_counts(scoring_db_profiles),
        "table_impacts": {table: dict(table_impacts[table]) for table in sorted(table_impacts)},
        "scoring_table_impacts": {
            table: dict(scoring_table_impacts[table]) for table in sorted(scoring_table_impacts)
        },
        "tasks": per_task,
    }


def _analyze_simulation(simulation: Mapping[str, Any]) -> dict[str, Any]:
    task_id = simulation.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise FailureAnalysisError("every simulation requires a nonblank string task_id")
    raw_reward_info = simulation.get("reward_info")
    reward_info = (
        {}
        if raw_reward_info is None
        else _mapping(raw_reward_info, f"simulation {task_id}.reward_info")
    )
    reward = reward_info.get("reward")
    if reward is not None and (isinstance(reward, bool) or not isinstance(reward, int | float)):
        raise FailureAnalysisError(f"simulation {task_id} reward must be numeric or null")
    reward = None if reward is None else float(reward)
    success = reward is not None and abs(reward - 1.0) <= 1e-6
    trial = simulation.get("trial")
    if trial is not None and (type(trial) is not int or trial < 0):
        raise FailureAnalysisError(f"simulation {task_id} trial must be nonnegative or null")
    basis = _string_sequence(
        reward_info.get("reward_basis") or [], f"simulation {task_id}.reward_basis"
    )
    messages = [
        _mapping(item, f"simulation {task_id}.messages[{index}]")
        for index, item in enumerate(
            _sequence(simulation.get("messages") or [], f"simulation {task_id}.messages")
        )
    ]
    action_checks = [
        _mapping(item, f"simulation {task_id}.action_checks[{index}]")
        for index, item in enumerate(
            _sequence(
                reward_info.get("action_checks") or [],
                f"simulation {task_id}.action_checks",
            )
        )
    ]
    actual_calls = _actual_calls(messages)
    reference_actions = _reference_action_diagnostics(action_checks, actual_calls)
    workflow = _workflow_diagnostics(messages)
    termination = simulation.get("termination_reason")
    if not isinstance(termination, str) or not termination:
        raise FailureAnalysisError(f"simulation {task_id} termination_reason must be text")
    scoring_failures = _scoring_failures(
        reward_info=reward_info,
        reward_basis=basis,
        success=success,
        termination=termination,
    )
    return {
        "task_id": task_id,
        "trial": trial,
        "reward": reward,
        "success": success,
        "reward_basis": basis,
        "termination_reason": termination,
        "scoring_failure_components": scoring_failures,
        "tool_execution": _tool_execution(messages, actual_calls),
        "reference_actions": reference_actions,
        "workflow": workflow,
        "bridge": _bridge_diagnostics(messages),
        "memory": _memory_diagnostics(messages),
    }


def _scoring_failures(
    *,
    reward_info: Mapping[str, Any],
    reward_basis: Sequence[str],
    success: bool,
    termination: str,
) -> list[str]:
    failures: list[str] = []
    if termination not in _NORMAL_TERMINATIONS:
        failures.append("PREMATURE_TERMINATION")
    if not reward_info:
        failures.append("MISSING_REWARD_INFO")
        return failures
    if "DB" in reward_basis:
        db_check = reward_info.get("db_check")
        if not isinstance(db_check, Mapping) or db_check.get("db_match") is not True:
            failures.append("DB_MISMATCH")
    if "ACTION" in reward_basis:
        checks = reward_info.get("action_checks") or []
        if checks and any(
            not isinstance(check, Mapping) or check.get("action_match") is not True
            for check in checks
        ):
            failures.append("ACTION_MISMATCH")
    component_fields = {
        "ENV_ASSERTION": ("env_assertions", "met"),
        "NL_ASSERTION": ("nl_assertions", "met"),
        "COMMUNICATE": ("communicate_checks", "met"),
    }
    for basis_name, (field, predicate) in component_fields.items():
        if basis_name not in reward_basis:
            continue
        checks = reward_info.get(field) or []
        if not checks or any(
            not isinstance(check, Mapping) or check.get(predicate) is not True for check in checks
        ):
            failures.append(f"{basis_name}_MISMATCH")
    if not success and not failures:
        failures.append("UNEXPLAINED_REWARD_MISMATCH")
    return failures


def _actual_calls(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in messages:
        for raw_call in message.get("tool_calls") or []:
            if not isinstance(raw_call, Mapping):
                continue
            name = raw_call.get("name")
            if not isinstance(name, str) or not name:
                continue
            arguments = raw_call.get("arguments")
            calls.append(
                {
                    "id": raw_call.get("id"),
                    "name": name,
                    "arguments": dict(arguments) if isinstance(arguments, Mapping) else {},
                }
            )
    return calls


def _tool_execution(
    messages: Sequence[Mapping[str, Any]], actual_calls: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    calls_by_tool = Counter(call["name"] for call in actual_calls)
    names_by_id = {
        call["id"]: call["name"]
        for call in actual_calls
        if isinstance(call.get("id"), str) and call["id"]
    }
    errors_by_tool: Counter[str] = Counter()
    result_messages = 0
    for message in messages:
        if message.get("role") != "tool":
            continue
        result_messages += 1
        if message.get("error") is True:
            name = names_by_id.get(message.get("id"), "<unknown>")
            errors_by_tool[name] += 1
    return {
        "calls": len(actual_calls),
        "calls_by_tool": dict(calls_by_tool),
        "result_messages": result_messages,
        "error_results": sum(errors_by_tool.values()),
        "errors_by_tool": dict(errors_by_tool),
    }


def _reference_action_diagnostics(
    checks: Sequence[Mapping[str, Any]], actual_calls: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    miss_reasons: Counter[str] = Counter()
    misses_by_tool: Counter[str] = Counter()
    matched = 0
    for check in checks:
        action = check.get("action")
        if not isinstance(action, Mapping):
            raise FailureAnalysisError("action check requires an action object")
        name = action.get("name")
        if not isinstance(name, str) or not name:
            raise FailureAnalysisError("reference action requires a nonblank name")
        tool_type = check.get("tool_type")
        if not isinstance(tool_type, str) or not tool_type:
            tool_type = "unknown"
        by_type[tool_type]["total"] += 1
        if check.get("action_match") is True:
            matched += 1
            by_type[tool_type]["matched"] += 1
            continue
        by_type[tool_type]["missed"] += 1
        misses_by_tool[name] += 1
        same_name = any(call["name"] == name for call in actual_calls)
        miss_reasons["SAME_TOOL_NONMATCHING_ARGUMENTS" if same_name else "TOOL_NOT_CALLED"] += 1
    return {
        "total": len(checks),
        "matched": matched,
        "missed": len(checks) - matched,
        "by_tool_type": {key: dict(by_type[key]) for key in sorted(by_type)},
        "miss_reasons": dict(miss_reasons),
        "misses_by_tool": dict(misses_by_tool),
    }


def _workflow_diagnostics(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    snapshots: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    text_while_actionable = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        raw_data = message.get("raw_data")
        if not isinstance(raw_data, Mapping):
            continue
        workflow = raw_data.get("cayu_tau3_workflow")
        if not isinstance(workflow, Mapping):
            continue
        snapshots.append((message, workflow))
        statuses = set(_workflow_statuses(workflow))
        content = message.get("content")
        if isinstance(content, str) and content.strip() and statuses & _ACTIONABLE_STATUSES:
            text_while_actionable += 1
    if not snapshots:
        return {
            "snapshot_count": 0,
            "revision": 0,
            "item_count": 0,
            "status_counts": {},
            "completion_mode_counts": {},
            "status_counts_by_completion_mode": {},
            "operation_customer_response_handoffs": 0,
            "items_with_operation_customer_response_handoffs": 0,
            "maximum_operation_customer_response_handoffs": 0,
            "product_selection_transition_rejections": 0,
            "plan_created": False,
            "plan_populated": False,
            "actionable_at_last_response": False,
            "unresolved_at_last_response": False,
            "text_responses_while_actionable": 0,
        }
    last_message, last_workflow = snapshots[-1]
    revision = last_workflow.get("revision")
    if type(revision) is not int or revision < 0:
        revision = 0
    raw_statuses = _workflow_statuses(last_workflow)
    statuses = set(raw_statuses)
    status_counts = Counter(raw_statuses)
    items = last_workflow.get("items")
    item_count = len(items) if isinstance(items, list) else 0
    completion_mode_counts: Counter[str] = Counter()
    status_counts_by_completion_mode: dict[str, Counter[str]] = defaultdict(Counter)
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, Mapping):
                continue
            completion_mode = item.get("completion_mode")
            status = item.get("status")
            if not isinstance(completion_mode, str) or not completion_mode:
                continue
            completion_mode_counts[completion_mode] += 1
            if isinstance(status, str) and status:
                status_counts_by_completion_mode[completion_mode][status] += 1
    raw_handoffs = last_workflow.get("customer_response_handoffs")
    operation_handoffs = (
        [count for count in raw_handoffs.values() if type(count) is int and count >= 0]
        if isinstance(raw_handoffs, Mapping)
        else []
    )
    selection_rejections = last_workflow.get("product_selection_transition_rejections", 0)
    if type(selection_rejections) is not int or selection_rejections < 0:
        selection_rejections = 0
    last_content = last_message.get("content")
    return {
        "snapshot_count": len(snapshots),
        "revision": revision,
        "item_count": item_count,
        "status_counts": _ordered_counts(status_counts),
        "completion_mode_counts": _ordered_counts(completion_mode_counts),
        "status_counts_by_completion_mode": {
            mode: _ordered_counts(counts)
            for mode, counts in sorted(status_counts_by_completion_mode.items())
        },
        "operation_customer_response_handoffs": sum(operation_handoffs),
        "items_with_operation_customer_response_handoffs": sum(
            count > 0 for count in operation_handoffs
        ),
        "maximum_operation_customer_response_handoffs": max(operation_handoffs, default=0),
        "product_selection_transition_rejections": selection_rejections,
        "plan_created": revision > 0,
        "plan_populated": item_count > 0,
        "actionable_at_last_response": bool(statuses & _ACTIONABLE_STATUSES),
        "unresolved_at_last_response": bool(statuses & _UNRESOLVED_STATUSES),
        "last_response_was_text": isinstance(last_content, str) and bool(last_content.strip()),
        "text_responses_while_actionable": text_while_actionable,
    }


def _workflow_statuses(workflow: Mapping[str, Any]) -> tuple[str, ...]:
    items = workflow.get("items")
    if not isinstance(items, list):
        return ()
    return tuple(
        status
        for item in items
        if isinstance(item, Mapping) and isinstance((status := item.get("status")), str) and status
    )


def _memory_diagnostics(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    empty_plan = {
        "enabled": False,
        "provider_calls": 0,
        "acceptances": 0,
        "rejections": 0,
        "cache_hits": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "explicit_operation_floor_rejections": 0,
    }
    empty_action = {
        "enabled": False,
        "provider_calls": 0,
        "acceptances": 0,
        "rejections": 0,
        "cache_hits": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    empty_response = {
        "enabled": False,
        "provider_calls": 0,
        "acceptances": 0,
        "rejections": 0,
        "cache_hits": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        raw_data = message.get("raw_data")
        memory = raw_data.get("cayu_tau3_memory") if isinstance(raw_data, Mapping) else None
        if not isinstance(memory, Mapping):
            continue
        retrieval = _safe_knowledge_retrieval(memory.get("knowledge_retrieval"))
        verifier = memory.get("semantic_plan_verifier")
        if not isinstance(verifier, Mapping):
            return {
                "knowledge_retrieval": retrieval,
                "semantic_plan_verifier": empty_plan,
                "semantic_action_verifier": empty_action,
                "semantic_response_verifier": empty_response,
            }
        enabled = verifier.get("enabled") is True
        counts = {
            key: value
            for key in (
                "provider_calls",
                "acceptances",
                "rejections",
                "cache_hits",
                "prompt_tokens",
                "completion_tokens",
                "explicit_operation_floor_rejections",
            )
            if type(value := verifier.get(key)) is int and value >= 0
        }
        action_verifier = memory.get("semantic_action_verifier")
        action_enabled = (
            isinstance(action_verifier, Mapping) and action_verifier.get("enabled") is True
        )
        action_counts = (
            {
                key: value
                for key in (
                    "provider_calls",
                    "acceptances",
                    "rejections",
                    "cache_hits",
                    "prompt_tokens",
                    "completion_tokens",
                )
                if type(value := action_verifier.get(key)) is int and value >= 0
            }
            if isinstance(action_verifier, Mapping)
            else {}
        )
        response_verifier = memory.get("semantic_response_verifier")
        response_enabled = (
            isinstance(response_verifier, Mapping) and response_verifier.get("enabled") is True
        )
        response_counts = (
            {
                key: value
                for key in (
                    "provider_calls",
                    "acceptances",
                    "rejections",
                    "cache_hits",
                    "prompt_tokens",
                    "completion_tokens",
                )
                if type(value := response_verifier.get(key)) is int and value >= 0
            }
            if isinstance(response_verifier, Mapping)
            else {}
        )
        return {
            "knowledge_retrieval": retrieval,
            "semantic_plan_verifier": {**empty_plan, **counts, "enabled": enabled},
            "semantic_action_verifier": {
                **empty_action,
                **action_counts,
                "enabled": action_enabled,
            },
            "semantic_response_verifier": {
                **empty_response,
                **response_counts,
                "enabled": response_enabled,
            },
        }
    return {
        "knowledge_retrieval": _safe_knowledge_retrieval(None),
        "semantic_plan_verifier": empty_plan,
        "semantic_action_verifier": empty_action,
        "semantic_response_verifier": empty_response,
    }


def _safe_knowledge_retrieval(value: object) -> dict[str, Any]:
    """Return only bounded, allowlisted retrieval identity fields from run metadata."""

    if not isinstance(value, Mapping):
        return {
            "mode": "unknown",
            "embedding_model": None,
            "embedding_dimensions": None,
            "embedding_representation": None,
            "fusion_configuration": None,
        }
    raw_mode = value.get("mode")
    mode = raw_mode if raw_mode in {"lexical", "hybrid"} else "unknown"
    raw_model = value.get("embedding_model")
    if raw_model is None:
        model = None
    elif raw_model in _SAFE_EMBEDDING_MODELS:
        model = raw_model
    else:
        model = "other"
    raw_dimensions = value.get("embedding_dimensions")
    dimensions = (
        raw_dimensions if type(raw_dimensions) is int and 0 < raw_dimensions <= 100_000 else None
    )
    raw_representation = value.get("embedding_representation")
    if raw_representation is None:
        representation = None
    elif raw_representation in _SAFE_EMBEDDING_REPRESENTATIONS:
        representation = raw_representation
    else:
        representation = "other"
    raw_fusion = value.get("fusion_configuration")
    if raw_fusion is None:
        fusion = None
    elif raw_fusion in _SAFE_FUSION_CONFIGURATIONS:
        fusion = raw_fusion
    else:
        fusion = "other"
    return {
        "mode": mode,
        "embedding_model": model,
        "embedding_dimensions": dimensions,
        "embedding_representation": representation,
        "fusion_configuration": fusion,
    }


def _bridge_diagnostics(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    internal_calls: Counter[str] = Counter()
    provider_phases: Counter[str] = Counter()
    provider_results: Counter[str] = Counter()
    provider_maxima = Counter()
    provider_turns: list[dict[str, Any]] = []
    remediation_totals = Counter()
    remediation_codes: Counter[str] = Counter()
    repair_classes: Counter[str] = Counter()
    fields = (
        "internal_continuations",
        "internal_tool_rounds",
        "internal_progress_events",
        "internal_stalled_rounds",
        "stalled_internal_rounds_since_progress",
        "internal_compacted_rounds",
        "internal_context_refreshes",
        "staged_external_batches",
        "staged_external_calls",
        "external_admission_rounds",
        "external_admission_rejections",
        "terminal_gate_rounds",
        "terminal_gate_rejections",
        "terminal_gate_rounds_since_progress",
        "provider_call_count",
        "empty_response_retries",
    )
    for message in messages:
        if message.get("role") != "assistant":
            continue
        raw_data = message.get("raw_data")
        bridge = raw_data.get("cayu_tau3_bridge") if isinstance(raw_data, Mapping) else None
        if not isinstance(bridge, Mapping):
            continue
        totals["assistant_turns_with_bridge_metadata"] += 1
        calls_by_tool = bridge.get("internal_calls_by_tool")
        if isinstance(calls_by_tool, Mapping):
            internal_calls.update(
                {
                    name: count
                    for name, count in calls_by_tool.items()
                    if isinstance(name, str) and type(count) is int and count >= 0
                }
            )
        remediation = bridge.get("internal_remediation")
        if isinstance(remediation, Mapping) and remediation.get("schema_version") == 1:
            for field in _REMEDIATION_TOTAL_FIELDS:
                value = remediation.get(field)
                if type(value) is int and value >= 0:
                    if field == "repair_rounds_max":
                        remediation_totals[field] = max(remediation_totals[field], value)
                    else:
                        remediation_totals[field] += value
            _update_safe_remediation_counts(
                remediation_codes,
                remediation.get("rejection_codes"),
                allowed=SAFE_INTERNAL_REMEDIATION_CODES,
            )
            _update_safe_remediation_counts(
                repair_classes,
                remediation.get("accepted_repair_classes"),
                allowed=SAFE_INTERNAL_REPAIR_CLASSES,
            )
        for field in fields:
            value = bridge.get(field)
            if type(value) is int and value >= 0:
                totals[field] += value
        provider_diagnostics = _provider_diagnostics(bridge)
        provider_turns.append(dict(provider_diagnostics))
        provider_phases.update(provider_diagnostics.pop("provider_calls_by_phase"))
        provider_results.update(provider_diagnostics.pop("provider_results_by_kind"))
        for field in _PROVIDER_MAX_FIELDS:
            provider_maxima[field] = max(
                provider_maxima[field],
                provider_diagnostics.pop(field),
            )
        totals.update(provider_diagnostics)
        round_limit = bridge.get("internal_tool_round_limit", 12)
        if (
            type(round_limit) is int
            and round_limit >= 0
            and bridge.get("internal_tool_rounds") == round_limit
        ):
            totals["turns_reaching_current_internal_round_limit"] += 1
    return {
        **dict(totals),
        **dict(provider_maxima),
        "internal_calls_by_tool": _ordered_counts(internal_calls),
        "provider_calls_by_phase": _ordered_counts(provider_phases),
        "provider_results_by_kind": _ordered_counts(provider_results),
        "provider_turns": provider_turns,
        "internal_remediation": {
            **{field: remediation_totals[field] for field in _REMEDIATION_TOTAL_FIELDS},
            "rejection_codes": _ordered_counts(remediation_codes),
            "accepted_repair_classes": _ordered_counts(repair_classes),
        },
    }


def _provider_diagnostics(bridge: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize bounded, content-free acting-provider telemetry for one turn."""

    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    cached_input_tokens = 0
    usage_records = 0
    raw_usages = bridge.get("provider_usages")
    if isinstance(raw_usages, list):
        for raw_usage in raw_usages[:_MAX_PROVIDER_CALL_TRACE_RECORDS]:
            if not isinstance(raw_usage, Mapping):
                continue
            usage_records += 1
            input_tokens += _safe_nonnegative_int(raw_usage.get("input_tokens"))
            output_tokens += _safe_nonnegative_int(raw_usage.get("output_tokens"))
            reasoning_tokens += _safe_nonnegative_int(
                raw_usage.get("reasoning_output_tokens")
                or (
                    raw_usage.get("output_tokens_details", {}).get("reasoning_tokens")
                    if isinstance(raw_usage.get("output_tokens_details"), Mapping)
                    else 0
                )
            )
            cached_input_tokens += _safe_nonnegative_int(
                raw_usage.get("cached_input_tokens")
                or (
                    raw_usage.get("input_tokens_details", {}).get("cached_tokens")
                    if isinstance(raw_usage.get("input_tokens_details"), Mapping)
                    else 0
                )
            )

    phases: Counter[str] = Counter()
    results: Counter[str] = Counter()
    message_bytes = 0
    system_message_bytes = 0
    tool_schema_bytes = 0
    maximum_message_bytes = 0
    maximum_message_count = 0
    maximum_system_message_bytes = 0
    maximum_tool_schema_bytes = 0
    maximum_tool_schema_count = 0
    forced_tool_calls = 0
    trace_records = 0
    raw_trace = bridge.get("provider_call_trace")
    raw_calls = raw_trace.get("calls") if isinstance(raw_trace, Mapping) else None
    if (
        isinstance(raw_trace, Mapping)
        and raw_trace.get("schema_version") == 1
        and isinstance(raw_calls, list)
    ):
        for raw_call in raw_calls[:_MAX_PROVIDER_CALL_TRACE_RECORDS]:
            if not isinstance(raw_call, Mapping):
                continue
            trace_records += 1
            phase = raw_call.get("phase")
            phases[phase if phase in _SAFE_PROVIDER_CALL_PHASES else "other"] += 1
            result = raw_call.get("result_kind")
            results[result if result in _SAFE_PROVIDER_RESULT_KINDS else "other"] += 1
            call_message_bytes = _safe_nonnegative_int(raw_call.get("message_bytes"))
            call_message_count = _safe_nonnegative_int(raw_call.get("message_count"))
            call_system_bytes = _safe_nonnegative_int(raw_call.get("system_message_bytes"))
            call_tool_bytes = _safe_nonnegative_int(raw_call.get("tool_schema_bytes"))
            call_tool_count = _safe_nonnegative_int(raw_call.get("tool_schema_count"))
            message_bytes += call_message_bytes
            system_message_bytes += call_system_bytes
            tool_schema_bytes += call_tool_bytes
            maximum_message_bytes = max(maximum_message_bytes, call_message_bytes)
            maximum_message_count = max(maximum_message_count, call_message_count)
            maximum_system_message_bytes = max(
                maximum_system_message_bytes,
                call_system_bytes,
            )
            maximum_tool_schema_bytes = max(maximum_tool_schema_bytes, call_tool_bytes)
            maximum_tool_schema_count = max(maximum_tool_schema_count, call_tool_count)
            forced_tool_calls += raw_call.get("forced_tool") is True

    provider_call_count = _safe_nonnegative_int(bridge.get("provider_call_count"))
    return {
        "acting_cached_input_tokens": cached_input_tokens,
        "acting_completion_tokens": output_tokens,
        "acting_prompt_tokens": input_tokens,
        "acting_reasoning_tokens": reasoning_tokens,
        "maximum_provider_request_message_bytes": maximum_message_bytes,
        "maximum_provider_request_message_count": maximum_message_count,
        "maximum_provider_request_system_message_bytes": maximum_system_message_bytes,
        "maximum_provider_request_tool_schema_bytes": maximum_tool_schema_bytes,
        "maximum_provider_request_tool_schema_count": maximum_tool_schema_count,
        "provider_calls_by_phase": _ordered_counts(phases),
        "provider_calls_without_trace": max(provider_call_count - trace_records, 0),
        "provider_calls_without_usage": max(provider_call_count - usage_records, 0),
        "provider_forced_tool_calls": forced_tool_calls,
        "provider_request_message_bytes": message_bytes,
        "provider_request_system_message_bytes": system_message_bytes,
        "provider_request_tool_schema_bytes": tool_schema_bytes,
        "provider_results_by_kind": _ordered_counts(results),
        "provider_trace_records": trace_records,
        "provider_usage_records": usage_records,
    }


def _safe_nonnegative_int(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def _update_safe_remediation_counts(
    destination: Counter[str],
    value: object,
    *,
    allowed: frozenset[str],
) -> None:
    if not isinstance(value, Mapping):
        return
    for key, count in value.items():
        if not isinstance(key, str) or type(count) is not int or count < 0:
            continue
        destination[key if key in allowed else "other"] += count


def _aggregate_workflow(simulations: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "tasks_with_plan_revision": sum(item["workflow"]["plan_created"] for item in simulations),
        "tasks_with_populated_plan": sum(
            item["workflow"]["plan_populated"] for item in simulations
        ),
        "tasks_without_populated_plan": sum(
            not item["workflow"]["plan_populated"] for item in simulations
        ),
        "failures_without_populated_plan": sum(
            not item["success"] and not item["workflow"]["plan_populated"] for item in simulations
        ),
        "successes_without_populated_plan": sum(
            item["success"] and not item["workflow"]["plan_populated"] for item in simulations
        ),
        "failures_actionable_at_last_response": sum(
            not item["success"] and item["workflow"]["actionable_at_last_response"]
            for item in simulations
        ),
        "successes_actionable_at_last_response": sum(
            item["success"] and item["workflow"]["actionable_at_last_response"]
            for item in simulations
        ),
        "failures_unresolved_at_last_response": sum(
            not item["success"] and item["workflow"]["unresolved_at_last_response"]
            for item in simulations
        ),
        "successes_unresolved_at_last_response": sum(
            item["success"] and item["workflow"]["unresolved_at_last_response"]
            for item in simulations
        ),
        "text_responses_while_actionable": sum(
            item["workflow"]["text_responses_while_actionable"] for item in simulations
        ),
        "operation_customer_response_handoffs": sum(
            item["workflow"]["operation_customer_response_handoffs"] for item in simulations
        ),
        "tasks_with_operation_customer_response_handoffs": sum(
            item["workflow"]["items_with_operation_customer_response_handoffs"] > 0
            for item in simulations
        ),
        "maximum_operation_customer_response_handoffs": max(
            (
                item["workflow"]["maximum_operation_customer_response_handoffs"]
                for item in simulations
            ),
            default=0,
        ),
        "product_selection_transition_rejections": sum(
            item["workflow"]["product_selection_transition_rejections"] for item in simulations
        ),
        "tasks_with_product_selection_transition_rejections": sum(
            item["workflow"]["product_selection_transition_rejections"] > 0 for item in simulations
        ),
    }


def _aggregate_correlations(simulations: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "failures_with_reference_action_miss": sum(
            not item["success"] and item["reference_actions"]["missed"] > 0 for item in simulations
        ),
        "failures_without_reference_action_miss": sum(
            not item["success"] and item["reference_actions"]["missed"] == 0 for item in simulations
        ),
        "successes_with_reference_action_miss": sum(
            item["success"] and item["reference_actions"]["missed"] > 0 for item in simulations
        ),
        "failures_with_tool_error": sum(
            not item["success"] and item["tool_execution"]["error_results"] > 0
            for item in simulations
        ),
        "failures_with_actionable_plan_state": sum(
            not item["success"] and item["workflow"]["actionable_at_last_response"]
            for item in simulations
        ),
    }


def _pass_hat_k(simulations: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    by_task: dict[str, list[bool]] = defaultdict(list)
    for simulation in simulations:
        by_task[simulation["task_id"]].append(simulation["success"])
    if not by_task:
        return {}
    max_k = min(len(outcomes) for outcomes in by_task.values())
    return {
        str(k): sum(
            math.comb(sum(outcomes), k) / math.comb(len(outcomes), k)
            for outcomes in by_task.values()
        )
        / len(by_task)
        for k in range(1, max_k + 1)
    }


def _diff_database(
    expected: Mapping[str, Any], predicted: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return table-level counts without record ids or values."""

    diffs: list[dict[str, Any]] = []
    for table in sorted(set(expected) | set(predicted)):
        expected_table = expected.get(table)
        predicted_table = predicted.get(table)
        if expected_table == predicted_table:
            continue
        expected_records = _table_records(expected_table)
        predicted_records = _table_records(predicted_table)
        expected_ids = set(expected_records)
        predicted_ids = set(predicted_records)
        common = expected_ids & predicted_ids
        changed_records = [
            record_id
            for record_id in common
            if expected_records[record_id] != predicted_records[record_id]
        ]
        changed_fields = 0
        for record_id in changed_records:
            expected_record = expected_records[record_id]
            predicted_record = predicted_records[record_id]
            if not isinstance(expected_record, Mapping) or not isinstance(
                predicted_record, Mapping
            ):
                changed_fields += 1
                continue
            fields = set(expected_record) | set(predicted_record)
            changed_fields += sum(
                expected_record.get(field) != predicted_record.get(field) for field in fields
            )
        metadata_changed = False
        if isinstance(expected_table, Mapping) and isinstance(predicted_table, Mapping):
            metadata_changed = {
                key: value for key, value in expected_table.items() if key != "data"
            } != {key: value for key, value in predicted_table.items() if key != "data"}
        elif expected_table != predicted_table:
            metadata_changed = True
        diffs.append(
            {
                "table": str(table),
                "counts": {
                    "missing_records": len(expected_ids - predicted_ids),
                    "unexpected_records": len(predicted_ids - expected_ids),
                    "changed_records": len(changed_records),
                    "changed_fields": changed_fields,
                    "metadata_changed": int(metadata_changed),
                },
            }
        )
    return diffs


def _table_records(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping) and isinstance(value.get("data"), Mapping):
        return value["data"]
    return {}


def _classify_db_diff(table_diffs: Sequence[Mapping[str, Any]]) -> str:
    if not table_diffs:
        return "MATCH"
    totals = Counter()
    for item in table_diffs:
        counts = item["counts"]
        totals.update(
            {
                "missing": counts["missing_records"],
                "unexpected": counts["unexpected_records"],
                "changed": counts["changed_records"] + counts["metadata_changed"],
            }
        )
    active = [key for key in ("missing", "unexpected", "changed") if totals[key] > 0]
    if active == ["missing"]:
        return "MISSING_EXPECTED_STATE"
    if active == ["unexpected"]:
        return "UNEXPECTED_STATE"
    if active == ["changed"]:
        return "CHANGED_STATE"
    return "MIXED_STATE_DIFFERENCE"


def _state_profile(table_diffs: Sequence[Mapping[str, Any]]) -> str:
    tables = {str(item["table"]) for item in table_diffs}
    if not tables:
        return "MATCH"
    control_tables = tables & _CONTROL_STATE_TABLES
    if control_tables == tables:
        return "CONTROL_ONLY"
    if not control_tables:
        return "BUSINESS_ONLY"
    return "CONTROL_AND_BUSINESS"


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FailureAnalysisError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise FailureAnalysisError(f"{field} must be an array")
    return value


def _string_sequence(value: Any, field: str) -> list[str]:
    values = _sequence(value, field)
    if not all(isinstance(item, str) and item for item in values):
        raise FailureAnalysisError(f"{field} must contain nonblank strings")
    return list(values)


def _ordered_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {key: counts[key] for key in sorted(counts)}


def _action_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {key: counts.get(key, 0) for key in ("total", "matched", "missed")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Saved tau results.json artifact")
    parser.add_argument("--output", type=Path, help="Write JSON report atomically")
    parser.add_argument(
        "--deep-replay",
        action="store_true",
        help="Replay predicted and golden states with the pinned local tau environment",
    )
    parser.add_argument("--tau2-repository", type=Path, default=_DEFAULT_TAU2_REPOSITORY)
    args = parser.parse_args(argv)
    results_path = args.results.resolve()
    with results_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    report = analyze_results(payload)
    report["source"] = {
        "results_path": str(results_path),
        "sha256": _sha256(results_path),
    }
    if args.deep_replay:
        attach_deep_replay(
            report,
            results_path=results_path,
            tau2_repository=args.tau2_repository,
        )
    if args.output is None:
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _write_json_atomic(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FailureAnalysisError",
    "analyze_results",
    "attach_deep_replay",
    "main",
]
