import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "frozen_task_monitor", Path(__file__).resolve().parents[1] / "scripts/monitor_frozen_task.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
summarize_events = module.summarize_events


def test_failed_compactor_terminal_event_is_not_successful_actor_completion():
    result = summarize_events(
        [
            ("model.started", None, {}),
            ("model.completed", None, {"status": "completed"}),
            ("model.started", None, {"purpose": "context_compaction"}),
            (
                "model.completed",
                None,
                {
                    "purpose": "context_compaction",
                    "compaction_outcome": "provider_error",
                    "error_type": "ModelProviderError",
                },
            ),
            ("context.compaction.failed", None, {}),
            ("interaction.failed", None, {}),
            ("session.failed", None, {}),
        ]
    )
    assert result["actor_model_completed"] == 1
    assert result["actor_model_error_events"] == 0
    assert result["compactor_model_failed"] == 1
    assert result["compactions_completed"] == 0
    assert result["runtime_failure"]


def test_successful_compaction_is_separate_from_actor_and_failure_counts():
    result = summarize_events(
        [
            ("model.started", None, {"purpose": "context_compaction"}),
            ("model.completed", None, {"purpose": "context_compaction"}),
            ("context.compaction.completed", None, {}),
            ("interaction.completed", None, {}),
        ]
    )
    assert result["actor_model_completed"] == 0
    assert result["compactor_model_terminal"] == result["compactions_completed"] == 1
    assert not result["runtime_failure"]


def test_error_tool_result_cannot_hide_in_completed_event():
    result = summarize_events([("tool.call.completed", "example", {"result": {"is_error": True}})])
    assert result["tool_result_errors"] == 1
    assert result["tool_counts"] == {"example": 1}


def test_capture_failures_are_visible_without_reading_private_explanation(tmp_path):
    directory = tmp_path / "provider-test"
    directory.mkdir()
    (directory / "one.error.json").write_text("PRIVATE PROVIDER EXPLANATION")
    (directory / "one.capture.json").write_text(
        json.dumps(
            {
                "schema": "cayu-tau3.provider-capture.v1",
                "records_written": 0,
                "sink_failures": 1,
                "records_dropped": 0,
            }
        )
    )
    result = module.summarize_captures(tmp_path)
    assert result["incomplete"] and result["sink_failures"] == 1
    assert result["invalid_status_files"] == 0
    assert "PRIVATE" not in json.dumps(result)
