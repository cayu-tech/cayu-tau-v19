import asyncio
import runpy
from pathlib import Path


def test_merged_compaction_fits_on_first_context_build_without_agent_retry():
    script = Path(__file__).resolve().parents[1] / "scripts/reproduce_partial_compaction_fit.py"
    reproduce = runpy.run_path(str(script))["reproduce"]
    result = asyncio.run(reproduce(expected_first_build="success"))
    assert result["automatic_first_build_fits"] is True
    assert result["automatic_first_build_fails"] is False
    assert result["bounded_reentry_with_same_limits_fits"] is False
    assert result["network_calls"] == 0
    assert result["scripted_summary_calls"] == 6
    assert result["max_prompt_chars"] <= 10_000
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["cursor"] == 14
    assert result["attempts"][0]["estimated_input_tokens"] <= 500
