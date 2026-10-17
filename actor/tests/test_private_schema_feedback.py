import asyncio
import json

from cayu.tools.base import ToolSpec

from cayu_tau3_banking.runtime_banking import BankingNativeTool


def test_private_schema_error_identifies_field_and_bound_without_running():
    spec = ToolSpec(
        name="private_test",
        description="test",
        input_schema={
            "type": "object",
            "properties": {"quote": {"type": "string", "minLength": 8}},
            "required": ["quote"],
            "additionalProperties": False,
        },
    )
    # No runtime state/context needed: invalid calls must never reach execution.
    tool = BankingNativeTool(spec, None)
    result = asyncio.run(tool.run(None, {"quote": "tiny"}))
    assert result.is_error
    details = json.loads(result.content.split("Repair these fields: ", 1)[1])
    assert details[0]["path"] == ["quote"]
    assert details[0]["rule"] == "minLength"
    assert details[0]["constraint"] == 8
    assert "tiny" not in result.content


def test_private_schema_feedback_bounds_number_of_errors():
    spec = ToolSpec(
        name="private_test",
        description="test",
        input_schema={"type": "object", "required": ["a", "b", "c", "d", "e"]},
    )
    result = asyncio.run(BankingNativeTool(spec, None).run(None, {}))
    details = json.loads(result.content.split("Repair these fields: ", 1)[1])
    assert len(details) == 3
    assert all(d["rule"] == "required" for d in details)
