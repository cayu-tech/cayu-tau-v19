import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "read_audit", Path(__file__).parents[1] / "scripts/audit_read_repetition.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_nested_tool_json_is_decoded_but_title_alone_is_not_full_body():
    body = "Product Alpha\nFee $5. Conditions apply."
    assert _module.body_present(body, [{"text": json.dumps({"text": body})}])
    assert not _module.body_present(body, [{"text": "Product Alpha fee unknown"}])


def test_body_cannot_be_assembled_across_unrelated_messages():
    assert not _module.body_present("Alpha fee $5", [{"text": "Alpha"}, {"text": "fee $5"}])
    assert not _module.body_present("", [{"text": ""}])
