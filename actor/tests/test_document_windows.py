import pytest
from jsonschema import Draft202012Validator

from cayu_tau3_banking.document_windows import document_window
from cayu_tau3_banking.product_navigation import READ_TOOL, tool_schemas


def test_window_preserves_unicode_offsets_and_removes_full_evidence():
    text = "前置き\n" * 30 + "Policy conditions" + "\ntrailing" * 100
    source = {
        "text": text,
        "entry_id": "policy",
        "entry_revision": 1,
        "evidence_ref": {"reference_id": "policy"},
        "text_complete": True,
    }
    result = document_window(source, {"start": 120, "end": 137, "context_chars": 20})
    assert result["text"] == text[100:157]
    assert result["source_window"]["total_characters"] == len(text)
    assert result["navigation_only"] and not result["text_complete"]
    assert "evidence_ref" not in result
    assert "evidence_ref" in source


def test_full_source_digest_is_not_misrepresented_as_window_digest():
    result = document_window(
        {"text": "abcdef", "content_sha256": "source-hash"},
        {"start": 1, "end": 3, "context_chars": 0},
    )
    assert result["source_content_sha256"] == "source-hash"
    assert "content_sha256" not in result


@pytest.mark.parametrize(
    "span",
    [
        {"start": -1, "end": 1},
        {"start": 0, "end": 99},
        {"start": 1, "end": 1},
        {"start": True, "end": 2},
        {"start": 0, "end": 2, "context_chars": 2001},
    ],
)
def test_window_rejects_invalid_offsets(span):
    with pytest.raises(ValueError):
        document_window({"text": "abc"}, span)


@pytest.mark.parametrize(
    "arguments",
    [
        {"product_id": "account", "source_span": {"start": 0, "end": 10}},
        {
            "documents": [{"entry_id": "a", "revision": 1}, {"entry_id": "b", "revision": 1}],
            "source_span": {"start": 0, "end": 10},
        },
        {
            "documents": [{"entry_id": "a", "revision": 1}],
            "source_span": {"start": 0, "end": 10, "field": "title"},
        },
    ],
)
def test_window_schema_rejects_ambiguous_source_or_wrong_field(arguments):
    schema = next(item["input_schema"] for item in tool_schemas() if item["name"] == READ_TOOL)
    assert not Draft202012Validator(schema).is_valid(arguments)
