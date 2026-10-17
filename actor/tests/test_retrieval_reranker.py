import asyncio
import json

import pytest
from cayu import ModelStreamEvent, ScriptedModelProvider
from cayu.providers.openai import build_openai_payload

from cayu_tau3_banking.retrieval_reranker import (
    RerankerFailure,
    build_id_request,
    build_request,
    select,
    validate_document_ids,
    validate_selection,
)


def test_id_only_contract_schema_and_exact_host_provenance():
    request = build_id_request("query", DOCS)
    payload = build_openai_payload(request, stream=True, reasoning_state="inline")
    assert payload["tools"][0]["parameters"]["required"] == ["candidate_ids"]
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    name="select_evidence",
                    arguments={"candidate_ids": ["D01", "D01"], "unused": "ignored"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    result = asyncio.run(select(provider, request, DOCS))
    assert len(result["selections"]) == 1
    assert "quote" not in result["selections"][0]
    assert len(result["selections"][0]["body_sha256"]) == 64
    with pytest.raises(ValueError, match="Unknown candidate"):
        validate_document_ids({"candidate_ids": ["invented"]}, DOCS)


DOCS = [{"candidate_id": "D01", "title": "Reference", "body": "The qualifying period is 30 days."}]
VALID = {"selections": [{"candidate_id": "D01", "quote": "qualifying period is 30 days"}]}


def test_request_schema_and_no_extra_candidate_fields():
    request = build_request("Which period applies?", DOCS)
    payload = build_openai_payload(request, stream=True, reasoning_state="inline")
    assert payload["tools"][0]["name"] == "select_evidence"
    assert payload["tools"][0]["parameters"]["properties"]["selections"]["maxItems"] == 4
    assert json.loads(request.messages[1].content[0].text) == {
        "query": "Which period applies?",
        "documents": DOCS,
    }
    with pytest.raises(ValueError, match="Unexpected candidate fields"):
        build_request("query", [{**DOCS[0], "expected": True}])


@pytest.mark.parametrize(
    "arguments",
    [
        {"selections": [{"candidate_id": "unknown", "quote": "qualifying period is 30 days"}]},
        {"selections": [{"candidate_id": "D01", "quote": "invented quote"}]},
        {"selections": VALID["selections"] * 17},
        {"other": True},
    ],
)
def test_invalid_selection_rejected(arguments):
    with pytest.raises(ValueError):
        validate_selection(arguments, DOCS)


def test_one_read_only_response_is_verified():
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(name="select_evidence", arguments=VALID),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    result = asyncio.run(select(provider, build_request("query", DOCS), DOCS))
    assert result["selections"][0]["quote_start_char"] == 4
    assert len(result["selections"][0]["body_sha256"]) == 64
    assert validate_selection({"selections": []}, DOCS) == []


def test_bad_quote_retains_response_and_exact_validation_stage():
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    name="select_evidence",
                    arguments={"selections": [{"candidate_id": "D01", "quote": "invented quote"}]},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    events = []
    with pytest.raises(RerankerFailure) as caught:
        asyncio.run(select(provider, build_request("query", DOCS), DOCS, capture=events.append))
    assert caught.value.diagnostic["stage"] == "evidence_validation"
    assert "exact bounded body passage" in caught.value.diagnostic["validation_reason"]
    assert len(events) == 2


def test_transport_failure_is_not_mislabeled_as_model_selection_error():
    class Broken:
        async def runtime_stream(self, request):
            raise ValueError("Sensitive transport message must not be serialized")
            yield  # Make this a deliberately failing async generator.

    with pytest.raises(RerankerFailure) as caught:
        asyncio.run(select(Broken(), build_request("query", DOCS), DOCS))
    assert caught.value.diagnostic["stage"] == "provider_stream"
    assert "Sensitive transport message" not in json.dumps(caught.value.diagnostic)


def test_nonstrict_metadata_and_long_exact_quote_are_normalized():
    body = "A verified public passage. " * 25
    documents = [{"candidate_id": "D01", "title": "Reference", "body": body}]
    result = validate_selection(
        {
            "maxItems": 1,
            "selections": [
                {
                    "candidate_id": "D01",
                    "quote": body,
                    "unused_metadata": "ignored",
                }
            ],
        },
        documents,
    )
    assert result[0]["quote"] == body[:400]
    assert result[0]["original_quote_characters"] == len(body)
    assert "unused_metadata" not in result[0]
    assert len(validate_selection({"selections": VALID["selections"] * 2}, DOCS)) == 1
    with pytest.raises(ValueError, match="exact bounded body passage"):
        validate_selection(
            {"selections": [{"candidate_id": "D01", "quote": body + "invented"}]}, documents
        )
