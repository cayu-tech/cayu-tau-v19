import asyncio
import json
import stat

import httpx
import pytest
from cayu import (
    AgentSpec,
    CayuApp,
    CheckpointCompactionContextPolicy,
    EventType,
    Message,
    ModelCompactor,
    RunRequest,
)
from cayu.providers import HttpxOpenAITransport, ModelRequest, OpenAIProvider

from cayu_tau3_banking.provider_diagnostics import DiagnosticOpenAITransport, safe_error_fields


@pytest.mark.parametrize("value", [[], {}, "SECRET", 4, None])
def test_unknown_identity_is_never_serialized(value):
    result = safe_error_fields({"type": value, "code": value, "message": "SECRET"})
    assert "SECRET" not in json.dumps(result)


def test_numeric_fields_are_validated():
    assert safe_error_fields({"status_code": True, "retry_after_s": float("inf")}) == {
        "type": None,
        "code": None,
    }
    assert safe_error_fields({"status_code": 429, "code": "usage_limit_reached"}) == {
        "type": None,
        "code": "usage_limit_reached",
        "status_code": 429,
    }


@pytest.mark.parametrize("kind", ["ReadError", "RemoteProtocolError", "SseEventLimitError"])
def test_known_transport_error_categories_survive_without_raw_error_text(kind):
    result = safe_error_fields({"type": kind, "message": "SECRET", "request_id": "SECRET"})
    assert result == {"type": kind, "code": None}
    assert "SECRET" not in json.dumps(result)


def test_capture_preserves_events_closes_stream_and_excludes_secrets(tmp_path):
    event = {
        "type": "response.failed",
        "response": {
            "error": {
                "code": "usage_limit_reached",
                "message": "credential-canary",
                "request_id": "credential-canary",
            }
        },
    }

    class Base:
        closed = False
        stream_closed = False

        async def stream_response_events(self, **kwargs):
            assert kwargs["headers"]["Authorization"] == "credential-canary"
            try:
                yield event
            finally:
                self.stream_closed = True

        async def aclose(self):
            self.closed = True

    base = Base()
    transport = DiagnosticOpenAITransport(base, tmp_path)

    async def run():
        result = [
            e
            async for e in transport.stream_response_events(
                headers={"Authorization": "credential-canary"},
                url="credential-canary",
                payload={
                    "model": "model",
                    "input": [],
                    "metadata": {"secret": "credential-canary"},
                },
            )
        ]
        assert result[0] is event
        await transport.aclose()

    asyncio.run(run())
    assert base.closed and base.stream_closed
    files = list(transport.directory.glob("*.json"))
    assert len(files) == 3
    assert all("credential-canary" not in p.read_text() for p in files)
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in files)
    assert stat.S_IMODE(transport.directory.stat().st_mode) == 0o700
    request = json.loads(next(transport.directory.glob("*.request.json")).read_text())
    assert request["omitted_field_count"] == 1


def test_transport_exception_is_recorded_and_propagated_unchanged(tmp_path):
    failure = ValueError("SECRET")

    class Base:
        async def stream_response_events(self, **kwargs):
            raise failure
            yield  # pragma: no cover

    transport = DiagnosticOpenAITransport(Base(), tmp_path)

    async def run():
        with pytest.raises(ValueError) as caught:
            async for _ in transport.stream_response_events(payload={}):
                pass
        assert caught.value is failure

    asyncio.run(run())
    assert len(list(transport.directory.glob("*.exception.json"))) == 1
    assert all("SECRET" not in p.read_text() for p in transport.directory.glob("*.json"))


@pytest.mark.parametrize("role", ["direct", "actor", "compactor"])
def test_real_provider_capture_correlates_request_body_and_upstream_explanation(tmp_path, role):
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (
                b'data: {"type":"error","error":{"type":"invalid_request_error",'
                b'"code":"invalid_prompt","param":"input[0].content",'
                b'"message":"Unsupported content: SECRET"}}\n\n'
            )

    async def run():
        http_transport = HttpxOpenAITransport()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={"x-request-id": "req_upstream"}, stream=Body()
                )
            )
        ) as client:
            http_transport._client._client = client
            capture = DiagnosticOpenAITransport(http_transport, tmp_path)
            provider = OpenAIProvider(api_key="SECRET", transport=capture)
            if role == "direct":
                events = [
                    e
                    async for e in provider.stream(
                        ModelRequest(model="test-model", messages=[Message.text("user", "test")])
                    )
                ]
                assert events[-1].payload["provider_error_code"] == "invalid_prompt"
            else:
                app = CayuApp(enable_logging=False)
                app.register_provider(provider, default=True)
                context_policy = (
                    CheckpointCompactionContextPolicy(
                        compactor=ModelCompactor(provider=provider, model="test-model"),
                        max_user_turns=1,
                        compact_after_messages=2,
                    )
                    if role == "compactor"
                    else None
                )
                app.register_agent(
                    AgentSpec(name="assistant", model="test-model"), context_policy=context_policy
                )
                events = [
                    e
                    async for e in app.run(
                        RunRequest(
                            agent_name="assistant",
                            messages=[
                                Message.text("user", "old"),
                                Message.text("assistant", "old answer"),
                                Message.text("user", "new"),
                            ],
                        )
                    )
                ]
                assert events[-1].type == EventType.SESSION_FAILED
            assert capture.capture_failures == 0
            return capture.directory

    directory = asyncio.run(run())
    errors = list(directory.glob("*.error.json"))
    assert len(errors) == 1  # No duplicate from transport plus wrapper.
    error = json.loads(errors[0].read_text())
    request = json.loads(next(directory.glob("*.request.json")).read_text())
    status = json.loads(next(directory.glob("*.capture.json")).read_text())
    assert request["capture_id"] == error["capture_id"] == status["capture_id"]
    assert error["error"]["request_id"] == "req_upstream"
    assert error["error"]["param"] == "input[0].content"
    assert error["error"]["message"] == "Unsupported content: [REDACTED_SECRET]"
    assert error["http_status_code"] == 200
    assert error["field_states"]["message"] == "redacted"
    assert status["records_written"] == 1 and status["sink_failures"] == 0


def test_diagnostic_sink_failure_is_reported_and_blocks_another_dispatch(tmp_path):
    event = {"type": "error", "message": "A useful provider explanation"}

    class Base:
        calls = 0

        async def stream_response_events(self, **kwargs):
            self.calls += 1
            yield event

    class FailingCapture(DiagnosticOpenAITransport):
        def _save(self, name, value):
            if name.endswith(".error.json"):
                raise OSError("private sink failure")
            super()._save(name, value)

    transport = FailingCapture(Base(), tmp_path)

    async def run():
        assert [e async for e in transport.stream_response_events(payload={})] == [event]
        with pytest.raises(RuntimeError, match="repair it before another request"):
            async for _ in transport.stream_response_events(payload={}):
                pass

    asyncio.run(run())
    assert transport.base.calls == 1
    status = json.loads(next(transport.directory.glob("*.capture.json")).read_text())
    assert status["sink_failures"] == 1 and status["records_written"] == 0
    assert "private sink failure" not in json.dumps(status)


def test_request_capture_redacts_credentials_without_mutating_dispatched_body(tmp_path):
    class Base:
        async def stream_response_events(self, **kwargs):
            assert kwargs["payload"]["input"] == "echo SECRET"
            yield {"type": "response.completed"}

    transport = DiagnosticOpenAITransport(Base(), tmp_path)

    async def run():
        return [
            e
            async for e in transport.stream_response_events(
                payload={"input": "echo SECRET"}, headers={"Authorization": "Bearer SECRET"}
            )
        ]

    asyncio.run(run())
    request = json.loads(next(transport.directory.glob("*.request.json")).read_text())
    assert request["payload_redacted"] is True
    assert request["payload"]["input"] == "echo [REDACTED_SECRET]"


@pytest.mark.parametrize(
    ("headers", "public_name_survives"),
    [
        ({"Originator": "cayu"}, True),
        (
            {
                "originator": "cayu",
                "Authorization": "Bearer credential-canary",
                "X-Custom-Token": "custom-canary",
            },
            True,
        ),
        ({"originator": "cayu", "Authorization": "Bearer cayu"}, False),
        ({"originator": "credential-canary", "X-Custom-Token": "cayu"}, False),
    ],
)
def test_public_originator_exemption_does_not_weaken_secret_redaction(
    tmp_path, headers, public_name_survives
):
    payload = {
        "instructions": "Use cayu public tools; credential-canary custom-canary",
        "tools": [
            {"name": f"cayu_{suffix}"}
            for suffix in (
                "browse_product_documents",
                "read_product_documents",
                "check_comparison_facts",
                "search_banking_knowledge",
            )
        ],
    }
    original = json.loads(json.dumps(payload))

    class Base:
        async def stream_response_events(self, **kwargs):
            assert kwargs["payload"] == original
            assert kwargs["headers"] == headers
            yield {"type": "response.completed"}

    transport = DiagnosticOpenAITransport(Base(), tmp_path)

    async def run():
        return [
            event
            async for event in transport.stream_response_events(payload=payload, headers=headers)
        ]

    asyncio.run(run())
    captured = json.loads(next(transport.directory.glob("*.request.json")).read_text())
    saved = captured["payload"]
    assert all(
        (tool["name"].startswith("cayu_") == public_name_survives) for tool in saved["tools"]
    )
    assert payload == original
    for secret in ("credential-canary", "custom-canary"):
        if any(secret in value for value in headers.values()):
            assert secret not in json.dumps(saved)
    if headers == {"Originator": "cayu"}:
        assert captured["payload_redacted"] is False
        assert saved == original


def test_capture_scope_does_not_leak_between_yields_or_fail_on_cross_task_close(tmp_path):
    from cayu.providers.diagnostics import _record_stream_error

    class Base:
        async def stream_response_events(self, **kwargs):
            yield {"type": "response.output_text.delta", "delta": "first"}
            yield {"type": "response.output_text.delta", "delta": "second"}

    transport = DiagnosticOpenAITransport(Base(), tmp_path)

    async def run():
        stream = transport.stream_response_events(payload={})
        await anext(stream)
        # An unrelated operation in the consumer's task must not inherit this
        # transport's private sink while it is suspended at yield.
        _record_stream_error({"type": "error", "message": "unrelated"}, headers={})
        await asyncio.create_task(anext(stream))
        await asyncio.create_task(stream.aclose())

    asyncio.run(run())
    assert not list(transport.directory.glob("*.error.json"))
    assert transport.capture_failures == 0
    assert len(list(transport.directory.glob("*.capture.json"))) == 1


def test_cancellation_while_waiting_for_provider_is_preserved(tmp_path):
    class Base:
        async def stream_response_events(self, **kwargs):
            started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover

    transport = DiagnosticOpenAITransport(Base(), tmp_path)

    async def run():
        nonlocal started
        started = asyncio.Event()
        stream = transport.stream_response_events(payload={})
        task = asyncio.create_task(anext(stream))
        await started.wait()
        task.cancel("caller cancelled")
        with pytest.raises(asyncio.CancelledError, match="caller cancelled"):
            await task
        await asyncio.create_task(stream.aclose())

    started = None
    asyncio.run(run())
    assert transport.capture_failures == 0
    assert len(list(transport.directory.glob("*.capture.json"))) == 1
