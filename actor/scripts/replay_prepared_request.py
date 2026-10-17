"""Replay one fully captured payload through normal subscription authentication.

Diagnostic only: no environment/tool execution, no benchmark score, no retries.
Credentials and raw errors are never persisted. The original capture is untouched.
"""

import argparse
import asyncio
import hashlib
import json
import tempfile
import time
from collections import Counter
from contextlib import aclosing
from pathlib import Path

from cayu import Message
from cayu.providers.base import ModelRequest

from cayu_tau3_banking.cli import build_provider
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.provider_diagnostics import DiagnosticOpenAITransport
from cayu_tau3_banking.retrieval_eval import safe_error_diagnostic


class ReplayTransport:
    def __init__(self, base, payload):
        self.base = base
        self.payload = payload
        self.calls = 0

    async def stream_response_events(self, **kwargs):
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("Diagnostic permits exactly one request")
        kwargs["payload"] = json.loads(json.dumps(self.payload))
        async with aclosing(self.base.stream_response_events(**kwargs)) as stream:
            async for event in stream:
                yield event

    async def aclose(self):
        await self.base.aclose()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    capture = json.loads(args.request.read_text())
    if capture.get("payload_redacted", False):
        raise ValueError("Exact payload replay is unavailable: the capture was redacted")
    if capture.get("omitted_field_count") != 0:
        raise ValueError("Exact payload replay requires a complete capture")
    if capture.get("schema") != "cayu-tau3.provider-request.v1":
        raise ValueError("Unsupported capture schema")
    payload = capture["payload"]
    output = Path(tempfile.mkdtemp(prefix="prepared-replay-", dir="artifacts"))
    save_private_json(
        output / "protocol.json",
        {
            "source": str(args.request.resolve()),
            "capture_sha256": hashlib.sha256(args.request.read_bytes()).hexdigest(),
            "live": args.live,
            "max_requests": 1,
            "tools_executed": 0,
            "limitation": "Identical captured body; fresh authentication and connection. Not a resumed task.",
        },
    )
    print(f"Output: {output}", flush=True)
    if not args.live:
        return
    provider = build_provider(
        provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
    )
    transport = ReplayTransport(DiagnosticOpenAITransport(provider.transport, output), payload)
    provider.transport = transport
    events = Counter()
    result = {}
    started = time.monotonic()
    try:
        async with asyncio.timeout(300):
            async with aclosing(
                provider.runtime_stream(
                    ModelRequest(
                        model=payload["model"],
                        messages=[Message.text("user", "Diagnostic transport replay")],
                    )
                )
            ) as stream:
                async for event in stream:
                    events[event.type.value] += 1
    except Exception as error:
        result["exception"] = safe_error_diagnostic(error)
    finally:
        await provider.aclose()
    result.update(
        seconds=time.monotonic() - started,
        events=dict(events),
        requests=transport.calls,
        tools_executed=0,
    )
    save_private_json(output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
