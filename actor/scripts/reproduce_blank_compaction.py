"""Offline reproduction: blank summaries fail outside ModelCompactor's retry path.

This is a diagnostic of the pinned framework, not a workaround or a banking test.
It makes no network calls and does not read benchmark tasks or credentials.
"""

import asyncio
import json

from cayu import AgentSpec, Message
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runtime import (
    CompactionRequest,
    InMemorySessionStore,
    ModelCompactor,
    RunRequest,
    SessionIdentity,
)
from cayu.runtime.retry_policy import RetryPolicy


class ScriptedSummaryProvider(ModelProvider):
    name = "blank-summary-reproducer"

    def __init__(self, first_summary: str):
        self.first_summary = first_summary
        self.calls = 0

    async def stream(self, request):
        self.calls += 1
        summary = self.first_summary if self.calls == 1 else "Valid recovery summary."
        if summary:
            yield ModelStreamEvent.text_delta(summary)
        yield ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 4}})


async def reproduce():
    store = InMemorySessionStore()
    session = await store.create(
        RunRequest(agent_name="assistant", messages=[]),
        identity=SessionIdentity(provider_name=ScriptedSummaryProvider.name, model="fake"),
    )
    rows = []
    for label, summary in (("empty", ""), ("whitespace", "   "), ("valid", "Valid summary.")):
        provider = ScriptedSummaryProvider(summary)
        compactor = ModelCompactor(
            provider=provider,
            model="fake",
            retry_policy=RetryPolicy(max_attempts=2, max_unknown_attempts=2),
        )
        row = {"case": label, "configured_max_attempts": 2}
        try:
            result = await compactor.compact(
                CompactionRequest(
                    session=session,
                    agent=AgentSpec(name="assistant", model="fake"),
                    messages=[Message.text("user", "Remember the pending request.")],
                )
            )
            row["summary"] = result.summary
        except ValueError as error:
            row.update(error_type=type(error).__name__, error=str(error))
        row["provider_calls"] = provider.calls
        rows.append(row)
    return rows


if __name__ == "__main__":
    print(json.dumps(asyncio.run(reproduce()), indent=2))
