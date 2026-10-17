import asyncio
from contextlib import aclosing

import pytest
from cayu import Message, ModelStreamEvent, ScriptedModelProvider
from cayu.providers.base import ModelRequest

from cayu_tau3_banking.runtime_helpers import TrackedHelperProvider
from cayu_tau3_banking.runtime_usage import RuntimeUsageAccounting


def request():
    return ModelRequest(model="test", messages=[Message.text("user", "Verify.")], tools=[])


def completed():
    return ModelStreamEvent.completed(
        {"usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}}
    )


def test_helper_usage_is_normalized_once_and_shared_with_actor_accounting():
    async def run():
        accounting = RuntimeUsageAccounting()
        helper = TrackedHelperProvider(ScriptedModelProvider([[completed()]]), accounting)
        async for _ in helper.runtime_stream(request()):
            assert accounting.active_helpers == 1
            assert not accounting.snapshot()["complete"]
        assert accounting.snapshot()["complete"]
        assert accounting.snapshot()["model_completions"] == 1
        assert accounting.snapshot()["input_tokens"] == 12
        assert accounting.snapshot()["output_tokens"] == 4

    asyncio.run(run())


def test_cancelled_helper_leaves_usage_unknown_and_releases_active_counter():
    async def run():
        accounting = RuntimeUsageAccounting()
        helper = TrackedHelperProvider(
            ScriptedModelProvider([[ModelStreamEvent.text_delta("partial"), completed()]]),
            accounting,
        )
        async with aclosing(helper.stream(request())) as stream:
            await anext(stream)
        assert accounting.active_helpers == 0
        assert not accounting.snapshot()["complete"]
        assert accounting.snapshot()["uncertain_usage_events"] == 1

    asyncio.run(run())


def test_duplicate_helper_completion_is_rejected_without_double_counting():
    async def run():
        accounting = RuntimeUsageAccounting()
        helper = TrackedHelperProvider(
            ScriptedModelProvider([[completed(), completed()]]), accounting
        )
        with pytest.raises(ValueError, match="duplicate completion"):
            async for _ in helper.stream(request()):
                pass
        assert accounting.active_helpers == 0
        assert accounting.snapshot()["model_completions"] == 1
        assert not accounting.snapshot()["complete"]

    asyncio.run(run())
