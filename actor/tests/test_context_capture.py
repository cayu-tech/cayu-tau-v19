import asyncio
import json
import stat
from types import SimpleNamespace

from cayu import Message
from cayu.context.base import ContextBuildResult

from cayu_tau3_banking.context_capture import CapturingContextPolicy


def test_capture_preserves_each_summary_and_messages_without_mutating_result(tmp_path):
    class Base:
        async def build_with_checkpoint(self, request, *, checkpoint):
            return ContextBuildResult(
                messages=[Message.text("user", f"recall-{request.step}")],
                checkpoint={
                    "context_compaction": {
                        "summary": f"summary-{request.step}",
                        "compacted_transcript_cursor": request.step,
                    },
                    "not_for_capture": "SECRET_SENTINEL",
                },
            )

    policy = CapturingContextPolicy(Base(), tmp_path)
    request = SimpleNamespace(
        session=SimpleNamespace(id="session"),
        interaction_id="turn",
        model_step_id="step",
        step=1,
        agent=SimpleNamespace(model="test", system_prompt="system"),
    )

    async def exercise():
        first = await policy.build_with_checkpoint(request, checkpoint=None)
        request.step = 2
        second = await policy.build_with_checkpoint(request, checkpoint=None)
        assert first.checkpoint["context_compaction"]["summary"] == "summary-1"
        assert second.checkpoint["context_compaction"]["summary"] == "summary-2"

    asyncio.run(exercise())
    files = list(policy.directory.glob("*.json"))
    assert len(files) == 2
    snapshots = sorted([json.loads(path.read_text()) for path in files], key=lambda x: x["step"])
    assert [x["compaction"]["summary"] for x in snapshots] == ["summary-1", "summary-2"]
    assert snapshots[0]["messages"][0]["content"][0]["text"] == "recall-1"
    assert all("SECRET_SENTINEL" not in path.read_text() for path in files)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in files)
    assert stat.S_IMODE(policy.directory.stat().st_mode) == 0o700
