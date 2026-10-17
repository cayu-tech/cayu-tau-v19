import asyncio
import json

import pytest
from cayu import ModelStreamEvent, ScriptedModelProvider

from cayu_tau3_banking.compaction import SUMMARY_INSTRUCTIONS
from cayu_tau3_banking.compaction_fidelity_probe import CASES, run


def test_probe_uses_lb_only_and_saves_exact_requests(tmp_path, monkeypatch):
    seen = []

    class Provider(ScriptedModelProvider):
        async def aclose(self):
            pass

    provider = Provider(
        [
            [
                ModelStreamEvent.text_delta("Scripted summary, not a semantic assessment."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
            for _ in CASES
        ]
    )

    def build(**options):
        seen.append(options)
        return provider

    monkeypatch.setattr("cayu_tau3_banking.compaction_fidelity_probe.build_provider", build)
    root = tmp_path / "probe"
    asyncio.run(run(root))
    assert seen[0]["provider"] == "codex-lb"
    assert seen[0]["codex_lb_base_url"] == "https://codex-lb.cayu.ai"
    assert len(provider.requests) == len(CASES)
    for name, request in zip(CASES, provider.requests, strict=True):
        assert request.messages[0].content[0].text == SUMMARY_INSTRUCTIONS
        assert request.messages[1].content[0].text == CASES[name]
        assert request.tools == []
        assert json.loads((root / f"{name}-request.json").read_text()) == request.model_dump(
            mode="json"
        )
    with pytest.raises(FileExistsError):
        asyncio.run(run(root))
