import asyncio
import json
import stat

import pytest
from test_knowledge import corpus, write_document

from cayu_tau3_banking.retrieval_capture import RetrievalCapture


@pytest.mark.parametrize("mode", ["literal", "lexical"])
def test_capture_is_private_and_does_not_inject_inventory(tmp_path, mode):
    memory = corpus(tmp_path)
    capture = RetrievalCapture(tmp_path / "captures", "task")

    async def run():
        ordinary, refs = await memory.search_with_evidence("card", mode=mode)
        captured, captured_refs = await memory.search_with_evidence(
            "card", mode=mode, capture=capture
        )
        payload = json.loads(captured)
        trace_id = payload.pop("retrieval_trace_id")
        assert payload == json.loads(ordinary)
        assert refs == captured_refs
        path = capture.directory / f"{trace_id}.json"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(capture.directory.stat().st_mode) == 0o700
        trace = json.loads(path.read_text())
        assert trace["trace_id"] == trace_id
        assert trace["task_id"] == "task"
        assert trace["inventory"]
        if mode == "lexical":
            assert trace["recall_result"]["candidates"]

    asyncio.run(run())


def test_capture_failure_is_not_silently_ignored(tmp_path):
    memory = corpus(tmp_path)

    async def failed(snapshot):
        raise OSError("disk unavailable")

    with pytest.raises(OSError, match="disk unavailable"):
        asyncio.run(memory.search_with_evidence("card", capture=failed))


def test_capability_capture_links_exact_inventory(tmp_path):
    from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus

    write_document(
        tmp_path,
        "transfer_policy",
        "Public procedure",
        "Call read_package_status_1234 to read package status.",
    )
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)
    capture = RetrievalCapture(tmp_path / "capture", "task")
    result, _ = asyncio.run(
        memory.search_with_evidence("read_package_status_1234", mode="capability", capture=capture)
    )
    payload = json.loads(result)
    trace = json.loads((capture.directory / f"{payload['retrieval_trace_id']}.json").read_text())
    assert trace["inventory"] == [["transfer_policy", 1]]


def test_concurrent_captures_preserve_inventory_beyond_delivered_page(tmp_path):
    from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus

    for index in range(25):
        write_document(tmp_path, f"source_{index:02}", "Routing policy", "Routing instructions.")
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)
    capture = RetrievalCapture(tmp_path / "capture", "task")

    async def run():
        return await asyncio.gather(
            *[
                memory.search_with_evidence("routing", limit=1, mode="lexical", capture=capture)
                for _ in range(2)
            ]
        )

    results = asyncio.run(run())
    payloads = [json.loads(content) for content, _ in results]
    assert payloads[0]["retrieval_trace_id"] != payloads[1]["retrieval_trace_id"]
    for payload in payloads:
        trace = json.loads(
            (capture.directory / f"{payload['retrieval_trace_id']}.json").read_text()
        )
        assert len(trace["inventory"]) == 25
        assert len(trace["page"]["documents"]) == 1
        assert payload["navigation"]["remaining_unseen_in_pool"] == 24
        assert "inventory" not in payload
