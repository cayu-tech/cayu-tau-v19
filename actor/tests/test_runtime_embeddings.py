import asyncio
import concurrent.futures
import json
import threading

import pytest
from cayu import TextEmbedding, TextEmbeddingProvider, TextEmbeddingRequest, TextEmbeddingResult

from cayu_tau3_banking.embedding_diagnostics import (
    EmbeddingInitializationError,
    embedding_failure_details,
)
from cayu_tau3_banking.retrieval_eval import safe_error_diagnostic
from cayu_tau3_banking.runtime_embeddings import OwnedEmbeddingProvider


class Embeddings(TextEmbeddingProvider):
    name = "test"

    def __init__(self):
        self.loops = []
        self.closed = False

    async def embed_texts(self, request):
        self.loops.append(asyncio.get_running_loop())
        await asyncio.sleep(0)
        return TextEmbeddingResult(
            model=request.model,
            embeddings=[TextEmbedding(index=0, vector=[1.0, 0.0])],
        )

    async def aclose(self):
        self.loops.append(asyncio.get_running_loop())
        self.closed = True


def request():
    return TextEmbeddingRequest(model="test", texts=["query"], dimensions=2)


def test_ingestion_and_parallel_query_loops_share_one_owned_client_and_close_it():
    provider = Embeddings()
    with OwnedEmbeddingProvider(lambda: provider) as owner:
        asyncio.run(owner.embed_texts(request()))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: asyncio.run(owner.embed_texts(request())), range(4)))
        assert all(result.embeddings[0].vector == [1.0, 0.0] for result in results)
    assert provider.closed
    assert len(set(provider.loops)) == 1
    assert not owner._thread.is_alive()
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(owner.embed_texts(request()))


def test_query_cancellation_settles_owned_request_before_shutdown():
    entered = threading.Event()
    settled = threading.Event()

    class Blocking(Embeddings):
        async def embed_texts(self, request):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                settled.set()

    provider = Blocking()
    with OwnedEmbeddingProvider(lambda: provider) as owner:

        async def run():
            task = asyncio.create_task(owner.embed_texts(request()))
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())
    assert settled.is_set()
    assert provider.closed


def test_failed_provider_construction_does_not_leak_owner_thread():
    def fail():
        raise ValueError("construction failed")

    with (
        OwnedEmbeddingProvider(fail) as owner,
        pytest.raises(ValueError, match="construction failed"),
    ):
        asyncio.run(owner.embed_texts(request()))
    assert not owner._thread.is_alive()


def test_embedding_failure_survives_backfill_wrapping_without_secrets():
    secret = "sensitive-key-and-provider-body"

    class Rejected(Embeddings):
        async def embed_texts(self, request):
            error = RuntimeError(secret)
            error.status_code = 401
            error.error_code = "invalid_api_key"
            error.body = {"secret": secret}
            raise error

    with OwnedEmbeddingProvider(Rejected) as owner:
        with pytest.raises(RuntimeError):
            asyncio.run(owner.embed_texts(request()))
        saved = owner.failure_diagnostic
        assert saved["exception_chain"][0]["http_status"] == 401
        assert saved["exception_chain"][0]["provider_code"] == "invalid_api_key"
        saved["exception_chain"].clear()
        diagnostic = safe_error_diagnostic(EmbeddingInitializationError(owner.failure_diagnostic))
        assert diagnostic["embedding_failure"]["exception_chain"]
        assert secret not in json.dumps(diagnostic)


def test_embedding_diagnostics_bound_cycles_and_reject_arbitrary_codes():
    error = RuntimeError("secret")
    error.code = "secret"
    error.status_code = True
    error.__cause__ = error
    details = embedding_failure_details(error)
    assert details["exception_chain"] == [{"error_type": "RuntimeError"}]


def test_parallel_channels_queue_instead_of_failing_at_eight_requests():
    class Bounded(Embeddings):
        active = 0
        peak = 0
        async def embed_texts(self, request):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0.03)
                return await super().embed_texts(request)
            finally:
                self.active -= 1
    provider = Bounded()
    with OwnedEmbeddingProvider(lambda: provider) as owner:
        async def run():
            return await asyncio.gather(*(owner.embed_texts(request()) for _ in range(16)))
        results = asyncio.run(run())
    assert len(results) == 16
    assert provider.peak == 8
    assert provider.active == 0
    assert provider.closed
