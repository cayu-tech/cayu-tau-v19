"""One pooled embedding client, owned by one loop across parallel conversations."""

import asyncio
import threading
from copy import deepcopy

from cayu import TextEmbeddingProvider

from cayu_tau3_banking.embedding_diagnostics import embedding_failure_details


class OwnedEmbeddingProvider(TextEmbeddingProvider):
    name = "banking-owned-embeddings"

    def __init__(self, factory):
        self._factory = factory
        self._provider = None
        self._pending = set()
        self._admission = None
        self._lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._closed = False
        self._shutdown = None
        self._failure_diagnostic = None
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="banking-embeddings", daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.run_until_complete(self._loop.shutdown_default_executor(timeout=30))
            self._loop.close()

    async def _embed(self, request):
        task = asyncio.current_task()
        # One task may request multiple retrieval channels. Queue those requests
        # on the owned loop instead of misclassifying saturation as an outage.
        self._pending.add(task)
        try:
            if self._admission is None:
                self._admission = asyncio.Semaphore(8)
            async with self._admission:
                if self._provider is None:
                    self._provider = self._factory()
                return await self._provider.embed_texts(request)
        except Exception as error:
            with self._lock:
                if self._failure_diagnostic is None:
                    self._failure_diagnostic = embedding_failure_details(error)
            raise
        finally:
            self._pending.discard(task)

    @property
    def failure_diagnostic(self):
        with self._lock:
            return deepcopy(self._failure_diagnostic)

    async def embed_texts(self, request):
        with self._lock:
            if self._closed:
                raise RuntimeError("Embedding provider is closed.")
            future = asyncio.run_coroutine_threadsafe(
                self._embed(request.model_copy(deep=True)), self._loop
            )
        # Cancellation reaches the owned request; no abandoned background billing.
        return await asyncio.wrap_future(future)

    async def _close_owned(self):
        tasks = tuple(self._pending)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._provider is not None:
            await self._provider.aclose()

    def close(self):
        if threading.current_thread() is self._thread:
            raise RuntimeError("Cannot synchronously close the embedding owner from its loop.")
        with self._stop_lock:
            self._close()

    def _close(self):
        with self._lock:
            if not self._thread.is_alive():
                return
            self._closed = True
            if self._shutdown is None:
                self._shutdown = asyncio.run_coroutine_threadsafe(self._close_owned(), self._loop)
            future = self._shutdown
        # Retain ownership on timeout; a later close can wait for the same cleanup.
        future.result(timeout=30)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise RuntimeError("Embedding owner did not stop.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
