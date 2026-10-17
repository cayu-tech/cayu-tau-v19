"""Opt-in private search inventories, never injected into model context."""

import asyncio
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


class RetrievalCapture:
    def __init__(self, directory: Path, task_id: str):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="retrieval-", dir=directory))
        self.task_id = task_id

    async def __call__(self, snapshot: dict) -> str:
        trace_id = uuid4().hex
        await asyncio.to_thread(
            self._save,
            {
                **snapshot,
                "task_id": self.task_id,
                "trace_id": trace_id,
                "captured_at": datetime.now(UTC).isoformat(),
                "boundary": "prepared search result, before tool completion; join the runtime result trace_id to establish delivery",
            },
        )
        return trace_id

    def _save(self, snapshot: dict) -> None:
        descriptor = os.open(
            self.directory / f"{snapshot['trace_id']}.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(snapshot, output, ensure_ascii=False)
            output.write("\n")
