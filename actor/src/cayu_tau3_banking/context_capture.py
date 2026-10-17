"""Opt-in private snapshots at the final application context-policy boundary.

These are composed-context snapshots, not provider wire logs or proof of dispatch.
No credentials, provider options, or arbitrary checkpoint metadata are serialized.
Conversation content itself may be sensitive; enable only for local diagnostics.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from cayu.context.base import RuntimeManagedContextPolicy


class CapturingContextPolicy(RuntimeManagedContextPolicy):
    def __init__(self, base, directory: Path):
        self.base = base
        directory.mkdir(parents=True, exist_ok=True)
        # A fresh private child prevents reuse/overwrite of another run's files.
        self.directory = Path(tempfile.mkdtemp(prefix="context-", dir=directory))

    async def build_with_checkpoint(self, request, *, checkpoint):
        result = await self.base.build_with_checkpoint(request, checkpoint=checkpoint)
        effective = result.checkpoint if result.checkpoint is not None else checkpoint
        compaction = (effective or {}).get("context_compaction") or {}
        snapshot = {
            "schema": "cayu-tau3.composed-context.v1",
            "boundary": "after application compaction and recall; before provider preparation",
            "session_id": request.session.id,
            "interaction_id": request.interaction_id,
            "model_step_id": request.model_step_id,
            "step": request.step,
            "model": request.agent.model,
            "system_prompt": request.agent.system_prompt,
            "messages": [message.model_dump(mode="json") for message in result.messages],
            "compaction": {
                key: compaction[key]
                for key in ("summary", "compacted_transcript_cursor")
                if key in compaction
            },
        }
        # Await persistence: never continue a diagnostic run with silently missing evidence.
        await asyncio.to_thread(self._save, snapshot)
        return result

    def _save(self, snapshot):
        data = json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
        descriptor = os.open(
            self.directory / f"{uuid4().hex}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(data)
