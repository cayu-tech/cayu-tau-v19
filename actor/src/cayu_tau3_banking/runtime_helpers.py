"""Account for application verifier calls without creating another acting loop."""

from contextlib import aclosing

from cayu import Event, EventType, normalize_usage_metrics
from cayu.providers import ModelProvider
from cayu.providers.base import ModelStreamEventType


class TrackedHelperProvider(ModelProvider):
    def __init__(self, provider, accounting):
        self.provider = provider
        self.accounting = accounting
        self.name = provider.name
        self.usage_dialect = provider.usage_dialect

    @property
    def stream_deadlines(self):
        return self.provider.stream_deadlines

    def runtime_stream(self, request):
        return self._tracked(request)

    def stream(self, request):
        return self._tracked(request)

    async def _tracked(self, request):
        completed = False
        failed = False
        self.accounting.active_helpers += 1
        try:
            async with aclosing(self.provider.runtime_stream(request)) as stream:
                async for event in stream:
                    if event.type is ModelStreamEventType.COMPLETED:
                        if completed:
                            raise ValueError("Helper emitted duplicate completion.")
                        completed = True
                        payload = event.payload
                        try:
                            metrics = normalize_usage_metrics(
                                provider_name=self.name,
                                model=payload.get("model"),
                                requested_model=request.model,
                                raw_usage=payload.get("usage"),
                                usage_dialect=self.usage_dialect,
                            )
                        except (TypeError, ValueError):
                            metrics = None
                        self.accounting.observe(
                            Event(
                                type=EventType.MODEL_COMPLETED,
                                session_id="application-helpers",
                                payload={}
                                if metrics is None
                                else {"usage_metrics": metrics.model_dump(mode="json")},
                            )
                        )
                    yield event
        except BaseException:
            failed = True
            raise
        finally:
            self.accounting.active_helpers -= 1
            if not completed or failed:
                self.accounting.observe(
                    Event(type=EventType.MODEL_ERROR, session_id="application-helpers")
                )
