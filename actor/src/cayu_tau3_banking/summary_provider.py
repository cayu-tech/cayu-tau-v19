"""Permit bounded redispatch of interrupted, demonstrably text-only summaries.

Cayu's compactor owns retries and accounting. This adapter never dispatches a
second request itself and is never installed on the acting agent's provider.
"""

from contextlib import aclosing

from cayu.providers import ModelProvider, OpenAISubscriptionProvider
from cayu.providers.base import ModelStreamEvent, ModelStreamEventType
from cayu.providers.openai import OpenAIProvider, build_openai_payload


class ReadOnlySummaryProvider(ModelProvider):
    def __init__(self, provider: OpenAIProvider | OpenAISubscriptionProvider):
        if not isinstance(provider, (OpenAIProvider, OpenAISubscriptionProvider)) or (
            isinstance(provider, OpenAIProvider) and provider.reasoning_state != "inline"
        ):
            raise ValueError("Summary recovery requires an inline OpenAI API or subscription adapter.")
        self.provider = provider
        self.name = provider.name
        self.billing_provider_name = provider.billing_provider_name
        self.usage_dialect = provider.usage_dialect

    @property
    def stream_deadlines(self):
        return self.provider.stream_deadlines

    async def billing_identity_for_request(self, request):
        return await self.provider.billing_identity_for_request(request)

    def billing_identity_for_completion(self, identity, payload):
        return self.provider.billing_identity_for_completion(identity, payload)

    def request_fingerprint_options(self, request):
        return self.provider.request_fingerprint_options(request)

    def request_footprint_options(self, request):
        return self.provider.request_footprint_options(request)

    def request_cache_policy(self, request):
        return self.provider.request_cache_policy(request)

    def request_cache_projection(self, request):
        return self.provider.request_cache_projection(request)

    def runtime_stream(self, request):
        return self._stream_summary(request)

    def stream(self, request):
        return self._stream_summary(request)

    async def _stream_summary(self, request):
        # Validate the actual native payload, including raw provider options.
        # No hosted tools, conversation continuation, background operation,
        # native attachments or arbitrary provider-specific capability may enter.
        payload = build_openai_payload(request, stream=True, reasoning_state="inline")
        if (
            request.tools
            or request.tool_discovery_projection is not None
            or request.targeted_tool_projection is not None
            or set(payload)
            - {"model", "input", "store", "instructions", "include", "stream", "reasoning"}
            or payload.get("store") is not False
            or any(
                message.get("role") != "user"
                or any(part.get("type") != "input_text" for part in message.get("content", []))
                for message in payload["input"]
            )
        ):
            raise ValueError("Recovery is restricted to stateless text-only summary requests.")
        completed = False
        async with aclosing(self.provider.runtime_stream(request)) as stream:
            async for event in stream:
                if event.type is ModelStreamEventType.COMPLETED:
                    completed = True
                if (
                    not completed
                    and event.type is ModelStreamEventType.ERROR
                    and event.payload.get("provider_error_type")
                    in {"RemoteProtocolError", "ReadError", "ReadTimeout", "service_unavailable_error"}
                    and event.payload.get("retryable") is False
                    and event.payload.get("status_code") in (None, 200)
                    and not event.payload.get("stream_cleanup_failed")
                    and not event.payload.get("quota_recovery")
                ):
                    # A second text-only generation may incur additional model
                    # spend, but cannot repeat a banking or hosted-tool effect.
                    event = ModelStreamEvent(
                        type=event.type,
                        payload={
                            **event.payload,
                            "retryable": True,
                            "summary_recovery": "interrupted_read_only_generation",
                        },
                    )
                yield event
