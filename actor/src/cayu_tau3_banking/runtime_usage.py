"""Content-free usage export from Cayu events to half-duplex τ replies.

One model round may produce multiple host callbacks. Counters are drained at
each reply boundary, never repeated on subsequent calls in the same round.
Missing usage remains unknown; it is never replaced by a zero-cost estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

from cayu import Event, EventType
from cayu.budgets.usage import (
    is_conversational_model_completion_payload,
    usage_metrics_from_event_payload,
)
from tau2.data_model.message import AssistantMessage


@dataclass
class _Counters:
    model_completions: int = 0
    missing_usage_completions: int = 0
    uncertain_usage_events: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_output_tokens: int = 0

    def payload(self) -> dict[str, int]:
        return {
            "model_completions": self.model_completions,
            "missing_usage_completions": self.missing_usage_completions,
            "uncertain_usage_events": self.uncertain_usage_events,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
        }


class RuntimeUsageAccounting:
    def __init__(self, *, untracked_helpers: bool = False):
        if type(untracked_helpers) is not bool:
            raise TypeError("untracked_helpers must be a boolean.")
        self.untracked_helpers = untracked_helpers
        self._interval = _Counters()
        self._total = _Counters()
        self._in_flight = False
        self.active_helpers = 0

    def observe(self, event: Event) -> None:
        if event.type == EventType.MODEL_STARTED:
            self._in_flight = True
            return
        if event.type in {EventType.MODEL_ERROR, EventType.MODEL_ATTEMPT_DISCARDED}:
            # An errored/discarded attempt may still be billable. A later
            # successful completion does not prove the failed attempt cost zero.
            self._interval.uncertain_usage_events += 1
            self._total.uncertain_usage_events += 1
            return
        if event.type != EventType.MODEL_COMPLETED:
            return
        if is_conversational_model_completion_payload(event.payload):
            self._in_flight = False
        try:
            metrics = usage_metrics_from_event_payload(event.payload)
        except (ValueError, TypeError):
            metrics = None
        for counters in (self._interval, self._total):
            counters.model_completions += 1
            if metrics is None:
                counters.missing_usage_completions += 1
                continue
            counters.input_tokens += metrics.input_tokens
            counters.output_tokens += metrics.output_tokens
            counters.cached_input_tokens += metrics.cache.cached_input_tokens
            counters.reasoning_output_tokens += metrics.reasoning_output_tokens

    def snapshot(self) -> dict:
        """Known lower-bound totals, retained even when no reply can be emitted."""

        return {
            "schema_version": 1,
            "scope": "cayu_runtime_model_events",
            "untracked_helpers": self.untracked_helpers,
            "model_in_flight": self._in_flight,
            "active_helpers": self.active_helpers,
            "complete": self._complete(self._total),
            **self._total.payload(),
        }

    def _complete(self, counters: _Counters) -> bool:
        return not (
            self.untracked_helpers
            or self.active_helpers
            or self._in_flight
            or counters.missing_usage_completions
            or counters.uncertain_usage_events
        )

    def attach(self, reply: AssistantMessage) -> AssistantMessage:
        interval = self._interval
        complete = self._complete(interval)
        reply.usage = (
            {"prompt_tokens": interval.input_tokens, "completion_tokens": interval.output_tokens}
            if complete
            else None
        )
        # Pricing belongs to the run's explicit billing configuration, not an
        # inferred model name or a dynamically fetched price table here.
        reply.cost = None
        reply.raw_data = {
            **(reply.raw_data or {}),
            "cayu_runtime_usage": {
                "schema_version": 1,
                "scope": "since_previous_agent_reply",
                "complete": complete,
                "untracked_helpers": self.untracked_helpers,
                **interval.payload(),
                "conversation": self.snapshot(),
            },
        }
        self._interval = _Counters()
        return reply
