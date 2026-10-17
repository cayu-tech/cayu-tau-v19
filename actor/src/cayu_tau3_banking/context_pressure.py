"""Advisory context-pressure warning; never adds a reviewer or blocks an action."""

from cayu import Message
from cayu.context.base import ObservedDeltaContextEstimator

COMPACTION_TRIGGER = 180_000
RECENT_CONTEXT_TARGET = 100_000
WARNING_RESERVE = 256
_WARNING_KEY = "tau_context_pressure_warning"


def warn_before_compaction(result, request, *, previous_checkpoint, threshold, reserved_output):
    """Warn once per durable compaction cursor using the composed request estimate.

    A response that jumps across the threshold may already have been compacted;
    this is an advisory, not a guarantee of a final note-writing turn.
    """
    if threshold is None:
        return result
    checkpoint = {**(previous_checkpoint or {}), **(result.checkpoint or {})}
    cursor = checkpoint.get("context_compaction", {}).get("compacted_transcript_cursor", 0)
    if checkpoint.get(_WARNING_KEY, {}).get("cursor") == cursor:
        return result
    estimator = ObservedDeltaContextEstimator()
    estimate = (
        estimator.estimate_anchored_request
        if result.messages == request.messages
        else estimator.estimate_full_request
    )
    pressure = estimate(
        usage=request.context_usage,
        messages=result.messages,
        overhead=request.pressure_overhead,
        reserved_output_tokens=reserved_output,
    ).estimated_context_window_tokens
    if pressure < threshold * 5 // 6:
        return result
    warning = Message.text(
        "user",
        "<context_pressure advisory='true'>"
        f"Estimated context including reserved space: {pressure} tokens; "
        f"automatic compaction threshold: {threshold}. "
        "Before another broad read, update only changed working notes: plausible alternatives, "
        "decisive values and conditions, exclusions with reasons, remaining decision-changing "
        "unknowns, and exact source IDs/offsets needed to continue. Use existing evidence; "
        "do not research merely to prepare notes or rewrite the whole worksheet. "
        "This is a context warning, not a task deadline or permission requirement. "
        "Proceed with supported customer work. Large results may trigger compaction before "
        "another turn; prefer bounded reads."
        "</context_pressure>",
    )
    checkpoint[_WARNING_KEY] = {"cursor": cursor, "estimated_tokens": pressure}
    # Cayu publishes checkpoint updates atomically only with an event receipt.
    # Preserve a real compaction receipt if present; do not label a warning as compaction.
    event = result.checkpoint_event_payload or {
        "checkpoint": _WARNING_KEY,
        "estimated_context_window_tokens": pressure,
        "trigger_estimated_context_tokens": threshold,
    }
    return result.model_copy(
        update={
            "messages": [*result.messages, warning],
            "checkpoint": checkpoint,
            "checkpoint_event_payload": event,
        }
    )
