"""Offline instruction replacement for captured-context diagnostics, not runtime."""

from collections.abc import Sequence

from cayu import Message
from cayu.providers.base import ModelRequest
from cayu.providers.openai import build_openai_payload


def replace_captured_instructions(
    messages: Sequence[Message], *, original: str, replacement: str
) -> list[Message]:
    """Replace the captured system message; never prepend a second instruction set."""
    if not original or not replacement or not replacement.strip():
        raise ValueError("Expected nonempty original and replacement instructions.")
    if (
        not messages
        or messages[0].role != "system"
        or sum(message.role == "system" for message in messages) != 1
    ):
        raise ValueError("Expected exactly one leading captured system message.")
    serialized = messages[0].model_dump(mode="json")
    content = serialized["content"]
    if len(content) != 1 or content[0]["type"] != "text" or content[0]["text"] != original:
        raise ValueError("Captured system message does not match the declared instructions.")
    content[0]["text"] = replacement
    return [Message.model_validate(serialized), *messages[1:]]


def checked_guidance_payload(request: ModelRequest, *, expected: str) -> dict:
    """Use the subscription provider's actual builder before any authentication."""
    payload = build_openai_payload(request, stream=True, reasoning_state="inline")
    if payload.get("instructions") != expected:
        raise ValueError("Prepared instructions must equal the intended prompt exactly once.")
    return payload
