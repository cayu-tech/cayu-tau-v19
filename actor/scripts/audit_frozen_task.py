"""Post-completion evidence extraction; never feeds reference answers to an actor."""

import argparse
import hashlib
import json
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.runtime_profile import profile_session


def parts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from parts(child)
    elif isinstance(value, list):
        for child in value:
            yield from parts(child)


def decode(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def audit(root: Path, task: str):
    protocol = json.loads((root / "protocol.json").read_text())
    if task not in protocol["task_ids"]:
        raise ValueError("Task is outside the frozen registration")
    directory = root / task
    assessment_path = directory / "assessment.json"
    if not assessment_path.is_file():
        raise ValueError("Wait for a completed assessment before inspecting this trajectory")
    for name, digest in json.loads((root / "source-manifest.json").read_text())["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen file changed: {name}")
    official = directory / "official/results.json"
    rows = [s for s in json.loads(official.read_text())["simulations"] if s["task_id"] == task]
    if len(rows) != 1:
        raise ValueError("Expected exactly one registered trial")
    simulation = rows[0]
    messages = simulation.get("messages") or []
    results = {m["id"]: m for m in messages if m["role"] == "tool"}
    external = []
    for index, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            result = results.get(call["id"])
            external.append(
                {
                    "message_index": index,
                    "caller": message["role"],
                    **call,
                    "result": None if result is None else decode(result["content"]),
                    "tool_error": None if result is None else result.get("error", False),
                }
            )
    private = {}
    for path in (directory / "contexts").glob("*/*.json"):
        for part in parts(json.loads(path.read_text()).get("messages", [])):
            if part.get("type") != "tool_call":
                continue
            value = {"name": part["tool_name"], "arguments": part["arguments"]}
            prior = private.setdefault(part["tool_call_id"], value)
            if prior != value:
                raise ValueError("Inconsistent call identity across saved contexts")
    result = {
        "task": task,
        "source_unchanged": True,
        "official_result_sha256": hashlib.sha256(official.read_bytes()).hexdigest(),
        "duration_seconds": simulation["duration"],
        "termination": simulation["termination_reason"],
        "reward_info": simulation["reward_info"],
        "dialogue": [
            {"index": i, "role": m["role"], "text": m["content"]}
            for i, m in enumerate(messages)
            if m["role"] in {"assistant", "user"} and m.get("content") and not m.get("tool_calls")
        ],
        "external_calls": external,
        "observed_context_calls": private,
        "product_packet_calls": {
            k: v
            for k, v in private.items()
            if v["name"] == "cayu_read_product_documents" and "product_id" in v["arguments"]
        },
        "profiles": {p.name: profile_session(p) for p in directory.glob("sessions/*.sqlite3")},
        "limits": "Read-only extraction, not a semantic grade. Reward metadata is evaluator-only. "
        "Context captures may not include a call made after the last captured request.",
    }
    save_private_json(directory / "post-run-evidence.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.task)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "task",
                    "duration_seconds",
                    "termination",
                    "source_unchanged",
                    "product_packet_calls",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
