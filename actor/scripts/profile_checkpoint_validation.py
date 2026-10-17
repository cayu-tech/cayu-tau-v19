"""Offline validation scaling on valid prefixes of a captured receipt ledger.

Never modifies or resumes the original SQLite store; never relaxes validation.
Private framework functions are used only as explicitly pinned diagnostic probes.
"""

import argparse
import copy
import cProfile
import hashlib
import importlib.metadata
import json
import pstats
import sqlite3
import statistics
import tempfile
import time
import zipfile
from pathlib import Path

from cayu.runtime._invocation_lifecycle import (
    _invocation_lifecycle_receipt_ledger_from_checkpoint,
    _InvocationLifecycleReceiptLedger,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (
        zipfile.ZipFile(args.archive) as archive,
        tempfile.TemporaryDirectory(prefix="tau-ledger-profile-") as td,
    ):
        members = [
            i
            for i in archive.infolist()
            if i.filename.startswith("sessions/") and i.filename.endswith(".sqlite3")
        ]
        if len(members) != 1 or members[0].file_size > 256 * 1024 * 1024:
            raise ValueError("Expected one bounded session database")
        path = Path(td) / "snapshot.sqlite3"
        path.write_bytes(archive.read(members[0]))
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            rows = db.execute("SELECT state_json FROM cayu_checkpoints").fetchall()
        if len(rows) != 1:
            raise ValueError("Expected one checkpoint")
        checkpoint = json.loads(rows[0][0])
    original = _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
    epochs = sorted({r.active_profile.run_epoch for r in original.receipts})
    results = []
    for epoch in epochs:
        receipts = tuple(r for r in original.receipts if r.active_profile.run_epoch <= epoch)
        active = max(
            (r for r in receipts if r.kind.value in {"create", "admit", "rebind"}),
            key=lambda r: r.active_profile.run_epoch,
        )
        release = f"release:{active.session_id}:{active.session_instance_id}:{active.active_profile.run_epoch}"
        reservation = (
            None
            if any(r.command_identity == release for r in receipts)
            else active.command_identity
        )
        ledger = _InvocationLifecycleReceiptLedger(
            receipts=receipts, release_capacity_command_identity=reservation
        )
        data = {"invocation_lifecycle_receipt": ledger.model_dump(mode="json")}
        before = json.dumps(data, sort_keys=True)
        samples = []
        for _ in range(7):
            start = time.perf_counter()
            verified = _invocation_lifecycle_receipt_ledger_from_checkpoint(data)
            samples.append(time.perf_counter() - start)
            assert verified.record_sha256 == ledger.record_sha256
        assert json.dumps(data, sort_keys=True) == before
        changed = copy.deepcopy(data)
        changed["invocation_lifecycle_receipt"]["record_sha256"] = "f" * 64
        try:
            _invocation_lifecycle_receipt_ledger_from_checkpoint(changed)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("Tampered ledger was accepted")
        results.append(
            {
                "epoch_prefix": epoch,
                "receipt_count": len(receipts),
                "canonical_json_bytes": len(before.encode()),
                "median_seconds": statistics.median(samples),
                "min_seconds": min(samples),
                "max_seconds": max(samples),
                "samples_seconds": samples,
                "tampering_rejected": True,
            }
        )
    profiler = cProfile.Profile()
    profiler.runcall(_invocation_lifecycle_receipt_ledger_from_checkpoint, checkpoint)
    stats = pstats.Stats(profiler)
    functions = sorted(
        (
            {
                "file": Path(file).name,
                "function": function,
                "primitive_calls": values[0],
                "calls": values[1],
                "self_seconds": values[2],
                "cumulative_seconds": values[3],
            }
            for (file, _line, function), values in stats.stats.items()
        ),
        key=lambda r: r["cumulative_seconds"],
        reverse=True,
    )[:15]
    with args.archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "archive_sha256": digest,
        "framework_install": json.loads(
            importlib.metadata.distribution("cayu").read_text("direct_url.json")
        ),
        "source": "Valid epoch prefixes of an existing completed task checkpoint's invocation-lifecycle receipt ledger",
        "ordinary_timings": results,
        "one_profiled_validation": functions,
        "limits": [
            "Offline per-call latency on this machine, not historical checkpoint time or total session overhead.",
            "Prefix sizes are controlled within one ledger; later records can have different serialized sizes.",
            "cProfile timings are perturbed and are not compared with unprofiled latency.",
            "No validation bypass, cache, runtime change or historical resume was applied.",
        ],
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            {
                "rows": [{k: v for k, v in r.items() if k != "samples_seconds"} for r in results],
                "top_profiled_functions": functions[:5],
            }
        )
    )


if __name__ == "__main__":
    main()
