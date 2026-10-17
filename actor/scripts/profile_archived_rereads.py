"""Run the existing exact-context reread audit on completed archive copies."""

import argparse
import asyncio
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from audit_read_repetition import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="+", type=Path)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for task in args.tasks:
        if not (task / "assessment.json").is_file():
            raise ValueError("Only completed tasks may be audited")
        with (
            zipfile.ZipFile(task / "diagnostics.zip") as archive,
            tempfile.TemporaryDirectory(prefix="tau-rereads-") as td,
        ):
            root = Path(td).resolve()
            infos = [
                i
                for i in archive.infolist()
                if (i.filename.startswith("sessions/") and i.filename.endswith(".sqlite3"))
                or (i.filename.startswith("contexts/") and i.filename.endswith(".json"))
            ]
            if sum(i.file_size for i in infos) > 512 * 1024 * 1024:
                raise ValueError("Diagnostic archive exceeds extraction allowance")
            for member in infos:
                destination = root / member.filename
                if not destination.resolve().is_relative_to(root) or member.is_dir():
                    raise ValueError("Invalid diagnostic path")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output)
            result = asyncio.run(audit(root, args.documents))
            result["task"] = task.name
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
    with args.output.open("x") as output:
        json.dump({"tasks": results}, output, indent=2)


if __name__ == "__main__":
    main()
