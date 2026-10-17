"""Start only a qualified frozen census; keep expanded diagnostics intact."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve(strict=True)
    sys.path.insert(0, str(root))
    import run_benchmark as runner
    from archive_support import archive_preserving

    runner.verify(root)
    protocol = json.loads((root / "protocol.json").read_text())
    qualification = json.loads((root / "linux-qualified.json").read_text())
    if qualification != {
        "passed": True,
        "totals": {"tests": protocol["expected_tests"], "failures": 0, "errors": 0, "skipped": 0},
        "manifest_sha256": runner.digest(root / "source-manifest.json"),
    }:
        raise ValueError("Missing exact Linux qualification")
    if not json.loads((root / "connectivity.json").read_text()).get("passed"):
        raise ValueError("Missing connectivity qualification")
    runner.archive_diagnostics = lambda directory: archive_preserving(directory, runner.write_json)
    runner.run(root, Path("/opt/cayu-tau3-credentials/openai_key.txt"))


if __name__ == "__main__":
    main()
