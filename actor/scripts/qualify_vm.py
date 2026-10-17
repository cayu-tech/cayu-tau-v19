"""Run every test in balanced disjoint file shards and record the exact frozen source."""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_full_benchmark import digest, verify, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    verify(root)
    files = sorted(str(p.relative_to(root)) for p in (root / "tests").glob("test_*.py"))
    durations = {}
    previous = root.parent / "official-banking-4trials-20260923-r2"
    for receipt in previous.glob("vm-tests-*.xml"):
        for case in ET.parse(receipt).iter("testcase"):
            filename = case.get("classname", "").replace(".", "/") + ".py"
            durations[filename] = durations.get(filename, 0.0) + float(case.get("time", 0))
    count = min(8, os.cpu_count() or 4)
    groups = [[] for _ in range(count)]
    weights = [0.0] * count
    for filename in sorted(files, key=lambda name: durations.get(name, 1.0), reverse=True):
        index = min(range(count), key=lambda i: weights[i])
        groups[index].append(filename)
        weights[index] += durations.get(filename, 1.0)
    assert sorted(name for group in groups for name in group) == files
    def run(shard):
        with (root / f"vm-tests-{shard}.log").open("xb") as log:
            return subprocess.run([sys.executable, "-m", "pytest", "-q", *groups[shard],
                f"--junitxml={root / f'vm-tests-{shard}.xml'}"], cwd=root,
                env={**os.environ, "PYTHONPATH": str(root / "src")},
                stdout=log, stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max_workers=count) as pool:
        codes = list(pool.map(run, range(count)))
    totals = dict(tests=0, failures=0, errors=0, skipped=0)
    for shard in range(count):
        for suite in ET.parse(root / f"vm-tests-{shard}.xml").iter("testsuite"):
            for key in totals:
                totals[key] += int(suite.get(key, 0))
    verify(root)
    passed = not any(codes) and totals["tests"] >= 1692 and not any(totals[k] for k in ("failures", "errors", "skipped"))
    write_json(root / "qualification.json", {"passed": passed, "platform": sys.platform,
        "source_manifest_sha256": digest(root / "source-manifest.json"),
        "test_files": files, "shard_exit_codes": codes, **totals})
    print({"passed": passed, **totals})
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
