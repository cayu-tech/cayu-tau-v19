"""Freeze a disclosed 12-task development test; no hidden evaluation inputs."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    project = Path(__file__).resolve().parents[1]
    previous = project.parent / 'cayu-tau3-repair-20260924/artifacts/full97-v9-20260925'
    for name, expected in json.loads((previous / 'source-manifest.json').read_text())['files'].items():
        assert digest(previous / name) == expected, name
    quota = project.parent / 'cayu-tau3-repair-20260924/artifacts/quota-handling-fix-20260925/manifest.json'
    for name, expected in json.loads(quota.read_text())['files'].items():
        assert digest(project / name) == expected, name
    root = project / 'artifacts/coherent-v10-screen12-20260925'
    root.mkdir(exist_ok=False)
    for name in ['src', 'scripts', 'tests']:
        shutil.copytree(project / name, root / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(project / 'pyproject.toml', root / 'pyproject.toml')
    shutil.copy2(previous / 'run_pilot.py', root / 'run_pilot.py')
    shutil.copy2(project / 'scripts/run_full_benchmark.py', root / 'run_benchmark.py')
    p = json.loads((previous / 'protocol.json').read_text())
    failures = ['task_' + t for t in ['046','054','056','058','074','080','095','100']]
    controls = ['task_' + t for t in ['043','059','075','096']]
    p.update(
        candidate_revision='coherent-v10-20260925', actor_source_identical=False,
        actor_snapshot_reference=str(previous), baseline_manifest_sha256=digest(previous / 'source-manifest.json'),
        task_ids=failures + controls, diagnostic_failures=failures, regression_controls=controls,
        expected_trajectories=12, max_parallel_task_processes=8,
        purpose='12-task development screen of coherent V10 plus staged quota fixes. Not an official benchmark or score estimate.',
        selection='Eight diagnosed V9 failures and four tasks passing in both V9 and the 52-pass baseline. Previously inspected public tasks; no ambiguous/reference-mismatch optimization targets.',
        causal_limit='Combined candidate update with historical comparisons; not a randomized ablation or an unseen holdout.',
        scoring='Upstream binary reward. Report all 12 attempts, diagnostic and control scores separately, with runtime/tokens/reads/compactions. No best-of selection.',
        qualification='Affected local tests and exact frozen Linux tests, source/catalog/framework checks before launch.',
        acceptance='Inspect every failure and controls, plus efficiency, before any full 97-task or official campaign.',
        treatment=[
            'Staged typed quota/nonretryable handling, summary retry preservation, early supervisor circuit detection; no whole-task retries',
            'Capture successful pre-execution condition receipts in advisory mode; exact planned step can settle after its own resource invalidation',
            'Reject unambiguous public product-title/category versus account_type mismatches; no silent argument rewrite',
            'Additional interest components require typed identity and exact delivered source; checking/card benefits cannot be added twice through other-components category',
            'Optional compact comparison rows in existing source-linked fact memory, automatically exposed after compaction',
            'Concise guidance targets missing decisive facts, cross-outcome prerequisites and operation-scoped uncertainty; no additional reviewer or mandatory decision packets',
        ],
        report=str(root / 'report.html'),
        whole_task_retries=0,
    )
    (root / 'protocol.json').write_text(json.dumps(p, indent=2) + '\n')
    (root / 'source-manifest.json').write_text(json.dumps({'files': {
        str(f.relative_to(root)): digest(f) for f in sorted(root.rglob('*')) if f.is_file()
    }}, indent=2) + '\n')
    (root / 'progress.json').write_text(json.dumps(dict(status='prepared', rows={}, running={}, pause_reason=None), indent=2))
    receipt = dict(root=str(root), manifest_sha256=digest(root/'source-manifest.json'),
                   tasks=12, workers=8, trials=1, seed=p['seed'], baseline=str(previous), quota_fixes_verified=True)
    (root / 'freeze-receipt.json').write_text(json.dumps(receipt, indent=2))
    archive = project / 'artifacts/coherent-v10-screen12-20260925.tar.gz'
    with tarfile.open(archive, 'x:gz') as output:
        output.add(root, arcname=root.name)
    print(json.dumps({**receipt, 'archive': str(archive), 'archive_sha256': digest(archive)}))


if __name__ == '__main__':
    main()
