"""Freeze a four-trajectory development comparison; never edit prior runs."""
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    candidate = Path(__file__).resolve().parents[1]
    previous = candidate.parent / 'cayu-tau3-evidence-20260923/artifacts/evidence-ablation-20260923/no-packets-300'
    for name, expected in json.loads((previous / 'source-manifest.json').read_text())['files'].items():
        if digest(previous / name) != expected:
            raise ValueError('Previous frozen source changed: ' + name)
    root = candidate / 'artifacts/simplified-screen-20260923'
    root.mkdir(exist_ok=False)
    tasks = ['task_054', 'task_079']
    for arm, source in [('control', previous), ('simplified', candidate)]:
        actor = root / (arm + '-300')
        actor.mkdir()
        for directory in ['src', 'scripts', 'tests']:
            shutil.copytree(source / directory, actor / directory,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(source / 'pyproject.toml', actor / 'pyproject.toml')
        shutil.copy2(source / 'scripts/run_lean_advice_task.py', actor / 'run_pilot.py')
        (actor / '.vendor').symlink_to(source.joinpath('.vendor').resolve(), target_is_directory=True)
        protocol = json.loads((previous / 'protocol.json').read_text())
        protocol.update(task_ids=tasks, expected_trajectories=2,
            purpose='Four-trajectory simplified-candidate development screen; not an official score',
            selection='Two previously inspected development tasks, 054 and 079, not in the current exclusions register. Not an unseen holdout.',
            causal_limit='Combined prompt, receipts and public-guidance package versus the frozen prior no-packets actor. Does not isolate an individual feature.',
            stopping_policy='No whole-task retries. Stop the comparison for review after any infrastructure failure; preserve both arms and every attempt.',
            treatment=['Frozen previous no-packets actor'] if arm == 'control' else [
                'No mandatory packets; automatic observed execution receipts',
                'Public-source tool ownership and operation navigation',
                'Material outcome updates rather than prewriting every action',
                'Correct batch search schema repair wording'],
            qualification='Local full suite with affected rechecks, then Linux VM regression checks before launch')
        if arm == 'simplified':
            protocol['candidate_profile'] = 'simplified-v3'
            args = protocol['cli_arguments']
            args[args.index('--candidate-profile') + 1] = 'simplified-v3'
        save(actor / 'protocol.json', protocol)
        save(actor / 'source-manifest.json', {'files': {
            str(p.relative_to(actor)): digest(p)
            for p in sorted(actor.rglob('*'))
            if p.is_file() and '.vendor' not in p.parts}})
    python = str(candidate.parent / 'cayu-tau3-latest-20260923/artifacts/paired-comparison-20260923/candidate-venv/bin/python')
    save(root / 'comparison.json', dict(arms=['control','simplified'], seeds=[300], tasks=tasks,
         schedule=[{'task': task, 'seed': 300} for task in tasks],
         python={arm: python for arm in ['control', 'simplified']},
         api_key_file='/opt/cayu-tau3-credentials/openai_key.txt', concurrency=2,
         whole_task_retries=0,
         acceptance='Report every result, failures, elapsed time, tokens and bookkeeping. Four attempts are a smoke comparison, insufficient for a general accuracy claim. Do not promote to the official campaign automatically.'))
    shutil.copy2(candidate / 'scripts/run_matched_comparison.py', root / 'run_comparison.py')
    print(root)


if __name__ == '__main__':
    main()
