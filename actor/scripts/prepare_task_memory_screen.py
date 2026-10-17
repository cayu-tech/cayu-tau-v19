"""Freeze two task-memory treatments without modifying any previous run."""
import hashlib
import json
import shutil
from pathlib import Path


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n')


def main():
    source = Path(__file__).resolve().parents[1]
    old = source.parent / 'cayu-tau3-evidence-20260923/artifacts/evidence-ablation-20260923/no-packets-300'
    for name, expected in json.loads((old/'source-manifest.json').read_text())['files'].items():
        if digest(old/name) != expected:
            raise ValueError('Previous frozen source changed: '+name)
    root = source/'artifacts/task-memory-screen-20260923'
    root.mkdir(exist_ok=False)
    tasks = ['task_054', 'task_005']
    for arm, profile in [('constraints', 'constraints-v4'), ('notebook', 'notebook-v4')]:
        actor = root/(arm+'-300'); actor.mkdir()
        for name in ['src', 'scripts', 'tests']:
            shutil.copytree(source/name, actor/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ['README.md', 'pyproject.toml']:
            shutil.copy2(source/name, actor/name)
        shutil.copy2(source/'scripts/run_lean_advice_task.py', actor/'run_pilot.py')
        (actor/'.vendor').symlink_to((source/'.vendor').resolve(), target_is_directory=True)
        p = json.loads((old/'protocol.json').read_text())
        p.update(task_ids=tasks, candidate_profile=profile, expected_trajectories=2,
            purpose='Four-trajectory development comparison of structured constraints and Markdown task notes; not an official score',
            selection='Previously inspected 054 sequencing failure and 005 recovery-procedure failure. Neither is currently excluded. Not unseen holdouts.',
            causal_limit='Compares two memory/interface packages, not note syntax alone. Historical prior 054 controls are contextual evidence, not a concurrent control arm.',
            runtime='Native advisory admission; '+ ('compact persistent constraints plus advisory outcomes' if arm=='constraints' else 'Markdown-only model-authored notes; structured plan/fact/comparison tools absent'),
            stopping_policy='No whole-task retries. Stop for review on infrastructure failure; preserve every attempt.',
            qualification='Local full-suite coverage plus affected rechecks, Linux focused regressions and frozen-manifest verification before launch',
            treatment=['Actor-authored cross-outcome ordering; no extra model reviewer',
                'Current task memory injected after compaction on each model decision',
                'Initial note required for known consequential APIs; changed notes must enter next context before action',
                'No task-specific ordering answers or oracle policy bundles',
                'Retrospective successful evidence allowed for newly recorded outcomes' if arm=='constraints' else 'One editable task-notes.md; no automatic receipts or structured working-memory tools'])
        args=p['cli_arguments'];args[args.index('--candidate-profile')+1]=profile
        # Correct inherited documentation to match existing source constants;
        # actual retrieval budgets are unchanged in this experiment.
        p['limits'].update(searches_per_customer=36, searches_per_conversation=576,
                           planning_stalls='inactive in advisory mode',plan_attempts_without_observation='inactive in advisory mode',task_memory_bytes=16000)
        p['navigation_limits'].update(discovery=[72,576],read=[96,768],check=[48,384] if arm=='constraints' else 'not exposed',exact_source_recovery=[6,24])
        p['metadata_correction']='Previous protocol prose retained older search/navigation limits; these fields now match the inherited actor constants. This does not increase live limits.'
        save(actor/'protocol.json',p)
        save(actor/'source-manifest.json',{'files':{str(f.relative_to(actor)):digest(f) for f in sorted(actor.rglob('*')) if f.is_file() and '.vendor' not in f.parts}})
    python=str(source.parent/'cayu-tau3-latest-20260923/artifacts/paired-comparison-20260923/candidate-venv/bin/python')
    save(root/'comparison.json',dict(arms=['constraints','notebook'],seeds=[300],tasks=tasks,
        schedule=[{'task':t,'seed':300} for t in tasks],python={a:python for a in ['constraints','notebook']},
        api_key_file='/opt/cayu-tau3-credentials/openai_key.txt',concurrency=2,whole_task_retries=0,
        acceptance='Report all four attempts, upstream score, unscored errors, completed actions, notes/constraints actually written and retained, model tokens, time and note-edit cost. No automatic official promotion.'))
    shutil.copy2(source/'scripts/run_matched_comparison.py',root/'run_comparison.py')
    print(root)


if __name__=='__main__':
    main()
