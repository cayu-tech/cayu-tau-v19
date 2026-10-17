"""Run the registered validation, then promote unchanged actor only on a clean pass."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_full_benchmark import verify, write_json


def promotion_decision(results):
    tasks = [t['id'] for t in results.get('tasks', [])]
    sims = results.get('simulations', [])
    if len(tasks) != 97 or len(set(tasks)) != 97:
        return False, 'incomplete_task_catalog'
    ids = [s['task_id'] for s in sims]
    if len(ids) != 97 or set(ids) != set(tasks):
        return False, 'incomplete_or_duplicate_trajectories'
    rewards = [(s.get('reward_info') or {}).get('reward') for s in sims]
    if any(r not in (0, 1) for r in rewards):
        return False, 'unscored_results_require_review'
    passes = sum(r == 1 for r in rewards)
    return passes >= 49, f'{passes}/97'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--four-trial', type=Path, required=True)
    parser.add_argument('--openai-api-key-file', type=Path, required=True)
    args = parser.parse_args()
    for root in [args.validation, args.four_trial]:
        verify(root)
    manifests = [json.loads((p/'source-manifest.json').read_text())['files'] for p in [args.validation,args.four_trial]]
    assert {k:v for k,v in manifests[0].items() if k.startswith('src/')} == {k:v for k,v in manifests[1].items() if k.startswith('src/')}
    state = args.validation.parent/'pipeline-status.json'
    def run(root):
        subprocess.run([sys.executable,str(root/'scripts/run_official_campaign.py'),'supervise','--root',str(root),'--openai-api-key-file',str(args.openai_api_key_file)],cwd=root,check=True)
    write_json(state,{'stage':'validation','validation':str(args.validation),'four_trial':str(args.four_trial)})
    run(args.validation)
    checkpoint = args.validation/'official/results.json'
    if not checkpoint.exists():
        write_json(state,{'stage':'needs_review','reason':'validation_has_no_results'})
        return
    approved, reason = promotion_decision(json.loads(checkpoint.read_text()))
    if not approved:
        write_json(state,{'stage':'needs_review','reason':reason,'four_trial_started':False})
        return
    verify(args.validation)
    write_json(state,{'stage':'four_trial','validation_result':reason,'actor_unchanged':True})
    run(args.four_trial)
    status=json.loads((args.four_trial/'campaign-status.json').read_text())
    write_json(state,{'stage':'four_trial_'+status['status'],'validation_result':reason,'four_trial_status':status})


if __name__=='__main__':
    main()
