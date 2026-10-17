#!/usr/bin/env python3
"""Portable packaging wrapper. Frozen V19r3 acting/evaluation code is unchanged."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parent
PINS = {'cayu': 'd8e530acafa3cb7ab9596340182cffc8d9467d5d',
        'tau2': 'fc0055dc4e0a316c3f83133267fbd6faaa770992'}
ORIGINAL_MANIFEST = '93300d01e05e9721de27e650377a270467665aecdd3792d361646d694b92714d'
# 'subscription' is the submitted setup; 'api' runs the same actor on an OpenAI API key.
ACTOR_PROVIDERS = ('subscription', 'api')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(command, **kwargs):
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def git(repo, *args):
    return subprocess.check_output(['git', '--git-dir', str(repo), *args], text=True).strip()


def verify():
    manifest = ROOT / 'PACKAGE_SHA256.json'
    if not manifest.exists():
        raise ValueError('Package checksum manifest is missing.')
    hashes = read(manifest)
    for name, expected in hashes.items():
        p = ROOT / name
        if not p.resolve().is_relative_to(ROOT) or not p.is_file() or sha(p) != expected:
            raise ValueError(f'Package integrity failure: {name}')
    source = ROOT / 'actor/source-manifest.json'
    if sha(source) != ORIGINAL_MANIFEST:
        raise ValueError('Original actor manifest changed.')
    files = read(source)['files']
    for name, expected in files.items():
        if sha(ROOT / 'actor' / name) != expected:
            raise ValueError(f'Frozen actor changed: {name}')
    for name, commit in PINS.items():
        repo = ROOT / 'dependencies' / (name + '.git')
        if git(repo, 'rev-parse', 'HEAD') != commit:
            raise ValueError(f'{name} commit mismatch')
        run(['git', '--git-dir', repo, 'fsck', '--full', '--no-reflogs'], capture_output=True)
    raw = gzip.decompress((ROOT / 'results/trajectories.json.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest() != read(ROOT / 'provenance/public-trajectory-origin.json')['uncompressed_sha256']:
        raise ValueError('Published trajectory bytes changed.')
    return {'passed': True, 'package_files_verified': len(hashes),
            'frozen_actor_files_verified': len(files), 'commits': PINS,
            'network_or_model_calls': False}


def work_python(work):
    return work / '.venv/bin/python'


def verify_work(work):
    receipt = read(work / 'relocation.json')
    if receipt['original_manifest_sha256'] != ORIGINAL_MANIFEST:
        raise ValueError('Workspace does not belong to this frozen agent.')
    original = read(ROOT / 'actor/source-manifest.json')['files']
    manifest = read(work / 'actor/source-manifest.json')['files']
    if original.keys() != manifest.keys():
        raise ValueError('Unexpected added/removed source files in execution manifest.')
    changed = {name for name in original if original[name] != manifest[name]}
    if changed != {'protocol.json'}:
        raise ValueError(f'Unexpected relocated source changes: {sorted(changed)}')
    for name, expected in manifest.items():
        if sha(work / 'actor' / name) != expected:
            raise ValueError(f'Execution source changed: {name}')
    if sha(work / 'actor/source-manifest.json') != receipt['execution_manifest_sha256']:
        raise ValueError('Execution manifest changed.')
    old = read(ROOT/'actor/protocol.json')
    current = read(work/'actor/protocol.json')
    provider = receipt.get('actor_provider', 'subscription')
    if provider not in ACTOR_PROVIDERS:
        raise ValueError(f'Unknown actor provider in relocation receipt: {provider}')
    old['cli_arguments'] = expected_cli_arguments(old['cli_arguments'], provider)
    for key in set(old) | set(current):
        if key not in {'tau_repository','corpus_embedding_cache','framework_repository'} and old.get(key) != current.get(key):
            raise ValueError(f'Non-path protocol change: {key}')
    return receipt


def expected_cli_arguments(arguments, provider):
    """The only permitted CLI change: the actor provider, recorded in relocation.json."""
    arguments = list(arguments)
    arguments[arguments.index('--provider') + 1] = provider
    return arguments


def setup(work, python, actor_provider='subscription'):
    verify()
    if ROOT == work or work.is_relative_to(ROOT):
        raise ValueError('Choose a new workspace outside the immutable package directory.')
    if work.exists():
        raise ValueError('Workspace already exists; choose a new empty path.')
    if not shutil.which('uv'):
        raise ValueError('Install uv first (python3 -m pip install uv).')
    work.mkdir(parents=True, mode=0o700)
    shutil.copytree(ROOT / 'actor', work / 'actor')
    # Local shallow Git snapshots preserve the exact commits without including history.
    run(['git', 'clone', '--quiet', '--no-local', (ROOT/'dependencies/tau2.git').as_uri(), work/'tau2'])
    run(['git', '-C', work/'tau2', 'checkout', '--quiet', '--detach', PINS['tau2']])
    (work/'actor/.vendor').mkdir(exist_ok=True)
    (work/'actor/.vendor/tau2').symlink_to(work/'tau2', target_is_directory=True)
    framework = (ROOT/'dependencies/cayu.git').as_uri()
    project = tomllib.loads((work/'actor/pyproject.toml').read_text())['project']
    old = 'cayu @ git+file:///Users/talgatt/Desktop/Projects/cayu-tau-performance-20260927@' + PINS['cayu']
    if project['dependencies'].count(old) != 1:
        raise ValueError('Unexpected frozen dependency declaration.')
    protocol = read(work/'actor/protocol.json')
    changes = {'tau_repository': str(work/'tau2'),
               'corpus_embedding_cache': str(ROOT/'embedding-cache'),
               'framework_repository': framework}
    original_protocol = read(ROOT/'actor/protocol.json')
    protocol.update(changes)
    protocol['cli_arguments'] = expected_cli_arguments(protocol['cli_arguments'], actor_provider)
    cli_changes = {} if actor_provider == 'subscription' else {
        '--provider': {'from': 'subscription', 'to': actor_provider}}
    save(work/'actor/protocol.json', protocol)
    names = read(ROOT/'actor/source-manifest.json')['files']
    save(work/'actor/source-manifest.json', {'files': {n: sha(work/'actor'/n) for n in names}})
    save(work/'relocation.json', {
        'created_unix': time.time(), 'original_manifest_sha256': ORIGINAL_MANIFEST,
        'execution_manifest_sha256': sha(work/'actor/source-manifest.json'),
        'changed_files': ['protocol.json'],
        'path_changes': {k: {'from': original_protocol[k], 'to': v} for k,v in changes.items()},
        'actor_code_changed': False, 'actor_provider': actor_provider,
        'cli_arguments_changed': bool(cli_changes), 'cli_argument_changes': cli_changes,
        'package_location': str(ROOT), 'environment_note': 'Fresh dependency resolution, not the historical VM lock.'})
    env = {**os.environ, 'UV_CACHE_DIR': str(work/'.uv-cache')}
    run(['uv', 'venv', '--python', python, work/'.venv'], env=env)
    constraints = ROOT/'constraints-verified.txt'
    constraint_args = ['--constraint',str(constraints)] if constraints.exists() else []
    # Install the exact dependency explicitly; retain the archived pyproject bytes
    # (including its historical URL) and avoid resolving that machine-specific URL.
    requirements = [r for r in project['dependencies'] if r != old]
    requirements += project.get('optional-dependencies',{}).get('dev',[])
    run(['uv', 'pip', 'install', '--python', work_python(work), *constraint_args,
         '-e', str(work/'tau2')+'[knowledge]', 'cayu @ git+'+framework+'@'+PINS['cayu'],
         *requirements], env=env)
    run(['uv','pip','install','--python',work_python(work),'--no-deps','-e',work/'actor'],env=env)
    installed = subprocess.check_output(['uv','pip','freeze','--python',str(work_python(work))], env=env, text=True)
    (work/'installed-packages.txt').write_text(installed)
    check(work)
    print(json.dumps({'status':'setup_complete','work_dir':str(work),'model_calls':0},indent=2))


def offline_env():
    env = dict(os.environ)
    for name in list(env):
        if any(word in name for word in ('API_KEY', 'ACCESS_TOKEN', 'REFRESH_TOKEN')) or name.startswith('CAYU_TAU_CODEX'):
            env.pop(name)
    env['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    return env


def check(work):
    verify(); verify_work(work)
    env = offline_env()
    run([work_python(work), '-m', 'cayu_tau3_banking.provenance', '--tau2-repository', work/'tau2'], env=env)
    run([work_python(work), '-m', 'cayu_tau3_banking.cli', '--help'], env=env, stdout=subprocess.DEVNULL)
    print('Offline source provenance and CLI imports passed. No credentials or model calls used.')


def score(path):
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    data = json.loads(raw); rows = data['simulations']; by_task = defaultdict(list)
    keys = [(r['task_id'],r['trial']) for r in rows]
    if len(keys) != len(set(keys)): raise ValueError('Duplicate task/trial slots.')
    counts = Counter((r.get('reward_info') or {}).get('reward') for r in rows)
    for r in rows: by_task[r['task_id']].append(r)
    complete = len(by_task)==97 and len(rows)==388 and all({r['trial'] for r in rs}=={0,1,2,3} for rs in by_task.values())
    binary = all(type((r.get('reward_info') or {}).get('reward')) in (int,float)
                 and r['reward_info']['reward'] in (0,1) for r in rows)
    result = {'tasks':len(by_task),'trajectories':len(rows),'passed':counts[1],'scored_failures':counts[0],
              'unscored':counts[None], 'complete_four_trial_matrix':complete,'pass_hat_percent':None}
    if complete and binary:
        result['pass_hat_percent'] = {str(k): 100*sum(math.comb(sum(r['reward_info']['reward']==1 for r in rs),k)/math.comb(4,k)
             for rs in by_task.values())/97 for k in range(1,5)}
        result['chronological_trials'] = [sum(r['trial']==i and r['reward_info']['reward']==1 for r in rows) for i in range(4)]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify','score','setup','check','offline-tests','live-check','run','status','continue'])
    parser.add_argument('--work-dir',type=Path)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--results',type=Path,default=ROOT/'results/trajectories.json.gz')
    parser.add_argument('--actor-provider',choices=ACTOR_PROVIDERS,default='subscription',
                        help='setup only: api runs the actor on the OpenAI API key instead of a Codex subscription')
    parser.add_argument('--auth-file',type=Path)
    parser.add_argument('--openai-api-key-file',type=Path)
    args=parser.parse_args()
    if args.command=='verify': print(json.dumps(verify(),indent=2));return
    if args.command=='score': print(json.dumps(score(args.results),indent=2));return
    if args.work_dir is None: parser.error('--work-dir is required')
    work=args.work_dir.resolve()
    if args.command=='setup':setup(work,args.python,args.actor_provider);return
    verify();receipt=verify_work(work)
    actor=work/'actor';py=work_python(work)
    if args.command=='check':check(work);return
    if args.command=='offline-tests':
        run([py,actor/'scripts/qualify_vm.py'],cwd=actor,env=offline_env());return
    if args.command=='status':
        for name in ['campaign-status.json','trial-1-boundary.json']:
            path=actor/name
            if path.exists(): print(name+'\n'+json.dumps(read(path),indent=2))
        return
    if args.command=='continue':
        boundary=actor/'trial-1-boundary.json'
        if not boundary.exists():raise ValueError('The first-trial boundary has not been reached.')
        save(actor/'continue-after-trial-1.json',{'continue_after_trial':1,
             'source_manifest_sha256':sha(actor/'source-manifest.json'),'created_unix':time.time()})
        print('Continuation authorized for this frozen execution manifest. All scored outcomes retained.');return
    api=receipt.get('actor_provider','subscription')=='api'
    if args.openai_api_key_file is None or (args.auth_file is None and not api):
        parser.error('--openai-api-key-file is required for live commands, and --auth-file unless '
                     'the workspace was set up with --actor-provider api')
    key=args.openai_api_key_file.resolve(strict=True)
    credentials=[key] if api else [key,args.auth_file.resolve(strict=True)]
    if any(path.is_relative_to(ROOT) for path in credentials):
        raise ValueError('Keep credentials outside the shareable package.')
    env={**os.environ,'LITELLM_LOCAL_MODEL_COST_MAP':'True'}
    env.pop('CAYU_TAU_CODEX_AUTH_FILE',None)
    if not api: env['CAYU_TAU_CODEX_AUTH_FILE']=str(credentials[1])
    # The frozen supervisor's connectivity probe always uses the subscription provider;
    # api_runner.py swaps only that probe and otherwise runs the unchanged supervisor.
    runner=ROOT/'api_runner.py' if api else actor/'scripts/run_official_campaign.py'
    if args.command=='live-check':
        run([py,runner,'probe','--root',actor,'--openai-api-key-file',key],cwd=actor,env=env)
        run([py,actor/'scripts/probe_parallel_recall.py','--openai-api-key-file',key],cwd=actor,env=env)
        return
    run([py,runner,'supervise','--root',actor,'--openai-api-key-file',key],cwd=actor,env=env)


if __name__=='__main__':main()
