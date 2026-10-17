"""Offline package checks, including pinned upstream score and embedding-cache parity."""
import argparse
import asyncio
from copy import deepcopy
import gzip
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import reproduce

os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',type=Path,required=True)
    args=parser.parse_args();work=args.work_dir.resolve();root=Path(__file__).resolve().parent
    reproduce.verify();receipt=reproduce.verify_work(work)
    assert receipt['changed_files']==['protocol.json']
    assert (work/'actor/.vendor/tau2').resolve()==(work/'tau2').resolve()
    original=reproduce.read(root/'actor/protocol.json')
    relocated=reproduce.read(work/'actor/protocol.json')
    provider=receipt.get('actor_provider','subscription')
    assert relocated['cli_arguments']==reproduce.expected_cli_arguments(original['cli_arguments'],provider)
    assert (original['cli_arguments']==relocated['cli_arguments'])==(provider=='subscription')
    assert reproduce.sha(root/'actor/pyproject.toml')==reproduce.sha(work/'actor/pyproject.toml')

    data=json.loads(gzip.decompress((root/'results/trajectories.json.gz').read_bytes()))
    ours=reproduce.score(root/'results/trajectories.json.gz')
    assert (ours['passed'],ours['scored_failures'],ours['unscored'])==(193,195,0)
    assert ours['chronological_trials']==[49,52,47,45]
    from tau2.data_model.simulation import Results
    from tau2.metrics.agent_metrics import compute_metrics
    official=compute_metrics(Results.model_validate(data))
    for k,value in official.pass_hat_ks.items():
        assert math.isclose(100*value,ours['pass_hat_percent'][str(k)],abs_tol=1e-10)
    with tempfile.TemporaryDirectory(prefix='tau-repro-check-') as td:
        path=Path(td)/'results.json'
        incomplete=deepcopy(data);incomplete['simulations'].pop();path.write_text(json.dumps(incomplete))
        assert reproduce.score(path)['pass_hat_percent'] is None
        duplicate=deepcopy(data);duplicate['simulations'].append(duplicate['simulations'][0]);path.write_text(json.dumps(duplicate))
        try:reproduce.score(path)
        except ValueError:pass
        else:raise AssertionError('Duplicate slot accepted')
        unscored=deepcopy(data);unscored['simulations'][0]['reward_info']=None;path.write_text(json.dumps(unscored))
        assert reproduce.score(path)['pass_hat_percent'] is None

    sys.path.insert(0,str(work/'actor/scripts'))
    from evaluate_section_embeddings import CachedEmbeddings
    from cayu_tau3_banking.cli import build_memory_corpus
    class NoNetworkEmbeddings:
        async def embed_texts(self,request):
            raise AssertionError('Cache miss would require an embedding API call')
    cache=CachedEmbeddings(NoNetworkEmbeddings())
    cache.restore(root/'embedding-cache',reproduce.read(root/'embedding-cache/manifest.json')['corpus_sha256'])
    corpus=asyncio.run(build_memory_corpus(work/'tau2/data/tau2/domains/banking_knowledge/documents',
        retrieval_mode='hybrid',embedding_model='text-embedding-3-large',embedding_dimensions=3072,
        embedding_provider=cache))
    assert cache.calls==0
    assert corpus.corpus_sha256==reproduce.read(root/'embedding-cache/manifest.json')['corpus_sha256']
    print(json.dumps({'passed':True,'changed_source_files':['protocol.json'],
        'unchanged_actor_code':True,'actor_provider':provider,'upstream_score_matches':True,'duplicate_incomplete_unscored_checks':'passed',
        'cached_corpus_build':'passed','cached_texts':len(cache.vectors),'embedding_api_calls':cache.calls,
        'pass_hat_percent':ours['pass_hat_percent']},indent=2))


if __name__=='__main__':main()
