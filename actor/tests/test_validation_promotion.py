import importlib.util
from pathlib import Path
import pytest

@pytest.fixture
def decide(monkeypatch):
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    spec=importlib.util.spec_from_file_location('validation_chain','scripts/run_validation_then_four.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.promotion_decision

def data(passes):
    return {'tasks':[{'id':str(i)} for i in range(97)],'simulations':[{'task_id':str(i),'reward_info':{'reward':int(i<passes)}} for i in range(97)]}

@pytest.mark.parametrize('passes,expected',[(48,False),(49,True),(97,True)])
def test_full_denominator_threshold(decide,passes,expected):
    assert decide(data(passes))[0] is expected

@pytest.mark.parametrize('kind',['unscored','missing','duplicate','missing_catalog'])
def test_incomplete_or_unscored_never_promotes(decide,kind):
    d=data(97)
    if kind=='unscored': d['simulations'][0]['reward_info']=None
    elif kind=='missing': d['simulations'].pop()
    elif kind=='duplicate': d['simulations'][0]['task_id']='1'
    else:d['tasks'].pop()
    assert not decide(d)[0]
