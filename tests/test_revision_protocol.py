import json
import numpy as np
import pytest
import torch
from pabhbf import revision as r
from pabhbf.agents.planner import ProtectedInterface

def test_temporal_model_is_permutation_invariant():
    torch.manual_seed(3)
    model=r.TemporalSet(4).eval()
    x=torch.randn(11,8,4)
    with torch.no_grad():
        torch.testing.assert_close(model(x),model(x[:,[7,1,5,3,6,4,0,2]]))

def test_wilson_boundary_and_simultaneous_upper_bound():
    low,high=r.wilson(0,100)
    assert abs(low)<1e-12 and 0.03<high<0.04
    assert r.wilson(20,100,.05/40)[1]>r.wilson(20,100)[1]

def test_drop_bootstrap_preserves_within_drop_dependence():
    # 100 independent drops, each with 16 perfectly correlated users.
    success=np.repeat([0,1]*50,16)
    ci=r.drop_bootstrap(success)
    naive=r.wilson(800,1600)
    assert ci[1]-ci[0]>2*(naive[1]-naive[0])
    assert r.drop_bootstrap(np.ones(160))==[1.,1.]

def test_duplicate_geometry_fails_even_with_different_ids(monkeypatch,tmp_path):
    monkeypatch.setattr(r,'OUT',tmp_path)
    monkeypatch.setattr(r,'config',lambda:dict(policy_train='train',policy_val='val'))
    monkeypatch.setattr(r,'load_split',lambda name,keys:dict(xy=np.array([[[1.,2.]]])))
    for role in ['attack_train','attack_val']:
        d=tmp_path/'data'/role;d.mkdir(parents=True)
        np.savez(d/'a.npz',xy=np.array([[[3.,4.]]]))
    with pytest.raises(AssertionError,match='geometry overlap'):
        r.check_splits()

def test_final_generation_requires_freeze(monkeypatch,tmp_path):
    monkeypatch.setattr(r,'OUT',tmp_path)
    with pytest.raises(RuntimeError,match='Freeze'):
        r.prepare('final_test')

def test_final_test_rejects_changed_frozen_input(monkeypatch,tmp_path):
    monkeypatch.setattr(r,'OUT',tmp_path)
    monkeypatch.setattr(r,'ROOT',tmp_path)
    (tmp_path/'model').write_text('modified')
    (tmp_path/'frozen.json').write_text(json.dumps(dict(files={'model':'wrong'})))
    with pytest.raises(RuntimeError,match='frozen input changed'):
        r.final_test()
    assert not (tmp_path/'test_access.json').exists()

def test_ack_is_constant_and_execution_state_remains_in_bs():
    class Planner:
        def run(self,logits,box,rows,streams):
            return box
    local=[]
    for private in [np.zeros(10),np.ones(10)*72]:
        api=ProtectedInterface(private,local.append)
        assert api.submit(Planner(),None,[0],1)==b'ACK'
    assert len(local)==2 and np.all(local[1]==72)

def test_trusted_search_executes_feasible_precoder_without_score_return():
    from pabhbf.agents.planner import TrustedToolbox,SearchPlanner
    from pabhbf.data.dataset import generate_split
    from pabhbf.utils.config import load_config,system_params
    from pabhbf.phy.codebook import polar_codebook
    p=system_params(load_config('smoke.yaml'))
    d=generate_split(p,4,991,10.,teacher=False)
    rng=np.random.default_rng(88)
    L=(rng.normal(size=(4,p.K)),rng.normal(size=(4,p.K,p.G)),rng.normal(size=(4,p.K)))
    box=TrustedToolbox(d['Hhat'][:,0],d['w'],d['snr_db'],polar_codebook(p))
    local=[]
    ack=ProtectedInterface(box,local.append).submit(SearchPlanner(3),L,np.arange(4),p.Ns)
    s,F1,F2=local.pop()
    assert ack==b'ACK' and box.exposure_bits==0
    assert not hasattr(box,'evaluate_candidate')
    assert box.calls['internal_candidate_score']==12
    assert np.all(np.isfinite(F1@F2))
    assert np.allclose(np.linalg.norm(F1@F2,axis=(1,2))**2,1.,atol=1e-6)
    assert all(len(set(row))==p.Ns for row in s)
