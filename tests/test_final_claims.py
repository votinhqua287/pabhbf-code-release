"""Independent arithmetic/provenance checks on the completed confirmatory results."""
import json
import re
import numpy as np
import pytest
from scipy.stats import t
from pabhbf.utils.config import ROOT

OUT=ROOT/'results/revision_final'

def results():
    if not (OUT/'test_complete.json').exists():
        pytest.skip('Confirmatory experiment has not completed; no final claim is certified')
    return json.loads((OUT/'final_summary.json').read_text())

def test_final_leakage_uses_every_attack_and_shorter_horizons():
    summary=results()
    for tag,Ts in summary['statistics'].items():
        last={s:0. for s in (1,2,3)}
        for T in sorted(map(int,Ts)):
            for s in (1,2,3):
                r=json.loads((OUT/'test'/f'{tag}_s{s}_T{T}.json').read_text())
                assert r['leakage']==max(a['hits']/a['n'] for a in r['attackers'].values())
                assert r['leakage']>=last[s]
                last[s]=r['leakage']

def test_generated_number_macros_match_sources_and_student_t_intervals():
    summary=results()
    macros=dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]+)\}',(OUT/'generated/numbers_revision.tex').read_text()))
    fields={'Leak':None,'Rmse':None,'Ratio':'ratio','Wsr':'wsr'}
    for key,c in summary['claims'].items():
        metric=c['metric'];values=[]
        # one registry per seed: the frozen 20,000-drop file (without the seed-1 5,000-drop rules) united with the
        # post-hoc 80,000-drop file of the same seed and window, when the learned representation has one
        by_seed={}
        for f in c['files']:
            seed=int(re.search(r'_s(\d)_T\d',f).group(1))
            r=json.loads((ROOT/f).read_text())
            by_seed.setdefault(seed,{}).update({k:a for k,a in r['attackers'].items() if not k.endswith('_5000drops')})
            if metric in ('Ratio','Wsr') and 'test80k' not in f:by_seed[seed]['__utility__']=r['utility']
        for seed in sorted(by_seed):
            reg={k:a for k,a in by_seed[seed].items() if k!='__utility__'}
            if metric=='Leak':values.append(max(a['hit_prob'] for a in reg.values()))
            elif metric=='Rmse':values.append(min(a['rmse_m'] for a in reg.values()))
            else:values.append(by_seed[seed]['__utility__'][fields[metric]])
        assert len(values)==3
        digits=2 if metric in ('Rmse','Wsr') else 3
        assert macros[key]==f'{np.mean(values):.{digits}f}'
        assert macros[key+'CI']==f'{t.ppf(.975,2)*np.std(values,ddof=1)/np.sqrt(3):.{digits}f}'
    for name,T,metric in [('SmallTemporalGain','8','Leak'),('SmallRateCost','1','Ratio')]:
        base=summary['statistics']['dz4_lam0'][T][metric]['values']
        adv=summary['statistics']['dz4_lam0.03'][T][metric]['values']
        delta=np.asarray(base)-np.asarray(adv)
        assert macros[name+'Pct']==f'{100*delta.mean():.1f}'
        assert macros[name+'CIPct']==f'{100*t.ppf(.975,2)*delta.std(ddof=1)/np.sqrt(3):.1f}'

def test_final_tools_return_only_constant_ack():
    results()
    for p in (OUT/'test').glob('tools_*.json'):
        for mode in json.loads(p.read_text()).values():
            assert mode['csi_dependent_return_bits']==0
            assert mode['ack_bits_per_batch']==24
            assert mode['timing_repeats']==3

def test_registry_maximum_is_uniform_over_seeds():
    """The 5,000-drop attackers of seed 1 are excluded from the registry maximum and never attain it."""
    summary=results()
    for tag in ('lam0.01','lam0','dz4_lam0','dz4_lam0.03'):
        for T in (1,4,8):
            r=json.loads((OUT/'test'/f'{tag}_s1_T{T}.json').read_text())
            small={k:a for k,a in r['attackers'].items() if k.endswith('_5000drops')}
            large={k:a for k,a in r['attackers'].items() if not k.endswith('_5000drops')}
            assert small and max(a['hit_prob'] for a in small.values())<max(a['hit_prob'] for a in large.values())
            big=OUT/'test80k'/f'{tag}_s1_T{T}.json'
            if big.exists():large.update(json.loads(big.read_text())['attackers'])
            assert summary['statistics'][tag][str(T)]['Leak']['values'][0]==max(a['hit_prob'] for a in large.values())


def test_sixteen_bit_task_only_encoder_dominates_the_selected_point():
    """Section VIII-C: same rate retention (to 0.1 pp), lower leakage at every window, larger RMSE."""
    s=results()['statistics']
    small,full=s['dz4_lam0'],s['lam0.01']
    assert round(100*small['1']['Ratio']['mean'],1)>=round(100*full['1']['Ratio']['mean'],1)
    assert all(small[T]['Leak']['mean']+small[T]['Leak']['ci95']<full[T]['Leak']['mean']-full[T]['Leak']['ci95'] for T in ('1','4','8'))
    assert small['1']['Rmse']['mean']>full['1']['Rmse']['mean']
    text=(ROOT/'manuscript/main.tex').read_text(encoding='utf-8') if (ROOT/'manuscript/main.tex').exists() else ''
    assert 'is not Pareto-efficient among the evaluated encoders' in text
