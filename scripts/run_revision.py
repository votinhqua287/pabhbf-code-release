"""Run validation jobs with bounded CPU concurrency; never opens the final test."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_final'
C=yaml.safe_load((ROOT/'configs/revision_final.yaml').read_text())
ENV=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3')

def run(args):
    name='_'.join(args).replace('--','')
    with (OUT/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable,'-u','-m','pabhbf.revision',*args],cwd=ROOT,env=ENV,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    print('completed',name,flush=True)

def batch(jobs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
            future.result()

if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='selection':
        run(['audit'])
        batch([['fit','--tag',t,'--seed','1','--stage','selection'] for t in C['candidates']])
        run(['select'])
    elif stage=='confirmatory':
        sel=json.loads((OUT/'selection.json').read_text())['selected']
        tags=list(dict.fromkeys(sel+['lam0','dz4_lam0','dz4_lam0.03']+C['baselines']))
        # Keep each model's time horizons sequential to avoid racing cache writes.
        def model_jobs(pair):
            tag,seed=pair
            for T in C['intervals'] if tag.startswith(('lam','dz')) else [1]:
                run(['fit','--tag',tag,'--seed',str(seed),'--T',str(T),'--stage','confirmatory'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(model_jobs,[(t,s) for t in tags for s in C['seeds']]))
    else:
        raise ValueError(stage)
