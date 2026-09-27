"""Generate manuscript claims only from the completed, frozen confirmatory test."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_final'
MS=ROOT/'manuscript' if (ROOT/'manuscript').exists() else ROOT/'outputs'  # the code package has no LaTeX tree
NAMES={
 'lam0.01':('Full',r'PAB-HBF, $B=128$, $\mu=0.01$'),
 'lam0':('Task',r'Task-only, $B=128$'),
 'dz4_lam0':('Small',r'Task-only, $B=16$'),
 'dz4_lam0.03':('SmallAdv',r'PAB-HBF, $B=16$, $\mu=0.03$'),
 'raw_csi':('Raw','B1 uncompressed CSI input'), 'pca':('Pca','B2 PCA'),
 'random_projection':('Random','B3 random projection'),
 'scalar_quant':('Scalar','B4 scalar quantization'),
 'autoencoder':('Ae','B7 autoencoder'), 'angle_domain':('Angle','B8 angle domain')}

GEN=OUT/'generated'  # generated TeX pieces; the copies inside main.tex are refreshed by splice()

def splice(name,body):
    """Replace the block between the GENERATED markers of main.tex, so that main.tex stays one self-contained file."""
    import re
    main=MS/'main.tex'
    if not main.exists():return  # without the manuscript the generated file in results/ is the only output
    text=main.read_text(encoding='utf8')
    pattern=re.compile(r'(% >>> GENERATED '+name+r' \(scripts/final_artifacts\.py\) >>>\n).*?(\n% <<< GENERATED '+name+r' <<<)',re.S)
    assert len(pattern.findall(text))==1,name
    main.write_text(pattern.sub(lambda m:m.group(1)+body.rstrip('\n')+m.group(2),text),encoding='utf8')

def read(path):return json.loads(path.read_text())
def stat(values):
    x=np.asarray(values,dtype=float)
    assert len(x)==3 and np.isfinite(x).all()
    return dict(mean=float(x.mean()),ci95=float(t.ppf(.975,2)*x.std(ddof=1)/np.sqrt(3)),values=x.tolist())
def pm(s,d=3,scale=1):return f"{scale*s['mean']:.{d}f}\\,$\\pm$\\,{scale*s['ci95']:.{d}f}"
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    complete=read(OUT/'test_complete.json')
    assert complete['frozen_sha']==digest(OUT/'frozen.json')
    frozen=read(OUT/'frozen.json')
    assert read(OUT/'selection.json')['selected']==['lam0.01','dz4_lam0']
    allrows={};sources={};macros={};registry={}
    for tag in frozen['tags']:
        allrows[tag]={}
        for T in ([1,4,8] if tag.startswith(('lam','dz')) else [1]):
            paths=[OUT/'test'/f'{tag}_s{s}_T{T}.json' for s in (1,2,3)]
            rows=[read(p) for p in paths]
            big=[OUT/'test80k'/f'{tag}_s{s}_T{T}.json' for s in (1,2,3)]
            scaled=all(p.exists() for p in big)
            if scaled: paths=paths+big
            def uniform(r,i):
                # uniform over seeds: the 5,000-drop attackers of seed 1 serve the data-scaling comparison only.
                # The post-hoc 80,000-drop attackers of the learned representations join the registry maximum.
                d={k:a for k,a in r['attackers'].items() if not k.endswith('_5000drops')}
                if scaled: d.update(read(big[i])['attackers'])
                return d
            for i,row in enumerate(rows):
                assert row['leakage']==max(a['hit_prob'] for a in row['attackers'].values())
                assert all(a['n']==32000 for a in row['attackers'].values())
                merged=uniform(row,i)
                assert all(a['n']==32000 for a in merged.values())
                # the registry maximum never falls below the frozen 20,000-drop maximum
                assert max(a['hit_prob'] for a in merged.values())>=row['leakage']
                assert min(a['rmse_m'] for a in merged.values())<=min(a['rmse_m'] for a in row['attackers'].values())
            res={'Leak':stat([max(a['hit_prob'] for a in uniform(r,i).values()) for i,r in enumerate(rows)]),
                 'Rmse':stat([min(a['rmse_m'] for a in uniform(r,i).values()) for i,r in enumerate(rows)])}
            if T==1:
                res.update(Wsr=stat([r['utility']['wsr'] for r in rows]),
                           Ratio=stat([r['utility']['ratio'] for r in rows]))
            allrows[tag][T]=res
            registry[f'{tag}_T{T}']=[dict(seed=r['seed'],attack_sizes=sorted({a['attack_train_size'] for a in uniform(r,i).values()}),
                                         winner=max(uniform(r,i),key=lambda a:uniform(r,i)[a]['hit_prob']),
                                         attackers=uniform(r,i)) for i,r in enumerate(rows)]
            prefix=NAMES[tag][0]+('' if T==1 else f'T{ {4:"Four",8:"Eight"}[T]}')
            for metric,s in res.items():
                key=prefix+metric
                places=2 if metric in ('Wsr','Rmse') else 3
                macros[key]=f"{s['mean']:.{places}f}"
                macros[key+'CI']=f"{s['ci95']:.{places}f}"
                if metric in ('Ratio','Leak'):macros[key+'Pct']=f"{100*s['mean']:.1f}"
                sources[key]=dict(files=[str(p.relative_to(ROOT)) for p in paths],metric=metric,statistic=s)
    paired={
        'SmallTemporalGain':stat(np.array(allrows['dz4_lam0'][8]['Leak']['values'])-np.array(allrows['dz4_lam0.03'][8]['Leak']['values'])),
        'SmallRateCost':stat(np.array(allrows['dz4_lam0'][1]['Ratio']['values'])-np.array(allrows['dz4_lam0.03'][1]['Ratio']['values']))}
    for name,s in paired.items():
        macros[name+'Pct']=f"{100*s['mean']:.1f}"
        macros[name+'CIPct']=f"{100*s['ci95']:.1f}"
    teacher=read(OUT/'test/lam0.01_s1_T1.json')['utility']['teacher_wsr']
    # Table order: the four learned representations, then the baselines. The T = 4 and T = 8 leakage joins the main table.
    ORDER=['lam0.01','lam0','dz4_lam0','dz4_lam0.03','raw_csi','pca','random_projection','scalar_quant','autoencoder','angle_domain']
    assert set(ORDER)==set(allrows)
    lines=[r'\begin{tabular}{lrrrrrrr}',r'\toprule',
           r'Scheme & $B$ (bits) & WSR (bit/s/Hz) & Rate ratio & $\widehat\Lambda_{\rho,1}^{\rm tr}$ & $\widehat\Lambda_{\rho,4}^{\rm tr}$ & $\widehat\Lambda_{\rho,8}^{\rm tr}$ & RMSE (m)\\',r'\midrule',
           f'Teacher (local CSI) & 0 & {teacher:.2f} & 1 & -- & -- & -- & --'+r'\\']
    for tag in ORDER:
        rs=allrows[tag];row=rs[1];bits=16384 if tag=='raw_csi' else 16 if tag.startswith('dz4') else 128
        longer=[pm(rs[T]['Leak']) if T in rs else '--' for T in (4,8)]
        lines.append(' & '.join([NAMES[tag][1],str(bits),pm(row['Wsr'],2),pm(row['Ratio']),pm(row['Leak'])]+longer+[pm(row['Rmse'],2)])+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    GEN.mkdir(exist_ok=True)
    (GEN/'tab_main_final.tex').write_text('\n'.join(lines)+'\n');splice('tab_main','\n'.join(lines))
    lines=[r'\begin{tabular}{lccc}',r'\toprule',r'Representation & $T=1$ & $T=4$ & $T=8$\\',r'\midrule']
    temporal=['lam0','lam0.01','dz4_lam0','dz4_lam0.03']
    short=[r'$128$ bits, $\mu=0$',r'$128$ bits, $\mu=0.01$',r'$16$ bits, $\mu=0$',r'$16$ bits, $\mu=0.03$']
    for tag,label in zip(temporal,short):
        lines.append(' & '.join([label]+[pm(allrows[tag][T]['Leak']) for T in (1,4,8)])+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (GEN/'tab_temporal_final.tex').write_text('\n'.join(lines)+'\n')  # kept for the record, not in the manuscript
    # Attack-data scaling: a seed-1 5k/20k comparison, and the post-hoc 80k registry over all three seeds.
    scaling={}
    for tag in temporal:
        row=read(OUT/'test'/f'{tag}_s1_T1.json')
        scaling[tag]={n:max(a['hit_prob'] for a in row['attackers'].values() if a['attack_train_size']==n) for n in (5000,20000)}
        for n,v in scaling[tag].items():macros[NAMES[tag][0]+('FiveK' if n==5000 else 'TwentyK')]=f'{v:.3f}'
    scaling_seeds={}
    for tag in temporal:
        for T in (1,8):
            sizes={}
            for size,folder in ((20000,'test'),(80000,'test80k')):
                files=[OUT/folder/f'{tag}_s{s}_T{T}.json' for s in (1,2,3)]
                if not all(p.exists() for p in files):continue
                sizes[size]=stat([max(a['hit_prob'] for a in read(p)['attackers'].values()
                                      if a['attack_train_size']==size) for p in files])
            if len(sizes)==2:
                gain=stat(np.array(sizes[80000]['values'])-np.array(sizes[20000]['values']))
                sizes['gain']=gain
                key=NAMES[tag][0]+('' if T==1 else 'TEight')
                macros[key+'TwentyKAll']=f"{sizes[20000]['mean']:.3f}"
                macros[key+'EightyKAll']=f"{sizes[80000]['mean']:.3f}"
                macros[key+'EightyKGain']=f"{gain['mean']:.3f}"
            scaling_seeds[f'{tag}_T{T}']=sizes
    # Attack families: every decision rule of the registry for the four learned representations, 20k and 80k.
    RULES={1:[('mlp_xy','Cartesian MLP'),('mlp_rtheta','Polar MLP'),('resmlp_xy','Residual MLP'),('knn','20-NN regressor'),
              ('grid_map','Disc-MAP classifier'),('grid_mean','Mean classifier')],
           8:[('mean_mlp','Mean-input MLP'),('mean_grid_map','Mean-input disc-MAP'),('mean_grid_mean','Mean-input mean'),
              ('deepsets','DeepSets')]}
    families={}
    def rule_mean(tag,T,rule,size):
        folder='test' if size==20000 else 'test80k'
        return float(np.mean([read(OUT/folder/f'{tag}_s{s}_T{T}.json')['attackers'][f'{rule}_{size}drops']['hit_prob'] for s in (1,2,3)]))
    def rule_rmse(tag,T,rule,size):
        folder='test' if size==20000 else 'test80k'
        return float(np.mean([read(OUT/folder/f'{tag}_s{s}_T{T}.json')['attackers'][f'{rule}_{size}drops']['rmse_m'] for s in (1,2,3)]))
    lines=[r'\begin{tabular}{lcccc}',r'\toprule',
           r'Rule, $(B,\mu)$ & $(128,0.01)$ & $(128,0)$ & $(16,0)$ & $(16,0.03)$\\',r'\midrule']
    for T,title in ((1,'Snapshot, $T=1$'),(8,'Repeated releases, $T=8$')):
        if T==8:lines.append(r'\midrule')
        lines.append(r'\multicolumn{5}{l}{\textit{'+title+r'}}\\')
        for rule,label in RULES[T]:
            cells=[]
            for tag in temporal[1:2]+temporal[0:1]+temporal[2:]:  # lam0.01, lam0, dz4_lam0, dz4_lam0.03
                pair=[]
                for size in (20000,80000):
                    v=rule_mean(tag,T,rule,size);families[f'{tag}_T{T}_{rule}_{size}']=v;pair.append(f'{v:.3f}')
                    families[f'{tag}_T{T}_{rule}_{size}_rmse']=rule_rmse(tag,T,rule,size)
                cells.append('/'.join(pair))
            lines.append(' & '.join([label]+cells)+r'\\')
    lines+=[r'\bottomrule',r'\end{tabular}']
    (GEN/'tab_attacks_final.tex').write_text('\n'.join(lines)+'\n');splice('tab_attacks','\n'.join(lines))
    for tag in ('lam0.01','lam0','dz4_lam0','dz4_lam0.03'):
        P=NAMES[tag][0]
        macros[P+'ClassifierOne']=f"{families[f'{tag}_T1_grid_map_80000']:.3f}"
        macros[P+'ClassifierOneTwentyK']=f"{families[f'{tag}_T1_grid_map_20000']:.3f}"
        macros[P+'ResidualOne']=f"{families[f'{tag}_T1_resmlp_xy_80000']:.3f}"
        macros[P+'ResidualOneTwentyK']=f"{families[f'{tag}_T1_resmlp_xy_20000']:.3f}"
        macros[P+'MeanOne']=f"{families[f'{tag}_T1_grid_mean_80000']:.3f}"
        macros[P+'RegressorOne']=f"{max(families[f'{tag}_T1_{r}_80000'] for r in ('mlp_xy','mlp_rtheta','resmlp_xy','knn')):.3f}"
        macros[P+'DeepSetsRmseEight']=f"{families[f'{tag}_T8_deepsets_80000_rmse']:.2f}"
        macros[P+'MapRmseEight']=f"{families[f'{tag}_T8_mean_grid_map_80000_rmse']:.2f}"
        # growth of the strongest rule from one to eight intervals (80,000-drop attackers)
        macros[P+'TemporalFactor']=f"{families[f'{tag}_T8_mean_grid_map_80000']/families[f'{tag}_T1_grid_map_80000']:.1f}"
    # the frozen 20,000-drop attackers reproduce their attacker-validation hit probabilities on the final test
    gaps=[]
    for tag in ('lam0.01','lam0','dz4_lam0','dz4_lam0.03'):
        for s in (1,2,3):
            val=read(OUT/'attacks/confirmatory'/f'{tag}_s{s}_T1'/'validation.json')['leakage']
            test=max(a['hit_prob'] for a in read(OUT/'test'/f'{tag}_s{s}_T1.json')['attackers'].values() if a['attack_train_size']==20000)
            gaps.append(abs(test-val))
    macros['ValTestMaxGap']=f'{max(gaps):.3f}';families['validation_test_max_gap']=max(gaps)
    knn_change=max(abs(read(OUT/'test80k'/f'{tag}_s{s}_T1.json')['attackers']['knn_80000drops']['hit_prob']
                       -read(OUT/'test'/f'{tag}_s{s}_T1.json')['attackers']['knn_20000drops']['hit_prob'])
                   for tag in ('lam0.01','lam0','dz4_lam0','dz4_lam0.03') for s in (1,2,3))
    macros['KnnMaxChange']=f'{knn_change:.3f}'
    toolrows=[read(OUT/'test'/f'tools_lam0.01_s{s}.json') for s in (1,2,3)]
    toolstats={}
    lines=[r'\begin{tabular}{lrr}',r'\toprule',r'Mode & Rate ratio & Compute time (ms)\\',r'\midrule']
    for name in toolrows[0]:
        rs=[r[name] for r in toolrows]
        assert all(r['csi_dependent_return_bits']==0 for r in rs)
        rr=stat([r['ratio'] for r in rs]);ms=stat([r['compute_ms_per_drop'] for r in rs])
        toolstats[name]=dict(ratio=rr,compute_ms=ms)
        prefix='Tool'+''.join(c for c in name.title() if c.isalpha())
        macros[prefix+'Ratio']=f"{rr['mean']:.3f}"
        macros[prefix+'Ms']=f"{ms['mean']:.2f}"
        label={'direct':'Direct','rule':'Rule-based','search4':'Search, $C=4$','learned':'Learned router'}.get(name,name.replace('_',' '))
        lines.append(' & '.join([label,pm(rr),pm(ms,2)])+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (GEN/'tab_tools_final.tex').write_text('\n'.join(lines)+'\n');splice('tab_tools','\n'.join(lines))
    numbers=('% Generated by scripts/final_artifacts.py from frozen test results.\n'+
        '\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items()))
    (GEN/'numbers_revision.tex').write_text(numbers+'\n');splice('numbers',numbers)
    plt.rcParams.update({'font.family':'serif','font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'pdf.fonttype':42})
    # House style: plain markers and lines without error bars (the intervals are in the tables and the text).
    fig,axes=plt.subplots(1,2,figsize=(7,2.7))
    fig.subplots_adjust(left=.075,right=.98,top=.965,bottom=.43,wspace=.30)
    markers=['o','s','^','D','v','P','*','X','h','>']
    for i,tag in enumerate(ORDER):
        r=allrows[tag][1];color=plt.get_cmap('tab10')(i)
        for ax,metric in zip(axes,['Leak','Rmse']):
            ax.plot(100*r['Ratio']['mean'],r[metric]['mean'],linestyle='none',marker=markers[i],ms=5,color=color,label=NAMES[tag][1])
            ax.grid(alpha=.2);ax.set_xlabel('Teacher-rate retention (%)')
    axes[0].set_ylabel('Hit probability within 1 m');axes[1].set_ylabel('Localization RMSE (m)')
    # Subfigure tags under the x-axis labels (no titles above the axes); the caption describes each panel.
    for ax,tag in zip(axes,['(a)','(b)']):
        ax.annotate(tag,xy=(0.5,0),xycoords=ax.xaxis.label,xytext=(0,-3),textcoords='offset points',
                    ha='center',va='top',fontsize=8)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=7)
    (MS/'figures').mkdir(parents=True,exist_ok=True)
    fig.savefig(MS/'figures/fig_pareto_final.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.45,2.35),layout='constrained')
    for tag,label,line,marker in zip(temporal,short,['-','--','-.',':'],['o','s','^','D']):
        ax.plot([1,4,8],[allrows[tag][T]['Leak']['mean'] for T in (1,4,8)],linestyle=line,marker=marker,ms=4,label=label)
    top=max(allrows[t][8]['Leak']['mean'] for t in temporal)
    ax.set(xlabel='Observed intervals $T$',ylabel='Hit probability within 1 m',xticks=[1,4,8],xlim=(1,8),ylim=(0.1,top+0.14))
    ax.grid(alpha=.2);ax.legend(loc='upper left',ncol=2)
    fig.savefig(MS/'figures/fig_temporal_final.pdf');plt.close(fig)
    report=dict(statistics=allrows,scaling=scaling,scaling80k=scaling_seeds,attack_families=families,tools=toolstats,paired_effects=paired,claims=sources,
                frozen_sha=digest(OUT/'frozen.json'),test_files={p.name:digest(p) for p in (OUT/'test').glob('*.json')},
                test80k_files={p.name:digest(p) for p in (OUT/'test80k').glob('*.json')} if (OUT/'test80k').exists() else {})
    (OUT/'final_summary.json').write_text(json.dumps(report,indent=2))
    (OUT/'attack_registry.json').write_text(json.dumps(registry,indent=2))
    md=['# Final numbers','', 'Means and Student-t 95% CI half-widths over three fitted model seeds. Test: 2,000 drops / 32,000 users.','',
        '| Scheme | Rate ratio | Leakage T=1 | Leakage T=4 | Leakage T=8 |','|---|---:|---:|---:|---:|']
    for tag,rs in allrows.items():
        vals=[f"{rs[1]['Ratio']['mean']:.4f}"]+[f"{rs[T]['Leak']['mean']:.4f}" if T in rs else 'not evaluated' for T in (1,4,8)]
        md.append('| '+tag+' | '+' | '.join(vals)+' |')
    md += ['', 'Exact inputs and all attacker outcomes: `results/revision_final/final_summary.json` and `attack_registry.json`.','',
           'Finite-test Wilson intervals are working-binomial summaries. Drop-bootstrap intervals preserve common drop context. Neither is the across-seed Student-t interval or an upper bound on an unrestricted attacker.']
    (ROOT/'FINAL_NUMBERS.md').write_text('\n'.join(md)+'\n')
    spliced=' (spliced into main.tex)' if (MS/'main.tex').exists() else ''
    print(f'Generated frozen-test macros, tables{spliced}, two figures, registry, and FINAL_NUMBERS.md.')

if __name__=='__main__':main()
