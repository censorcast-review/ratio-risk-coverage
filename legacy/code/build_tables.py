from pathlib import Path
import json,numpy as np

root=Path(__file__).resolve().parents[1]
dst=root/'paper/tables';dst.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads(p.read_text())
point=load(root/'results/analysis/POINT_ANALYSIS.json')
studies={s:load(root/f'results/{s}/SELECTIVE_RESULTS.json') for s in ['selective','selective_strong']}
analyses={s:load(root/f'results/analysis/{s}_ANALYSIS.json') for s in studies}
def write(name,caption,label,cols,header,rows):
    text=r'\begin{table}[t]'+'\n'+r'\centering\small'+'\n'+r'\caption{'+caption+'}\n'+r'\label{'+label+'}\n'+r'\begin{tabular}{'+cols+'}\n'+r'\toprule'+'\n'+header+r'\\'+'\n'+r'\midrule'+'\n'
    text+='\n'.join(' & '.join(row)+r'\\' for row in rows)
    text+='\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'+r'\end{table}'+'\n'
    (dst/name).write_text(text)

pmap={r['method']:r for r in point['seed_aggregates']}
rows=[['Earlier HistGB Poisson','1','0.72713','--'],['Earlier CENSORCAST','1','0.70584','--']]
for key,label in [('observed_poisson','LightGBM Poisson'),('uncensored_l1','LightGBM L1, uncensored rows'),('observed_l1','LightGBM L1, observed sales'),('global_scale','L1 + global rescaling'),('category_scale','L1 + category rescaling'),('ungated_em_correction','L1 + ungated EM correction'),('censor_adapter','L1 + censor-gated adapter')]:
    r=pmap[key];rows.append([label,'3',f"{r['shadow_wape_mean']:.5f}",f"{r['shadow_wape_sd']:.5f}"])
write('point_table.tex','Forecasting ablations on the full 120-day M5 design shadow. SD is across three training seeds, not a confidence interval. Earlier forecasts are fixed single-run controls.','tab:point','lrrr','Forecaster & Fits & Mean WAPE & Seed SD',rows)

rows=[]
names={'mean_error':'Mean absolute error','quantile_error':'Error quantile (0.85)','relative_error':'Predicted relative error','contract_excess':'Contract excess'}
for s,title in [('selective','Earlier'),('selective_strong','L1 + adapter')]:
    for name,label in names.items():
        rs=[r for r in studies[s]['records'] if r['score']==name and r['threshold_mode']=='pooled']
        c=np.mean([r['metrics']['shadow']['coverage'] for r in rs]);w=[r['metrics']['shadow']['wape'] for r in rs];auc=np.mean([r['score_diagnostics']['auc_top20_absolute_error'] for r in rs])
        rows.append([title,label,f'{auc:.3f}',f'{c:.3f}',f'{np.mean(w):.3f}' if all(v is not None for v in w) else '--'])
write('selector_table.tex','Matched risk targets with one pooled threshold. Means across three risk seeds on the 114-day design shadow. AUC detects high absolute error. All rows use cap 0.64013 and floor 0.35; zero coverage means no feasible calibration threshold, not a theorem of impossibility.','tab:selectors','llrrr','Forecaster & Score & AUC & Coverage & WAPE',rows)

rows=[]
for s,title in [('selective','Earlier'),('selective_strong','L1 + adapter')]:
    for group,v in analyses[s]['category_intervals']['contract_excess'].items():
        rows.append([title,group.title(),f"{v['coverage']:.4f}",f"{v['wape']:.4f}",f"{v['coverage_lcb_bonf8']:.4f}",f"{v['wape_ucb_bonf8']:.4f}"])
write('group_table.tex','Pooled contract-excess threshold, first risk seed. Adjusted bounds use bootstrap quantiles at 0.05/8 and 1--0.05/8 across coverage/WAPE for the whole population and three categories. They are descriptive, not a calibrated familywise guarantee.','tab:groups','llrrrr','Forecaster & Group & Coverage & WAPE & Cov. lower & WAPE upper',rows)

rows=[]
for s,title in [('selective','Earlier'),('selective_strong','L1 + adapter')]:
    for r in studies[s]['records']:
        if r['score'] not in ['contract_excess','relative_error'] or r['threshold_mode']!='pooled':continue
        m=r['metrics']['shadow'];rows.append([title,names[r['score']],str(r['seed']),f"{m['coverage']:.5f}",f"{m['wape']:.5f}",f"{r['thresholds']['ALL']:.5f}"])
write('seed_table.tex','Per-seed pooled thresholds and shadow outcomes. These are repeated fits on the same development data.','tab:seeds','llrrrr','Forecaster & Score & Seed & Coverage & WAPE & Threshold',rows)
rows=[]
for s,title in [('selective','Earlier'),('selective_strong','L1 + adapter')]:
    for r in studies[s]['records']:
        if r['score'] not in ['contract_excess','relative_error'] or r['threshold_mode']!='pooled' or r['seed']!=20260906:continue
        for scheme,k in [('Item','shadow_bootstrap'),('Week','shadow_week_bootstrap')]:
            b=r[k];c=b['coverage_ci95'];w=b['wape_ci95']
            rows.append([title,names[r['score']],scheme,f'[{c[0]:.4f}, {c[1]:.4f}]',f'[{w[0]:.4f}, {w[1]:.4f}]'])
write('week_table.tex','Dependence sensitivity for the first seed: paired item-cluster versus forecast-week bootstrap, each with 3,000 draws. Neither incorporates repeated development selection.','tab:week','lllcc','Forecaster & Score & Cluster & Coverage interval & WAPE interval',rows)

rs=analyses['selective_strong']['seed_aggregates']
direct=next(r for r in rs if r['score']=='contract_excess' and r['mode']=='pooled')
rel=next(r for r in rs if r['score']=='relative_error' and r['mode']=='pooled')
delta=analyses['selective_strong']['direct_minus_relative_coverage']
sentence=(f"{100*direct['coverage_mean']:.2f}\\% mean acceptance for contract excess versus {100*rel['coverage_mean']:.2f}\\% for predicted relative error, "
    f"with accepted WAPE {np.mean(direct['wape_all']):.4f} and {np.mean(rel['wape_all']):.4f}, respectively. "
    f"The first-seed coverage gain is {100*delta['coverage_difference']:.2f} percentage points "
    f"(descriptive paired interval [{100*delta['ci95'][0]:.2f}, {100*delta['ci95'][1]:.2f}]).")
groups=analyses['selective_strong']['category_intervals']['contract_excess']
sentence2=(f"The stronger forecast improves the pooled acceptance result but retains category failures: "
           f"Hobbies accepted WAPE is {groups['HOBBIES']['wape']:.4f} and Household is {groups['HOUSEHOLD']['wape']:.4f}, "
           f"both above the unchanged cap. Category-specific screening still returns no Hobbies candidate.")
(dst/'result_macros.tex').write_text(r'\newcommand{\StrongRiskSentence}{'+sentence+'}\n'+r'\newcommand{\StrongGroupSentence}{'+sentence2+'}\n')
print('Manuscript tables generated from completed results.')
