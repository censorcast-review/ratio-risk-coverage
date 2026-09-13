"""Generate paper tables and figures from the completed immutable result files."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'paper';E=ROOT/'evidence/revision'
def read(p):return json.loads(p.read_text())
def f(x,d=4):return f'{x:.{d}f}'
def table(name,caption,label,cols,head,rows):
    text='\\begin{table}[htbp]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\small\n\\begin{tabular}{'+cols+'}\n\\toprule\n'
    text+=' & '.join(head)+r' \\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
    text+='\n\\bottomrule\n\\end{tabular}\n\\end{table}\n';(P/name).write_text(text)

u=read(E/'audit/UCI_AUDIT.json');m=read(E/'audit/M5_AUDIT.json')
rows=[]
for name,z in [('M5 A',m['calibration_a']['severity']),('M5 B',m['calibration_b']['severity']),('M5 later',m['shadow']['severity']),('UCI validation',u['severity']['validation']),('UCI test',u['severity']['test'])]:
    rows.append([name,f"{z['rows']:,}",f(100*z['capacity_hit_fraction'],2),f(100*z['strictly_censored_fraction'],2),f(100*z['hidden_mass_fraction'],2)])
table('revision_severity.tex',r'Actual censoring severity. Hit and strict columns are row percentages; hidden mass is $100\sum(Y-S)/\sum Y$. The target remains recorded sales before imposed censoring.','tab:severity','lrrrr',['Population','Rows',r'Hit (\%)',r'Strict (\%)',r'Hidden mass (\%)'],rows)

rows=[]
for name,key in [('Frozen scaled L1','baseline'),('Frozen correction','proposal')]:
    z=u['frozen_test_metrics'][key];rows.append([name,f(z['wape']),f(z['mae']),f(z['rmse']),f(z['mean_series_rmsse']),f(100*z['signed_bias'],2)])
table('revision_uci.tex','The original two-policy UCI temporal test, at full coverage. RMSSE is the macro average over 2,000 products with fixed training normalizers. Additional metrics are descriptive reuse of frozen predictions.','tab:uci','lrrrrr',['Frozen policy','WAPE','MAE','RMSE','RMSSE',r'Bias (\%)'],rows)

rows=[[f"L1, $s={v['scale']:.5f}$" if i==4 else f"L1, $s={v['scale']:g}$",f(v['wape'],6),f(100*v['signed_bias'],2)] for i,v in enumerate(u['validation_scale_diagnostic'])]
rows.append(['Frozen correction',f(u['frozen_correction_validation_wape'],6),'--'])
table('revision_scale.tex','UCI validation-only scale audit. The final L1 row uses the exact weighted-median optimum. No wider-scale policy is evaluated on the test.','tab:scale','lrr',['Validation policy','WAPE',r'Bias (\%)'],rows)

rows=[]
for v in u['concentration']:
    rows.append([v['unit'].capitalize(),f(100*v['top_fraction'],1),f(100*v['target_share'],2),f(100*v['baseline_error_share'],2),f(100*v['share_of_total_error_reduction'],2),f(v['remaining_baseline_wape'],5),f(v['remaining_proposal_wape'],5)])
table('revision_concentration.tex',r'UCI outcome-ranked concentration, on the unchanged frozen forecasts. The top-set shares are percentages. Complement columns show WAPE after descriptive removal of that top set. A reduction share above $100\%$ means the complement worsens.','tab:concentration','lrrrrrr',['Unit',r'Top (\%)','Mass','Base error','Reduction','Base rest','Corr. rest'],rows)

fig,axes=plt.subplots(1,2,figsize=(6.8,2.6),gridspec_kw={'width_ratios':[1.05,1]})
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
z=np.load(E/'audit/UCI_VALIDATION_PREDICTIONS.npz');y=z['truth'].astype(float);b=z['unscaled_l1'];ss=np.linspace(1.7,4,101)
scores=np.array([np.abs(y-s*b).sum()/y.sum() for s in ss]);a=axes[0]
a.plot(ss,scores,color='#1b5c88',lw=1.8,label='Scaled L1')
a.axhline(u['frozen_correction_validation_wape'],color='#c36b24',ls='--',lw=1.4,label='Frozen correction')
a.scatter([u['continuous_optimum_scale']],[u['validation_scale_diagnostic'][-1]['wape']],s=25,color='#1b5c88',zorder=3)
a.set(xlabel='L1 scale (validation only)',ylabel='WAPE');a.legend(frameon=False,fontsize=8,loc='upper left')
sub=[next(v for v in u['concentration'] if v['unit']==unit and v['top_fraction']==.01) for unit in ['row','product']]
xx=np.arange(2)
for j,(key,label,color) in enumerate([('target_share','Demand','#1b5c88'),('baseline_error_share','Baseline error','#8eabc1'),('share_of_total_error_reduction','Error reduction','#c36b24')]):
    axes[1].bar(xx+(j-1)*.23,[100*v[key] for v in sub],width=.21,label=label,color=color)
axes[1].set_xticks(xx,['Top 1% rows','Top 1% products']);axes[1].set(ylabel='Share (%)',ylim=(0,85));axes[1].legend(frameon=False,fontsize=7.5,loc='upper right')
for a in axes:a.spines[['top','right']].set_visible(False);a.tick_params(labelsize=8)
fig.tight_layout(pad=.9);(P/'figures').mkdir(exist_ok=True);fig.savefig(P/'figures/revision_diagnostics.pdf');fig.savefig(P/'figures/revision_diagnostics.png',dpi=180);plt.close(fig)

seeds=[20260906,20260907,20260908];base=E/'m5_observable'
results=read(base/'RESULTS.json')['rows']
def mr(seed,block):return next(x for x in results if x['seed']==seed and x['block']==block)
names={'l1':'L1, observable features','scaled_l1':'Exact global scale','category_scaled_l1':'Exact category scales',
       'observed_mean_gate':'Ordinary Poisson + gate','censored_mean_gate':'Censored Poisson + gate','censored_mean_ungated':'Censored Poisson, ungated'}
rows=[]
for key,label in names.items():
    vals=[mr(s,'shadow')['metrics'][key]['wape'] for s in seeds]
    rows.append([label]+[f(mr(seeds[0],block)['metrics'][key]['wape']) for block in ['calibration_a','calibration_b','shadow']]+[f'{np.mean(vals):.6f} ({np.std(vals,ddof=1):.6f})'])
table('revision_m5.tex',r'Matched observable-input M5 comparison. A, B, and later are the first predefined seed; the last column gives later-block mean (sample SD) across all three seeds. All policies retain every row. The finite strength grid selects zero in each correction family.','tab:observable','lrrrr',['Observable model','A','B','Later','Later mean (SD)'],rows)
first=mr(seeds[0],'shadow')['metrics'];gain=100*(1-first['scaled_l1']['wape']/first['l1']['wape'])
(P/'observable_summary.tex').write_text(f"Across all three seeds, each finite-grid correction family selects $\\alpha=0$. On the first seed, exact global rescaling lowers later-block WAPE from {first['l1']['wape']:.5f} to {first['scaled_l1']['wape']:.5f}, a {gain:.2f}\\% relative reduction; category scales give {first['category_scaled_l1']['wape']:.5f}. This is a calibration gain, with no selected upward count correction (Table~\\ref{{tab:observable}}).\n")

rows=[]
for key,label in names.items():
    r=first[key];rows.append([label,f(r['mae']),f(r['rmse']),f(r['mean_series_rmsse']),f(100*r['signed_bias'],2)])
table('revision_m5_metrics.tex','Additional first-seed metrics for the new observable M5 study, on the aligned later block. RMSSE uses 17,299 series with nonzero training normalizers; every primary row is retained.','tab:observablemetrics','lrrrr',['Observable model','MAE','RMSE','RMSSE',r'Bias (\%)'],rows)

found=read(ROOT/'evidence/m5/FOUNDATION_RESULTS.json')['rows']
def fw(model):return next(r['wape'] for r in found if r['model']==model and r['block']=='shadow' and r['category']=='ALL' and 'horizon' not in r)
rows=[['Chronos-Bolt Small','Sales','Zero-shot',f(fw('chronos_bolt_small'))],
      ['Chronos-2 univariate','Sales','Zero-shot',f(fw('chronos2_univariate'))],
      ['Chronos-2 covariates','Observable','Zero-shot',f(fw('chronos2_covariates'))],
      ['Original L1','Observable','Task fit',f(fw('observed_l1'))],
      ['New scaled L1','Observable','Task fit',f(first['scaled_l1']['wape'])],
      ['Original correction','Privileged flag','Task fit',f(fw('original_censor_adapter'))]]
table('revision_foundation.tex','Later-block reference configurations on the same M5 targets. Inputs and training regimes differ. The new observable correction equals the new scaled L1 in the finite-grid study. No architectural superiority or equality of training budgets is claimed.','tab:reference','lllr',['Configuration','Inputs','Learning','WAPE'],rows)

rows=[];summary=[]
for seed in seeds:
    j=read(E/'joint_calibration'/f'seed_{seed}/POLICY_FREEZE.json');orig=read(base/f'seed_{seed}/POLICY_FREEZE.json')['choices']['scaled_l1']['wape']
    for key,label in [('observed_mean_gate','Ordinary + gate'),('censored_mean_gate','Censored + gate'),('censored_mean_ungated','Censored, ungated')]:
        v=j['choices'][key]
        rows.append([str(seed),label,f(v['beta_base'],5),f(v['beta_uplift'],5),f(v['wape'],8),f(1e6*(orig-v['wape']),3)])
        summary.append(dict(seed=seed,improvement_wape=orig-v['wape'],**v))
table('revision_joint.tex',r'Joint continuous calibration on M5 validation only. Each reported mean/gate family selects its best specified power. Last column is improvement over exact rescaling in units of $10^{-6}$ WAPE. All numerical optimization gaps are at most $10^{-7}$.','tab:joint','llrrrr',['Seed','Family',r'$\beta_0$',r'$\beta_1$','WAPE',r'Gain ($10^{-6}$)'],rows)
positive=[v for v in summary if v['beta_uplift']>0]
assert len(positive)==1
v=positive[0]
(P/'joint_summary.tex').write_text(f"The second seed admits a positive censored uplift under continuous calibration, but its validation improvement over exact rescaling is only ${v['improvement_wape']*1e6:.2f}\\times 10^{{-6}}$ WAPE; the first and third seeds select zero. This small refinement is reported in Appendix Table~\\ref{{tab:joint}}, without promoting it to a new temporal confirmation.\n")
print('All revision assets generated from completed evidence')
