"""Render manuscript tables/figures from completed, frozen results. No fitting."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
M5 = json.loads((ROOT/'evidence/m5/FOUNDATION_RESULTS.json').read_text())
UCI = json.loads((ROOT/'evidence/uci/run_001/CONFIRMATION_RESULTS.json').read_text())
MODELS = ['chronos_bolt_small','chronos2_univariate','chronos2_covariates','observed_l1','original_censor_adapter']
NAMES = ['Chronos-Bolt Small','Chronos-2 (univariate)','Chronos-2 (covariates)','Observed-sales L1','CENSORCAST (benchmark)']
def row(model, block='shadow', category='ALL', horizon=None):
    return next(r for r in M5['rows'] if r['model']==model and r['block']==block and r['category']==category and r.get('horizon')==horizon)
def table(path, caption, label, columns, header, rows):
    text = '\\begin{table}[htbp]\n\\centering\\small\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+' \\\\\n\\midrule\n'+'\n'.join(' & '.join(r)+' \\\\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (PAPER/path).write_text(text)

rows=[]
for model,name in zip(MODELS,NAMES):
    v=[row(model,b)['wape'] for b in ['calibration_a','calibration_b','shadow']]
    vals=[f'{x:.4f}' for x in v]
    if model==MODELS[-1]: vals=['\\textbf{'+s+'}' for s in vals]
    rows.append([name]+vals+[f"{row(model)['mae']:.4f}"])
table('m5_table.tex','Full-coverage M5 comparison on the same origin-aligned rows within each block. A/B/later contain 119/120/114 days. The comparison is retrospective. The benchmark correction uses strict censoring information; Chronos receives observed histories (and the stated covariates), with no task-specific fitting.','tab:m5','lrrrr','Method & A WAPE & B WAPE & Later WAPE & Later MAE',rows)

rows=[]
for r,name in zip(UCI['rows_results'],['Baseline: $2b$','Correction: $2[b+2p^2(m-b)_+]$']):
    rows.append([name,f"{r['wape']:.4f}",f"{r['mae']:.4f}",f"{100*r['volume_bias']:.2f}",'100'])
table('uci_table.tex','One-time temporal confirmation on Online Retail II: 2,000 products, 149 days, 298,000 rows. Both fitted policies and their predictions were fixed before test evaluation. Bias is signed total forecast error divided by total recorded sales, in percent. All rows and all target mass are retained.','tab:uci','lrrrr','Policy & WAPE & MAE & Bias (\\%) & Rows (\\%)',rows)

rows=[]
for model,name in zip(MODELS,NAMES):
    rows.append([name]+[f"{row(model,category=c)['wape']:.4f}" for c in ['FOODS','HOBBIES','HOUSEHOLD']]+[f"{100*row(model)['wpe']:.2f}"])
table('m5_groups.tex','Full-coverage later-block M5 results by category, with pooled signed volume bias. The small Hobbies deterioration relative to L1 is retained.','tab:groups','lrrrr','Method & Foods & Hobbies & Household & Bias (\\%)',rows)
rows=[]
for h in range(1,8):
    rows.append([str(h)]+[f"{row(m,horizon=h)['wape']:.4f}" for m in MODELS])
table('m5_horizons.tex','M5 later-block WAPE by forecast horizon, on the common aligned population. C2-U/C2-C denote Chronos-2 univariate/covariate runs.','tab:horizons','rrrrrr','Horizon & Bolt & C2-U & C2-C & L1 & Correction',rows)

dev=json.loads((ROOT/'evidence/uci/run_001/DEVELOPMENT_RESULTS.json').read_text())
rows=[]
for m,n in zip(['l1','poisson','seasonal_naive','rolling_mean'],['Observed-sales L1','Observed-sales Poisson','Seasonal naive','28-day mean']):
    r=min((r for r in dev['records'] if r['config']['kind']=='base' and r['config']['model']==m),key=lambda r:r['wape'])
    rows.append([n,str(r['config']['scale']),f"{r['wape']:.6f}"])
rows.append(['Gated correction',str(dev['proposal']['config']['scale']),f"{dev['proposal']['wape']:.6f}"])
table('uci_development.tex','Online Retail II development selection. Each baseline family uses the same six-scale grid. The selected correction has $\\alpha=2$, $\\gamma=2$. These are validation scores; unselected policies were not evaluated on the final test.','tab:ucidev','lrr','Family & Selected scale & Validation WAPE',rows)

ab=json.loads((ROOT/'evidence/diagnostics/m5_adapter_development.json').read_text())
rows=[]
for m,n in [('plain','L1 base'),('global_scale','Global rescaling'),('category_scale','Category rescaling'),('ungated_em_correction','Ungated EM uplift'),('censor_adapter','Gated correction')]:
    rs=[v for v in ab['records'] if v['method']==m]
    val=[v['selection']['wape'] for v in rs];later=[v['shadow']['wape'] for v in rs]
    rows.append([n,f'{np.mean(val):.5f}',f'{np.mean(later):.5f}',f'[{min(later):.5f}, {max(later):.5f}]'])
table('m5_components.tex','Historical M5 component controls, three point-fit seeds. Each choice uses only the earlier point-selection block. This ablation evaluates 120 later days (2,194,800 rows), distinct from the common 114-day foundation comparison. The range describes three fits, not a confidence interval.','tab:components','lrrr','Control & Selection mean & Later mean & Later range',rows)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(6.6,2.7))
cmp=[next(c for c in M5['paired_comparisons'] if c['model']==m and c['block']=='shadow' and c['reference']=='proposal') for m in MODELS[:-1]]
names=['M5: vs. Chronos-Bolt','M5: vs. Chronos-2 (U)','M5: vs. Chronos-2 (C)','M5: vs. L1','UCI: frozen temporal test']
vals=[-c['delta_wape']*100 for c in cmp]+[UCI['delta_wape_proposal_minus_baseline']*100]
cis=[[-c['ci95'][1]*100,-c['ci95'][0]*100] for c in cmp]+[[v*100 for v in UCI['paired_item_bootstrap_ci95']]]
for i,(v,ci) in enumerate(zip(vals,cis)):
    color='#176b87' if i<4 else '#b04b21'
    ax.errorbar(v,4-i,xerr=[[v-ci[0]],[ci[1]-v]],fmt='o',color=color,capsize=3,ms=5)
ax.axvline(0,color='#777777',lw=.8,ls='--')
ax.set_yticks(range(5),names[::-1]);ax.set_xlabel('WAPE difference × 100 (correction − comparator)')
ax.set_xlim(-3.35,.15);ax.grid(axis='x',alpha=.15);fig.tight_layout()
fig.savefig(PAPER/'figures/full_coverage_gain.pdf',bbox_inches='tight');fig.savefig(PAPER/'figures/full_coverage_gain.png',dpi=180,bbox_inches='tight');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(6.6,2.8),gridspec_kw={'width_ratios':[1,1.15]})
ext=json.loads((ROOT/'evidence/selection/EXTERNAL_COMPARISON.json').read_text())
# Exact values here are checked against the external statistics by verify_manuscript.py.
em=ext['primary']['metrics']
cr=np.array([em[k]['row_coverage']*100 for k in ['composed_excess_lambda_1','mixed_utility_lambda_0_25']])
dm=np.array([em[k]['demand_coverage']*100 for k in ['composed_excess_lambda_1','mixed_utility_lambda_0_25']])
for i,(x,y) in enumerate(zip(cr,dm)):
    axs[0].plot(x,y,'o',color=['#176b87','#b04b21'][i]);axs[0].annotate([r'$\lambda=1$',r'$\lambda=0.25$'][i],(x,y),xytext=(5,-13 if i==0 else 5),textcoords='offset points')
axs[0].set(xlabel='Row coverage (%)',ylabel='Demand coverage (%)',xlim=(80.5,89),ylim=(81.5,90))
xs=np.linspace(0,1/3,200);axs[1].plot(xs,(1/3-xs)/(1-xs/3),color='#176b87',label=r'$1\leq w\leq2$ envelope')
axs[1].plot(1/15,1/18,'o',color='#b04b21',label='Fixed two-label example')
axs[1].set(xlabel='Row-coverage gap',ylabel='Exposure-coverage gap',xlim=(0,.35),ylim=(0,.35));axs[1].legend(fontsize=7.5,frameon=False,loc='upper right')
for ax in axs: ax.grid(alpha=.15)
fig.tight_layout();fig.savefig(PAPER/'figures/coverage_geometry.pdf',bbox_inches='tight');fig.savefig(PAPER/'figures/coverage_geometry.png',dpi=180,bbox_inches='tight');plt.close(fig)
print('Generated five evidence tables, horizon table, and two figures from saved results.')
