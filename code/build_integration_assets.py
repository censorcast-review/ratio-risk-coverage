from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];p=R/'paper';r=json.loads((R/'evidence/accuracy_selection/RESULTS.json').read_text());f=json.loads((R/'evidence/accuracy_selection/FROZEN_SELECTORS.json').read_text())
keys=['group_l1','learned_l1','chronos2_univariate_raw'];names=['Group-scaled L1','Residual L1','Chronos-2 (raw)'];rows=[]
for k,n in zip(keys,names):
 v=r['predictors'][k]
 for lam in ['1.0','0.25']:
  x=v['policies'][lam];rows.append(f"{n} & {v['full']['wape']:.5f} & {float(lam):g} & {100*x['row_coverage']:.2f} & {100*x['demand_coverage']:.2f} & {x['wape']:.5f} "+r'\\')
t=r'''\begin{table}[H]
\centering\small
\caption{New retrospective comparison on the same 2,085,060 M5 development evaluation rows. Each predictor has a newly fitted error head; all share a newly fitted demand head, head-training rows, cap, and calibration dates. Higher coverage is better; lower WAPE is better. Full WAPE uses every row. These are fitted-policy outcomes, not population optima or a new external test.}
\label{tab:accuracy-selection}
\begin{tabular}{lrrrrr}\toprule
Predictor & Full WAPE & $\lambda$ & Rows (\%) & Demand (\%) & Accepted WAPE\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}
\end{table}
''';(p/'accuracy_table.tex').write_text(t)
rows=[]
for k,n in zip(keys,names):
 for l in ['0.25','0.0']:
  v=r['contrasts'][k+':'+l];ci=np.array(v['ci95'])*100;rows.append(f"{n} & {float(l):g} & {100*v['row_delta']:+.2f} & $[{ci[0,0]:.2f},{ci[0,1]:.2f}]$ & {100*v['demand_delta']:+.2f} & $[{ci[1,0]:.2f},{ci[1,1]:.2f}]$ "+r'\\')
(p/'accuracy_ci_table.tex').write_text(r'''\begin{table}[htbp]\centering\small
\caption{New within-predictor contrasts versus $\lambda=1$, in percentage points. Marginal 95\% paired item-bootstrap intervals condition on the fits and research history; 4,000 draws, 1,829 items. The pure-demand endpoint $\lambda=0$ is included without a denominator floor.}
\label{tab:accuracy-ci}
\begin{tabular}{lrrrrr}\toprule
Predictor & $\lambda$ & $\Delta$ rows & Interval & $\Delta$ demand & Interval\\\midrule
'''+ '\n'.join(rows)+r'\bottomrule\end{tabular}\end{table}')
rows=[]
for k,n in zip(keys,names):
 for label,labeln in [('common','Common'),('row_only','Row only'),('other_only','Mixed only')]:
  v=r['budgets'][k+':0.25'][label];rows.append(f"{n} & {labeln} & {v['rows']:,} & {100*v['demand_coverage']:.2f} & {v['wape']:.4f} & {v['mean_forecast']:.3f} & {v['excess']:,.1f} "+r'\\')
(p/'accuracy_budget_table.tex').write_text(r'''\begin{table}[htbp]\centering\small
\caption{Budget decomposition for $\lambda=1$ versus $\lambda=0.25$ in the new retrospective comparison. Demand share uses the whole evaluation denominator; excess is absolute-error mass minus $r$ times demand mass. Negative common excess supplies slack.}
\label{tab:accuracy-budget}
\begin{tabular}{llrrrrr}\toprule
Predictor & Set & Rows & Demand (\%) & WAPE & Mean $f$ & Excess\\\midrule
'''+ '\n'.join(rows)+r'\bottomrule\end{tabular}\end{table}')
rows=[]
for k,n in zip(keys,names):
 for l in ['1.0','0.25','0.0']:
  v=r['predictors'][k]['policies'][l];g=r['groups'][k][l];ca=f['calibration_metrics'][k][l];rows.append(f"{n} & {float(l):g} & {ca['0']['wape']:.4f}/{ca['1']['wape']:.4f} & {g['FOODS']['wape']:.4f} & {g['HOBBIES']['wape']:.4f} & {g['HOUSEHOLD']['wape']:.4f} "+r'\\')
(p/'accuracy_groups_table.tex').write_text(r'''\begin{table}[htbp]\centering\small
\caption{Calibration pooled and later category WAPE for the new predictors. The cap is 0.64013. All nine policies meet both calibration point constraints; Hobbies and Household miss the later cap.}
\label{tab:accuracy-groups}
\begin{tabular}{lrrrrr}\toprule
Predictor & $\lambda$ & Cal. A/B & Foods & Hobbies & Household\\\midrule
'''+ '\n'.join(rows)+r'\bottomrule\end{tabular}\end{table}')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(8.3,3.0),gridspec_kw={'width_ratios':[1.12,1]})
colors=['#546d83','#c25029','#197879']
for k,n,col in zip(keys,names,colors):
 v=r['predictors'][k]['policies'];xs=[100*v[l]['row_coverage'] for l in ['1.0','0.25','0.0']];ys=[100*v[l]['demand_coverage'] for l in ['1.0','0.25','0.0']]
 axes[0].plot(xs[:2],ys[:2],color=col,lw=1.8,label=n);axes[0].plot(xs[1:],ys[1:],color=col,lw=1,ls=':');
 for i,mark in enumerate(['o','s','^']):axes[0].scatter(xs[i],ys[i],marker=mark,color=col,s=32,zorder=3)
axes[0].set(xlabel='Accepted rows (%)',ylabel='Covered demand (%)',title='Accuracy improves both coverages');axes[0].grid(alpha=.15);axes[0].legend(fontsize=8,loc='lower left');
k='learned_l1';a=r['budgets'][k+':0.25']['row_only'];b=r['budgets'][k+':0.25']['other_only'];N=r['predictors'][k]['full']['rows'];xx=np.arange(2)
axes[1].bar(xx-.18,[100*a['rows']/N,100*b['rows']/N],.34,color='#546d83',label='Rows');axes[1].bar(xx+.18,[100*a['demand_coverage'],100*b['demand_coverage']],.34,color='#c25029',label='Demand')
axes[1].set(xticks=xx,xticklabels=['Row only','Mixed only'],ylabel='Share of all evaluation (%)',title='The stronger predictor still reallocates');axes[1].legend(fontsize=8);axes[1].set_ylim(0,11)
for i,s in enumerate([a,b]):axes[1].text(i,9.3,f"WAPE {s['wape']:.3f}",ha='center',fontsize=9)
fig.tight_layout(w_pad=1.5);fig.savefig(p/'figures/accuracy_coverage.pdf',bbox_inches='tight');plt.close(fig)
