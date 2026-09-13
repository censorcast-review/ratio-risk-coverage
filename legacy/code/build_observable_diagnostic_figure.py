"""Compact diagnostic table and figure from retained aggregate audits only."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output; out.mkdir(parents=True,exist_ok=True)
    audit=json.loads((out/'OBSERVABLE_MOMENT_DIAGNOSTICS.json').read_text())
    original=json.loads((a.source/'results/capacity_hit_extension/CAPACITY_HIT_VERIFIED_SUMMARY.json').read_text())
    cap=audit['cap']
    def get(method,block='shadow'):
        return next(v for v in audit['means'] if v['method']==method and v['block']==block)
    tex=r'''\begin{table}[t]
\centering\small
\caption{Sales-and-capacity NB $\kappa=2$ at its observed-likelihood-selected, model-implied threshold. Bias is $(\widehat E/E-1)$ or $(\widehat M/M-1)$ on the same accepted rows. Each entry averages three fit-specific ratios. Positive error-mass bias exceeds positive demand-mass bias, making the implied ratio conservative in aggregate.}
\label{tab:observablemoments}
\begin{tabular}{lrrrrr}
\toprule
Period & Rows (\%) & Implied WAPE & Realized WAPE & Error bias (\%) & Demand bias (\%)\\
\midrule
'''
    for block,name in [('calibration_a','A'),('calibration_b','B'),('shadow','Shadow')]:
        x=get('capacity_hit_nb_2',block)
        vals=[100*x['row_coverage']['mean'],x['model_implied_wape']['mean'],x['realized_wape']['mean'],100*x['error_mass_relative_bias']['mean'],100*x['demand_mass_relative_bias']['mean']]
        tex+=f'{name} & {vals[0]:.2f} & {vals[1]:.4f} & {vals[2]:.4f} & +{vals[3]:.2f} & +{vals[4]:.2f}'+r'\\'+'\n'
    tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'
    (out/'observable_moment_table.tex').write_text(tex)
    text=r'''At the selected $\kappa=2$ thresholds, implied/realized WAPE is
$0.6401/0.5568$ in A, $0.6376/0.5562$ in B, and $0.6365/0.5461$ on shadow
(three-fit means). On the shadow accepted set, predicted absolute-error mass
exceeds realized error by $20.59\%$, while predicted demand exceeds realized
demand by only $3.46\%$. Thus the pooled pass reflects conservative aggregate
moments, not evidence of accurate conditional risk estimates. Conversely,
capacity-hit Poisson and NB $\kappa=10$ accept all shadow rows and imply WAPE
$0.5603$ and $0.6009$, respectively, while realized WAPE is $0.6790$.
No threshold for $\kappa=0.5$ meets the imposed pooled cap and row floor in both
calibration periods. These controls expose dispersion sensitivity; they do not
establish category-wise validity.

At the same empirical row-probability coverage of $48.17\%$, uniform random
acceptance with the identical fixed point forecast has demand coverage
$48.17\%$ and a ratio of expected error to expected demand of $0.6790$;
the selected NB rule gives $54.80\%$ and $0.5461$. This exact fractional-
acceptance control shows that the retained rows are more favorable than
uniform abstention. It does not replace a comparison of learned score
rankings at matched coverage; the retained aggregate records do not identify
those alternative operating points.
'''
    (out/'observable_diagnostic_text.tex').write_text(text)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(9.0,3.05),gridspec_kw={'width_ratios':[1.23,1]})
    methods=['capacity_hit_nb_0p5','capacity_hit_nb_2','capacity_hit_nb_10','capacity_hit_poisson']
    labels=['NB 0.5\n0% rows','NB 2\n48.17% rows','NB 10\n100% rows','Poisson\n100% rows']
    for field,color,marker,delta,label in [('model_implied_wape','#b05a29','s',-.08,'Model-implied'),('realized_wape','#126f67','o',.08,'Realized')]:
        data=[get(method)[field] for method in methods]
        indices=[i for i,d in enumerate(data) if d is not None]
        vv=[data[i] for i in indices];mm=np.array([v['mean'] for v in vv]);lo=np.array([v['min'] for v in vv]);hi=np.array([v['max'] for v in vv])
        axs[0].errorbar(np.array(indices)+delta,mm,yerr=[mm-lo,hi-mm],fmt=marker,color=color,capsize=3,label=label,markersize=5,ls='none')
    axs[0].text(0,.588,'No accepted\nmass',ha='center',va='center',fontsize=8,color='#666666')
    axs[0].set(xticks=np.arange(4),xticklabels=labels,xlim=(-.55,3.55),ylim=(.49,.725),ylabel='Pooled WAPE',title='Moment conservatism depends on dispersion')
    axs[0].axhline(cap,color='#a53947',ls='--',lw=1,label='Cap')
    axs[0].legend(loc='lower right',fontsize=7,frameon=False)
    rec=next(v for v in original['method_means'] if v['method']=='capacity_hit_nb_2' and v['score']=='excess' and v['calibration']=='model_implied_observable')
    groups=['ALL','FOODS','HOBBIES','HOUSEHOLD'];vv=[rec['metrics']['shadow'][g]['wape'] for g in groups]
    mm=np.array([v['mean'] for v in vv]);lo=np.array([v['min'] for v in vv]);hi=np.array([v['max'] for v in vv])
    axs[1].errorbar(np.arange(4),mm,yerr=[mm-lo,hi-mm],fmt='o',color='#126f67',capsize=3,markersize=5)
    axs[1].axhline(cap,color='#a53947',ls='--',lw=1)
    axs[1].set(xticks=np.arange(4),xticklabels=['Pooled','Foods','Hobbies','Household'],ylim=(.45,.88),ylabel='Realized WAPE',title='NB 2 still fails on Hobbies')
    axs[1].tick_params(axis='x',labelsize=8)
    fig.tight_layout(pad=.8,w_pad=1.5)
    fig.savefig(out/'observable_moment_diagnostic.pdf');fig.savefig(out/'observable_moment_diagnostic.png',dpi=180);plt.close(fig)
    caption=r'''\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{figures/observable_moment_diagnostic.pdf}
\caption{Sales-and-capacity likelihood controls on the consumed development shadow period. Left: the pooled implied and realized ratios are shown at each model's saved threshold, with row coverage under the model label. Shape 0.5 has no accepted mass; Poisson and NB 10 occupy separate columns despite identical realized full-acceptance results. Right: NB 2 category risks. Points are three-fit means; bars span the fit range, not confidence intervals.}
\label{fig:observablecapacity}
\end{figure}
'''
    (out/'observable_diagnostic_figure.tex').write_text(caption)
    print('Created moment table, manuscript paragraphs, and PDF/PNG diagnostic figure.')

if __name__=='__main__':main()
