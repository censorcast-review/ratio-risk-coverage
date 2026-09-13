"""Build paper tables and a finite-menu figure from the complete audit output."""
from pathlib import Path
import json, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1]; E=R/'evidence/policy_choice'
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=R/'reproduction_outputs/policy_choice_assets')
P=parser.parse_args().output
(P/'figures').mkdir(parents=True,exist_ok=True)
z=json.loads((E/'RESULTS.json').read_text()); cap=.85*.7530939208
names={'group_l1':'M5: L1','learned_l1':'M5: residual','chronos2_univariate_raw':'M5: Chronos-2','bike':'Bike Sharing'}
short={'row':'Excess','demand':'Ratio','mixed':'Mixed','weight_descending':'Exposure descending','error':'Error'}
primary=[x for x in z['menus'] if (x['model']=='bike' and x['factor']==.85) or (x['model']!='bike' and x['cap']==cap)]

lines=[r'\begin{table}[t]',r'\centering\small',
 r'\caption{Coverage changes the selected rule. Thresholds and the case/exposure choices use calibration outcomes only. Differences are exposure-choice minus case-choice on evaluation data (percentage points). Intervals are paired cluster-bootstrap 99.375\% intervals, adjusted across the eight primary coverage contrasts; all eight exclude zero. These additional menu comparisons are retrospective.}',
 r'\label{tab:policy-choice}',r'\begin{tabular}{llrr}',r'\toprule',
 r'Setting & Selected by cases / exposure & $\Delta c$ & $\Delta d$ [adjusted interval] \\',r'\midrule']
for x in primary:
    q=x['contrast'];lo,hi=q['ci99_375'][1];a,b=(x['selected'][o] for o in ('c','d'))
    lines.append(f"{names[x['model']]} & {short[a]} / {short[b]} & {100*q['dc']:.2f} & {100*q['dd']:.2f} [{100*lo:.2f}, {100*hi:.2f}] \\")
    lines[-1]+='\\'
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(P/'policy_choice_table.tex').write_text('\n'.join(lines)+'\n')

lines=[r'\begin{table}[t]',r'\centering\small',
 r'\caption{Two-window risk screen on M5. Thresholds use A with design cap $0.95r$; a separate B item-bootstrap check tests signed excess at reporting cap $r=0.64013$, with Bonferroni adjustment over all 15 candidates. All six selected policies pass that approximate B screen. Later risk intervals below are pointwise 95\%, conditional on fits and selected policies.}',
 r'\label{tab:policy-screen}',r'\begin{tabular}{lllrrl}',r'\toprule',
 r'Predictor & Utility & Rule & $c$ (\%) & $d$ (\%) & Later risk [95\% interval] \\',r'\midrule']
for x in z['screens']:
    for obj in ('c','d'):
        m=x['selected'][obj]; t=x['test'][m]; lo,hi=t['risk_ci95']
        lines.append(f"{names[x['model']]} & {'Cases' if obj=='c' else 'Exposure'} & {short[m]} & {100*t['c']:.2f} & {100*t['d']:.2f} & {t['risk']:.4f} [{lo:.4f}, {hi:.4f}] \\")
        lines[-1]+='\\'
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(P/'policy_screen_table.tex').write_text('\n'.join(lines)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(7.1,2.75),layout='constrained')
colors={'row':'#2471A3','mixed':'#178571','demand':'#A24B95','weight_descending':'#BE681A'}
markers={'row':'o','mixed':'s','demand':'^','weight_descending':'D'}
for ax,model in zip(axs,['learned_l1','bike']):
    rec=next(x for x in primary if x['model']==model)
    for m in colors:
        t=rec['test'][m]; ax.scatter(100*t['c'],100*t['d'],s=55,color=colors[m],marker=markers[m],zorder=3,label=short[m])
        selected=''
        if m==rec['selected']['c']: selected='Selected for cases'
        if m==rec['selected']['d']: selected='Selected for exposure'
        if selected:
            offset=(-6,-18) if m=='row' else (-6,9)
            align='right'
            if model=='learned_l1' and m==rec['selected']['d']:
                offset=(5,8); align='left'
            ax.annotate(selected,(100*t['c'],100*t['d']),xytext=offset,textcoords='offset points',ha=align,fontsize=8,fontweight='bold')
    ax.set_title(names[model],fontsize=10);ax.set_xlabel('Case coverage (%)');ax.set_ylabel('Exposure coverage (%)');ax.grid(alpha=.18)
    ax.set_xlim(32,98); ax.set_ylim((78,97) if model=='learned_l1' else (68,97))
    ax.text(.03,.04,'Error ranking: no feasible\ncalibration threshold',transform=ax.transAxes,fontsize=7.5,color='#555555')
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=8)
fig.savefig(P/'figures/policy_choice.pdf',bbox_inches='tight');plt.close(fig)

# All candidate outcomes and all cap settings remain machine-readable; the
# appendix table makes the change of selected family visible without searching JSON.
lines=[r'\begin{table}[t]',r'\centering\scriptsize',
 r"\caption{Declared cap sensitivity of utility-based menu selection. M5 uses absolute caps; Bike lists multipliers of full calibration risk. Changes are evaluation exposure-choice minus case-choice, in percentage points. ``All'' denotes an identical accept-all policy, regardless of its tie-breaking method label. These are descriptive additional comparisons.}",
 r'\label{tab:choice-caps}',r'\begin{tabular}{lllr r}',r'\toprule',r'Setting & Cap / multiplier & Case / exposure rule & $\Delta c$ & $\Delta d$ \\',r'\midrule']
for x in z['menus']:
    a,b=(x['selected'][o] for o in ('c','d'));q=x['contrast'];c=x.get('factor',x['cap'])
    choice='All / All' if a==b and x['policies'][a]['threshold']=='all' else short.get(a,'None')+' / '+short.get(b,'None')
    dc,dd=('--','--') if q is None else (f"{100*q['dc']:.2f}",f"{100*q['dd']:.2f}")
    lines.append(f"{names[x['model']]} & {c:.3f} & {choice} & {dc} & {dd} \\");lines[-1]+='\\'
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(P/'policy_choice_caps.tex').write_text('\n'.join(lines)+'\n')
print('Built policy-choice tables and figure')
