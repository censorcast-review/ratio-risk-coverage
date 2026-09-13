"""Create manuscript assets from the matched audit; never writes evidence."""
from pathlib import Path
import json,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
def main(out):
 out.mkdir(parents=True,exist_ok=True);res=json.loads((R/'evidence/matched_censoring/RESULTS.json').read_text());names={'censored':'Artificially censored','complete':'Complete recorded sales'};short={'row':'Excess','demand':'Ratio','mixed':'Mixed','error':'Error','weight_descending':'Exposure descending',None:'None'}
 lines=[];heads=[];detail=[]
 for arm,a in res.items():
  for row in a['menus']:
   c,d=row['selected']['c'],row['selected']['d'];co=row['contrast'];dc='--' if co is None else f"{100*co['dc']:+.2f}";dd='--' if co is None else f"{100*co['dd']:+.2f}";pair=short[c]+' / '+short[d]
   if row['e_trees']==row['w_trees']==220:
    detail.append(f"{names[arm]} & {row['kind']} {row['value']:.3f} & {row['cap']:.4f} & {pair} & {dc} & {dd} "+r'\\')
    if row['kind']=='relative' and row['value']==.95:
     ci='--' if co is None else f"[{100*co['ci95'][1][0]:.2f}, {100*co['ci95'][1][1]:.2f}]"
     lines.append(f"{names[arm]} & {a['full']['risk']:.4f} & {row['cap']:.4f} & {dc} & {dd} & {ci} "+r'\\')
   if row['kind']=='relative' and row['value']==.95:
    e=a['head_diagnostics']['e'+str(row['e_trees'])]['mse'];w=a['head_diagnostics']['w'+str(row['w_trees'])]['mse'];heads.append(f"{names[arm]} & {row['e_trees']} / {row['w_trees']} & {e:.3f} & {w:.3f} & {pair} & {dc} & {dd} "+r'\\')
 (out/'matched_main_table.tex').write_text(r'''\begin{table}[t]\centering\small
\caption{Paired history intervention. Identical feature definitions and training budgets; capacity, fill and hit features are excluded from both arms. The primary cap is .95 times each arm's maximum full-calibration risk. Coverage differences are exposure-choice minus case-choice, in percentage points; intervals are descriptive 95\% paired item-bootstrap intervals. These are new fits on previously evaluated development data.}\label{tab:matched-history}
\begin{tabular}{lrrrrl}\toprule
History and fit target & Full WAPE & Cap & $\Delta c$ & $\Delta d$ & $\Delta d$ interval\\\midrule
'''+ '\n'.join(lines)+r'\bottomrule\end{tabular}\end{table}'+'\n')
 (out/'matched_head_table.tex').write_text(r'''\begin{table}[t]\centering\scriptsize
\caption{Untuned tree-count sensitivity at each arm's primary cap. Error/exposure head tree counts vary factorially, with the point predictor fixed; tree counts were not selected by validation or early stopping. Larger heads are not uniformly more accurate, so these comparisons do not test whether improved moment estimates resolve the choice. MSE is evaluated against realized error or sales, not observed conditional moments.}\label{tab:matched-heads}
\begin{tabular}{llrrlrr}\toprule
Arm & Trees $e/w$ & Error MSE & Exposure MSE & Case / exposure rule & $\Delta c$ & $\Delta d$\\\midrule
'''+ '\n'.join(heads)+r'\bottomrule\end{tabular}\end{table}'+'\n')
 (out/'matched_caps_table.tex').write_text(r'''\begin{table}[t]\centering\scriptsize
\caption{All declared cap settings for the two refitted arms, using the 220-tree heads. Relative caps use the arm-specific maximum full A/B calibration risk. A missing contrast means no eligible policy, not equivalence.}\label{tab:matched-caps}
\begin{tabular}{llrlrr}\toprule
Arm & Cap specification & Risk cap & Case / exposure rule & $\Delta c$ & $\Delta d$\\\midrule
'''+ '\n'.join(detail)+r'\bottomrule\end{tabular}\end{table}'+'\n')
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(1,2,figsize=(8,3.65))
 tags={'row':'X','mixed':'M','demand':'R','weight_descending':'D','error':'E',None:'--'}
 def rule_tag(row,objective):
  method=row['selected'][objective]
  # At full coverage all families tie; label the actual accept-all policy,
  # rather than implying that the deterministic Error tie-break is active.
  if method is not None and row['policies'][method]['threshold']=='all':return 'A'
  return tags[method]
 for arm,col,marker in [('censored','#b45d31','o'),('complete','#197879','x')]:
  rows=sorted((x for x in res[arm]['menus'] if x['kind']=='relative' and x['e_trees']==x['w_trees']==220),key=lambda row:row['value'])
  x=[v['value'] for v in rows]
  for j,key in enumerate(['dc','dd']):
   y=[100*v['contrast'][key] if v['contrast'] is not None else np.nan for v in rows]
   # A hollow circle and an x remain visible when the two arms coincide.
   ax[j].plot(x,y,marker=marker,color=col,label=names[arm],markersize=6,
              markerfacecolor='none',markeredgewidth=1.3,linewidth=1.3,
              linestyle='-' if arm=='censored' else '--')
   for row,xx,yy in zip(rows,x,y):
    if not np.isfinite(yy):continue
    label=rule_tag(row,'c')+'/'+rule_tag(row,'d')
    if row['value']>=1.:
     offset=10 if arm=='censored' else -15
    elif j==1:
     offset=-15 if arm=='censored' else 10
    elif arm=='censored':
     offset=10 if row['value']==.85 else -15
    else:
     offset=-15 if row['value']==.85 else 10
    ax[j].annotate(label,(xx,yy),xytext=(0,offset),textcoords='offset points',
                   color=col,fontsize=9,ha='center',va='center',
                   bbox={'facecolor':'white','edgecolor':'none','pad':.6,'alpha':.92})
 for j,label in enumerate(['Case coverage difference (pp)','Exposure coverage difference (pp)']):
  ax[j].set(xlabel='Cap / full calibration risk',ylabel=label,xlim=(.831,1.069),xticks=[.85,.9,.95,1.,1.05])
  ax[j].axhline(0,color='gray',lw=.7,zorder=0);ax[j].grid(alpha=.15)
 ax[0].set_ylim(-48,8);ax[1].set_ylim(-4.5,23)
 fig.legend(*ax[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,fontsize=9.5,bbox_to_anchor=(.53,1.))
 fig.text(.5,.082,'Labels show case / exposure rule: X = Excess; M = Mixed; R = Ratio;',ha='center',fontsize=8.5)
 fig.text(.5,.039,'D = Exposure descending; A = accept-all (both objectives).',ha='center',fontsize=8.5)
 fig.subplots_adjust(left=.085,right=.988,bottom=.255,top=.865,wspace=.28)
 fig.savefig(out/'matched_history.pdf');plt.close(fig)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=R/'reproduction_outputs/matched_assets');x=a.parse_args();main(x.output)
