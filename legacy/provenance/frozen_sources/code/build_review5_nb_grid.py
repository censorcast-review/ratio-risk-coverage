"""Compact fine-grid evidence table and diagnostic figure."""
from pathlib import Path
import io,json
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from r2_io import write
ROOT=Path(__file__).resolve().parents[1]
def main():
 r=json.loads((ROOT/'results/review5_nb_grid/RESULTS.json').read_text());recs=sorted(r['records'],key=lambda z:z['shape']);sel=r['selected_shape_by_observed_validation']
 def fmt(v,d=4):return '--' if v is None else f'{v:.{d}f}'
 rows=[]
 for z in recs:
  m=z['metrics']['shadow']['ALL'];k=f"{z['shape']:g}"+(' *' if z['shape']==sel else '')
  vals=[k,fmt(z['observed_validation_nll'])]
  for block in ['calibration_a','calibration_b','shadow']:
   mm=z['metrics'][block]['ALL'];vals.append(fmt(mm['implied_wape'])+' / '+fmt(mm['wape']))
  vals.append(fmt(100*m['row_coverage'],2));rows.append(' & '.join(vals)+r' \\')
 text=r'''\begin{table}[t]
\centering\small
\caption{NB fine grid on consumed development data, first seed. Each pair is implied / realized WAPE at the model-moment threshold calibrated on A/B. The star minimizes observed validation NLL among $\{1,1.5,2,3,5\}$; $0.5$ and $10$ are retained anchors. Dashes denote no admissible threshold, not zero WAPE. Rows are development-shadow coverage.}
\label{tab:nb-fine-grid}
\begin{tabular}{rrrrrr}
\toprule
$\kappa$ & Valid. NLL & Calibration A & Calibration B & Shadow & Rows (\%)\\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table}
'''
 write(ROOT/'paper/tables/nb_fine_grid.tex',text.encode())
 ks=[z['shape'] for z in recs];x=np.arange(len(ks))
 im=[z['metrics']['shadow']['ALL']['implied_wape'] for z in recs];real=[z['metrics']['shadow']['ALL']['wape'] for z in recs]
 co=[z['matched_supervised']['shadow']['exact_count_metrics']['ALL']['wape'] for z in recs];cv=[100*z['metrics']['shadow']['ALL']['row_coverage'] for z in recs]
 with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False}):
  fig,ax=plt.subplots(1,2,figsize=(6.5,2.5),gridspec_kw={'width_ratios':[1.2,1]})
  for yy,label,color,marker in [(im,'NB implied','#d97706','s'),(real,'NB realized','#156082','o')]:
   ax[0].plot(x,[np.nan if v is None else v for v in yy],label=label,color=color,marker=marker,ms=4,lw=1.2)
  missing=[i for i,z in enumerate(recs) if z['metrics']['shadow']['ALL']['accepted_n']==0]
  if missing:
   ax[0].plot(missing,[.06]*len(missing),'x',color='#666666',ms=5,transform=ax[0].get_xaxis_transform(),label='No admissible threshold')
   ax[0].text(np.mean(missing),.52,'WAPE undefined',color='#666666',ha='center',fontsize=6.7)
  ax[0].set_xticks(x,[f'{k:g}'+('*' if k==sel else '') for k in ks]);ax[0].set_xlabel(r'NB shape $\kappa$')
  ax[0].set_title('(a) Dispersion sensitivity',loc='left',fontsize=8);ax[0].legend(fontsize=6.6,loc='upper left');ax[0].set_ylabel('Shadow WAPE')
  good=[i for i,v in enumerate(real) if v is not None]
  ax[1].plot([cv[i] for i in good],[real[i] for i in good],color='#156082',marker='o',ms=4,lw=1.2,label='NB realized')
  ax[1].plot([cv[i] for i in good],[co[i] for i in good],color='#6b7280',marker='^',ms=4,lw=1.2,ls='--',label='Matched supervised')
  for i in good:
   ax[1].plot([cv[i],cv[i]],[real[i],co[i]],color='#999999',lw=.6)
   offset=(-25,-14) if ks[i]==5 else ((-58,8) if ks[i]==10 else (3,5))
   label=(r'$\kappa=10$: all' if ks[i]==10 else f'$\\kappa={ks[i]:g}$')
   ax[1].annotate(label,(cv[i],real[i]),xytext=offset,textcoords='offset points',fontsize=6.5)
  ax[1].set_xlim(40,104);ax[1].set_xlabel('Matched row coverage (%)');ax[1].set_title('(b) Identical-forecast ranking',loc='left',fontsize=8);ax[1].legend(fontsize=6.7,loc='upper left')
  for a in ax:
   a.axhline(.85*.7530939208313988,color='#991b1b',ls=':',lw=1);a.set_ylim(.50,.71);a.grid(axis='y',alpha=.15)
  fig.tight_layout(pad=.7)
  for suffix in ['pdf','png']:
   b=io.BytesIO();fig.savefig(b,format=suffix,dpi=220,bbox_inches='tight');write(ROOT/f'paper/figures/nb_fine_grid.{suffix}',b.getvalue())
  plt.close(fig)
 figure=r'''\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/nb_fine_grid.pdf}
\caption{First-seed consumed-development diagnostics. Left: likelihood-family sensitivity at model-implied A/B thresholds; crosses mark an undefined WAPE because no admissible threshold exists. Right: at each nonempty NB row count, the supervised excess ranking uses the identical fixed forecast and exactly matches row coverage through an outcome-independent, fixed-seed boundary-tie draw. Matching is descriptive, not a transferred policy. Dotted lines show the WAPE cap; stars mark the validation-likelihood choice.}
\label{fig:nb-fine-grid}
\end{figure}
'''
 write(ROOT/'paper/tables/nb_fine_grid_figure.tex',figure.encode())
 selected=next(z for z in recs if z['shape']==sel);m=selected['metrics']['shadow']['ALL'];c=selected['matched_supervised']['shadow']['exact_count_metrics']['ALL']
 paragraph=(r'\paragraph{Dispersion and coverage-matched control.} A finer first-seed grid $\kappa\in\{1,1.5,2,3,5\}$ selects '+f'${sel:g}$'+r' by sales-and-capacity validation likelihood; the $0.5$ and $10$ fits remain descriptive anchors (Table~\ref{tab:nb-fine-grid}). The likelihood choice is unchanged, but acceptance is shape-sensitive: the tested shapes $\kappa\in\{0.5,1,1.5\}$ have no admissible threshold, $\kappa=3$ reaches 79.48\% rows with shadow WAPE 0.6311 but fails the realized A/B caps, and $\kappa=5$ reaches 99.41\% with WAPE 0.6748. '
  +f"At the selected threshold, shadow implied and realized WAPE are {fmt(m['implied_wape'])} and {fmt(m['wape'])}; row coverage is {100*m['row_coverage']:.2f}\\%. "
  +r'To isolate ranking from forecast quality, we fit a 220-tree supervised excess head to the same fixed unadapted forecast, training rows, and origin features. '
  +f"Truncating its ranking to the exact NB row count gives WAPE {fmt(c['wape'])} and demand coverage {100*c['demand_coverage']:.2f}\\% (NB: {100*m['demand_coverage']:.2f}\\%). "
  +r'The boundary is sampled uniformly with a fixed seed, independent of outcomes; fractional-boundary expectations are also reported. Matching uses the NB count separately in each block, so this is a descriptive equal-coverage comparison, not a deployed threshold.'+'\n')
 write(ROOT.parent.parent/'tmp/review5_nb_text.tex',paragraph.encode());print(text);print(paragraph)
if __name__=='__main__':main()
