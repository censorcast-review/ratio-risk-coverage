from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];E=R/'evidence/forward_controls'
def read(p):return json.loads((E/p).read_text())
m=read('m5/m5_RESULTS.json');f=read('fresh/fresh_RESULTS.json');nm=read('noq_m5/RESULTS.json');nf=read('noq_fresh/RESULTS.json')
def table(name,caption,label,cols,header,rows):
 t='\\begin{table}[htbp]\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\centering\\small\n\\begin{tabular}{'+cols+'}\\toprule\n'+header+' \\\\ \\midrule\n'+'\n'.join(' & '.join(row)+' \\\\' for row in rows)+'\n\\bottomrule\\end{tabular}\n\\end{table}\n';(R/'paper'/name).write_text(t)
rows=[]
for k,n in [('base','Group scale'),('forecast','Forecast-value bins'),('rc','RC (eight risk bins)'),('top','Single top-risk uplift'),('merged','RC + minimum support'),('stack','Forward-time learned ($q$)'),('noq','Forward-time learned (no $q$)')]:
 vals=[m['metrics'][k]['wape'],f['eligible']['metrics'][k]['wape'],f['all']['metrics'][k]['wape']] if k!='noq' else [nm['all']['wape'],nf['eligible']['wape'],nf['all']['wape']]
 rows.append([n,*[f'{v:.5f}' for v in vals]])
table('forward_main_table.tex','Matched post-calibration comparisons. M5 uses the primary seed; FreshRetailNet columns share the original frozen cohort but the new comparisons are retrospective. Lower WAPE is better. The learned calibrators are separately tuned on chronological validation blocks; no evaluation block selects them.','tab:forward-main','lrrr','Calibrator & M5 Later & FRN eligible & FRN all sales',rows)
rows=[]
for k,n in [('rc_minus_forecast','RC $-$ forecast bins'),('rc_minus_top','RC $-$ top uplift'),('rc_minus_stack','RC $-$ learned'),('merged_minus_base','Support rule $-$ group')]:
 for u,un in [('series','Series'),('store','Store'),('product','Product'),('day','Date')]:
  v=f['eligible']['contrasts'][k][u];rows.append([n,un,f"{v['estimate']:.5f}",'$['+','.join(f'{x:.5f}' for x in v['ci95'])+']$'])
table('forward_ci_table.tex','FreshRetailNet eligible-row paired differences, with 4,000 cluster bootstrap draws and seed 20260910. Negative favors the first method. Intervals are marginal, conditional on fitted policies, and do not account for adaptive research history. Full recorded-sales counterparts are distributed in JSON.','tab:forward-ci','llrl','Contrast & Unit & Difference & 95\\% interval',rows)
rows=[]
for tag,n in [('m5','M5'),('fresh','FRN')]:
 for pre,label in [('', 'With $q$'),('noq_', 'Without $q$')]:
  path=f'{tag}/{tag}_SELECTION.json' if not pre else f'noq_{tag}/SELECTION.json';d=read(path);b=d['selected'];rows.append([n,label,str(b['depth']),str(b['trees']),f"{b['wape']:.5f}"])
table('forward_selection_table.tex','Chronological validation selections from identical six-setting grids, including parent scaling as fallback. Each family is selected independently; no-$q$ is a separately disclosed adaptive follow-up. These are validation, not evaluation, WAPE values.','tab:forward-selection','llrrr','Dataset & Learned inputs & Depth & Trees & Validation WAPE',rows)
rows=[['M5','Product',f"{nm['noq_minus_q']['estimate']:.5f}",'$['+','.join(f'{x:.5f}' for x in nm['noq_minus_q']['ci95'])+']$']]
for unit,n in [('series','Series'),('store','Store'),('product','Product'),('day','Date')]:
 d=nf['eligible']['noq_minus_q'][unit];rows.append(['FRN eligible',n,f"{d['estimate']:.5f}",'$['+','.join(f'{x:.5f}' for x in d['ci95'])+']$'])
table('forward_noq_table.tex','Learned calibrator without $q$ minus learned calibrator with $q$. The M5 difference favors learned risk; on natural data the smaller difference includes zero under product resampling. Both versions retain history and observable availability features.','tab:forward-noq','llrl','Dataset & Unit & Difference & 95\\% interval',rows)
print('Built 4 forward-control tables')
