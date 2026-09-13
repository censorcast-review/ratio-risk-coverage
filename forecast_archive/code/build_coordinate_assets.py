"""Render new tables exclusively from executed, saved experiment results."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];d=json.loads((R/'evidence/coordinate_tests/analysis/RESULTS.json').read_text())
def table(name,caption,label,cols,header,rows):
 t='\\begin{table}[htbp]\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\centering\\small\n\\begin{tabular}{'+cols+'}\\toprule\n'+header+' \\\\ \\midrule\n'+'\n'.join(' & '.join(row)+' \\\\' for row in rows)+'\n\\bottomrule\\end{tabular}\n\\end{table}\n';(R/'paper'/name).write_text(t)
rows=[];ci=[]
for key,name in [('chronos_bolt_small','Chronos-Bolt Small'),('chronos2_univariate','Chronos-2 univariate'),('chronos2_covariates','Chronos-2 covariates')]:
 a=d['foundation'][key];m=a['metrics'];gain=100*(1-m['risk']['wape']/m['base']['wape'])
 rows.append([name,f"{a['raw']['wape']:.5f}",*[f"{m[k]['wape']:.5f}" for k in ['base','forecast','risk']],f'{gain:.2f}\\%'])
 for k,control in [('risk_minus_base','Group scale'),('risk_minus_forecast','Forecast bins')]:
  v=a[k];ci.append([name,control,f"{v['estimate']:.5f}",'$['+','.join(f'{x:.5f}' for x in v['ci95'])+']$'])
table('coordinate_foundation_table.tex','Within-backbone calibration of frozen forecasts on M5 Later. All three calibration methods use block A targets (days 1555--1673); RC and forecast bins use fixed eight-bin, .75 shrinkage. Gain is relative to group scaling. This retrospective extension uses a different calibration interval from Table~\\ref{tab:main}; rows are not a matched ranking against the primary tree model.','tab:foundation-layer','lrrrrr','Frozen backbone & Raw & Group & Forecast bins & RC & Gain',rows)
table('coordinate_foundation_ci.tex','Paired product-bootstrap intervals for the fixed foundation-calibration contrasts (1,829 products, 4,000 draws). These intervals condition on the fitted policies and observed window; they are not simultaneous or post-selection guarantees.','tab:foundation-ci','llrl','Backbone & Comparator & RC minus comparator & 95\\% interval',ci)
rows=[]
for k,name in [('original','Full inputs'),('risk_input_only','Remove from risk only'),('both_inputs','Remove from both heads')]:
 a=d[k];v=a['risk_minus_forecast'];rows.append([name,*[f"{a['metrics'][s]['wape']:.5f}" for s in ['base','forecast','risk']],'$['+','.join(f'{x:.5f}' for x in v['ci95'])+']$'])
table('coordinate_capacity_table.tex','Direct-capacity-input removal, primary-seed M5 Later. Removed inputs are origin capacity, origin fill ratio, and capacity/28-day mean. Historical hit rates and the mechanical process remain. Interval: RC minus forecast bins, paired product bootstrap.','tab:capacity-removal','lrrrl','Inputs & Group & Forecast bins & RC & 95\\% interval',rows)
rows=[]
for k,name in [('base','Group scale'),('forecast','Forecast bins'),('risk','RC (geometric)'),('arithmetic','Arithmetic shrinkage'),('ess_arithmetic','Weight-ESS arithmetic'),('isotonic32','32-bin L1 isotonic')]:
 rows.append([name,f"{d['validation_half']['metrics'][k]['wape']:.5f}",f"{d['original']['metrics'][k]['wape']:.5f}",f"{d['validation_half']['categories']['HOBBIES'][k]['wape']:.5f}",f"{d['original']['categories']['HOBBIES'][k]['wape']:.5f}"])
table('coordinate_controls_table.tex','Fixed post-calibration controls. Validation fits through day 1373 and scores 1374--1433; Later policies refit on all original validation days. The arithmetic alternatives improve Later but lose to RC on held-out validation and are not selected as replacements.','tab:coordinate-controls','lrrrr','Method & Val. pooled & Later pooled & Val. Hobbies & Later Hobbies',rows)
rows=[[tau,*[f"{a[k]:.5f}" for k in ['base','forecast','risk']]] for tau,a in d['asymmetric'].items()]
table('coordinate_pinball_table.tex','Normalized pinball loss on M5 Later; lower is better. Each policy is calibrated for its own fixed quantile. Forecast-value bins outperform risk bins at both asymmetric loss ratios. No inventory simulation or realized profit is measured.','tab:pinball','lrrr','$\\tau$ & Group scale & Forecast bins & Risk bins',rows)
rows=[]
for k,name in [('series_id','Store--product'),('store','Store'),('product','Product'),('dt','Date')]:
 a=d['fresh_all_clusters'][k];rows.append([name,str(a['clusters']),'$['+','.join(f'{x:.5f}' for x in a['ci95'])+']$'])
table('coordinate_fresh_all_ci.tex','FreshRetailNet all-recorded-sales sensitivity, including stockout rows. RC minus base is $-.00129$; all four 95\\% intervals include zero (4,000 draws, seed 20260909). This secondary target is recorded sales, not latent demand.','tab:fresh-all-ci','lrl','Resampling unit & Clusters & 95\\% interval',rows)
print('Created six tables from RESULTS.json')
