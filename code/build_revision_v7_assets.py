"""Generate v7 tables from immutable saved audit outputs; no new computation."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]; P=R/'paper'
M=json.loads((R/'evidence/matched_censoring/RESULTS.json').read_text())
S=json.loads((R/'evidence/revision_v7_stability/RESULTS.json').read_text())
B=json.loads((R/'evidence/revision_v7_complete_screen/RESULTS.json').read_text())
F=json.loads((R/'evidence/revision_v7_complete_screen/FROZEN.json').read_text())
METHODS=['error','row','mixed','demand','weight_descending']
N={'error':'Error','row':'Excess','mixed':'Mixed','demand':'Ratio','weight_descending':'Descending','no_choice':'None'}
def table(file,cap,label,cols,head,rows,size='small'):
 t='\\begin{table}[t]\\centering\\'+size+'\n\\caption{'+cap+'}\\label{'+label+'}\n\\begin{tabular}{'+cols+'}\\toprule\n'+head+r'\\\midrule'+'\n'
 t+='\n'.join(' & '.join(str(x) for x in row)+r' \\' for row in rows)+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n'
 (P/file).write_text(t)
def pct(x): return f'{100*x:.2f}'
def val(x,n=4): return '--' if x is None else f'{x:.{n}f}'
def ci(x,scale=1,n=2):return '['+', '.join(f'{scale*v:.{n}f}' for v in x)+']'
rows=[]
for arm,label in [('censored','Censored'),('complete','Complete sales')]:
 for z in M[arm]['menus']:
  if z['kind']=='relative' and z['value']==.95:
   rows.append([label,f"{z['e_trees']}/{z['w_trees']}",N[z['selected']['c']]+' / '+N[z['selected']['d']],f"{100*z['contrast']['dc']:+.2f}",f"{100*z['contrast']['dd']:+.2f}",ci(z['contrast']['ci95'][1],100)])
table('matched_main_table.tex',r"All eight paired head settings at the protocol's primary relative cap (.95 of maximum full A/B calibration risk). Censored/complete caps are .6647/.6327 and full evaluation WAPE .6853/.6630. Differences are exposure-choice minus case-choice, in percentage points. Intervals are descriptive, fit-conditional 95\% item-bootstrap intervals. Descending means predicted exposure descending. 220/220 is a reference, not the sole prespecified head setting.",'tab:matched-history','ll l r r l',r'Arm & Trees $e/w$ & Case / exposure rule & $\Delta c$ & $\Delta d$ & $\Delta d$ interval',rows)
rows=[]
for z in B['plans']:
 rows.append([f"{z['e_trees']}/{z['w_trees']}",f"{100*z['contrast']['dc']:+.2f}",f"{100*z['contrast']['dd']:+.2f}",val(z['selected_policies']['c']['metrics']['risk_upper_family20'],5),val(z['selected_policies']['d']['metrics']['risk_upper_family20'],5)])
table('revision_v7_screen_main_table.tex',r'Complete-sales A-design/B-screen procedure at fixed reporting cap .63268. All four cells select Excess for cases and Descending for exposure. Coverage differences (pp) and upper risk summaries concern the previously evaluated Later window. $U_{20}$ is the approximate one-sided bootstrap upper quantile with tail .05/20; it is not a finite-sample guarantee.','tab:v7-screen-main','lrrrr',r'Trees $e/w$ & $\Delta c$ & $\Delta d$ & Case $U_{20}$ & Exposure $U_{20}$',rows)
rows=[]
for z in S['settings']:
 label=z['name'].replace('complete_','').replace('_','/') if z['name']!='bike' else 'Bike'
 for m in METHODS:
  ps=z['point']['policies'][m]; ca=ps['calibration']; feasible=ps['threshold'] is not None
  rows.append([label,N[m],pct(min(b['c'] for b in ca)),pct(min(b['d'] for b in ca)),val(max(b['risk'] for b in ca) if feasible else None,6),'yes' if feasible else 'no'])
table('revision_v7_calibration_table.tex',r'All five original calibration candidates in each complete-sales head cell and Bike. Coverage columns show the minimum over the original calibration windows (\%); Bike has one window. M5 reporting/design cap is .632683; Bike reporting cap is .132806 and design cap .126166. Error has no feasible threshold under the stated cap and .35 row floor; it remains in the menu record.','tab:v7-calibration','llrrrl',r'Trees $e/w$ & Rule & Min. case & Min. exposure & Max. risk & Feasible',rows,'scriptsize')
rows=[]
for z in S['settings']:
 label=z['name'].replace('complete_','').replace('_','/') if z['name']!='bike' else 'Bike'
 for mode in ['fixed','full']:
  if z[mode] is None:continue
  for obj in ['c','d']:
   st=z[mode]['selections'][obj]
   rows.append([label,'Fixed' if mode=='fixed' else 'Redesign',obj]+[f"{100*st['frequencies'][m]['frequency']:.2f}" for m in METHODS+['no_choice']])
table('revision_v7_frequency_table.tex',r'Selection frequency (\%) with fixed original thresholds (4,000 cluster draws) or all thresholds redesigned within each resample (200 draws). Error, Excess, Mixed, Ratio, Descending and no-choice columns are all retained. Both modes hold the fitted scores, reporting cap and $T$ fixed. M5 clusters are items shared across A/B; Bike clusters are days. Fixed-mode failures are eligibility checks on boundary thresholds, not full-procedure failure rates.','tab:v7-frequencies','lllrrrrrr',r'Setting & Mode & Utility & Error & Excess & Mixed & Ratio & Desc. & None',rows,'scriptsize')
rows=[]
for z in S['settings']:
 label=z['name'].replace('complete_','').replace('_','/') if z['name']!='bike' else 'Bike'
 for obj in ['c','d']:
  point=z['point']['statistics']['selections'][obj]['eligible_winner_runner_up_margin']['mean']
  fixed=z['fixed']['selections'][obj]['eligible_winner_runner_up_margin']
  rows.append([label,obj,pct(point),ci([fixed['q025'],fixed['q975']],100),fixed['n']])
table('revision_v7_margin_table.tex',r'Winner-minus-runner-up minimum-calibration-coverage margins (pp), among eligible candidates. The fixed-threshold interval is the empirical central 95\% resampling range conditional on at least two eligible candidates; $n$ gives that denominator out of 4,000. It is nonnegative by definition and does not measure whether a named pair switches order. Signed pair margins are reported in the text and machine-readable results.','tab:v7-margins','llrlr',r'Setting & Utility & Original margin & Fixed resampling range & $n$',rows)
rows=[]
for p in F['plans']:
 for m in METHODS:
  z=p['policies'][m]
  rows.append([f"{p['e_trees']}/{p['w_trees']}",N[m],pct(z['A']['c']),pct(z['A']['d']),val(z['B']['risk'],5),val(z['B_adjusted_excess_upper'],3),'yes' if z['B_screen_pass'] else 'no'])
table('revision_v7_B_table.tex',r'Complete 20-candidate B screen. $U^{g}_{20}$ is the .9975 quantile of resampled mean signed excess per item, using reporting cap .632683 and A design cap .601049. Passing requires a defined A candidate, positive B exposure and $U^{g}_{20}\leq0$. All 16 A-feasible candidates pass; B removes no additional candidate. Error has undefined risk and is not eligible even though its empty excess is zero.','tab:v7-B','llrrrrl',r'Trees $e/w$ & Rule & A case (\%) & A exposure (\%) & B risk & $U^g_{20}$ & Pass',rows,'scriptsize')
rows=[]
for z in B['summary']:
 rows.append([z['cell'].replace('e','').replace('_w','/'),z['utility'],f"{z['case_percent']:.2f}",f"{z['exposure_percent']:.2f}",val(z['risk'],5),ci(z['risk_ci95'],1,5),val(z['risk_upper_selected8'],5),f"{z['case_cost_pp']:.2f}",f"{z['exposure_cost_pp']:.2f}"])
table('revision_v7_cost_table.tex',r'All eight selected policies on Later. Case utility selects Excess; exposure utility selects Descending. Costs are unscreened minus A-design/B-screen coverage (pp), relative to each cell\textquotesingle s original utility-selected policy. Risk intervals and $U_8$ (tail .05/8) are approximate and fit-conditional. These shared-head policies are not eight independent replications.','tab:v7-cost','llrrrlrrr',r'Trees & Utility & Case \% & Exp. \% & Risk & Pointwise 95\% CI & $U_8$ & Cost $c$ & Cost $d$',rows,'scriptsize')
rows=[]
for z in B['plans']:
 rows.append([f"{z['e_trees']}/{z['w_trees']}",f"{100*z['contrast']['dc']:+.2f}",ci(z['contrast']['ci95'][0],100),f"{100*z['contrast']['dd']:+.2f}",ci(z['contrast']['ci95'][1],100)])
table('revision_v7_contrast_table.tex',r'A-design/B-screen Later contrasts and descriptive central 95\% paired item-bootstrap intervals, in percentage points. These compare the exposure-selected and case-selected policies after the full procedure.','tab:v7-contrast','lrlrl',r'Trees $e/w$ & $\Delta c$ & Interval & $\Delta d$ & Interval',rows)
print('Generated 8 v7 data tables')
