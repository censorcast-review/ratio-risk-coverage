"""Build v8 tables solely from the locked retrospective results."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];P=R/'paper'
def read(n):return json.loads((R/f'evidence/revision_v8_{n}/RESULTS.json').read_text())
T=read('tolerance');F=read('floor');E=read('exchange')
FF=json.loads((R/'evidence/revision_v8_floor/FROZEN.json').read_text())
N={'error':'Error','row':'Excess','mixed':'Mixed','demand':'Ratio','weight_descending':'Descending','no_choice':'None'}
Ms=list(N)[:-1]
def heads(z):return '/'.join(map(str,z['heads'])) if 'heads' in z else f"{z['e_trees']}/{z['w_trees']}"
def arm(z):return 'Complete' if z['arm']=='complete' else 'Censored'
def num(x,n=3):return '--' if x is None else f'{x:.{n}f}'
def ci(v,n=3):return '['+', '.join(num(x,n) for x in v)+']'
def table(name,caption,label,cols,header,rows,size='small'):
 out='\\begin{table}[t]\\centering\\'+size+'\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{'+cols+'}\\toprule\n'+header+r'\\\midrule'+'\n'
 out+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n';(P/name).write_text(out)
rows=[]
for floor in F['floors']:
 ps=[z for z in F['plans'] if z['floor']==floor];cs=[z['by_family']['60']['contrast'] for z in ps]
 upl=max(v['metrics']['risk_upper_family60'] for z in ps for v in z['by_family']['60']['selected_policies'].values())
 rows.append([f'{floor:.2f}',N[ps[0]['selected']['c']]+'/'+N[ps[0]['selected']['d']],ci([100*min(c['dc'] for c in cs),100*max(c['dc'] for c in cs)],2),ci([100*min(c['dd'] for c in cs),100*max(c['dd'] for c in cs)],2),f'{upl:.5f}'])
table('revision_v8_floor_main_table.tex',r'Complete-sales A-design/B-screen floor sensitivity, all four head pairs per row. Ranges show Later exposure-choice minus case-choice coverage (pp), not confidence intervals. $U_{60}$ is the largest approximate upper risk summary among selected conditions, adjusted over all 60 candidate conditions. Fixed reporting cap is .632683; the same selection occurs under the inherited 20-candidate adjustment.','tab:v8-floor-main','llllr',r'Floor & Case/exposure rules & $\Delta c$ range & $\Delta d$ range & Max. $U_{60}$',rows)
rows=[[arm(z),heads(z),f"{z['exchange_ratio']:.4f}",ci(z['ratio_ci95'],4)] for z in E['exchange_cells']]
table('revision_v8_exchange_table.tex',r'All eight original policy exchanges $\Delta d/(-\Delta c)$ at each arm\textquotesingle s primary relative cap. Central 95\% percentile intervals use 4,000 item resamples and condition on fits, caps, thresholds and choices. All case-loss denominators are positive in every draw; these intervals omit calibration reselection.','tab:v8-exchange','llrl',r'Arm & Heads $e/w$ & Exchange & Conditional interval',rows)
rows=[[heads(z),num(z['complete_minus_censored'],4),ci(z['ci95'],4),ci(z['four_contrast_bonferroni_interval'],4)] for z in E['paired_arm_contrasts']]
table('revision_v8_exchange_difference_table.tex',r'Complete-minus-censored exchange differences, with shared item multiplicities across arms. The four-contrast intervals use empirical .00625/.99375 quantiles. Adjustment gives approximate conditional summaries, not simultaneous population certification or a causal effect of censoring.','tab:v8-exchange-difference','lrll',r'Heads $e/w$ & Difference & Pointwise interval & Four-contrast interval',rows)
rows=[]
for z in E['calibration']:
 rows.append([arm(z),num(z['relative_factor'],2),z['eligible_count'],N[z['exposure_winner']], 'yes' if z['ratio_feasible'] else 'no','yes' if z['descending_feasible'] else 'no',num(z['ratio_minus_descending_pp'],4),num(z['ratio_minus_strongest_other_pp'],4)])
table('revision_v8_strict_cap_table.tex',r'Original 220/220 calibration menus at three relative caps. Margins are minimum-calibration-exposure differences in pp. R/D denote Ratio/Descending. Undefined comparisons are shown as -- when a candidate is infeasible; zero accepted mass is not a feasible competing policy. The strongest feasible other candidate is Mixed at .85/.90 and Descending at .95.','tab:v8-strict','lrrlllrr',r'Arm & Cap factor & Eligible & Winner & R feasible & D feasible & R--D & R--Other',rows,'scriptsize')
rows=[]
for z in T['settings']:
 b=next(b for b in z['bands'] if b['epsilon']==.005);v=b['evaluation'];chg=v['change_vs_original_exposure'];gap=v['contrast_vs_case']
 rows.append([arm(z),heads(z),N[z['original_selected']['d']]+'/'+N[b['selected']['d']],num(100*b['calibration_exposure_concession']),num(100*chg['c'],2),num(-100*chg['d'],3),num(100*gap['c'],2),num(100*gap['d'],2)])
table('revision_v8_tolerance_table.tex',r'All eight cells at the declared .5 pp exposure tolerance. Case choice remains Excess. Cal. cost is the minimum-calibration-exposure concession; Later gain/cost compare the tolerance-selected exposure policy with the original exposure choice. Final columns compare the new exposure choice with the unchanged case choice. The .621 pp Later cost in complete 660/220 exceeds the calibration allowance.','tab:v8-tolerance','llllrrrr',r'Arm & Heads & Original/new & Cal. cost & Later $c$ gain & Later $d$ cost & $\Delta c$ & $\Delta d$',rows,'scriptsize')
rows=[[arm(z),heads(z)]+[N[b['selected']['d']] for b in z['bands']] for z in T['settings']]
table('revision_v8_tolerance_grid.tex',r'Complete tolerance grid: exposure-rule choices with original thresholds held fixed. Column values are exposure-coverage allowances in pp. Case choices remain Excess throughout. This is an explicit utility preference, not a confidence region. All candidate metrics and 40 condition outcomes are retained in the accompanying results.','tab:v8-tolerance-grid','lllllll',r'Arm & Heads & 0 & .1 & .25 & .5 & 1.0',rows,'scriptsize')
rows=[]
for z in T['stability']:
 for b in z['bands']:
  rows.append(['Complete 220/220' if z['setting'].startswith('complete') else 'Bike',num(b['epsilon']*100,2)]+[num(100*b['frequencies'][m]['frequency'],1) for m in Ms+['no_choice']])
table('revision_v8_tolerance_frequency.tex',r'Exposure-rule frequency (\%) after applying each tolerance to the same 200 threshold-redesign draws from the earlier audit. Fitted scores, cap, $T$, and draw identities are unchanged. The .5 pp rule makes Ratio more frequent in the complete-sales reference and less frequent in Bike; a wider band need not monotonically favor Ratio.','tab:v8-tolerance-frequency','llrrrrrr',r'Setting & Tolerance pp & Error & Excess & Mixed & Ratio & Desc. & None',rows,'scriptsize')
rows=[]
for z in FF['plans']:
 vals=[num(z['policies'][m]['screens']['60']['signed_excess_upper']) if z['policies'][m]['threshold'] is not None else '--' for m in Ms]
 rows.append([num(z['floor'],2),heads(z)]+vals)
table('revision_v8_floor_screen_table.tex',r'All 60 floor/head/ranking conditions: B signed-excess upper summaries at tail .05/60, per item. -- means no A-feasible threshold. All defined entries are negative and pass B, so B eliminates no additional candidate. The floor change affects candidate eligibility on A. Full threshold and mass records are retained in the accompanying source.','tab:v8-floor-screen','llrrrrr',r'Floor & Heads & Error & Excess & Mixed & Ratio & Descending',rows,'scriptsize')
rows=[]
for z in F['summary']:
 if z['family']!=60:continue
 rows.append([num(z['floor'],2),z['cell'].replace('e','').replace('_w','/'),z['utility'],N[z['method']],num(z['case_percent'],2),num(z['exposure_percent'],2),num(z['risk'],5),num(z['risk_upper_family60'],5),num(z['coverage_change_vs_floor35_pp'][0],2),num(z['coverage_change_vs_floor35_pp'][1],2)])
table('revision_v8_floor_later_table.tex',r'All 24 selected condition records under the 60-candidate screen family on the previously used Later window. $c,d$ are percentages; changes are pp relative to the same head/utility at floor .35. Rows include duplicated policies and are not independent replications. $U_{60}$ is an approximate one-sided .9991667 quantile based on 4,000 draws.','tab:v8-floor-later','llllrrrrrr',r'Floor & Heads & Utility & Rule & $c$ & $d$ & Risk & $U_{60}$ & Change $c$ & Change $d$',rows,'scriptsize')
print('Generated 9 v8 tables')
