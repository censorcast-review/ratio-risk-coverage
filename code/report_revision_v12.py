"""Report all frozen seed cells and the single external analysis."""
from pathlib import Path
import json, numpy as np
ROOT=Path(__file__).resolve().parents[1]
PAIRS=[(220,220),(660,220),(220,660),(660,660)]
NAMES={'row':'Excess','demand':'Ratio','weight_descending':'Descending','mixed':'Mixed','error':'Error',None:'None'}
records=[]
for seed in [20260910,20260913,20260914]:
    src=ROOT/'evidence/matched_censoring' if seed==20260910 else ROOT/f'evidence/revision_v12_seeds/seed_{seed}'
    assert (src/'COMPLETE.json').exists()
    for arm in ['censored','complete']:
        q=json.loads((src/arm/'RESULTS.json').read_text())
        for p in q['menus']:
            if p['kind']=='relative' and p['value']==.95:
                records.append(dict(seed=seed,arm=arm,heads=[p['e_trees'],p['w_trees']],
                    cap=p['cap'],selected=p['selected'],contrast=p['contrast'],full_risk=q['full']['risk']))
summary=[]
for arm in ['censored','complete']:
    for h in PAIRS:
        rows=[p for p in records if p['arm']==arm and p['heads']==list(h)]
        assert len(rows)==3
        valid=[p for p in rows if p['contrast'] is not None]
        vals=np.asarray([[p['contrast']['dc']*100,p['contrast']['dd']*100] for p in valid])
        summary.append(dict(arm=arm,heads=list(h),feasible_seeds=len(valid),
            mean_pp=vals.mean(0).tolist() if len(vals) else None,
            sample_sd_pp=vals.std(0,ddof=1).tolist() if len(vals)>1 else None,
            dc_range_pp=[float(vals[:,0].min()),float(vals[:,0].max())] if len(vals) else None,
            dd_range_pp=[float(vals[:,1].min()),float(vals[:,1].max())] if len(vals) else None,
            exposure_choices=[p['selected']['d'] for p in rows],
            same_direction_seeds=sum(p['contrast']['dc']<0 and p['contrast']['dd']>0 for p in valid)))
ratio_differences=[]
for seed in [20260910,20260913,20260914]:
    for h in PAIRS:
        pair={p['arm']:p for p in records if p['seed']==seed and p['heads']==list(h)}
        if all(p['contrast'] is not None and p['contrast']['dc']<0 for p in pair.values()):
            ratios={a:p['contrast']['dd']/-p['contrast']['dc'] for a,p in pair.items()}
            ratio_differences.append(dict(seed=seed,heads=list(h),ratios=ratios,difference=ratios['complete']-ratios['censored']))
report=dict(records=records,summary=summary,ratio_differences=ratio_differences,
    independent_units='Three training procedures per head/arm; same data, shared models across head cells',
    original_seed_was_previously_inspected=True)
(ROOT/'evidence/revision_v12_seeds/SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')

table=r'''\begin{table}[t]\centering\small
\caption{Paired audit across the original seed 20260910 and two newly fixed seeds (20260913, 20260914). Entries are means $\pm$ sample SD across three seeds, in percentage points; SD is not a confidence interval. Caps are recomputed per seed and arm. Head settings share fitted components. Complete-sales 220/220 variability includes the R/R/D rule switch. Exposure choices are listed in seed order (R: Ratio; D: Descending); the case choice is Excess throughout. Appendix~\ref{app:seeds-v12} preserves every seed/cell and cap.}\label{tab:matched-history}
\begin{tabular}{llcrr}\toprule
Arm & Trees $e/w$ & Exposure choices & $\Delta c$ & $\Delta d$\\\midrule
'''
for p in summary:
    assert p['feasible_seeds']==3
    choices='/'.join('R' if s=='demand' else 'D' if s=='weight_descending' else NAMES[s] for s in p['exposure_choices'])
    a,b=p['mean_pp'];sa,sb=p['sample_sd_pp'];arm='Censored' if p['arm']=='censored' else 'Complete'
    table+=f"{arm} & {p['heads'][0]}/{p['heads'][1]} & {choices} & ${a:.2f}\\pm{sa:.2f}$ & $+{b:.2f}\\pm{sb:.2f}$ \\\\\n"
table+=r'\bottomrule\end{tabular}\end{table}'+'\n'
(ROOT/'paper/matched_main_table.tex').write_text(table)

app=r'''\section{Additional seeds and an external analysis frozen before data opening}
\label{app:additional-v12}
\subsection{Additional paired seeds}\label{app:seeds-v12}
Two seeds (20260913 and 20260914) were fixed after inspecting the original 20260910 results. Each repeats ten fits: two 300-tree median predictors and eight 220/660-tree squared-loss heads. Original periods, 600,000-row sampling, features, and parameters are retained. The seed changes row sampling and bin construction; both arms share the sampled rows within a seed. The item-bootstrap seed is held at 20260910 (2,000 draws) to separate its Monte Carlo variation from refitting. The cap formula remains .95 times maximum full A/B calibration risk per arm. These repetitions use the same previously evaluated M5 data; they are not three new external confirmation trials.

Table~\ref{tab:seed-v12-all} reports all 24 seed/cells. Original calibration stability, tolerance, floor, exchange-interval and budget audits retain seed 20260910; they are not rerun or averaged across the additional seeds. Original per-cell point values also remain in Table~\ref{tab:matched-heads}. One saved calibration NPZ in seed 20260914 was incomplete despite matching its original completion hash. Its seven arrays were replayed from the frozen models without fitting; all four surviving arrays matched exactly and original threshold/results records were retained. The original archive and recovery protocol are supplied in the source. New evidence and independent checks are in \texttt{evidence/revision\_v12\_seeds/} and \texttt{docs/revision\_v12/}.

\begin{table}[H]\centering\small
\caption{Every paired seed/cell. Seed suffixes 10, 13, 14 mean 20260910, 20260913, 20260914. Censored and complete are abbreviated C and F. Case choice is Excess throughout. These are point effects in percentage points, not independent dataset replications.}\label{tab:seed-v12-all}
\begin{tabular}{cllcrrr}\toprule
Seed & Arm & Heads & Exposure & Cap & $\Delta c$ & $\Delta d$\\\midrule
'''
for p in records:
    assert p['selected']['c']=='row' and p['contrast'] is not None
    c=p['contrast'];h=p['heads'];arm='C' if p['arm']=='censored' else 'F';expo=NAMES[p['selected']['d']]
    app+=f"{str(p['seed'])[-2:]} & {arm} & {h[0]}/{h[1]} & {expo} & {p['cap']:.5f} & {100*c['dc']:.2f} & +{100*c['dd']:.2f} \\\\\n"
app+=r'''\bottomrule\end{tabular}\end{table}

\subsection{External protocol and complete candidate report}\label{app:external-v12}
We thank the James M. Kilts Center for Marketing at the University of Chicago Booth School of Business for the Dominick's data. This academic analysis uses the Oatmeal movement CSV, selected for public access and bounded scope before raw outcomes were opened. Favorita was considered but not downloaded because authentication was unavailable. No outcome-based choice between datasets or categories was made. No earlier use of Dominick's was found in the study records. The protocol is a local, time-stamped pre-opening record, not a public-registry preregistration.

Weeks 53--208 fit the point predictor; 209--260 fit heads; 261--312 are A; 313--364 are B; 365--400 are final evaluation. Valid records have the provider's OK flag equal to one and finite nonnegative MOVE. Missing records are not imputed as sales. The cohort requires at least 52 valid point-training records and positive point-training sales, with no future-window selection. It contains 4,839 store--UPC series, 66 products and 86 stores. Point/head/A/B/evaluation record counts are 511,980/186,347/153,255/139,881/85,039.

Forecasts are rolling one week ahead. Features are store/UPC identifiers, time, seasonal sine/cosine, sales lags 1/2/4/13/26/52, and past-only rolling means, standard deviations, maxima and missing fractions over 4/13/26 weeks. The 300-tree median predictor and two 220-tree squared-loss heads use seed 20260912 and the paired study's learning rate, leaf count and minimum leaf size. No tuning or early stopping is performed. A determines $r=.95R_{\rm all,A}=.91739424$ and thresholds at $.95r=.87152453$, floor .40. Each utility selects on A; B only screens the fixed choices, without fallback or reselection.

Circular moving blocks of four calendar weeks retain all UPC/store records in each week; B and evaluation use 20,000 draws each, with separate fixed seeds. Five-candidate upper risk summaries use tail .05/5. The declared joint success criterion requires both A-selected policies to pass B and the final 95\% paired intervals to have upper $\Delta c<0$ and lower $\Delta d>0$. It is not met: $\Delta c=-4.6343$ ($[-6.1496,-3.4492]$) and $\Delta d=2.5299$ ($[-.6887,5.8453]$). Positive point direction and B passage provide partial evidence, but not the declared interval confirmation. No retuning followed.

An overstrict implementation assertion initially stopped evaluation because weeks 370, 371 and 400 had no records. The completion retains all 36 calendar weeks with zero contribution in empty weeks, using the same frozen policies, models, arrays, bootstrap settings and success criterion. It performs no new fits or selection and reproduces the existing B results exactly. Original failure and amendment records are preserved. This does not change the observational meaning of missing weeks or give a population guarantee.

\begin{table}[H]\centering\small
\caption{All five external candidates. Coverages are percentages. A risk design uses .871525; B compares its adjusted upper risk summary with .917394. Descending is infeasible on A; Error is feasible on A but fails B. Selected Excess and Ratio both pass B.}\label{tab:external-v12-all}
\begin{tabular}{lrrcrrr}\toprule
Rule & A $c$ & A $d$ & B risk upper & E $c$ & E $d$ & E risk\\\midrule
'''
q=json.loads((ROOT/'evidence/revision_v12_external/RESULTS.json').read_text());f=json.loads((ROOT/'evidence/revision_v12_external/FROZEN_A.json').read_text())
for m in NAMES:
    if m not in f['plans']:continue
    a=f['plans'][m]['calibration'][0];b=q['B'][m];e=q['E'][m];u='--' if b['risk_U5'] is None else f"{b['risk_U5']:.5f}";risk='--' if e['risk'] is None else f"{e['risk']:.5f}"
    app+=f"{NAMES[m]} & {a['c']*100:.2f} & {a['d']*100:.2f} & {u} & {e['c']*100:.2f} & {e['d']*100:.2f} & {risk} \\\\\n"
app+=r'\bottomrule\end{tabular}\end{table}'+'\n'
(ROOT/'paper/revision_v12_results_appendix.tex').write_text(app)
print(json.dumps(report['summary'],indent=2))
