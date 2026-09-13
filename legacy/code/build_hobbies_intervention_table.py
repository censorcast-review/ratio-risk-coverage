"""Build the compact development-only intervention table from saved aggregates."""
from pathlib import Path
import json
from r2_io import write
ROOT=Path(__file__).resolve().parents[1]
def build():
    v=json.loads((ROOT/'results/error_interventions/RESULTS.json').read_text())
    names={'shared_error_21':'Shared, 21 features','hobbies_error_21':'Hobbies, 21 features','hobbies_error_29':'Hobbies, 29 features'}
    rows=[]
    for z in v['records']:
        b=z['blocks']['shadow'];h=b['head'];p=b['policy'];q=b['samplewise_prefix_diagnostic']
        risk='--' if p['wape'] is None else f"{p['wape']:.4f}"
        rows.append(f"{names[z['method']]} & {h['error_mse']:.4f} & {h['positive_day_error_mse']:.4f} & {100*p['row_coverage']:.2f} & {risk} & {100*q['row_coverage']:.2f} \\\\")
    caption=("Development-only Hobbies intervention (first seed). Both specialists use the same Hobbies subset of the original risk-training sample; the 29-feature head adds origin-safe intermittency summaries to the original 21 features. The point forecast and demand head remain fixed. MSE concerns absolute-error prediction; positive-day MSE conditions on realized positive demand. Fixed-threshold rows and WAPE transfer the single threshold selected jointly on calibration A/B. Prefix rows use shadow outcomes to describe the best score prefix under the cap and 35\\% row floor, and are not deployable thresholds. An empty policy has zero coverage and undefined WAPE.")
    text='\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrrr}\n\\toprule\nError head & MSE & Positive MSE & Rows (\\%) & WAPE & Prefix (\\%) \\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n\\caption{'+caption+'}\n\\label{tab:hobbies_intervention}\n\\end{table}\n'
    write(ROOT/'paper/tables/hobbies_intervention.tex',text.encode())
if __name__=='__main__':build()
