"""Build current manuscript assets from preserved experiments and audited cells."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
EVIDENCE = ROOT / "evidence" / "risk_calibration"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


primary = load(EVIDENCE / "primary_policy" / "RESULTS.json")["rows"]
replication = load(EVIDENCE / "seed_replication" / "RESULTS.json")["rows"]
rows = [{"seed": 20260906, **row} for row in primary] + replication
attribution = load(EVIDENCE / "attribution_controls" / "RESULTS.json")
fresh = load(ROOT / "evidence" / "freshretail" / "frozen_eval" / "RESULTS.json")
analysis = load(EVIDENCE / "primary_analysis" / "RESULTS.json")["rows"][-1]
uci = load(EVIDENCE / "uci_transfer" / "RESULTS.json")

summary: dict = {}
for block in ["calibration_a", "calibration_b", "shadow"]:
    selected = [row for row in rows if row["block"] == block]
    summary[block] = {}
    for method in ["base", "risk"]:
        wape = np.asarray([row[method]["wape"] for row in selected])
        mae = np.asarray([row[method]["mae"] for row in selected])
        summary[block][method] = {
            "wape_mean": float(wape.mean()),
            "wape_sd": float(wape.std(ddof=1)),
            "mae_mean": float(mae.mean()),
            "mae_sd": float(mae.std(ddof=1)),
        }
summary["attribution"] = attribution
summary["freshretail"] = fresh
summary["uci_audit"] = uci
(ROOT / "verification" / "PRESENTATION_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

# Main M5 table.
labels = {"calibration_a": "A", "calibration_b": "B", "shadow": "Later"}
lines = [
    "\\begin{table}[t]",
    "\\caption{Full-coverage M5 WAPE across three fit seeds (mean $\\pm$ sample SD). The policy family is chosen on point validation; all three blocks were previously consumed and are retrospective.}",
    "\\label{tab:main}",
    "\\centering\\scriptsize",
    "\\setlength{\\tabcolsep}{3pt}",
    "\\begin{tabular}{lcccc}\\toprule",
    r"Method & A & B & Later & Later MAE \\ \midrule",
]
base_cells = [f"{summary[b]['base']['wape_mean']:.5f} $\\pm$ {summary[b]['base']['wape_sd']:.5f}" for b in labels]
risk_cells = [f"\\textbf{{{summary[b]['risk']['wape_mean']:.5f}}} $\\pm$ {summary[b]['risk']['wape_sd']:.5f}" for b in labels]
lines.append("Hierarchy L1 + scale & " + " & ".join(base_cells) + f" & {summary['shadow']['base']['mae_mean']:.4f} $\\pm$ {summary['shadow']['base']['mae_sd']:.4f} " + r"\\")
lines.append("CENSORCAST-RC & " + " & ".join(risk_cells) + f" & \\textbf{{{summary['shadow']['risk']['mae_mean']:.4f}}} $\\pm$ {summary['shadow']['risk']['mae_sd']:.4f} " + r"\\")
lines += ["\\bottomrule\\end{tabular}", "\\end{table}"]
(PAPER / "risk_main_table.tex").write_text("\n".join(lines) + "\n")

# Corrected matched score ablation.
corrected = attribution["corrected_score_ablation"]
score_rows = [
    ("Base category scale", corrected["base"]["wape"]),
    ("Forecast-quantile scale", corrected["scores"]["base_forecast"]["later"]["wape"]),
    ("Historical hit-rate scale", corrected["scores"]["historical_hit_rate_28"]["later"]["wape"]),
    ("Fill-ratio scale", corrected["scores"]["origin_fill_ratio"]["later"]["wape"]),
    ("Capacity-pressure scale", corrected["scores"]["capacity_pressure"]["later"]["wape"]),
    ("Learned hit-risk scale", corrected["scores"]["learned_hit_risk"]["later"]["wape"]),
]
lines = [
    "\\begin{table}[t]",
    "\\caption{Matched eight-bin calibration-score ablation on the primary-seed later block. Every row uses the same hierarchy L1 forecast, category conditioning, and shrinkage.}",
    "\\label{tab:scoreablation}",
    "\\centering\\small",
    "\\begin{tabular}{lc}\\toprule",
    r"Calibration score & WAPE \\ \midrule",
]
for name, value in score_rows:
    displayed = f"\\textbf{{{value:.5f}}}" if name.startswith("Learned") else f"{value:.5f}"
    lines.append(f"{name} & {displayed} " + r"\\")
lines += ["\\bottomrule\\end{tabular}", "\\end{table}"]
(PAPER / "risk_ablation_table.tex").write_text("\n".join(lines) + "\n")

# Feature integration and privileged-label reference.
integration = attribution["integration_and_oracle"]
crossfit = load(ROOT / "evidence/cell_audit/crossfit/RESULTS.json")
integration_rows = [
    ("Hierarchy L1 + category scale", integration["hierarchy_base"]),
    ("L1 + in-sample $q$ feature", integration["q_as_feature"]),
    ("L1 + item-cross-fitted $q$ feature", crossfit["later"]["category_scaled"]),
    ("CENSORCAST-RC post-calibration", integration["proposed_posthoc_calibration"]),
    ("Truth-label L1 reference", integration["truth_label_oracle"]),
]
lines = [
    "\\begin{table}[t]",
    "\\caption{Primary-seed Later controls under fixed model settings. The two feature controls use in-sample or five-fold item-held-out risk predictions during L1 training. The truth-label row uses benchmark demand during training and is an architecture-matched reference, not a deployable method or mathematical upper bound.}",
    "\\label{tab:integration}",
    "\\centering\\small",
    "\\begin{tabular}{lrrrr}\\toprule",
    r"Model & WAPE & MAE & RMSE & Bias (\%) \\ \midrule",
]
for name, value in integration_rows:
    lines.append(
        f"{name} & {value['wape']:.5f} & {value['mae']:.5f} & {value['rmse']:.5f} & ${100 * value['bias']:.2f}$ " + r"\\"
    )
lines += ["\\bottomrule\\end{tabular}", "\\end{table}"]
(PAPER / "risk_integration_table.tex").write_text("\n".join(lines) + "\n")

# Audited cells: keep the comparison baseline consistent with the main tables.
# Historical attribution JSON retains its misnamed raw-head column for provenance.
import sys
import pandas as pd
sys.path.insert(0, str(ROOT / "code"))
from analyze_m5_cells_floor import render_tables as render_m5_cells
from analyze_m5_floor_split import render as render_floor_split
from analyze_natural_cells import write_table as render_fresh_cells

cell_audit = ROOT / "evidence" / "cell_audit"
m5_cells = load(cell_audit / "m5" / "RESULTS.json")
render_m5_cells(m5_cells, PAPER)
render_floor_split(load(cell_audit / "m5" / "SPLIT_RESULTS.json"), m5_cells, PAPER)
render_fresh_cells(
    PAPER / "fresh_cells_table.tex",
    pd.read_csv(cell_audit / "natural" / "CELLS_ALL_POPULATIONS.csv"),
    load(ROOT / "evidence" / "freshretail" / "frozen_eval" / "FINAL_POLICY.json"),
)

# Decomposition.
lines = [
    "\\begin{table}[t]",
    "\\caption{Primary-seed later-block decomposition. Improvement is broad across horizons but not every product category.}",
    "\\label{tab:decomp}",
    "\\centering\\small",
    "\\begin{tabular}{lcc}\\toprule",
    r"Subset & Base & CENSORCAST-RC \\ \midrule",
]
for category in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
    lines.append(
        f"{category.title()} & {analysis['categories'][category]['base']['wape']:.5f} & "
        f"{analysis['categories'][category]['risk']['wape']:.5f} " + r"\\"
    )
for label, key in [("Strictly censored rows", "strict"), ("Other rows", "other")]:
    lines.append(
        f"{label} & {analysis['strata'][key]['base']['wape']:.5f} & "
        f"{analysis['strata'][key]['risk']['wape']:.5f} " + r"\\"
    )
lines += ["\\bottomrule\\end{tabular}", "\\end{table}"]
(PAPER / "risk_decomp_table.tex").write_text("\n".join(lines) + "\n")

# Frozen FreshRetailNet table.
fresh_base = fresh["primary_fully_available"]["base"]
fresh_rc = fresh["primary_fully_available"]["risk_conditioned"]
fresh_ci = fresh["primary_fully_available"]["paired_series_bootstrap"]
lines = [
    "\\begin{table}[t]",
    "\\caption{One-shot evaluation on the official FreshRetailNet future split for a frozen 2,000-series cohort. Primary rows have no recorded stockout hours. The interval is a paired 4,000-draw series bootstrap for the WAPE difference.}",
    "\\label{tab:fresh}",
    "\\centering\\small",
    "\\begin{tabular}{lrrrr}\\toprule",
    r"Method & WAPE & MAE & RMSE & Bias (\%) \\ \midrule",
    f"L1 + management-group scale & {fresh_base['wape']:.5f} & {fresh_base['mae']:.5f} & {fresh_base['rmse']:.5f} & ${100 * fresh_base['signed_percentage_bias']:.2f}$ " + r"\\",
    f"CENSORCAST-RC & \\textbf{{{fresh_rc['wape']:.5f}}} & \\textbf{{{fresh_rc['mae']:.5f}}} & \\textbf{{{fresh_rc['rmse']:.5f}}} & ${100 * fresh_rc['signed_percentage_bias']:.2f}$ " + r"\\",
    "\\midrule",
    f"RC $-$ base WAPE & \\multicolumn{{4}}{{c}}{{$ {fresh_ci['estimate']:.5f}$; 95\\% CI $[{fresh_ci['ci95'][0]:.5f},{fresh_ci['ci95'][1]:.5f}]$}} " + r"\\",
    "\\bottomrule\\end{tabular}",
    "\\end{table}",
]
(PAPER / "freshretail_table.tex").write_text("\n".join(lines) + "\n")

# UCI audit table, kept in the appendix only.
lines = [
    "\\begin{table}[htbp]",
    "\\caption{Online Retail II validation audit in the extreme mechanically censored regime. This table is not principal evidence.}",
    "\\label{tab:uci_transfer}",
    "\\centering\\small",
    "\\begin{tabular}{lccc}\\toprule",
    r"Method & WAPE & RMSE & Bias \\ \midrule",
    f"Exact-scaled L1 & {uci['baseline']['wape']:.5f} & {uci['baseline']['rmse']:.3f} & {100 * uci['baseline']['bias']:.1f}\\% " + r"\\",
    f"CENSORCAST-RC & {uci['risk']['wape']:.5f} & {uci['risk']['rmse']:.3f} & {100 * uci['risk']['bias']:.1f}\\% " + r"\\",
    "\\bottomrule\\end{tabular}",
    "\\end{table}",
]
(PAPER / "risk_uci_table.tex").write_text("\n".join(lines) + "\n")

# M5 figure.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
x = np.arange(3)
base = [summary[b]["base"]["wape_mean"] for b in labels]
risk_values = [summary[b]["risk"]["wape_mean"] for b in labels]
axes[0].plot(x, base, "o-", label="Hierarchy L1 + scale", lw=1.8)
axes[0].plot(x, risk_values, "o-", label="CENSORCAST-RC", lw=1.8)
axes[0].set_xticks(x, [labels[b] for b in labels])
axes[0].set_ylabel("WAPE")
axes[0].grid(alpha=0.25)
axes[0].legend(frameon=False, fontsize=7)
horizons = np.arange(1, 8)
delta = [
    100 * (analysis["horizons"][str(h)]["risk"]["wape"] - analysis["horizons"][str(h)]["base"]["wape"])
    for h in horizons
]
axes[1].bar(horizons, delta, color="#2878B5")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_xlabel("Forecast horizon")
axes[1].set_xticks(horizons)
axes[1].set_ylabel("WAPE change (pp)")
axes[1].grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(PAPER / "figures" / "risk_results.pdf", bbox_inches="tight")
fig.savefig(PAPER / "figures" / "risk_results.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print(json.dumps({"status": "BUILT", "later": summary["shadow"], "fresh": fresh["primary_fully_available"]}, indent=2))
