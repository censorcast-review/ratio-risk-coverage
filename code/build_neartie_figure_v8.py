#!/usr/bin/env python3
"""Draw the complete-sales 220/220 calibration near tie from frozen v7 data.

The shaded area is a declared 0.5 percentage-point tolerance, not an uncertainty
interval. No models or thresholds are fitted, and no evaluation array is read.
By default writes only to reproduction_outputs/revision_v8_neartie/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--check-json", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out_dir or root / "reproduction_outputs/revision_v8_neartie"
    out.mkdir(parents=True, exist_ok=True)
    check_path = args.check_json or out / "NEARTIE_FIGURE_CHECK.json"
    check_path.parent.mkdir(parents=True, exist_ok=True)
    source = root / "evidence/revision_v7_stability/complete_220_220/POINT.json"
    point = json.loads(source.read_text())
    names = {"row": "Excess", "mixed": "Mixed", "demand": "Ratio",
             "weight_descending": "Descending"}
    coords = {}
    for key in names:
        assert point["eligibility"][key], key
        windows = point["policies"][key]["calibration"]
        coords[key] = [100 * min(w[m] for w in windows) for m in ("c", "d")]
        stored = point["statistics"]["candidate_eligibility"][key]
        for coordinate, metric in zip(coords[key], ("case", "exposure")):
            assert np.isclose(coordinate, 100 * stored[f"min_{metric}_coverage"]["mean"],
                              rtol=0, atol=1e-10)
    assert point["policies"]["error"]["threshold"] is None
    assert not point["eligibility"]["error"]
    ratio = np.asarray(coords["demand"])
    descending = np.asarray(coords["weight_descending"])
    gap = ratio - descending
    assert np.allclose(gap, [21.83441380926345, 0.027043519263303395],
                       rtol=0, atol=1e-10)
    epsilon_pp = 0.5  # Declared v8 sensitivity setting, not fitted from outcomes.
    ymax = max(v[1] for v in coords.values())
    boundary = ymax - epsilon_pp
    tolerant = [key for key, v in coords.items() if v[1] >= boundary]
    assert set(tolerant) == {"demand", "weight_descending"}

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.8,
        "axes.labelsize": 7.8, "axes.titlesize": 8.1,
        "xtick.labelsize": 7.1, "ytick.labelsize": 7.1,
        "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.45))
    fig.subplots_adjust(left=0.084, right=0.986, bottom=0.205, top=0.86, wspace=0.40)
    colors = {"row": "#626C78", "mixed": "#626C78", "demand": "#007E80",
              "weight_descending": "#C36B24"}
    markers = {"row": "s", "mixed": "D", "demand": "o", "weight_descending": "^"}
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(color="#E4E6E8", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=0.5)
        ax.set_xlabel("Minimum calibration case coverage (%)")
        ax.axhspan(boundary, ymax, facecolor="#E8DBB5", alpha=0.7, edgecolor="none", zorder=0)
        ax.axhline(ymax, color="#C1A361", linewidth=0.7, zorder=1)
        ax.axhline(boundary, color="#AC8A42", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)

    ax = axes[0]
    ax.set_title("(a) All four feasible candidates", loc="left", pad=7)
    ax.set(xlim=(42, 94), ylim=(78, 94.4), xticks=[50, 60, 70, 80, 90],
           yticks=[80, 85, 90])
    ax.set_ylabel("Minimum calibration\nexposure coverage (%)")
    for key, (x, y) in coords.items():
        ax.scatter(x, y, marker=markers[key], color=colors[key], s=31,
                   linewidths=0.6, edgecolors="white", zorder=5)
    ax.text(descending[0], 92.1, "Descending", ha="center", color=colors["weight_descending"])
    ax.text(ratio[0], 92.1, "Ratio", ha="center", color=colors["demand"])
    ax.text(coords["mixed"][0] - 1.1, 87.95, "Mixed", ha="right", color=colors["mixed"])
    ax.text(coords["row"][0] - 1.1, 81.7, "Excess", ha="right", color=colors["row"])
    ax.annotate("Declared tolerance: 0.5 pp", xy=(59.3, boundary),
                xytext=(44.5, 84.6), fontsize=7.2, color="#72581F",
                arrowprops={"arrowstyle": "-", "linewidth": 0.7, "color": "#AC8A42"},
                zorder=4)

    ax = axes[1]
    ax.set_title("(b) Ratio vs. Descending (y zoom)", loc="left", pad=7)
    ax.set(xlim=(44, 77.5), ylim=(90.983, 91.043), xticks=[50, 60, 70],
           yticks=[90.99, 91.00, 91.01, 91.02, 91.03, 91.04])
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    for key in ("demand", "weight_descending"):
        x, y = coords[key]
        ax.scatter(x, y, marker=markers[key], color=colors[key], s=39,
                   linewidths=0.7, edgecolors="white", zorder=5)
    ax.annotate(f"Descending\n{descending[1]:.5f}%", xy=descending,
                xytext=(45.2, 91.009), ha="left", va="bottom", fontsize=7.1,
                color=colors["weight_descending"],
                arrowprops={"arrowstyle": "-", "color": colors["weight_descending"],
                            "linewidth": 0.6, "shrinkB": 4})
    ax.text(ratio[0], ratio[1] + 0.0031, f"Ratio\n{ratio[1]:.5f}%", ha="center",
            va="bottom", fontsize=7.1, color=colors["demand"])
    # Differences are between candidate coordinates, not uncertainty bounds.
    ax.annotate("", xy=(ratio[0], 90.988), xytext=(descending[0], 90.988),
                arrowprops={"arrowstyle": "<->", "linewidth": 0.8, "color": "#48505A"})
    ax.text((ratio[0] + descending[0]) / 2, 90.9895,
            f"Case gap: {gap[0]:.2f} pp", ha="center", va="bottom", fontsize=7.0)
    ax.annotate("", xy=(73.7, ratio[1]), xytext=(73.7, descending[1]),
                arrowprops={"arrowstyle": "<->", "linewidth": 0.8, "color": "#48505A"})
    ax.text(74.5, (ratio[1] + descending[1]) / 2, f"{gap[1]:.5f} pp",
            rotation=90, ha="left", va="center", fontsize=6.9)

    outputs = {}
    for extension in ("pdf", "png"):
        path = out / f"revision_v8_neartie.{extension}"
        fig.savefig(path, dpi=240, metadata={"Creator": "build_neartie_figure_v8.py"})
        outputs[extension] = {"path": str(path), "sha256": sha256(path)}
    plt.close(fig)
    check = {
        "source": str(source.relative_to(root)), "source_sha256": sha256(source),
        "script_sha256": sha256(Path(__file__)), "setting": point["name"],
        "report_cap": point["cap"], "units": "percentage points / percent coverage",
        "coordinate_definition": "100 * min(calibration A, calibration B), separately for each metric",
        "candidate_coordinates_percent": {names[k]: {"case": v[0], "exposure": v[1]}
                                          for k, v in coords.items()},
        "ratio_minus_descending_pp": {"case": float(gap[0]), "exposure": float(gap[1])},
        "epsilon_pp": epsilon_pp, "tolerance_boundary_percent": boundary,
        "tolerance_members": [names[k] for k in tolerant],
        "excluded_candidate": "Conditional-error ranking has no feasible threshold; omitted, not plotted at zero.",
        "checks": {"direct_window_minima_match_stored_point_statistics": True,
                   "exact_expected_margins": True, "both_ratio_and_descending_in_band": True,
                   "source_read_only": True, "no_evaluation_data_read": True},
        "display": {"size_inches": [6.4, 2.45], "right_y_axis_enlarged": True,
                    "tolerance_band_right_panel_clipped_to_view": True,
                    "shading_is_uncertainty_interval": False},
        "caption_proposal": "Calibration near tie in the complete-sales 220/220 reference setting. "
            "Each coordinate is the minimum coverage across calibration windows A and B. "
            "Ratio exceeds Descending by only 0.02704 exposure percentage points but by 21.83 case points. "
            "The shaded region is within the declared retrospective tolerance of 0.5 exposure points "
            "of the best candidate; it is not a confidence interval. The right panel enlarges the y-axis "
            "and clips the tolerance region to the displayed range. The conditional-error candidate "
            "has no feasible threshold and is omitted.",
        "outputs": outputs,
    }
    check_path.write_text(json.dumps(check, indent=2) + "\n")
    print(json.dumps({"outputs": outputs, "checks": str(check_path), "margins_pp": gap.tolist()}))


if __name__ == "__main__":
    main()
