#!/usr/bin/env python3
"""Verify the exact three-context construction; no retail inputs are read.

Enumerates the feasible acceptance polytope using Fraction arithmetic, checks
both coverage optima with scipy.optimize.linprog, and renders its projection.
Run from any directory: python code/verify_objective_geometry.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
from fractions import Fraction as F
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import linprog
from r2_io import dump, write
from reproduction_io import safe_output


ROOT = Path(__file__).resolve().parents[1]
PROB = (F(2, 5), F(1, 2), F(1, 10))
DEMAND = (F(1, 4), F(1, 10), F(10))
FORECAST = (F(1, 4), F(0), F(3))
ERROR = tuple(abs(y - f) for y, f in zip(DEMAND, FORECAST))
CAP = F(1, 2)
ROW_FLOOR = F(7, 20)
TOTAL_DEMAND = sum(p * y for p, y in zip(PROB, DEMAND))
EXCESS = tuple(e - CAP * y for e, y in zip(ERROR, DEMAND))
ROW_REWARD = PROB
DEMAND_REWARD = tuple(p * y / TOTAL_DEMAND for p, y in zip(PROB, DEMAND))


def dot(x, y):
    return sum((a * b for a, b in zip(x, y)), F(0))


def solve_exact(rows, rhs):
    """Gaussian elimination of a 3x3 rational system; None if singular."""
    aug = [list(row) + [value] for row, value in zip(rows, rhs)]
    for col in range(3):
        pivot = next((k for k in range(col, 3) if aug[k][col]), None)
        if pivot is None:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        divisor = aug[col][col]
        aug[col] = [v / divisor for v in aug[col]]
        for k in range(3):
            if k != col:
                multiplier = aug[k][col]
                aug[k] = [v - multiplier * u for v, u in zip(aug[k], aug[col])]
    return tuple(row[-1] for row in aug)


def constraints():
    """Each (name, coefficients, rhs) denotes coefficients @ a <= rhs."""
    result = []
    for k, name in enumerate(("C", "L", "H")):
        unit = tuple(F(int(j == k)) for j in range(3))
        result.append((f"{name} upper", unit, F(1)))
        result.append((f"{name} lower", tuple(-v for v in unit), F(0)))
    result.append(("WAPE excess", tuple(p * m for p, m in zip(PROB, EXCESS)), F(0)))
    result.append(("row floor", tuple(-p for p in PROB), -ROW_FLOOR))
    return result


def exact_vertices(inequalities):
    vertices = set()
    for active in itertools.combinations(inequalities, 3):
        a = solve_exact([r[1] for r in active], [r[2] for r in active])
        if a is not None and all(dot(row, a) <= rhs for _, row, rhs in inequalities):
            vertices.add(a)
    return sorted(vertices)


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def number(value):
    return {"exact": str(value), "decimal": float(value)}


def describe(a):
    demand = dot(tuple(p * y for p, y in zip(PROB, DEMAND)), a)
    error = dot(tuple(p * e for p, e in zip(PROB, ERROR)), a)
    assert demand > 0
    return {
        "acceptance_probabilities_C_L_H": [number(v) for v in a],
        "row_coverage": number(dot(ROW_REWARD, a)),
        "demand_coverage": number(demand / TOTAL_DEMAND),
        "accepted_demand_mass": number(demand),
        "accepted_error_mass": number(error),
        "wape": number(error / demand),
    }


def make_figure(hull, row_point, demand_point, figure_dir):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    points = np.array([[float(v) * 100 for v in p] for p in hull])
    ax.fill(points[:, 0], points[:, 1], color="#DFE7ED", alpha=0.9, edgecolor="#A8B7C4", linewidth=1.1)
    row = np.array([float(v) * 100 for v in row_point])
    demand = np.array([float(v) * 100 for v in demand_point])
    ax.plot([demand[0], row[0]], [demand[1], row[1]], color="#AF5039", linewidth=2.6, zorder=3)
    ax.scatter([row[0], demand[0]], [row[1], demand[1]], s=42, color="#AF5039", edgecolor="white", linewidth=0.8, zorder=4)
    ax.annotate("Demand optimum\n42.50% rows · 30.43% demand", xy=demand, xytext=(1, 13),
                textcoords="offset points", ha="left", va="bottom", fontsize=9.2, color="#263541")
    ax.annotate("Row optimum\n91.25% rows · 23.91% demand", xy=row, xytext=(-3, 30),
                textcoords="offset points", ha="right", va="bottom", fontsize=9.2, color="#263541")
    ax.text(67, 23, "Efficient segment\nWAPE = 0.5 throughout", ha="center", va="top", fontsize=9.3, color="#913F2C")
    ax.text(62, 14, "Feasible policies", ha="center", color="#687D8D", fontsize=10)
    ax.axvline(35, color="#B0BBC4", linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    ax.set(xlabel="Row coverage (%)", ylabel="Demand coverage (%)", xlim=(32, 96), ylim=(0, 38),
           xticks=[35, 45, 55, 65, 75, 85, 95], yticks=[0, 5, 10, 15, 20, 25, 30, 35])
    ax.set_title("Exact coverage trade-off with perfect conditional heads", loc="left", pad=20)
    ax.text(0, 1.015, "Deterministic three-context construction; WAPE cap 0.5; row floor 35%",
            transform=ax.transAxes, fontsize=8.6, color="#59636B", va="bottom")
    ax.grid(axis="y", color="#E5E8EB", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.1)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        stream=io.BytesIO()
        options={"metadata": {"Title": "Exact synthetic coverage geometry", "CreationDate": None, "ModDate": None}} if extension=="pdf" else {"dpi":240}
        fig.savefig(stream,format=extension,**options)
        write(figure_dir / ("objective_geometry."+extension),stream.getvalue())
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT / "reproduction_outputs" / "objective_geometry",help="Directory for JSON and both rendered figures; archived evidence paths are refused.")
    args=parser.parse_args()
    output=safe_output(args.output)
    inequalities = constraints()
    vertices = exact_vertices(inequalities)
    assert vertices
    exact_optima = {}
    for label, reward in (("row", ROW_REWARD), ("demand", DEMAND_REWARD)):
        maximum = max(dot(reward, a) for a in vertices)
        maximizers = [a for a in vertices if dot(reward, a) == maximum]
        assert len(maximizers) == 1, "This construction must have a unique optimum for each objective."
        exact_optima[label] = maximizers[0]
    assert exact_optima["row"] == (F(1), F(1), F(1, 8))
    assert exact_optima["demand"] == (F(1), F(0), F(1, 4))

    scipy_checks = []
    for label, reward in (("row", ROW_REWARD), ("demand", DEMAND_REWARD)):
        result = linprog(-np.array(reward, dtype=float),
                         A_ub=np.array([v[1] for v in inequalities], dtype=float),
                         b_ub=np.array([v[2] for v in inequalities], dtype=float),
                         bounds=[(0, 1)] * 3, method="highs")
        assert result.success, result.message
        assert np.allclose(result.x, np.array(exact_optima[label], dtype=float), atol=1e-10)
        scipy_checks.append({"objective": label, "success": bool(result.success),
                             "acceptance_probabilities": result.x.tolist(), "status": result.message})

    projection = [(dot(ROW_REWARD, a), dot(DEMAND_REWARD, a)) for a in vertices]
    hull = convex_hull(projection)
    pareto_vertices = [point for point in hull if not any(
        other[0] >= point[0] and other[1] >= point[1] and other != point for other in hull)]
    row_point = (F(73, 80), F(11, 46))
    demand_point = (F(17, 40), F(7, 23))
    assert set(pareto_vertices) == {row_point, demand_point}

    # Exact parameterization of the entire efficient segment, checked at endpoints
    # and 101 rational interior grid points. Algebra in the appendix proves it for
    # every real z in [0, 1/2]; the grid check is an implementation cross-check.
    for k in range(101):
        z = F(k, 200)
        x = F(1, 40) - z / 40
        a = (F(1), 2 * z, 10 * x)
        assert all(dot(row, a) <= rhs for _, row, rhs in inequalities)
        assert dot(tuple(p * m for p, m in zip(PROB, EXCESS)), a) == 0
        assert dot(ROW_REWARD, a) == F(17, 40) + F(39, 40) * z
        assert dot(DEMAND_REWARD, a) == F(7, 23) - F(3, 23) * z

    report = {
        "status": "PASS",
        "analysis_type": "exact deterministic mathematical construction; no retail data",
        "inputs": {"contexts": [
            {"name": name, "probability": number(p), "demand": number(y), "forecast": number(f),
             "conditional_error": number(e), "conditional_excess": number(m), "relative_demand_error": number(e / y)}
            for name, p, y, f, e, m in zip(("C", "L", "H"), PROB, DEMAND, FORECAST, ERROR, EXCESS)],
            "cap": number(CAP), "row_floor": number(ROW_FLOOR), "total_demand": number(TOTAL_DEMAND)},
        "exact_polytope_vertex_count": len(vertices),
        "exact_polytope_vertices": [describe(a) for a in vertices],
        "exact_projection_hull": [[number(v) for v in point] for point in hull],
        "optima": {label: describe(a) for label, a in exact_optima.items()},
        "optimum_unique_for_each_objective": True,
        "pareto_segment": {"parameter": "accepted L probability mass z in [0, 1/2]",
                           "accepted_H_probability_mass": "1/40 - z/40",
                           "row_coverage": "17/40 + 39*z/40", "demand_coverage": "7/23 - 3*z/23",
                           "wape": "1/2", "rational_grid_checks": 101},
        "difference_row_minus_demand_objective": {
            "row_coverage": number(F(39, 80)), "demand_coverage": number(-F(3, 46))},
        "scipy_version": scipy.__version__, "scipy_linprog_cross_checks": scipy_checks,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    dump(output / "OBJECTIVE_GEOMETRY.json",report)
    make_figure(hull, row_point, demand_point, output)
    print(json.dumps({"status": "PASS", "exact_vertices": len(vertices),
                      "row_optimum": report["optima"]["row"], "demand_optimum": report["optima"]["demand"]}, indent=2))


if __name__ == "__main__":
    main()
