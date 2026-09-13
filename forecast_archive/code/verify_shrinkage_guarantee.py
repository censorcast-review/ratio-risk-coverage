"""Independent finite-case audit of geometric shrinkage's training-loss bound.

This checks mathematical examples only: it does not fit or re-evaluate a model.
Run with standard Python; results are printed without modifying evidence files.
"""

import itertools
import json
import math
import random


def loss(rows, scale):
    return math.fsum(abs(y - scale * b) for y, b in rows)


def median(rows, upper):
    terms = sorted((y / b, b) for y, b in rows if b > 0)
    if not terms:
        return 0.0
    half = math.fsum(b for _, b in terms) / 2
    total = 0.0
    for ratio, weight in terms:
        total += weight
        if total >= half:
            return min(upper, max(0.0, ratio))
    raise AssertionError("No weighted median")


def check(rows, c, eps, upper, counts):
    m = median(rows, upper)
    ce, me = max(c, eps), max(m, eps)
    baseline = loss(rows, c)
    baseline_floored = loss(rows, ce)
    weight = math.fsum(b for _, b in rows)
    previous = baseline_floored
    tol = 1e-10 * max(1.0, baseline, baseline_floored, upper * weight)
    for lam in (0, .01, .1, .25, .5, .75, .9, .99, 1):
        s = math.exp((1 - lam) * math.log(ce) + lam * math.log(me))
        value = loss(rows, s)
        assert value <= previous + tol, ("monotonicity", rows, c, eps, m, lam)
        assert value <= baseline_floored + tol, ("floored baseline", rows, c, eps, m, lam)
        assert value <= baseline + (ce - c) * weight + tol, ("original baseline", rows, c, eps, m, lam)
        previous = value
        counts["lambda_checks"] += 1
    counts["configurations"] += 1


def main():
    counts = {"configurations": 0, "lambda_checks": 0, "omitted_row_checks": 0}
    atoms = list(itertools.product((0, 1, 3), (0, .25, 1, 3)))
    for size in (1, 2, 3):
        for rows in itertools.combinations_with_replacement(atoms, size):
            for upper in (2, 50):
                for c in (0, .001, .1, .5, 1, 2):
                    for eps in (1e-8, .01, .5):
                        check(rows, c, eps, upper, counts)
    rng = random.Random(20260908)
    for _ in range(3000):
        upper = rng.choice((2, 50))
        rows = [(rng.choice((0.0, rng.uniform(0, 200))),
                 rng.choice((0.0, 10 ** rng.uniform(-8, 2))))
                for _ in range(rng.randint(1, 20))]
        c = rng.uniform(0, upper)
        eps = rng.choice((1e-8, .01, .5))
        check(rows, c, eps, upper, counts)

        # Numerical implementation may omit near-zero forecasts in the fit.
        threshold = 1e-3
        kept = [(y, b) for y, b in rows if b > threshold]
        omitted = [(y, b) for y, b in rows if b <= threshold]
        m = median(kept, upper)
        ce = max(c, eps)
        lam = rng.random()
        s = math.exp((1 - lam) * math.log(ce) + lam * math.log(max(m, eps)))
        slack = ((ce - c) * math.fsum(b for _, b in kept)
                 + abs(s - c) * math.fsum(b for _, b in omitted))
        tol = 1e-8 * max(1.0, loss(rows, c))
        assert loss(rows, s) <= loss(rows, c) + slack + tol
        counts["omitted_row_checks"] += 1

    # The epsilon bound is necessary and attained exactly in this example.
    eps = .01
    assert loss([(0.0, 1.0)], eps) - loss([(0.0, 1.0)], 0.0) == eps
    # Shrinkage is not in general an exact optimizer of the original objective.
    assert loss([(3.0, 1.0)], math.sqrt(3.0)) > loss([(3.0, 1.0)], 3.0)
    print(json.dumps({"status": "PASS", **counts,
                      "floor_slack_sharp_example": {"y": 0, "b": 1, "c": 0, "m": 0},
                      "exact_optimality_counterexample": {"y": 3, "b": 1, "c": 1, "m": 3, "lambda": .5}}, indent=2))


if __name__ == "__main__":
    main()
