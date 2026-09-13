"""Finite-menu utility audit. Empirical feasibility is not a risk guarantee.

The same threshold design and eligibility set are used for both utilities.
No evaluation targets are accepted by the design functions.
"""
import numpy as np

METHODS = ('error', 'row', 'mixed', 'demand', 'weight_descending')


def scores(e, w, cap, method, T):
    if method == 'error':
        return e.copy()
    if method == 'weight_descending':
        return -w
    excess = e - cap * w
    if method == 'row':
        return excess
    if method not in ('mixed', 'demand'):
        raise ValueError(method)
    den = w / T if method == 'demand' else .25 + .75 * w / T
    return np.divide(excess, den, out=np.where(excess > 0, np.inf,
                     np.where(excess < 0, -np.inf, 0.)), where=den > 0)


def mask(score, threshold):
    if threshold == 'all':
        return np.ones(len(score), bool)
    return score <= threshold if threshold is not None else np.zeros(len(score), bool)


def metrics(L, W, a):
    mass = float(W[a].sum()); loss = float(L[a].sum())
    return dict(rows=int(a.sum()), c=float(a.mean()), d=mass/float(W.sum()),
                loss=loss, weight=mass, risk=loss/mass if mass > 0 else None)


def largest_threshold(score, L, W, blocks, cap, floor):
    """All complete score ties; risk need not be monotone along the ranking."""
    if not (np.isfinite(L).all() and np.isfinite(W).all()):
        raise ValueError('Nonfinite loss or exposure')
    if np.any(L < 0) or np.any(W < 0) or np.isnan(score).any():
        raise ValueError('Nonnegative loss/exposure and non-NaN scores required')
    keys = np.unique(blocks)
    full = [metrics(L[blocks == k], W[blocks == k], np.ones(sum(blocks == k), bool)) for k in keys]
    if all(x['risk'] is not None and x['risk'] <= cap for x in full):
        return 'all', full
    # Include a possible negative-infinity score tie. Positive-infinity ties
    # are included only by explicit accept-all, avoiding arithmetic sentinels.
    ts = np.unique(score[score < np.inf]); good = np.ones(len(ts), bool)
    for k in keys:
        ii = np.flatnonzero(blocks == k)
        order = ii[np.argsort(score[ii], kind='stable')]
        counts = np.searchsorted(score[order], ts, side='right')
        demand = np.r_[0., np.cumsum(W[order])][counts]
        error = np.r_[0., np.cumsum(L[order])][counts]
        good &= (demand > 0) & (counts >= floor*len(ii)) & (error <= cap*demand + 1e-9)
    ix = np.flatnonzero(good)
    if not len(ix):
        return None, [metrics(L[blocks == k], W[blocks == k], np.zeros(sum(blocks == k), bool)) for k in keys]
    t = float(ts[ix[-1]])
    if not np.isfinite(t):
        raise ValueError('Use an explicit mask for an infinite-only selected threshold')
    a = mask(score, t)
    return t, [metrics(L[blocks == k], W[blocks == k], a[blocks == k]) for k in keys]


def choose(menu, objective, eligible=None):
    if objective not in ('c', 'd'):
        raise ValueError(objective)
    candidates = [m for m in METHODS if menu[m]['threshold'] is not None and
                  (eligible is None or eligible.get(m, False))]
    if not candidates:
        return None
    # Python's stable max preserves the declared order on an exact tie.
    return max(candidates, key=lambda m: min(x[objective] for x in menu[m]['calibration']))


def sufficient_statistics(L, W, a, units, n_units):
    return np.column_stack([np.bincount(units, weights=a, minlength=n_units),
                            np.bincount(units, weights=a*W, minlength=n_units),
                            np.bincount(units, weights=a*L, minlength=n_units)])


def bootstrap_counts(n_units, seed, draws=4000):
    return np.random.default_rng(seed).multinomial(n_units, np.full(n_units, 1/n_units), size=draws)


def paired_intervals(A, B, total, counts):
    denominator = counts @ total
    values = (counts @ (B-A))[:, :2] / denominator
    return {'ci95': np.quantile(values, [.025, .975], axis=0).T.tolist(),
            'ci99_375': np.quantile(values, [.003125, .996875], axis=0).T.tolist()}
