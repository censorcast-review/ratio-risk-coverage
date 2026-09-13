"""Direct-gate research implementation; execution status is recorded separately.

This is a small custom primal-dual MLP, NOT a SelectiveNet reproduction.
Scores passed for reference methods must be ascending (smaller = accept first).
No model fitting or policy selection reads B or E outcomes.
"""
from dataclasses import asdict, dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class GateConfig:
    seed: int = 20260913
    hidden: tuple = (32, 16)
    epochs: int = 40
    batch_size: int = 2048
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    dual_learning_rate: float = 0.2
    dual_maximum: float = 100.0
    gradient_clip: float = 5.0
    feature_clip: float = 10.0
    case_floor: float = 0.40
    cap_multiplier: float = 0.95
    design_multiplier: float = 0.95
    screen_alpha: float = 0.05
    screen_multiplicity: int = 4


@dataclass
class Split:
    X: np.ndarray
    loss: np.ndarray
    exposure: np.ndarray
    blocks: Optional[np.ndarray] = None

    def validate(self):
        self.X = np.asarray(self.X, dtype=float)
        self.loss = np.asarray(self.loss, dtype=float)
        self.exposure = np.asarray(self.exposure, dtype=float)
        n = len(self.loss)
        if self.X.ndim != 2 or self.X.shape[0] != n or self.exposure.shape != (n,):
            raise ValueError('X, loss, and exposure must have matching rows')
        if n == 0 or not np.isfinite(self.loss).all() or not np.isfinite(self.exposure).all():
            raise ValueError('nonempty finite outcome arrays are required')
        if (self.loss < 0).any() or (self.exposure < 0).any() or self.exposure.sum() <= 0:
            raise ValueError('loss/exposure must be nonnegative; total exposure positive')
        if np.isinf(self.X).any():
            raise ValueError('feature infinity is invalid; NaN is allowed for imputation')
        if self.blocks is not None:
            self.blocks = np.asarray(self.blocks)
            if self.blocks.shape != (n,) or not np.issubdtype(self.blocks.dtype, np.integer) or (self.blocks < 0).any():
                raise ValueError('blocks must be nonnegative integer indices')
        return self


def _sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _forward(X, p):
    z1 = X @ p['W1'] + p['b1']
    h1 = np.maximum(z1, 0)
    z2 = h1 @ p['W2'] + p['b2']
    h2 = np.maximum(z2, 0)
    z3 = (h2 @ p['W3'] + p['b3']).ravel()
    a = _sigmoid(z3)
    return a, (X, z1, h1, z2, h2, z3)


def _gradient(p, cache, a, coefficient):
    X, z1, h1, z2, h2, z3 = cache
    # The clipped forward sigmoid has zero derivative outside its active range.
    d3 = coefficient * a * (1-a) * (np.abs(z3) < 35) / len(a)
    d3 = d3[:, None]
    d2 = (d3 @ p['W3'].T) * (z2 > 0)
    d1 = (d2 @ p['W2'].T) * (z1 > 0)
    return {'W1': X.T @ d1, 'b1': d1.sum(0),
            'W2': h1.T @ d2, 'b2': d2.sum(0),
            'W3': h2.T @ d3, 'b3': d3.sum(0)}


@dataclass
class GateModel:
    parameters: dict
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    config: GateConfig
    utility: str
    training_exposure_mean: float
    trace: list

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.median) or np.isinf(X).any():
            raise ValueError('invalid feature matrix')
        return np.clip((np.where(np.isnan(X), self.median, X)-self.mean)/self.scale,
                       -self.config.feature_clip, self.config.feature_clip)

    def score(self, X):
        # Unclipped logits preserve the ordering even if probabilities saturate.
        _, cache = _forward(self.transform(X), self.parameters)
        return -cache[-1]

    def predict(self, X):
        """Acceptance logit: larger values mean accept earlier (not probability)."""
        return -self.score(X)

    def predict_acceptance_probability(self, X):
        return _sigmoid(self.predict(X))

    def save(self, path):
        import json
        np.savez_compressed(path, **self.parameters, median=self.median,
                            mean=self.mean, scale=self.scale,
                            metadata=np.array(json.dumps({'config': asdict(self.config),
                                'utility': self.utility, 'training_exposure_mean': self.training_exposure_mean,
                                'trace': self.trace})))


def fit_direct_gate(train: Split, r: float, utility: str, config: GateConfig = GateConfig()):
    """Fit on TRAIN outcomes only; r is the precomputed A-only cap.

    X must use a frozen, common numeric feature encoding. Nominal integer IDs
    must be one-hot encoded by the caller using TRAIN-derived levels (+ OOV).
    """
    train.validate()
    if utility not in ('case', 'exposure') or not np.isfinite(r) or r <= 0:
        raise ValueError('utility must be case/exposure and cap must be positive')
    if config.epochs < 1 or config.batch_size < 1 or not (0 <= config.case_floor < 1):
        raise ValueError('invalid training configuration')
    rng = np.random.default_rng(config.seed)
    median = np.array([np.nanmedian(c) if not np.isnan(c).all() else 0.0 for c in train.X.T])
    imputed = np.where(np.isnan(train.X), median, train.X)
    mean = imputed.mean(0)
    scale = imputed.std(0)
    scale[scale < 1e-12] = 1.0
    X = np.clip((imputed-mean)/scale, -config.feature_clip, config.feature_clip)
    n, d = X.shape
    h1, h2 = config.hidden
    p = {'W1': rng.normal(0, np.sqrt(2/max(d, 1)), (d, h1)), 'b1': np.zeros(h1),
         'W2': rng.normal(0, np.sqrt(2/h1), (h1, h2)), 'b2': np.zeros(h2),
         'W3': rng.normal(0, 0.05, (h2, 1)), 'b3': np.array([0.0])}
    first, second = {k: np.zeros_like(v) for k, v in p.items()}, {k: np.zeros_like(v) for k, v in p.items()}
    ew = float(train.exposure.mean())
    u = np.ones(n) if utility == 'case' else train.exposure/ew
    g = (train.loss-r*train.exposure)/ew
    lam, gamma, step = min(1/r, config.dual_maximum), 0.0, 0
    trace = []
    for epoch in range(config.epochs):
        permutation = rng.permutation(n)
        for start in range(0, n, config.batch_size):
            idx = permutation[start:start+config.batch_size]
            a, cache = _forward(X[idx], p)
            grad = _gradient(p, cache, a, -u[idx]+lam*g[idx]-gamma)
            norm = np.sqrt(sum(float(np.square(v).sum()) for v in grad.values()))
            factor = min(1.0, config.gradient_clip/max(norm, 1e-30))
            step += 1
            for k in p:
                q = grad[k]*factor
                first[k] = .9*first[k]+.1*q
                second[k] = .999*second[k]+.001*q*q
                if k.startswith('W'):
                    p[k] *= 1-config.learning_rate*config.weight_decay
                p[k] -= config.learning_rate*(first[k]/(1-.9**step))/(np.sqrt(second[k]/(1-.999**step))+1e-8)
        # Full TRAIN aggregates make the dual update independent of minibatch size.
        aa = np.concatenate([_forward(X[s:s+config.batch_size], p)[0]
                             for s in range(0, n, config.batch_size)])
        signed_excess = float(np.mean(aa*g))
        floor_violation = float(config.case_floor-aa.mean())
        lam = float(np.clip(lam+config.dual_learning_rate*signed_excess, 0, config.dual_maximum))
        gamma = float(np.clip(gamma+config.dual_learning_rate*floor_violation, 0, config.dual_maximum))
        trace.append({'epoch': epoch+1, 'utility': float(np.mean(aa*u)), 'soft_case_coverage': float(aa.mean()),
                      'normalized_signed_excess': signed_excess, 'lambda': lam, 'gamma': gamma})
        if not all(np.isfinite(v).all() for v in p.values()):
            raise FloatingPointError('nonfinite optimization; report failure without retry/tuning')
    return GateModel(p, median, mean, scale, config, utility, ew, trace)


def fit_gate(X_train, loss_train, exposure_train, cap_from_A, utility,
             seed=20260913, floor=0.40, config=None):
    """Minimal learner API. B and E features/outcomes are deliberately absent."""
    if config is None:
        config = GateConfig(seed=seed, case_floor=floor)
    elif config.seed != seed or config.case_floor != floor:
        raise ValueError('seed/floor arguments must agree with the frozen config')
    return fit_direct_gate(Split(X_train, loss_train, exposure_train), cap_from_A, utility, config)


def metrics(loss, exposure, mask):
    mask = np.asarray(mask, dtype=bool)
    loss, exposure = np.asarray(loss), np.asarray(exposure)
    W, L = float(exposure[mask].sum()), float(loss[mask].sum())
    return {'accepted_n': int(mask.sum()), 'n': len(mask), 'accepted_loss': L, 'accepted_exposure': W,
            'case_coverage': float(mask.mean()), 'exposure_coverage': W/float(exposure.sum()),
            'risk': L/W if W > 0 else None}


def design_threshold(score, loss, exposure, r_design, floor, utility):
    """Exact feasible prefixes, grouping all exact score ties together."""
    score, loss, exposure = np.asarray(score), np.asarray(loss), np.asarray(exposure)
    if not np.isfinite(score).all() or score.ndim != 1 or len(score) != len(loss):
        raise ValueError('one finite score per row is required')
    if utility not in ('case', 'exposure'):
        raise ValueError('unknown utility')
    order = np.argsort(score, kind='stable')
    sorted_score = score[order]
    endpoints = np.r_[np.flatnonzero(sorted_score[:-1] != sorted_score[1:]), len(score)-1]
    L, W = np.cumsum(loss[order])[endpoints], np.cumsum(exposure[order])[endpoints]
    n = endpoints+1
    feasible = (W > 0) & (n/len(score) >= floor) & (L <= r_design*W+1e-12*np.maximum(W, 1))
    ix = np.flatnonzero(feasible)
    if not len(ix):
        return {'eligible': False, 'threshold': None, 'reason': 'no A-feasible nonzero-exposure prefix'}
    objective = n/len(score) if utility == 'case' else W/exposure.sum()
    # Equal utility chooses more rows; exact residual ties choose earlier threshold.
    winner = max(ix, key=lambda j: (objective[j], n[j], -j))
    threshold = float(sorted_score[endpoints[winner]])
    mask = score <= threshold
    return {'eligible': True, 'threshold': threshold, 'A': metrics(loss, exposure, mask)}


def _counts_check(split, counts):
    counts = np.asarray(counts)
    if split.blocks is None or counts.ndim != 2 or counts.shape[0] < 2:
        raise ValueError('bootstrap requires block indices and a draw-by-block matrix')
    if not np.isfinite(counts).all() or (counts < 0).any() or counts.shape[1] <= split.blocks.max():
        raise ValueError('invalid block-count draws')
    if (counts.sum(1) <= 0).any():
        raise ValueError('each draw must sample at least one block')
    return counts


def _block_values(split, mask, number):
    return np.stack([np.bincount(split.blocks, weights=x, minlength=number)
                     for x in (np.ones(len(mask)), mask.astype(float), split.exposure,
                               mask*split.exposure, mask*split.loss)], axis=1)


def screen_policy(split, mask, r, counts, alpha, multiplicity):
    counts = _counts_check(split, counts)
    draws = counts @ _block_values(split, mask, counts.shape[1])
    if (draws[:, 0] <= 0).any() or (draws[:, 3] <= 0).any():
        return {'passed': False, 'reason': 'bootstrap has zero sampled rows or accepted exposure',
                'alpha': alpha, 'multiplicity': multiplicity}
    q = 1-alpha/multiplicity
    excess = (draws[:, 4]-r*draws[:, 3])/draws[:, 0]
    upper = float(np.quantile(excess, q, method='linear'))
    return {'passed': bool(upper < 0), 'signed_excess_upper': upper,
            'risk_percentile_upper': float(np.quantile(draws[:, 4]/draws[:, 3], q, method='linear')),
            'alpha': alpha, 'multiplicity': multiplicity, 'draws': len(counts),
            'interpretation': 'descriptive block-bootstrap screen, not a population guarantee'}


def paired_contrast(split, first, second, counts=None):
    m1, m0 = metrics(split.loss, split.exposure, first), metrics(split.loss, split.exposure, second)
    result = {'delta_case_pp': 100*(m1['case_coverage']-m0['case_coverage']),
              'delta_exposure_pp': 100*(m1['exposure_coverage']-m0['exposure_coverage'])}
    if counts is not None:
        counts = _counts_check(split, counts)
        b1 = counts @ _block_values(split, first, counts.shape[1])
        b0 = counts @ _block_values(split, second, counts.shape[1])
        if (b1[:, 0] <= 0).any() or (b1[:, 2] <= 0).any():
            result['interval_status'] = 'undefined: empty rows or total exposure in a resample'
        else:
            for name, values in [('case', 100*(b1[:, 1]-b0[:, 1])/b1[:, 0]),
                                 ('exposure', 100*(b1[:, 3]-b0[:, 3])/b1[:, 2])]:
                result[f'delta_{name}_95_interval_pp'] = np.quantile(values, [.025, .975], method='linear').tolist()
            result['interval_status'] = 'conditional on frozen policies; no refitting or reselection'
    return result


def run_direct_gate_comparison(train, A, B, E, reference_scores, config=GateConfig(),
                               screen_counts=None, evaluation_counts=None):
    """Return (JSON-compatible report, fitted models).

    reference_scores = {method: {'A': ..., 'B': ..., 'E': ...}}, in declared
    tie-break order, using the five existing methods. Fits exactly two gates.
    Counts are supplied by the caller from a separately frozen resampling plan.
    B/E values cannot affect fits, thresholds, or A-only menu selection.
    """
    for split in (train, A, B, E):
        split.validate()
    if len(reference_scores) != 5:
        raise ValueError('exactly five declared reference score families required')
    if len({x.X.shape[1] for x in (train, A, B, E)}) != 1:
        raise ValueError('common feature encoding required')
    r = config.cap_multiplier*float(A.loss.sum()/A.exposure.sum())
    r_design = config.design_multiplier*r
    candidates, models, selected = {}, {}, {}
    for utility in ('case', 'exposure'):
        for method, by_split in reference_scores.items():
            key = f'{utility}:{method}'
            candidates[key] = design_threshold(by_split['A'], A.loss, A.exposure,
                                                 r_design, config.case_floor, utility)
        eligible = [m for m in reference_scores if candidates[f'{utility}:{m}']['eligible']]
        if eligible:
            # Python max keeps the first declared method when the objective ties.
            chosen = max(eligible, key=lambda m: candidates[f'{utility}:{m}']['A'][f'{utility}_coverage'])
            selected[f'menu_{utility}'] = (chosen, candidates[f'{utility}:{chosen}'], reference_scores[chosen])
        model = fit_direct_gate(train, r, utility, config)
        models[f'direct_{utility}'] = model
        scores = {name: model.score(split.X) for name, split in [('A', A), ('B', B), ('E', E)]}
        design = design_threshold(scores['A'], A.loss, A.exposure, r_design, config.case_floor, utility)
        candidates[f'{utility}:direct'] = design
        if design['eligible']:
            selected[f'direct_{utility}'] = ('direct', design, scores)
    policies, e_masks = {}, {}
    for name in ('menu_case', 'menu_exposure', 'direct_case', 'direct_exposure'):
        if name not in selected:
            policies[name] = {'eligible': False, 'reason': 'no eligible A design; no B fallback'}
            continue
        method, design, scores = selected[name]
        policy = {'eligible': True, 'method': method, 'threshold': design['threshold'], 'A': design['A']}
        for splitname, split in [('B', B), ('E', E)]:
            mask = scores[splitname] <= design['threshold']
            policy[splitname] = metrics(split.loss, split.exposure, mask)
            if splitname == 'E':
                e_masks[name] = mask
            elif screen_counts is not None:
                policy['B_screen'] = screen_policy(B, mask, r, screen_counts,
                                                     config.screen_alpha, config.screen_multiplicity)
        policy.setdefault('B_screen', {'passed': None, 'reason': 'no frozen bootstrap counts supplied'})
        policies[name] = policy
    contrasts = {}
    for first, second in [('menu_exposure', 'menu_case'), ('direct_exposure', 'direct_case'),
                          ('direct_case', 'menu_case'), ('direct_exposure', 'menu_exposure')]:
        name = f'{first}_minus_{second}'
        contrasts[name] = paired_contrast(E, e_masks[first], e_masks[second], evaluation_counts) if first in e_masks and second in e_masks else {'status': 'undefined: an A design is ineligible'}
    report = {'status': 'research prototype; real-data status must be supplied by the caller',
              'config': asdict(config), 'cap_A_only': r, 'design_cap': r_design,
              'reference_order': list(reference_scores), 'A_candidates': candidates,
              'policies': policies, 'E_contrasts': contrasts,
              'selection_note': 'B and E never change a threshold, family, model, or cap',
              'scope': 'custom direct gate; not SelectiveNet and not evidence about all learned selectors'}
    return report, models
