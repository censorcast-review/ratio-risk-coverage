"""Calibration-only fixed-candidate and full-threshold cluster sensitivity.

Protocol is locked independently before execution. No evaluation files are used.
Weighted cumulative calculations are tested against literal resampled data.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import time
import zipfile

import numpy as np
from policy_audit import METHODS, scores, mask, metrics, largest_threshold, choose

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'evidence/revision_v7_stability'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def quantiles(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return None if not len(x) else dict(n=int(len(x)),
        mean=float(x.mean()), q025=float(np.quantile(x, .025)),
        median=float(np.median(x)), q975=float(np.quantile(x, .975)))


def make_scores(e, w, cap, method, T, bike=False):
    # Match original arithmetic order, not just an equivalent rank scaling.
    if bike and method == 'demand':
        excess = e - cap*w
        return np.divide(excess, w, out=np.where(excess > 0, np.inf,
            np.where(excess < 0, -np.inf, 0.)), where=w > 0)
    return scores(e, w, cap, method, T)


class Ranking:
    """Whole-tie candidate grid; each bootstrap drops absent score ties."""
    def __init__(self, score, L, W, blocks, units, n_units, cap, floor, tol):
        self.score = score
        self.ts = np.unique(score[score < np.inf])
        self.cap, self.floor, self.tol = cap, floor, tol
        self.windows = []
        for block in np.unique(blocks):
            ii = np.flatnonzero(blocks == block)
            order = ii[np.argsort(score[ii], kind='stable')]
            self.windows.append(dict(units=units[order], W=W[order], L=L[order],
                ends=np.searchsorted(score[order], self.ts, side='right'),
                total=np.column_stack([
                    np.bincount(units[ii], minlength=n_units),
                    np.bincount(units[ii], weights=W[ii], minlength=n_units),
                    np.bincount(units[ii], weights=L[ii], minlength=n_units)])))

    def redesign(self, multiplicities):
        totals = np.array([multiplicities @ win['total'] for win in self.windows])
        # Original accept-all branch compares ratios directly, with no tolerance.
        if np.all((totals[:, 1] > 0) &
                  (totals[:, 2] <= self.cap*totals[:, 1])):
            return 'all', totals, totals
        good = np.ones(len(self.ts), bool)
        pooled_count = np.zeros(len(self.ts))
        cumulatives = []
        for win, total in zip(self.windows, totals):
            mult = multiplicities[win['units']]
            ends = win['ends']
            count = np.r_[0., np.cumsum(mult)][ends]
            weight = np.r_[0., np.cumsum(mult*win['W'])][ends]
            loss = np.r_[0., np.cumsum(mult*win['L'])][ends]
            good &= ((weight > 0) & (count >= self.floor*total[0]) &
                     (loss <= self.cap*weight + self.tol))
            pooled_count += count
            cumulatives.append((count, weight, loss))
        # A score present only in a zero-multiplicity cluster is not observed.
        good &= np.diff(np.r_[0., pooled_count]) > 0
        valid = np.flatnonzero(good)
        if not len(valid):
            return None, np.zeros_like(totals), totals
        j = valid[-1]
        if not np.isfinite(self.ts[j]):
            raise ValueError('Nonfinite selected threshold in a partial policy')
        stat = np.array([[v[j] for v in win] for win in cumulatives])
        return float(self.ts[j]), stat, totals


def derive(statistics, totals, exists, cap, floor, tol):
    """Input draws x methods x windows x [count, exposure, loss]."""
    c = statistics[..., 0] / totals[:, None, :, 0]
    d = np.divide(statistics[..., 1], totals[:, None, :, 1],
                  out=np.zeros_like(statistics[..., 1]),
                  where=totals[:, None, :, 1] > 0)
    risk = np.divide(statistics[..., 2], statistics[..., 1],
                     out=np.full_like(statistics[..., 2], np.nan),
                     where=statistics[..., 1] > 0)
    eligible = exists & np.all((statistics[..., 1] > 0) &
        (statistics[..., 0] >= floor*totals[:, None, :, 0]) &
        (statistics[..., 2] <= cap*statistics[..., 1] + tol), axis=2)
    return np.stack([c.min(axis=2), d.min(axis=2)], axis=2), eligible, risk


def summarize(statistics, totals, exists, cap, floor, tol):
    utilities, eligible, risk = derive(statistics, totals, exists, cap, floor, tol)
    ndraw = len(eligible)
    out = dict(draws=ndraw, candidate_eligibility={}, selections={}, margins={})
    for j, m in enumerate(METHODS):
        out['candidate_eligibility'][m] = dict(count=int(eligible[:, j].sum()),
            frequency=float(eligible[:, j].mean()),
            min_case_coverage=quantiles(utilities[:, j, 0]),
            min_exposure_coverage=quantiles(utilities[:, j, 1]))
    selected = np.full((ndraw, 2), -1, int)
    for k, objective in enumerate(('c', 'd')):
        eligible_utility = np.where(eligible, utilities[:, :, k], -np.inf)
        winner = eligible_utility.argmax(axis=1)
        winner[~eligible.any(axis=1)] = -1
        selected[:, k] = winner
        freq = {m: dict(count=int((winner == j).sum()),
                       frequency=float((winner == j).mean()))
                for j, m in enumerate(METHODS)}
        freq['no_choice'] = dict(count=int((winner < 0).sum()),
                                frequency=float((winner < 0).mean()))
        ordered = np.sort(eligible_utility, axis=1)
        margin = np.full(ndraw, np.nan)
        has_two = eligible.sum(axis=1) >= 2
        margin[has_two] = ordered[has_two, -1] - ordered[has_two, -2]
        out['selections'][objective] = dict(frequencies=freq,
            eligible_winner_runner_up_margin=quantiles(margin),
            replicates_with_at_least_two_candidates=int(has_two.sum()),
            margin_below_0_1pp=int(np.sum(margin < .001)),
            margin_below_0_5pp=int(np.sum(margin < .005)))
        for ma, mb in [('demand', 'weight_descending'), ('demand', 'mixed')]:
            a, b = METHODS.index(ma), METHODS.index(mb)
            both_exist = exists[:, a] & exists[:, b]
            both_eligible = eligible[:, a] & eligible[:, b]
            delta = utilities[:, a, k] - utilities[:, b, k]
            out['margins'][f'{objective}__{ma}_minus_{mb}'] = dict(
                both_thresholds_exist=quantiles(delta[both_exist]),
                both_eligible=quantiles(delta[both_eligible]),
                positive_frequency_when_both_exist=float(np.mean(delta[both_exist] > 0))
                    if both_exist.any() else None)
    return out, dict(utilities=utilities, eligible=eligible, risk=risk,
                     selected=selected, exists=exists)


def build_cluster_stats(L, W, blocks, units, n_units, score_map, policies):
    total, stat = [], []
    for block in np.unique(blocks):
        b = blocks == block
        total.append(np.column_stack([
            np.bincount(units[b], minlength=n_units),
            np.bincount(units[b], weights=W[b], minlength=n_units),
            np.bincount(units[b], weights=L[b], minlength=n_units)]))
    for method in METHODS:
        a = mask(score_map[method], policies[method]['threshold'])
        each = []
        for block in np.unique(blocks):
            b = blocks == block
            each.append(np.column_stack([
                np.bincount(units[b], weights=a[b], minlength=n_units),
                np.bincount(units[b], weights=a[b]*W[b], minlength=n_units),
                np.bincount(units[b], weights=a[b]*L[b], minlength=n_units)]))
        stat.append(each)
    return np.array(total), np.array(stat)


def literal_metrics(L, W, score_map, blocks, policies):
    out = {}
    for method in METHODS:
        a = mask(score_map[method], policies[method]['threshold'])
        out[method] = dict(threshold=policies[method]['threshold'], calibration=[
            metrics(L[blocks == b], W[blocks == b], a[blocks == b])
            for b in np.unique(blocks)])
    return out


def check_frozen(reproduced, frozen):
    for method in METHODS:
        assert reproduced[method]['threshold'] == frozen[method]['threshold']
        for got, expected in zip(reproduced[method]['calibration'],
                                 frozen[method]['calibration']):
            for key in ('rows', 'c', 'd', 'loss', 'weight', 'risk'):
                if expected[key] is None:
                    assert got[key] is None, (method, key)
                else:
                    assert np.isclose(got[key], expected[key], rtol=1e-12,
                                      atol=1e-9), (method, key, got[key], expected[key])


def toy_check():
    rng = np.random.default_rng(159)
    checks = 0
    for iteration in range(40):
        n = 60
        units = np.repeat(np.arange(10), 6)
        blocks = np.tile(np.repeat([0, 1], 3), 10)
        score = rng.integers(-3, 7, size=n).astype(float)
        if iteration % 4 == 0:
            score[:2] = -np.inf
            score[-3:] = np.inf
        W = rng.integers(0, 5, size=n).astype(float)
        L = rng.uniform(0, 3, size=n)
        cap = float(rng.uniform(.2, 1.5))
        counts = rng.multinomial(10, np.ones(10)/10)
        prep = Ranking(score, L, W, blocks, units, 10, cap, .35, 1e-9)
        got, stats, totals = prep.redesign(counts)
        ix = np.repeat(np.arange(n), counts[units])
        expected, mm = largest_threshold(score[ix], L[ix], W[ix], blocks[ix], cap, .35)
        assert got == expected, (got, expected, counts)
        for j, point in enumerate(mm):
            assert np.allclose(stats[j], [point['rows'], point['weight'], point['loss']],
                               rtol=1e-12, atol=1e-9)
        checks += 1
    return dict(literal_cluster_resample_equivalence_cases=checks,
                absent_cluster_ties=True, positive_and_negative_infinity=True)


def analyse_setting(out, name, L, W, e, w, blocks, units, unit_names, plan,
                    T, bike=False, full=False):
    folder = out/name
    folder.mkdir(exist_ok=False)
    cap = plan['cap']
    design_cap = .95*cap if bike else cap
    tol = 1e-12 if bike else 1e-9
    floor = .35
    n_units = len(unit_names)
    scores_by_method = {m: make_scores(e, w, cap, m, T, bike) for m in METHODS}
    policies = plan['policies']
    original = literal_metrics(L, W, scores_by_method, blocks, policies)
    check_frozen(original, policies)
    original_choices = {o: choose(original, o) for o in ('c', 'd')}
    if 'selected' in plan:
        assert original_choices == plan['selected']
    total, stat = build_cluster_stats(L, W, blocks, units, n_units,
                                     scores_by_method, policies)
    point_totals = total.sum(axis=1)[None, ...]
    point_stats = stat.sum(axis=2)[None, ...]
    exists = np.array([policies[m]['threshold'] is not None for m in METHODS])
    point_summary, point_raw = summarize(point_stats, point_totals,
        exists[None, :], design_cap, floor, tol)
    point = dict(name=name, cap=cap, design_cap=design_cap, T=T, rows=len(W),
        clusters=n_units, windows=len(np.unique(blocks)),
        cluster_label='calendar day' if bike else 'item',
        policies=original, selected=original_choices, statistics=point_summary,
        eligibility={m: bool(point_raw['eligible'][0, j]) for j, m in enumerate(METHODS)})
    dump(folder/'POINT.json', point)
    print('POINT', name, original_choices, flush=True)
    seed, draws = 20260913, 4000
    counts = np.random.default_rng(seed).multinomial(n_units,
        np.full(n_units, 1/n_units), size=draws)
    fixed_totals = (counts @ total.transpose(1, 0, 2).reshape(n_units, -1)).reshape(
        draws, total.shape[0], 3)
    fixed_stats = (counts @ stat.transpose(2, 0, 1, 3).reshape(n_units, -1)).reshape(
        draws, len(METHODS), total.shape[0], 3)
    fixed_summary, fixed_raw = summarize(fixed_stats, fixed_totals,
        np.broadcast_to(exists, (draws, len(METHODS))), design_cap, floor, tol)
    fixed_summary.update(scope='Fixed original thresholds; eligibility and menu choice rechecked. No threshold refit.', seed=seed)
    dump(folder/'FIXED_RESULTS.json', fixed_summary)
    np.savez_compressed(folder/'FIXED_DRAWS.npz', counts=counts, units=unit_names,
        statistics=fixed_stats, totals=fixed_totals, **fixed_raw)
    print('FIXED COMPLETE', name, flush=True)
    del counts, fixed_stats, fixed_totals, fixed_raw
    validation = dict(original_frozen_metrics='PASS', original_choices='PASS')
    if full:
        preparations = {}
        for m in METHODS:
            preparations[m] = Ranking(scores_by_method[m], L, W, blocks, units,
                                       n_units, design_cap, floor, tol)
            t, s, _ = preparations[m].redesign(np.ones(n_units, int))
            assert t == policies[m]['threshold'], (name, m, t, policies[m]['threshold'])
            expected = np.array([[x['rows'], x['weight'], x['loss']]
                                 for x in policies[m]['calibration']])
            assert np.allclose(s, expected, rtol=1e-12, atol=1e-8)
        validation['all_one_threshold_redesign'] = 'PASS, all five methods'
        dump(folder/'VALIDATION.json', validation)
        seed, draws = 20260914, 200
        counts = np.random.default_rng(seed).multinomial(n_units,
            np.full(n_units, 1/n_units), size=draws)
        output_stats = np.zeros((draws, len(METHODS), total.shape[0], 3))
        output_totals = np.zeros((draws, total.shape[0], 3))
        output_exists = np.zeros((draws, len(METHODS)), bool)
        thresholds = np.full((draws, len(METHODS)), np.nan)
        all_flags = np.zeros((draws, len(METHODS)), bool)
        start = time.monotonic()
        for i, count in enumerate(counts):
            for j, method in enumerate(METHODS):
                t, s, tt = preparations[method].redesign(count)
                output_stats[i, j] = s
                output_totals[i] = tt
                output_exists[i, j] = t is not None
                all_flags[i, j] = t == 'all'
                if isinstance(t, float):
                    thresholds[i, j] = t
            if (i+1) % 20 == 0:
                print('FULL PROGRESS', name, i+1, '/', draws,
                      'seconds', round(time.monotonic()-start, 1), flush=True)
        full_summary, full_raw = summarize(output_stats, output_totals,
            output_exists, design_cap, floor, tol)
        full_summary.update(seed=seed,
            scope='Every ranking threshold redesigned within each cluster bootstrap, then both menu choices rerun. Fits/scores/cap/T fixed.',
            elapsed_seconds=time.monotonic()-start)
        dump(folder/'FULL_RESULTS.json', full_summary)
        np.savez_compressed(folder/'FULL_DRAWS.npz', counts=counts, units=unit_names,
            statistics=output_stats, totals=output_totals, thresholds=thresholds,
            accept_all=all_flags, **full_raw)
        print('FULL COMPLETE', name, flush=True)
    else:
        dump(folder/'VALIDATION.json', validation)
    return dict(name=name, point=point, fixed=fixed_summary,
                full=full_summary if full else None)


def main(out):
    protocol = json.loads((out/'PROTOCOL.json').read_text())
    expected = (out/'PROTOCOL.sha256').read_text().split()[0]
    assert sha(out/'PROTOCOL.json') == expected
    for relative, digest in protocol['input_sha256'].items():
        assert sha(ROOT/relative) == digest, relative
    dump(out/'START.json', dict(utc=now(), protocol_sha256=expected,
        source_sha256=sha(__file__), numpy_version=np.__version__))
    validation = toy_check()
    dump(out/'IMPLEMENTATION_VALIDATION.json', validation)
    print('TOY VALIDATION', validation, flush=True)
    complete = ROOT/'evidence/matched_censoring/complete'
    frozen = json.loads((complete/'FROZEN.json').read_text())
    with np.load(complete/'CALIBRATION.npz', allow_pickle=False) as z:
        cal = {k: z[k] for k in z.files}
    with np.load(ROOT/'legacy/results/objectives/cache_design/metadata.npz', allow_pickle=False) as z:
        item_ids = z['item_id']
    unit_names, series_units = np.unique(item_ids, return_inverse=True)
    assert len(cal['y']) == len(item_ids)*63
    assert np.array_equal(cal['blocks'], np.tile(np.r_[np.zeros(35, int),
                                                      np.ones(28, int)], len(item_ids)))
    units = np.repeat(series_units, 63)
    results = []
    for et, wt in protocol['m5']['head_pairs']:
        plan = next(p for p in frozen['plans'] if p['kind'] == 'relative'
            and p['value'] == .95 and p['e_trees'] == et and p['w_trees'] == wt)
        results.append(analyse_setting(out, f'complete_{et}_{wt}',
            abs(cal['y']-cal['f']), cal['y'], cal[f'e{et}'], cal[f'w{wt}'],
            cal['blocks'], units, unit_names, plan, frozen['T'],
            full=(et, wt) == (220, 220)))
    del cal, units
    bike = ROOT/'evidence/nonretail/bike'
    with np.load(bike/'CALIBRATION_default.npz', allow_pickle=False) as z:
        L, W, e, w = [z[k] for k in ('L', 'W', 'e', 'w')]
    with np.load(bike/'SPLITS.npz', allow_pickle=False) as z:
        cal_indices = z['cal']
    # Read date field only. No held-out outcome is parsed or loaded.
    with zipfile.ZipFile(ROOT/'evidence/nonretail/bike.zip') as z:
        file = next(n for n in z.namelist() if n.endswith('hour.csv'))
        date_bytes = z.read(file).splitlines()
    header = date_bytes[0].decode().split(',')
    column = header.index('dteday')
    cal_dates = np.array([date_bytes[int(i)+1].split(b',')[column].decode()
                          for i in cal_indices])
    del date_bytes
    unit_names, units = np.unique(cal_dates, return_inverse=True)
    assert len(units) == len(W)
    bf = json.loads((bike/'FROZEN.json').read_text())
    chosen = [p for p in bf['policies'] if p['variant'] == 'default' and p['factor'] == .85]
    plan = dict(cap=chosen[0]['cap'], policies={p['method']: dict(
        threshold=p['threshold'], calibration=[p['calibration']]) for p in chosen})
    results.append(analyse_setting(out, 'bike', L, W, e, w, np.zeros(len(W), int),
        units, unit_names, plan, chosen[0]['T'], bike=True, full=True))
    dump(out/'RESULTS.json', dict(scope=protocol['scope'],
        protocol_sha256=expected, settings=results))
    dump(out/'COMPLETE.json', dict(utc=now(), model_fits=0,
        evaluation_outcomes_loaded=0, new_external_openings=0,
        source_sha256=sha(__file__), files={str(p.relative_to(out)): sha(p)
            for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'COMPLETE.json'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.output)
