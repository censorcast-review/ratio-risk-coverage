"""Exact tolerance intervals on already frozen calibration candidate menus.

Run --initialize before --run. This reads no evaluation targets and creates no
resampling draws. The original floating-point selector is boundary-probed; an
independent rational-arithmetic implementation checks mathematical endpoints.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import datetime
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/revision_v9_epsilon'
METHODS = ('error', 'row', 'mixed', 'demand', 'weight_descending')
RATIO = METHODS.index('demand')
HEADS = [(220, 220), (660, 220), (220, 660), (660, 660)]
TOLERANCE = 1e-12
INPUTS = ['code/run_tolerance_v8.py', 'code/policy_audit.py',
          'evidence/matched_censoring/complete/FROZEN.json'] + [
    f'evidence/revision_v7_stability/{name}/{file}'
    for name in ('complete_220_220', 'bike')
    for file in ('POINT.json', 'FULL_DRAWS.npz')]


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def initialize(out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(utc=stamp(), analysis='Exact epsilon acceptance intervals for Ratio under Eq. (7)',
        status='Retrospective extension informed by the v8 review; not prospective preregistration.',
        scope='Four complete-sales frozen point menus at the original relative cap .95 and '
              'the existing 200 redesigned calibration menus for complete 220/220 and Bike.',
        epsilon_units='Coverage fraction in computation; percentage points in all endpoint reports.',
        priority='Eligible candidates within epsilon of maximum minimum-window exposure coverage; '
                 'maximize minimum-window case coverage, then exposure coverage, then METHODS order.',
        methods=METHODS, ratio_method='demand', existing_comparison_tolerance=TOLERANCE,
        formula='Let g_j = max_i d_i - d_j on eligible candidates and K_j=(c_j,d_j,-j). '
                'Ratio is chosen exactly on [g_R, min_{j:K_j>K_R} g_j), intersected with epsilon>=0. '
                'An ineligible Ratio or lower>=upper gives the empty set; no outranking candidate '
                'gives an infinite upper endpoint. Lower closed, upper open.',
        numerical_reporting='Mathematical intervals use exact rational representations of stored '
                'binary64 utilities. Existing selector tolerance shifts positive activation '
                'boundaries downward by at most 1e-10 percentage points; do not imply greater '
                'scientific measurement precision from the exact stored arithmetic.',
        draw_summary='For each existing draw report empty/nonempty interval, both finite endpoints, '
                'limiting competitor, and membership at .5 pp. Report empirical quantiles '
                '(min, 2.5%, 25%, median, 75%, 97.5%, max) of nonempty lower and finite upper '
                'endpoints; no confidence or population interpretation. No joint four-cell '
                'draw interval is available because only reference complete heads were resampled.',
        validation='Reconstruct existing draw utilities and eligibility from raw sufficient '
                'statistics, check archived arrays; enumerate all activation breakpoints '
                'with rational arithmetic and independently replay the original floating '
                'selector below/at/above every activation (interior probes separated by '
                '1e-8 percentage points). Explicitly validate endpoint inclusions.',
        model_fits=0, threshold_fits=0, new_draws=0, evaluation_array_reads=0,
        code_sha256=sha(__file__), input_sha256={p: sha(ROOT/p) for p in INPUTS})
    dump(out/'PROTOCOL.json', protocol)
    (out/'PROTOCOL.sha256').write_text(sha(out/'PROTOCOL.json') + '  PROTOCOL.json\n')
    print('PROTOCOL LOCKED', sha(out/'PROTOCOL.json'), flush=True)


def fraction(value):
    return Fraction.from_float(float(value))


def float_select(utilities, eligible, epsilon):
    # Independently reproduce the published v8 selector, without importing it.
    active = [j for j in range(len(METHODS)) if eligible[j]]
    if not active:
        return -1
    best = max(float(utilities[j, 1]) for j in active)
    allowed = [j for j in active if utilities[j, 1] >= best-epsilon-TOLERANCE]
    return sorted(allowed, key=lambda j: (-utilities[j, 0], -utilities[j, 1], j))[0]


def rational_select(values, eligible, epsilon):
    ids = [j for j, flag in enumerate(eligible) if flag]
    if not ids:
        return -1
    maximum = max(values[j][1] for j in ids)
    ids = [j for j in ids if maximum-values[j][1] <= epsilon]
    return max(ids, key=lambda j: (values[j][0], values[j][1], -j))


def interval(utilities, eligible):
    values = [(fraction(c), fraction(d)) for c, d in utilities]
    ids = [j for j, flag in enumerate(eligible) if flag]
    if not ids or not eligible[RATIO]:
        return dict(empty=True, reason='Ratio ineligible', lower_pp=None, upper_pp=None,
                    lower_closed=False, upper_closed=False, upper_competitors=[],
                    includes_half_pp=False), None, None
    maximum = max(values[j][1] for j in ids)
    ratio_key = (*values[RATIO], -RATIO)
    lower = maximum-values[RATIO][1]
    blockers = [j for j in ids if (*values[j], -j) > ratio_key]
    upper = min((maximum-values[j][1] for j in blockers), default=None)
    empty = upper is not None and lower >= upper
    blockers_at_upper = [METHODS[j] for j in blockers if maximum-values[j][1] == upper]
    result = dict(empty=empty, reason='Secondary-priority competitor already active' if empty else None,
        lower_pp=float(lower*100), upper_pp=float(upper*100) if upper is not None else None,
        upper_infinite=upper is None, lower_closed=not empty, upper_closed=False,
        upper_competitors=blockers_at_upper,
        includes_half_pp=not empty and lower <= Fraction(1, 200) and
             (upper is None or Fraction(1, 200) < upper),
        operational_lower_pp=max(0., float(lower-TOLERANCE)*100),
        operational_upper_pp=float(upper-TOLERANCE)*100 if upper is not None else None,
        outranking_competitors=[dict(method=METHODS[j], activation_pp=float((maximum-values[j][1])*100),
                                 case_percent=float(values[j][0]*100), exposure_percent=float(values[j][1]*100))
                               for j in blockers])
    return result, lower, upper


def verify_interval(utilities, eligible, record, lower, upper):
    values = [(fraction(c), fraction(d)) for c, d in utilities]
    ids = [j for j, flag in enumerate(eligible) if flag]
    if not ids:
        return 0
    maximum = max(values[j][1] for j in ids)
    gaps = sorted(set([Fraction(0)] + [maximum-values[j][1] for j in ids]))
    probes = set(gaps + [max(gaps)+Fraction(1, 100)])
    for a, b in zip(gaps[:-1], gaps[1:]):
        probes.add((a+b)/2)
    checks = 0
    for epsilon in probes:
        expected = not record['empty'] and lower <= epsilon and (upper is None or epsilon < upper)
        assert (rational_select(values, eligible, epsilon) == RATIO) == expected
        checks += 1
    for endpoint in gaps:
        # Zero aside, operational tolerance means Eq. (7)'s nominal boundary is
        # already inside the selector's admissible band. Probe on either side.
        for step in (-1e-10, 1e-10):
            epsilon = float(endpoint)+step
            if epsilon < 0:
                continue
            expected = not record['empty'] and float(lower) <= epsilon and (upper is None or epsilon < float(upper))
            assert (float_select(utilities, eligible, epsilon) == RATIO) == expected
            checks += 1
    assert (float_select(utilities, eligible, .005) == RATIO) == record['includes_half_pp']
    return checks+1


def quantiles(values):
    if not values:
        return None
    probs = [0., .025, .25, .5, .75, .975, 1.]
    result = np.quantile(values, probs)
    return {name: float(value) for name, value in zip(
        ['min', 'q025', 'q25', 'median', 'q75', 'q975', 'max'], result)}


def run(out):
    protocol = json.loads((out/'PROTOCOL.json').read_text())
    assert protocol['code_sha256'] == sha(__file__)
    assert all(sha(ROOT/p) == expected for p, expected in protocol['input_sha256'].items())
    frozen = json.loads((ROOT/'evidence/matched_censoring/complete/FROZEN.json').read_text())
    points, point_bounds, checks = [], [], 0
    for e, w in HEADS:
        plan = next(p for p in frozen['plans'] if p['kind'] == 'relative' and p['value'] == .95
                    and p['e_trees'] == e and p['w_trees'] == w)
        policies = plan['policies']
        eligible = np.array([policies[m]['threshold'] is not None for m in METHODS])
        utility = np.array([[min(t[k] for t in policies[m]['calibration']) for k in ('c', 'd')]
                            for m in METHODS])
        record, lower, upper = interval(utility, eligible)
        checks += verify_interval(utility, eligible, record, lower, upper)
        record.update(cell=f'{e}/{w}', e_trees=e, w_trees=w,
            candidates=[dict(method=m, eligible=bool(eligible[j]),
                             case_percent=float(utility[j, 0]*100),
                             exposure_percent=float(utility[j, 1]*100)) for j, m in enumerate(METHODS)])
        points.append(record)
        point_bounds.append((lower, upper))
    joint_lower = max(x[0] for x in point_bounds)
    joint_upper = min(x[1] for x in point_bounds if x[1] is not None)
    intersection = dict(lower_pp=float(joint_lower*100), upper_pp=float(joint_upper*100),
        lower_closed=True, upper_closed=False, empty=joint_lower >= joint_upper,
        lower_limiting_cells=[p['cell'] for p, (lo, hi) in zip(points, point_bounds) if lo == joint_lower],
        upper_limiting_cells=[p['cell'] for p, (lo, hi) in zip(points, point_bounds) if hi == joint_upper])
    stability = []
    for setting in ('complete_220_220', 'bike'):
        folder = ROOT/f'evidence/revision_v7_stability/{setting}'
        point = json.loads((folder/'POINT.json').read_text())
        with np.load(folder/'FULL_DRAWS.npz') as archive:
            arrays = {k: archive[k] for k in ('statistics', 'totals', 'utilities', 'eligible', 'exists')}
        stat, total = arrays['statistics'], arrays['totals']
        utilities = np.stack([(stat[..., 0]/total[:, None, :, 0]).min(axis=2),
                              (stat[..., 1]/total[:, None, :, 1]).min(axis=2)], axis=2)
        assert np.allclose(utilities, arrays['utilities'], rtol=1e-14, atol=1e-14)
        tolerance = 1e-12 if setting == 'bike' else 1e-9
        eligible = arrays['exists'] & np.all((stat[..., 1] > 0) &
            (stat[..., 0] >= .35*total[:, None, :, 0]) &
            (stat[..., 2] <= point['design_cap']*stat[..., 1]+tolerance), axis=2)
        assert np.array_equal(eligible, arrays['eligible'])
        records = []
        for draw, (u, ok) in enumerate(zip(utilities, eligible)):
            record, lower, upper = interval(u, ok)
            checks += verify_interval(u, ok, record, lower, upper)
            record['draw'] = draw
            records.append(record)
        nonempty = [r for r in records if not r['empty']]
        summary = dict(setting=setting, draws=len(records), nonempty=len(nonempty),
            empty=len(records)-len(nonempty),
            empty_reasons={reason: sum(r['reason'] == reason for r in records)
                for reason in sorted({r['reason'] for r in records if r['empty']})},
            half_pp_ratio_count=sum(r['includes_half_pp'] for r in records),
            half_pp_ratio_frequency=sum(r['includes_half_pp'] for r in records)/len(records),
            lower_quantiles_pp=quantiles([r['lower_pp'] for r in nonempty]),
            finite_upper_count=sum(r['upper_pp'] is not None for r in nonempty),
            infinite_upper_count=sum(r['upper_pp'] is None for r in nonempty),
            finite_upper_quantiles_pp=quantiles([r['upper_pp'] for r in nonempty if r['upper_pp'] is not None]),
            width_quantiles_pp=quantiles([r['upper_pp']-r['lower_pp'] for r in nonempty if r['upper_pp'] is not None]),
            upper_competitor_counts={m: sum(m in r['upper_competitors'] for r in nonempty) for m in METHODS})
        stability.append(summary)
        dump(out/f'{setting}_INTERVALS.json', dict(summary=summary, draws=records))
    results = dict(utc=stamp(), protocol_sha256=sha(out/'PROTOCOL.json'), points=points,
        intersection=intersection, stability=stability,
        validation=dict(status='PASS', boundary_and_interior_checks=checks,
                        rational_endpoint_checks=True, original_float_selector_replay=True,
                        original_draw_utility_and_eligibility_reconstruction=True),
        limitations=['Fixed stored calibration utilities and fitted heads; no new evidence of generalization.',
            'Empirical draw endpoint quantiles are descriptive, not confidence intervals for a population interval.',
            'No four-cell joint redesign distribution exists in the archive.',
            'Numerical tolerance changes activation endpoints by at most 1e-10 percentage points.'])
    dump(out/'RESULTS.json', results)
    dump(out/'HASHES.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(dict(intersection=intersection, stability=stability, validation=results['validation']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--initialize', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    initialize(args.output) if args.initialize else run(args.output)
