"""Analytic tolerance intervals on the frozen A-design/B-screen menus.

This retrospective audit only reads v8 calibration artifacts.  It does not
fit heads, redesign thresholds, resample, or open later evaluation artifacts.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import datetime
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'evidence/revision_v9_screen_tolerance'
INPUTS = [
    'evidence/revision_v8_floor/PROTOCOL.json',
    'evidence/revision_v8_floor/FROZEN.json',
    'evidence/revision_v8_floor/CALIBRATION_ITEM_STATISTICS.npz',
]
METHODS = ['error', 'row', 'mixed', 'demand', 'weight_descending']
LABELS = dict(error='Error', row='Excess', mixed='Mixed', demand='Ratio',
              weight_descending='Descending')
TAU = Fraction(1, 10**12)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def initialize(out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        'utc': now(),
        'scope': 'Retrospective analytic audit of the v8 frozen A-design/B-screen menus, informed by the v8 review; not prospective preregistration.',
        'input_sha256': {p: sha(ROOT / p) for p in INPUTS},
        'code_sha256': sha(__file__),
        'floors': [.35, .40, .45],
        'screen_families': [60, 20],
        'primary_screen_family': 60,
        'methods': METHODS,
        'candidate_eligibility': 'Existing frozen threshold exists and existing B screen passes. No threshold redesign, B recomputation, or removal based on later outcomes.',
        'utilities': 'Single A-window case and exposure coverage, reconstructed exactly from frozen accepted row/weight counts and existing calibration aggregate denominators. Check against stored floating values.',
        'selection': 'Among eligible candidates within epsilon of maximum A exposure, maximize A case coverage, then A exposure, then METHODS order. Case policy unchanged.',
        'analytic_interval': 'For target k, L=max_j(d_j)-d_k. U=min_j[max_i(d_i)-d_j] over candidates with lexicographic priority above k. Selection is [L,U), or empty if L>=U; infinity if no higher-priority candidate. Report exact count-based rational endpoints and decimal percentage points.',
        'implementation_tolerance': 'Also report intervals after the inherited 1e-12 coverage-fraction admission slack: replace each admission gap g by max(0,g-1e-12). This differs by at most 1e-10 percentage points.',
        'queries': 'All individual intervals and all-four-cell intersections at each floor/family; choices at epsilon=0,.5,1,2 percentage points; A Descending-minus-Ratio exposure differences; the candidate ending each interval.',
        'validation': 'An independent cross-multiplied rational eligibility oracle at every boundary, immediately below/above, midpoints and the specified grid; interval-membership reconstruction; U20/U60 equality. No production selector imports.',
        'model_fits': 0,
        'threshold_redesigns': 0,
        'new_resampling_draws': 0,
        'evaluation_openings': 0,
        'new_external_openings': 0,
    }
    write(out / 'PROTOCOL.json', protocol)
    (out / 'PROTOCOL.sha256').write_text(sha(out / 'PROTOCOL.json') + '  PROTOCOL.json\n')
    print('PROTOCOL LOCKED', sha(out / 'PROTOCOL.json'), flush=True)


def frac_json(value):
    if value is None:
        return None
    return {'numerator': value.numerator, 'denominator': value.denominator,
            'coverage_fraction': float(value), 'percentage_points': float(100 * value)}


def exact_integer(value):
    result = int(round(float(value)))
    assert float(value) == result
    return result


def priority(record):
    return record['c'], record['d'], -METHODS.index(record['method'])


def derive_intervals(menu, slack):
    maximum = max(r['d'] for r in menu)
    admissions = {r['method']: max(Fraction(0), maximum-r['d']-slack) for r in menu}
    intervals = {}
    for target in menu:
        better = [r for r in menu if priority(r) > priority(target)]
        lower = admissions[target['method']]
        upper = min((admissions[r['method']] for r in better), default=None)
        terminators = [] if upper is None else [r['method'] for r in better if admissions[r['method']] == upper]
        intervals[target['method']] = dict(lower=lower, upper=upper,
            empty=upper is not None and lower >= upper, terminators=terminators)
    return intervals, admissions


def direct_oracle(menu, epsilon, slack):
    # Independent comparison uses integer cross-products for utility gaps;
    # there is no call to the interval formula or a production selector.
    maximum = sorted((r['d'] for r in menu))[-1]
    allowed = []
    for rec in menu:
        gap = maximum-rec['d']
        rhs = epsilon+slack
        if gap.numerator*rhs.denominator <= rhs.numerator*gap.denominator:
            allowed.append(rec)
    allowed.sort(key=lambda r: METHODS.index(r['method']))
    allowed.sort(key=lambda r: r['d'], reverse=True)
    allowed.sort(key=lambda r: r['c'], reverse=True)
    return allowed[0]['method']


def interval_choice(intervals, epsilon):
    active = [method for method, z in intervals.items() if not z['empty']
              and epsilon >= z['lower'] and (z['upper'] is None or epsilon < z['upper'])]
    assert len(active) == 1, (epsilon, active)
    return active[0]


def serialize_interval(z):
    return dict(lower=frac_json(z['lower']), upper=frac_json(z['upper']),
                lower_inclusive=True, upper_inclusive=False,
                upper_unbounded=z['upper'] is None, empty=z['empty'],
                ending_candidates=z['terminators'])


def run(out):
    if (out / 'RESULTS.json').exists():
        raise FileExistsError('Preserve existing results; use a new --output directory.')
    protocol = json.loads((out / 'PROTOCOL.json').read_text())
    assert sha(__file__) == protocol['code_sha256']
    for path, expected in protocol['input_sha256'].items():
        assert sha(ROOT / path) == expected, path
    source_protocol = json.loads((ROOT / INPUTS[0]).read_text())
    assert source_protocol['methods'] == METHODS
    frozen = json.loads((ROOT / INPUTS[1]).read_text())
    with np.load(ROOT / INPUTS[2]) as z:
        totals = z['A_total'].sum(axis=0)
    n, weight = map(exact_integer, totals)
    plans = []
    internal = {}
    oracle_checks = 0
    stored_utility_checks = 0
    for plan in frozen['plans']:
        for family in protocol['screen_families']:
            menu = []
            excluded = []
            for method in METHODS:
                rec = plan['policies'][method]
                good = rec['threshold'] is not None and rec['screens'][str(family)]['pass']
                if not good:
                    excluded.append(method)
                    continue
                c = Fraction(int(rec['A']['rows']), n)
                d = Fraction(exact_integer(rec['A']['weight']), weight)
                assert abs(float(c)-rec['A']['c']) < 1e-15
                assert abs(float(d)-rec['A']['d']) < 1e-15
                stored_utility_checks += 2
                menu.append(dict(method=method, c=c, d=d))
            intervals, admissions = derive_intervals(menu, Fraction(0))
            implementation, imp_admissions = derive_intervals(menu, TAU)
            for choices, breaks, slack in [(intervals, admissions, Fraction(0)),
                                           (implementation, imp_admissions, TAU)]:
                points = set(breaks.values()) | {Fraction(0), Fraction(1,200), Fraction(1,100), Fraction(1,50)}
                ordered = sorted(breaks.values())
                points |= {(a+b)/2 for a,b in zip(ordered, ordered[1:])}
                for boundary in ordered:
                    probe = min(Fraction(1,10**15), boundary/2) if boundary else Fraction(1,10**15)
                    if boundary > 0:
                        points.add(boundary-probe)
                    points.add(boundary+probe)
                points.add(max(ordered)+1)
                for epsilon in sorted(points):
                    assert interval_choice(choices,epsilon) == direct_oracle(menu,epsilon,slack)
                    oracle_checks += 1
            by_method = {r['method']:r for r in menu}
            difference = (by_method['weight_descending']['d']-by_method['demand']['d']) if all(m in by_method for m in ('weight_descending','demand')) else None
            original = max(menu,key=lambda r:(r['d'],-METHODS.index(r['method'])))['method']
            assert original == plan['selected_by_family'][str(family)]['d']
            rec = dict(floor=plan['floor'],cell=plan['cell'],e_trees=plan['e_trees'],w_trees=plan['w_trees'],family=family,
                excluded_candidates=excluded, eligible_candidates=[r['method'] for r in menu],
                original_exposure_choice=original,
                candidate_utilities={r['method']: {'case':frac_json(r['c']), 'exposure':frac_json(r['d'])} for r in menu},
                descending_minus_ratio_exposure=frac_json(difference),
                intervals={m:serialize_interval(z) for m,z in intervals.items()},
                implementation_intervals={m:serialize_interval(z) for m,z in implementation.items()},
                choices_at_epsilon_pp={str(float(100*eps)):direct_oracle(menu,eps,TAU) for eps in (Fraction(0),Fraction(1,200),Fraction(1,100),Fraction(1,50))})
            plans.append(rec)
            internal[(plan['floor'], family, plan['cell'])] = intervals['demand']
    intersections=[]
    for floor in protocol['floors']:
        for family in protocol['screen_families']:
            all_intervals=[z for (f,k,_),z in internal.items() if f==floor and k==family]
            assert len(all_intervals)==4
            lower=max(z['lower'] for z in all_intervals)
            finite=[z['upper'] for z in all_intervals if z['upper'] is not None]
            upper=min(finite) if finite else None
            empty=any(z['empty'] for z in all_intervals) or (upper is not None and lower>=upper)
            intersections.append(dict(floor=floor,family=family,lower=frac_json(lower),upper=frac_json(upper),
                lower_inclusive=True,upper_inclusive=False,empty=empty,
                implementation_lower=frac_json(max(Fraction(0),lower-TAU)),
                implementation_upper=frac_json(None if upper is None else max(Fraction(0),upper-TAU))))
    for floor in protocol['floors']:
        for cell in sorted({p['cell'] for p in plans}):
            primary=next(p for p in plans if p['floor']==floor and p['cell']==cell and p['family']==60)
            inherited=next(p for p in plans if p['floor']==floor and p['cell']==cell and p['family']==20)
            assert {k:v for k,v in primary.items() if k!='family'} == {k:v for k,v in inherited.items() if k!='family'}
    result = dict(utc=now(),protocol_sha256=sha(out/'PROTOCOL.json'),source_totals={'A_rows':n,'A_exposure':weight},
        plans=plans,ratio_intersections=intersections,labels=LABELS,
        interpretation='At floor .35 the published .5 pp band does not recover Ratio in this A-design/B-screen experiment. A sufficiently larger band can do so; floor is therefore the cause of switching only relative to the declared .5 pp alternative, not for all possible epsilon.',
        limitations=['All intervals are conditional on frozen fitted heads, thresholds, A utilities and inherited B-screen results.',
                     'They are sensitivity ranges, not confidence intervals, a noise estimate, population guarantees, or a recommendation to increase epsilon.',
                     'Only calibration artifacts were read; no later performance is reported for these composed selection rules.'],
        model_fits=0,threshold_redesigns=0,new_resampling_draws=0,evaluation_openings=0,new_external_openings=0)
    write(out/'RESULTS.json',result)
    verification=dict(utc=now(),status='PASS',stored_utility_checks=stored_utility_checks,
        rational_boundary_oracle_checks=oracle_checks,plans=24,candidate_screen_family_equality=True,
        interval_endpoint_convention='lower inclusive; upper exclusive, because the higher-case candidate enters at the upper endpoint',
        code_sha256=sha(__file__),protocol_sha256=sha(out/'PROTOCOL.json'),results_sha256=sha(out/'RESULTS.json'))
    write(out/'VERIFICATION.json',verification)
    lines=['# v9: tolerance on the frozen A-design/B-screen menus','',
        'Retrospective, calibration-only analytic audit. No refits, threshold redesigns, new draws or later-array reads. U20 and U60 give identical candidates and selections. Epsilon values below are percentage points; intervals include the lower endpoint and exclude the upper endpoint. Existing implementation admission slack is 1e-10 pp.','',
        '| Floor | Heads | Descending minus Ratio, A exposure | Ratio interval | Choice at 0 / .5 / 1 / 2 pp |','|---|---|---:|---|---|']
    for rec in plans:
        if rec['family']!=60: continue
        z=rec['intervals']['demand']; lower=z['lower']['percentage_points']; upper=z['upper']['percentage_points'] if z['upper'] else float('inf')
        diff=rec['descending_minus_ratio_exposure']; gap='ineligible' if diff is None else f"{diff['percentage_points']:.9f}"
        choices=' / '.join(LABELS[x] for x in rec['choices_at_epsilon_pp'].values())
        lines.append(f"| {rec['floor']:.2f} | {rec['e_trees']}/{rec['w_trees']} | {gap} | [{lower:.9f}, {upper:.9f}) | {choices} |")
    lines += ['', 'All-four-cell Ratio intersections:']
    for rec in intersections:
        if rec['family']==60:
            lines.append(f"- Floor {rec['floor']:.2f}: [{rec['lower']['percentage_points']:.9f}, {rec['upper']['percentage_points']:.9f}) pp.")
    lines += ['', result['interpretation'], '', 'The interval ends when the higher-case Mixed candidate joins. All candidate intervals, exact rational count-based boundaries, the numerical-slack variants and boundary replay checks are retained in RESULTS.json and VERIFICATION.json.', '',
              'Reproduce in a new output directory:', '', '```bash', 'python code/analyze_screen_tolerance_v9.py --output reproduction_outputs/v9_screen_tolerance --initialize', 'python code/analyze_screen_tolerance_v9.py --output reproduction_outputs/v9_screen_tolerance --run', '```','']
    (out/'SUMMARY.md').write_text('\n'.join(lines))
    print(json.dumps({'verification':verification,'intersections':intersections},indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--initialize',action='store_true')
    group.add_argument('--run',action='store_true')
    args=parser.parse_args()
    initialize(args.output) if args.initialize else run(args.output)
