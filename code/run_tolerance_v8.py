"""Retrospective exposure-tolerance selection on frozen candidate menus.

Initialize the declared protocol before running.  Calibration choices are frozen
before consumed evaluation outcomes are reopened.  No model or threshold fits.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json

import numpy as np
from policy_audit import METHODS, choose, mask, metrics, scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/revision_v8_tolerance'
EPSILONS = [0., .001, .0025, .005, .01]
PRIMARY = .005
HEADS = [(220, 220), (660, 220), (220, 660), (660, 660)]
NUMERIC_TOLERANCE = 1e-12


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def initialize(out):
    out.mkdir(parents=True, exist_ok=False)
    inputs = ['code/policy_audit.py']
    for arm in ('censored', 'complete'):
        for name in ('FROZEN.json', 'CALIBRATION.npz', 'EVALUATION.npz', 'RESULTS.json'):
            inputs.append(f'evidence/matched_censoring/{arm}/{name}')
    for name in ('complete_220_220', 'bike'):
        for file in ('FULL_DRAWS.npz', 'POINT.json', 'FULL_RESULTS.json'):
            inputs.append(f'evidence/revision_v7_stability/{name}/{file}')
    protocol = dict(
        utc=now(), scope='Retrospective selection reanalysis using consumed data and existing fitted heads. '
        'Not preregistered confirmation, risk certification, or improved moment estimation.',
        primary_epsilon=PRIMARY, epsilon_grid=EPSILONS,
        epsilon_units='coverage fraction; primary .005 means 0.5 percentage points',
        epsilon_rationale='An explicitly allowed calibration exposure utility concession. '
        'Not estimated from noise or tuned to select Ratio; the entire fixed grid is reported.',
        head_pairs=HEADS, arms=['censored', 'complete'], cap_relative_factor=.95,
        original_floor=.35, methods=list(METHODS), numerical_tolerance=NUMERIC_TOLERANCE,
        original_policy='Frozen fitted heads, original cap, all original whole-tie candidate thresholds, '
        'and original threshold-existence eligibility; case choice unchanged.',
        exposure_selection='Among eligible candidates with min-window exposure coverage at least '
        'the maximum minus epsilon (comparison tolerance 1e-12), maximize min-window case coverage, '
        'then exposure coverage, then original METHODS order. Epsilon zero is lexicographic '
        'tie-breaking; positive epsilon changes the declared objective.',
        freeze='Write all calibration choices and hash before loading EVALUATION.npz or '
        'reading original RESULTS.json outcomes during the run. Input file hashing is not outcome access.',
        evaluation='Later consumed evaluation: both coverage utilities, risk, contrasts against '
        'unchanged case choice and changes from original exposure choice; evaluation exposure '
        'concession is not bounded by calibration epsilon.',
        stability='Reuse exactly the existing 200 threshold-redesigned draws for complete 220/220 '
        'and Bike; fitted heads, scores, caps, and normalization fixed. All selection/no-choice '
        'frequencies for every epsilon. Not new draws or eight-cell resampling.',
        validation='Independent ordered lexicographic oracle; original choices; calibration '
        'statistics reconstructed from arrays; every selected concession <=epsilon+1e-12; '
        'epsilon=0 reproduces original when exposure maximum unique; raw draw reconstruction.',
        model_fits=0, threshold_fits=0, new_external_openings=0,
        code_sha256=sha(__file__), input_sha256={p: sha(ROOT/p) for p in inputs})
    dump(out/'PROTOCOL.json', protocol)
    (out/'PROTOCOL.sha256').write_text(sha(out/'PROTOCOL.json') + '  PROTOCOL.json\n')
    print('PROTOCOL LOCKED', sha(out/'PROTOCOL.json'), flush=True)


def select_band(utility, eligible, epsilon):
    """First column case, second exposure. Stable METHODS final tie breaker."""
    ids = np.flatnonzero(eligible)
    if not len(ids):
        return -1
    best_d = float(utility[ids, 1].max())
    admissible = ids[utility[ids, 1] >= best_d-epsilon-NUMERIC_TOLERANCE]
    return int(max(admissible, key=lambda j: (utility[j, 0], utility[j, 1], -j)))


def oracle(utility, eligible, epsilon):
    candidates = [(float(d), float(c), index) for index, (c, d) in enumerate(utility)
                  if bool(eligible[index])]
    if not candidates:
        return -1
    maximum = sorted(candidates, reverse=True)[0][0]
    allowed = [(c, d, -index) for d, c, index in candidates
               if maximum-d <= epsilon+NUMERIC_TOLERANCE]
    return -sorted(allowed, reverse=True)[0][2]


def validate_choice(utility, eligible, epsilon, chosen, original):
    assert chosen == oracle(utility, eligible, epsilon)
    if chosen < 0:
        assert not np.any(eligible)
        return
    assert eligible[chosen]
    best = max(utility[j, 1] for j in np.flatnonzero(eligible))
    assert best-utility[chosen, 1] <= epsilon+NUMERIC_TOLERANCE
    if epsilon == 0 and sum(utility[j, 1] >= best-NUMERIC_TOLERANCE
                            for j in np.flatnonzero(eligible)) == 1:
        assert chosen == original


def toy_checks():
    rng = np.random.default_rng(20260915)
    count = 0
    for _ in range(150):
        utility = rng.integers(0, 100, size=(5, 2))/100
        eligible = rng.integers(0, 2, size=5).astype(bool)
        ids = np.flatnonzero(eligible)
        original = int(max(ids, key=lambda j: utility[j, 1])) if len(ids) else -1
        for epsilon in EPSILONS:
            got = select_band(utility, eligible, epsilon)
            validate_choice(utility, eligible, epsilon, got, original)
            count += 1
    # Exact tie: secondary case utility can deliberately differ from stable argmax.
    utility = np.array([[.4, .9], [.7, .9], [.9, .899], [.3, .8], [.2, .7]])
    assert select_band(utility, np.ones(5, bool), 0.) == 1
    assert select_band(utility, np.ones(5, bool), .001) == 2
    return count+2


def check_metrics(got, expected):
    for key in ('rows', 'c', 'd', 'loss', 'weight', 'risk'):
        if expected[key] is None:
            assert got[key] is None
        else:
            assert np.isclose(got[key], expected[key], rtol=1e-12, atol=1e-9), (key, got, expected)


def analyze_calibration(out, protocol):
    settings, candidate_rows, selection_rows = [], [], []
    for arm in protocol['arms']:
        folder = ROOT/f'evidence/matched_censoring/{arm}'
        frozen = json.loads((folder/'FROZEN.json').read_text())
        with np.load(folder/'CALIBRATION.npz') as z:
            cal = {k: z[k] for k in z.files}
        L, W = abs(cal['y']-cal['f']), cal['y']
        windows = np.unique(cal['blocks'])
        for et, wt in HEADS:
            plan = next(p for p in frozen['plans'] if p['kind'] == 'relative'
                and p['value'] == .95 and (p['e_trees'], p['w_trees']) == (et, wt))
            policies = plan['policies']
            utility = np.zeros((len(METHODS), 2))
            eligible = np.array([policies[m]['threshold'] is not None for m in METHODS])
            candidates = {}
            for j, method in enumerate(METHODS):
                score = scores(cal[f'e{et}'], cal[f'w{wt}'], plan['cap'], method, frozen['T'])
                a = mask(score, policies[method]['threshold'])
                mm = [metrics(L[cal['blocks'] == b], W[cal['blocks'] == b],
                              a[cal['blocks'] == b]) for b in windows]
                for got, expected in zip(mm, policies[method]['calibration']):
                    check_metrics(got, expected)
                    if eligible[j]:
                        assert got['c'] >= .35
                        assert got['weight'] > 0
                        assert got['loss'] <= plan['cap']*got['weight']+1e-9
                utility[j] = [min(m[k] for m in mm) for k in ('c', 'd')]
                candidates[method] = dict(eligible=bool(eligible[j]),
                    threshold=policies[method]['threshold'], calibration=mm,
                    min_case_coverage=float(utility[j, 0]),
                    min_exposure_coverage=float(utility[j, 1]))
                candidate_rows.append(dict(arm=arm, e_trees=et, w_trees=wt,
                    method=method, eligible=bool(eligible[j]),
                    min_case_coverage=utility[j, 0], min_exposure_coverage=utility[j, 1],
                    threshold=policies[method]['threshold']))
            original = {o: choose(policies, o) for o in ('c', 'd')}
            assert original == plan['selected']
            original_d = METHODS.index(original['d']) if original['d'] else -1
            bands = []
            for epsilon in EPSILONS:
                chosen = select_band(utility, eligible, epsilon)
                validate_choice(utility, eligible, epsilon, chosen, original_d)
                method = METHODS[chosen] if chosen >= 0 else None
                entry = dict(epsilon=epsilon, selected=dict(c=original['c'], d=method),
                    changed_from_original=method != original['d'],
                    calibration_exposure_concession=float(utility[original_d, 1]-utility[chosen, 1])
                        if chosen >= 0 else None,
                    calibration_case_change=float(utility[chosen, 0]-utility[original_d, 0])
                        if chosen >= 0 else None,
                    calibration_exposure_change=float(utility[chosen, 1]-utility[original_d, 1])
                        if chosen >= 0 else None,
                    selected_min_case_coverage=float(utility[chosen, 0]) if chosen >= 0 else None,
                    selected_min_exposure_coverage=float(utility[chosen, 1]) if chosen >= 0 else None)
                bands.append(entry)
                selection_rows.append(dict(arm=arm, e_trees=et, w_trees=wt, epsilon=epsilon,
                    original_case=original['c'], original_exposure=original['d'],
                    selected_exposure=method, **{k: entry[k] for k in (
                    'calibration_exposure_concession', 'calibration_case_change',
                    'selected_min_case_coverage', 'selected_min_exposure_coverage')}))
            settings.append(dict(arm=arm, e_trees=et, w_trees=wt, cap=plan['cap'],
                T=frozen['T'], candidates=candidates, original_selected=original, bands=bands))
        del cal, L, W
    csv_write(out/'CALIBRATION_CANDIDATES.csv', candidate_rows)
    csv_write(out/'CALIBRATION_SELECTIONS.csv', selection_rows)
    return settings


def summary(x):
    x = np.asarray(x, float)
    if not len(x):
        return None
    return dict(mean=float(x.mean()), min=float(x.min()), max=float(x.max()),
                q025=float(np.quantile(x, .025)), median=float(np.median(x)),
                q975=float(np.quantile(x, .975)))


def frequencies(selected):
    return {name: dict(count=int(np.sum(selected == j)),
                      frequency=float(np.mean(selected == j)))
            for j, name in list(enumerate(METHODS))+[(-1, 'no_choice')]}


def analyze_stability(out):
    results, rows, arrays = [], [], {}
    for name in ('complete_220_220', 'bike'):
        folder = ROOT/f'evidence/revision_v7_stability/{name}'
        point = json.loads((folder/'POINT.json').read_text())
        with np.load(folder/'FULL_DRAWS.npz') as z:
            data = {k: z[k] for k in ('statistics', 'totals', 'utilities', 'eligible', 'selected', 'exists')}
        stat, total = data['statistics'], data['totals']
        case = stat[..., 0]/total[:, None, :, 0]
        exposure = stat[..., 1]/total[:, None, :, 1]
        utility = np.stack([case.min(axis=2), exposure.min(axis=2)], axis=2)
        assert np.allclose(utility, data['utilities'], rtol=1e-14, atol=1e-14)
        tol = 1e-12 if name == 'bike' else 1e-9
        eligible = data['exists'] & np.all((stat[..., 1] > 0) &
            (stat[..., 0] >= .35*total[:, None, :, 0]) &
            (stat[..., 2] <= point['design_cap']*stat[..., 1]+tol), axis=2)
        assert np.array_equal(eligible, data['eligible'])
        original = np.where(eligible, utility[:, :, 1], -np.inf).argmax(axis=1)
        original[~eligible.any(axis=1)] = -1
        assert np.array_equal(original, data['selected'][:, 1])
        old_summary = json.loads((folder/'FULL_RESULTS.json').read_text())
        assert frequencies(original) == old_summary['selections']['d']['frequencies']
        bands = []
        chosen_grid = np.full((len(EPSILONS), len(utility)), -1, int)
        for k, epsilon in enumerate(EPSILONS):
            concession, case_gain = [], []
            chosen = chosen_grid[k]
            for i, (u, ok, orig) in enumerate(zip(utility, eligible, original)):
                chosen[i] = select_band(u, ok, epsilon)
                validate_choice(u, ok, epsilon, int(chosen[i]), int(orig))
                if chosen[i] >= 0:
                    concession.append(u[orig, 1]-u[chosen[i], 1])
                    case_gain.append(u[chosen[i], 0]-u[orig, 0])
            ff = frequencies(chosen)
            bands.append(dict(epsilon=epsilon, frequencies=ff,
                changed_from_original=int(np.sum(chosen != original)),
                calibration_exposure_concession=summary(concession),
                calibration_case_gain=summary(case_gain)))
            for method, freq in ff.items():
                rows.append(dict(setting=name, epsilon=epsilon, method=method, **freq))
        results.append(dict(setting=name, draws=len(utility), original_frequencies=frequencies(original),
            bands=bands, scope='Same v7 full threshold-redesign draws; no refitting or new bootstrap draws.'))
        arrays[name+'_original'] = original
        arrays[name+'_band_choices'] = chosen_grid
    csv_write(out/'STABILITY_FREQUENCIES.csv', rows)
    np.savez_compressed(out/'STABILITY_CHOICES.npz', epsilons=np.array(EPSILONS), **arrays)
    return results


def evaluate(settings, out, frozen_sha):
    assert sha(out/'FROZEN_CHOICES.json') == frozen_sha
    dump(out/'EVALUATION_OPEN.json', dict(utc=now(), frozen_choices_sha256=frozen_sha,
        disclosure='Previously consumed evaluation outcomes, opened only after new calibration choices froze.'))
    rows = []
    for arm in ('censored', 'complete'):
        folder = ROOT/f'evidence/matched_censoring/{arm}'
        with np.load(folder/'EVALUATION.npz') as z:
            data = {k: z[k] for k in ('f', 'y', 'e220', 'e660', 'w220', 'w660')}
        original_results = json.loads((folder/'RESULTS.json').read_text())
        L, W = abs(data['y']-data['f']), data['y']
        for setting in [s for s in settings if s['arm'] == arm]:
            et, wt = setting['e_trees'], setting['w_trees']
            existing = next(p for p in original_results['menus'] if p['kind'] == 'relative'
                and p['value'] == .95 and (p['e_trees'], p['w_trees']) == (et, wt))
            test = {}
            for method in METHODS:
                score = scores(data[f'e{et}'], data[f'w{wt}'], setting['cap'], method, setting['T'])
                a = mask(score, setting['candidates'][method]['threshold'])
                test[method] = metrics(L, W, a)
                check_metrics(test[method], existing['test'][method])
            setting['evaluation_candidates'] = test
            case = test[setting['original_selected']['c']]
            orig = test[setting['original_selected']['d']]
            for band in setting['bands']:
                method = band['selected']['d']
                selected = test[method] if method else None
                band['evaluation'] = dict(case_selected=case, exposure_selected=selected,
                    contrast_vs_case={k: selected[k]-case[k] for k in ('c', 'd')},
                    change_vs_original_exposure={k: selected[k]-orig[k] for k in ('c', 'd')},
                    exposure_concession=orig['d']-selected['d']) if selected else None
                if selected:
                    rows.append(dict(arm=arm, e_trees=et, w_trees=wt, epsilon=band['epsilon'],
                        original_exposure=setting['original_selected']['d'], selected_exposure=method,
                        case_coverage=selected['c'], exposure_coverage=selected['d'], risk=selected['risk'],
                        contrast_case=selected['c']-case['c'], contrast_exposure=selected['d']-case['d'],
                        case_change_vs_original=selected['c']-orig['c'],
                        exposure_change_vs_original=selected['d']-orig['d'],
                        calibration_exposure_concession=band['calibration_exposure_concession']))
        del data, L, W
    csv_write(out/'EVALUATION.csv', rows)


def write_readme(out, results):
    lines = ['# Exposure tolerance selection: retrospective v8 analysis', '',
        'The primary allowed calibration exposure concession was fixed at **0.5 percentage points** '
        'before this run. The complete declared grid is 0, 0.1, 0.25, 0.5, and 1.0 percentage points. '
        'This is an explicit utility preference, not an estimate of noise and not a parameter tuned '
        'to obtain Ratio. Positive tolerance changes the objective. At zero, only an exact numerical '
        'tie receives a case-coverage secondary criterion.', '',
        'For each of the eight frozen paired menus, retain original eligibility and thresholds. '
        'Among candidates within epsilon of the best minimum-window exposure coverage, maximize '
        'minimum-window case coverage, then exposure, then original method order. The case policy '
        'is unchanged. All five candidates and all declared bands are retained.', '',
        'The protocol and code/input hashes precede computation. All choices froze before consumed '
        'Later outcomes were reopened. Results are retrospective and conditional on fitted heads; '
        'they do not establish population risk control or generalization. The calibration concession '
        'bound does not bound the evaluation exposure concession.', '',
        '| Arm | Error/exposure trees | Original exposure choice | 0.5 pp choice | Calibration exposure concession (pp) | Evaluation case change (pp) | Evaluation exposure change (pp) |',
        '|---|---:|---|---|---:|---:|---:|']
    for setting in results['settings']:
        band = next(b for b in setting['bands'] if b['epsilon'] == PRIMARY)
        change = band['evaluation']['change_vs_original_exposure']
        lines.append(f"| {setting['arm']} | {setting['e_trees']}/{setting['w_trees']} | "
            f"{setting['original_selected']['d']} | {band['selected']['d']} | "
            f"{100*band['calibration_exposure_concession']:.4f} | {100*change['c']:+.4f} | {100*change['d']:+.4f} |")
    lines += ['', '## Conditional stability on the same v7 draws', '',
        'Exactly the existing 200 full threshold-redesign bootstrap draws were reused for complete '
        '220/220 and Bike. There are no new bootstrap replicates, model fits, or eight-cell stability '
        'experiments. Each epsilon reselects among each draw\'s existing redesigned candidates. '
        'All method and no-choice frequencies appear in STABILITY_FREQUENCIES.csv.', '']
    for setting in results['stability']:
        band = next(b for b in setting['bands'] if b['epsilon'] == PRIMARY)
        parts = [f"{m}: {x['count']}/200" for m, x in band['frequencies'].items()]
        lines.append(f"- {setting['setting']}, primary tolerance: " + '; '.join(parts) + '.')
    lines += ['', 'Method labels: error = conditional error; row = Excess; mixed = Mixed; '
        'demand = Ratio; weight_descending = exposure descending.', '',
        'Verification includes direct reconstruction of calibration and evaluation metrics, '
        'independent ordered lexicographic choices, exact-tie toy cases, preservation of original '
        'choices at zero for unique primary maxima, calibration concession bounds, reconstruction '
        'of all reused draw utilities and eligibility, and original draw frequency equality. '
        'The complete machine-readable record is RESULTS.json; hashes and execution timestamps '
        'are in PROTOCOL.json, FROZEN_CHOICES.json, START.json, EVALUATION_OPEN.json, and COMPLETE.json.', '',
        'Reproduce into a new output directory:', '',
        '```bash', 'python code/run_tolerance_v8.py --output /tmp/tolerance-v8 --initialize',
        'python code/run_tolerance_v8.py --output /tmp/tolerance-v8 --run', '```', '']
    (out/'README.md').write_text('\n'.join(lines))


def run(out):
    assert not (out/'START.json').exists(), 'Use a new output path for a new run.'
    protocol = json.loads((out/'PROTOCOL.json').read_text())
    protocol_sha = (out/'PROTOCOL.sha256').read_text().split()[0]
    assert sha(out/'PROTOCOL.json') == protocol_sha
    assert sha(__file__) == protocol['code_sha256']
    for relative, digest in protocol['input_sha256'].items():
        assert sha(ROOT/relative) == digest, relative
    dump(out/'START.json', dict(utc=now(), protocol_sha256=protocol_sha,
        code_sha256=sha(__file__), numpy_version=np.__version__))
    toy_count = toy_checks()
    settings = analyze_calibration(out, protocol)
    stability = analyze_stability(out)
    dump(out/'FROZEN_CHOICES.json', dict(utc=now(), protocol_sha256=protocol_sha,
        settings=settings, stability=stability, evaluation_outcomes_opened_during_run=0))
    frozen_sha = sha(out/'FROZEN_CHOICES.json')
    (out/'FROZEN_CHOICES.sha256').write_text(frozen_sha+'  FROZEN_CHOICES.json\n')
    print('CHOICES FROZEN', frozen_sha, flush=True)
    evaluate(settings, out, frozen_sha)
    result = dict(scope=protocol['scope'], primary_epsilon=PRIMARY, epsilon_grid=EPSILONS,
        protocol_sha256=protocol_sha, frozen_choices_sha256=frozen_sha,
        settings=settings, stability=stability)
    dump(out/'RESULTS.json', result)
    dump(out/'VALIDATION.json', dict(status='PASS', toy_oracle_checks=toy_count,
        point_band_choices_checked=40, reused_draw_band_choices_checked=2000,
        frozen_calibration_metric_reconstruction='all 8 menus, 5 candidates, both windows',
        evaluation_metric_reconstruction='all 8 menus, all 5 candidates',
        original_stability_frequencies='exact equality for complete 220/220 and Bike',
        bounds='all selected calibration exposure concessions <=epsilon+1e-12',
        exact_zero_unique_max='original selection reproduced',
        choices_before_evaluation='PASS', frozen_choices_unchanged=sha(out/'FROZEN_CHOICES.json') == frozen_sha))
    write_readme(out, result)
    dump(out/'COMPLETE.json', dict(utc=now(), code_sha256=sha(__file__),
        model_fits=0, threshold_fits=0, new_external_openings=0,
        files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
               if p.is_file() and p.name != 'COMPLETE.json'}))
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--initialize', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    initialize(args.output) if args.initialize else run(args.output)
