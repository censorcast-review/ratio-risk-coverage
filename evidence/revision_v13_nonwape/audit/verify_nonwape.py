"""Independent v13 verifier: saved arrays/receipts only; no fits/raw data.

Does not import the experiment runner or its statistical helpers. Reconstructs
all paired bootstrap draws with the declared RNG and evaluates saved policies.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

EXPECTED_PROTOCOL = 'ea6184685bf7699a134edc823c7b4d54b4ad0f60c8d848a8a163148c8b0f08bd'
ORDER = ['Error', 'Excess', 'Mixed', 'Ratio', 'Descending']
ALL = ORDER + ['GateCase', 'GateExposure']
ATOL = 1e-12


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def timestamp(value):
    return datetime.fromisoformat(value)


class Checks:
    def __init__(self):
        self.count = 0
        self.failures = []
        self.max_numeric_difference = 0.

    def check(self, condition, label):
        self.count += 1
        if not condition:
            self.failures.append(label)

    def equal(self, actual, expected, label):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict), label + ':dict')
            if not isinstance(actual, dict):
                return
            for k, value in expected.items():
                self.check(k in actual, label + ':key:' + str(k))
                if k in actual:
                    self.equal(actual[k], value, label + '/' + str(k))
            return
        if isinstance(expected, (list, tuple)):
            self.check(isinstance(actual, (list, tuple)), label + ':list')
            if not isinstance(actual, (list, tuple)):
                return
            self.check(len(actual) == len(expected), label + ':length')
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.equal(a, b, label + '/' + str(i))
            return
        if expected is None or isinstance(expected, (str, bool, int)):
            self.check(actual == expected, label)
            return
        if isinstance(expected, (float, np.floating)):
            ok = isinstance(actual, (float, int, np.floating, np.integer))
            diff = abs(actual-expected) if ok else float('inf')
            if math.isfinite(diff):
                self.max_numeric_difference = max(self.max_numeric_difference, diff)
            self.check(ok and diff <= ATOL, label)
            return
        raise TypeError(type(expected))

    def array(self, actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected)
        self.check(a.shape == b.shape, label + ':shape')
        if a.shape != b.shape:
            return
        if b.dtype.kind in 'bui':
            self.check(np.array_equal(a, b), label + ':exact')
        else:
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any():
                self.max_numeric_difference = max(self.max_numeric_difference,
                                                  float(np.max(np.abs(a[finite]-b[finite]))))
            self.check(np.allclose(a, b, rtol=0, atol=ATOL, equal_nan=True), label + ':numeric')


def read_arrays(path, checks, label):
    with zipfile.ZipFile(path) as z:
        checks.check(z.testzip() is None, label + ':zip_crc')
    with np.load(path, allow_pickle=False) as f:
        return {k: np.array(f[k], copy=True) for k in f.files}


def hashes(dataset, role, rows, salt):
    values = []
    for row in rows:
        text = 'v13-' + salt + '-20260913|' + dataset + '|' + role + '|' + str(int(row))
        values.append(int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:16], 16))
    return np.asarray(values, dtype=np.uint64)


def measure(a, selected):
    n = len(a['exposure'])
    accepted = int(np.count_nonzero(selected))
    mass = float(np.sum(a['exposure'][selected]))
    lost = float(np.sum(a['loss'][selected]))
    return dict(accepted_n=accepted, n=n, accepted_loss=lost,
                accepted_exposure=mass, case_coverage=accepted/n,
                exposure_coverage=mass/float(np.sum(a['exposure'])),
                risk=lost/mass if mass else None)


def accept(a, method, policy):
    if not policy['eligible']:
        return np.zeros(a['loss'].size, dtype=bool)
    score, tie = a[method], a['tie']
    cut, hash_cut = policy['threshold'], np.uint64(policy['boundary_hash'])
    return np.less(score, cut) | (np.equal(score, cut) & np.less_equal(tie, hash_cut))


def design(a, method, rd, floor):
    score, tie, losses, weights = a[method], a['tie'], a['loss'], a['exposure']
    permutation = sorted(range(len(score)), key=lambda i: (float(score[i]), int(tie[i])))
    L, W, last = 0., 0., None
    for number, i in enumerate(permutation, start=1):
        L += float(losses[i])
        W += float(weights[i])
        if number >= floor*len(score) and W > 0 and L <= rd*W:
            last = i
    if last is None:
        return {'eligible': False, 'threshold': None, 'boundary_hash': None}
    p = {'eligible': True, 'threshold': float(score[last]), 'boundary_hash': int(tie[last])}
    p['A'] = measure(a, accept(a, method, p))
    return p


def bootstrap(a, masks, seed, count):
    """Independent aggregation ordering; same specified multinomial RNG."""
    n = len(a['loss'])
    source = [a['exposure']]
    # Reverse candidate order and put FP/exposure/count in a different order
    # from the experiment's serialization and aggregation code.
    reverse = list(reversed(ALL))
    for name in reverse:
        m = masks[name]
        source.extend([a['loss']*m, a['exposure']*m, m.astype(np.float64)])
    basis = np.asarray(source, dtype=np.float64).T
    sums = np.empty((count, basis.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    probabilities = np.repeat(1.0/n, n)
    for first in range(0, count, 128):
        last = min(first+128, count)
        resampled_counts = rng.multinomial(n, probabilities, size=last-first)
        sums[first:last] = resampled_counts.dot(basis)
    result = {}
    r = a['_cap']
    for j, method in enumerate(reverse):
        L, W, N = sums[:, 1+3*j:4+3*j].T
        risk = np.full(count, np.nan)
        np.divide(L, W, out=risk, where=W > 0)
        result[method] = {'c': N/n, 'd': W/sums[:, 0], 'risk': risk, 'g': (L-r*W)/n}
    return result


def comparison(first, second, measured, draws, alpha):
    if first is None or second is None:
        return {'available': False}
    result = {'available': True, 'first': first, 'second': second, 'interval_level': 1-alpha}
    for letter, field in [('c', 'case_coverage'), ('d', 'exposure_coverage')]:
        result['delta_'+letter+'_pp'] = 100.0*(measured[first][field]-measured[second][field])
        differences = 100.0*(draws[first][letter]-draws[second][letter])
        result['interval_'+letter+'_pp'] = (np.quantile(differences, [alpha/2, 1-alpha/2],
                    method='linear').tolist() if np.isfinite(differences).all() else None)
    return result


def check_dataset(root, dataset, protocol, global_checks):
    c = Checks()
    d, e = root/'evidence'/dataset, root/'evidence'
    result = read(d/'RESULT.json')
    summary = {'dataset': dataset, 'status_in_experiment': result['status']}
    if result['status'] != 'completed':
        summary.update(status='NOT_VERIFIABLE_AS_COMPLETED', checks=0, failures=[])
        return summary
    fixed, screen, complete, data, started = [read(d/f) for f in
                     ('A_FIXED.json', 'B_SCREEN.json', 'COMPLETE.json', 'DATA.json', 'STARTED.json')]
    freeze, download = read(e/'FREEZE_RECEIPT.json'), read(e/('DOWNLOAD_'+dataset+'.json'))
    psha = digest(e/'PROTOCOL.json')
    for name, rec in [('download', download), ('started', started), ('A_fixed', fixed), ('result', result)]:
        c.equal(rec['protocol_sha256'], psha, name+':protocol_hash')
    c.equal(screen['A_fixed_sha256'], digest(d/'A_FIXED.json'), 'B:A_hash_chain')
    c.equal(result['A_fixed_sha256'], digest(d/'A_FIXED.json'), 'E:A_hash_chain')
    c.equal(result['B_screen_sha256'], digest(d/'B_SCREEN.json'), 'E:B_hash_chain')
    dates = [protocol['frozen_utc'], freeze['utc'], download['started_utc'], download['completed_utc'],
             started['utc'], data['utc'], fixed['utc'], screen['utc'], result['utc'], complete['utc']]
    dt = [timestamp(v) for v in dates]
    for i in range(1, len(dt)):
        c.check(dt[i-1] <= dt[i], 'local_timestamp_order:'+str(i))
    for name, expected in complete['files'].items():
        c.check((d/name).is_file(), 'complete_file_exists:'+name)
        if (d/name).is_file():
            c.equal(digest(d/name), expected, 'complete_hash:'+name)
    for name, expected in fixed['model_sha256'].items():
        c.equal(digest(d/name), expected, 'A_model_hash:'+name)
    summary['unmanifested_files'] = sorted(p.name for p in d.iterdir()
                    if p.is_file() and p.name not in complete['files'] and p.name != 'COMPLETE.json')
    summary['complete_sha256'] = digest(d/'COMPLETE.json')
    splits = read_arrays(d/'SPLITS.npz', c, 'splits')
    nt, ne = data['source_train_n'], data['source_test_n']
    train_order = sorted(range(nt), key=lambda i: int(hashes(dataset, 'official_train', [i], 'split')[0]))
    test_order = sorted(range(ne), key=lambda i: int(hashes(dataset, 'official_test', [i], 'split')[0]))
    edge, third = int(.7*nt), ne//3
    expected_splits = {'T': train_order[:edge], 'H': train_order[edge:], 'A': test_order[:third],
                       'B': test_order[third:2*third], 'E': test_order[2*third:]}
    for key, values in expected_splits.items():
        c.array(splits[key], np.array(values, dtype=np.int64), 'split:'+key)
        c.equal(data['counts'][key], len(values), 'split_count:'+key)
    arrays = {role: read_arrays(d/(role+'_ARRAYS.npz'), c, 'arrays:'+role) for role in ['A', 'B', 'E']}
    for role, a in arrays.items():
        n = len(splits[role])
        c.equal(len(a['loss']), n, role+':length')
        c.array(a['tie'], hashes(dataset, 'official_test', splits[role], 'ties'), role+':tie_hash')
        c.check(bool(np.isfinite(a['loss']).all() and np.isfinite(a['exposure']).all()), role+':finite_outcomes')
        c.check(bool((a['exposure'] >= 1).all() and (a['loss'] >= 0).all() and
                     (a['loss'] <= a['exposure']).all()), role+':FP_and_exposure_bounds')
        c.check(bool((a['loss'] == np.floor(a['loss'])).all() and
                     (a['exposure'] == np.floor(a['exposure'])).all()), role+':integer_counts')
        for method in ALL:
            c.check(a[method].shape == (n,) and bool(np.isfinite(a[method]).all()), role+':score:'+method)
    A = arrays['A']
    cap = .95*sum(float(x) for x in A['loss'])/sum(float(x) for x in A['exposure'])
    rd = .95*cap
    c.equal(fixed['r'], cap, 'cap_from_A')
    c.equal(fixed['design_cap'], rd, 'design_cap_from_A')
    c.equal(result['r'], cap, 'result_cap')
    c.equal(result['design_cap'], rd, 'result_design_cap')
    c.equal(fixed['floor'], .4, 'A_floor')
    c.equal(result['floor_A'], .4, 'result_floor')
    for role, a in arrays.items():
        a['_cap'] = cap
        err, w = a['Error'], a['exposure']
        c.check(bool((err >= 0).all() and (err <= w).all()), role+':error_clip')
        c.array(a['Excess'], err-cap*w, role+':Excess_definition')
        c.array(a['Mixed'], (err-cap*w)/(.5+.5*w/fixed['mean_H_w']), role+':Mixed_definition')
        c.array(a['Ratio'], err/w, role+':Ratio_definition')
        c.array(a['Descending'], -w, role+':Descending_definition')
    policies = {method: design(A, method, rd, .4) for method in ALL}
    for method in ALL:
        c.equal(fixed['policies'][method], policies[method], 'A_policy:'+method)
    chosen = {}
    for utility, field in [('case', 'case_coverage'), ('exposure', 'exposure_coverage')]:
        candidate, optimum = None, -1.
        for method in ORDER:
            if policies[method]['eligible'] and policies[method]['A'][field] > optimum:
                candidate, optimum = method, policies[method]['A'][field]
        chosen[utility] = candidate
    c.equal(fixed['menu_selected'], chosen, 'menu_selection')
    measures, boots = {}, {}
    for role in ['B', 'E']:
        a = arrays[role]
        masks = {method: accept(a, method, policies[method]) for method in ALL}
        for method in ALL:
            c.array(a['mask_'+method], masks[method], role+':mask:'+method)
        measures[role] = {method: measure(a, mask) for method, mask in masks.items()}
        saved = read_arrays(d/(role+'_BOOTSTRAP.npz'), c, role+':saved_bootstrap')
        seed = protocol['datasets'][dataset]['seed']*100 + (1 if role == 'B' else 2)
        boots[role] = bootstrap(a, masks, seed, 20000)
        for method in ALL:
            for field in ['c', 'd', 'risk', 'g']:
                c.array(saved[method+'_'+field], boots[role][method][field], role+':draws:'+method+':'+field)
            c.equal(result[role][method], measures[role][method], role+':metrics:'+method)
    q = 1-.05/14
    c.equal(screen['quantile'], q, 'B:quantile')
    c.equal(screen['family_size'], 14, 'B:family_size')
    passed = {}
    for method in ALL:
        rec = dict(measures['B'][method])
        bd = boots['B'][method]
        upper = float(np.quantile(bd['g'], q, method='linear'))
        finite = bool(np.isfinite(bd['risk']).all())
        passed[method] = bool(policies[method]['eligible'] and finite and upper < 0)
        rec.update(signed_excess_upper=upper,
                   risk_percentile_upper=float(np.quantile(bd['risk'], q, method='linear')) if finite else None,
                   passed=passed[method], eligible_A=policies[method]['eligible'])
        c.equal(screen['results'][method], rec, 'B:screen:'+method)
        c.equal(result['B'][method], rec, 'result:Bscreen:'+method)
    primary = comparison(chosen['exposure'], chosen['case'], measures['E'], boots['E'], .05/4)
    primary['both_pass_B'] = all(chosen[k] is not None and passed[chosen[k]] for k in ['case', 'exposure'])
    c.equal(result['primary'], primary, 'primary')
    success = bool(primary['available'] and primary['both_pass_B'] and
            primary['interval_c_pp'] is not None and primary['interval_d_pp'] is not None and
            primary['interval_c_pp'][1] < 0 and primary['interval_d_pp'][0] > 0)
    c.equal(result['joint_success'], success, 'joint_success')
    for utility, method in [('case', 'GateCase'), ('exposure', 'GateExposure')]:
        secondary = comparison(method if policies[method]['eligible'] else None,
                               chosen[utility], measures['E'], boots['E'], .05/8)
        secondary['both_pass_B'] = bool(policies[method]['eligible'] and chosen[utility] is not None
                                       and passed[method] and passed[chosen[utility]])
        c.equal(result['secondary_gate_minus_menu'][utility], secondary, 'secondary:'+utility)
    for role in ['A', 'B', 'E']:
        w = arrays[role]['exposure']
        exp = {'n': len(w), 'mean': float(np.mean(w)), 'sd': float(np.std(w)),
               'minimum': float(min(w)), 'maximum': float(max(w)),
               'quantiles': dict(zip(['0','.25','.5','.75','1'],
                                     [float(v) for v in np.quantile(w, [0,.25,.5,.75,1])])),
               'share_one_prediction': float(np.count_nonzero(w==1)/len(w)),
               'share_zero_predictions': float(np.count_nonzero(w==0)/len(w))}
        c.equal(result['exposure'][role], exp, 'exposure:'+role)
    summary.update(status='PASS' if not c.failures else 'FAIL', checks=c.count,
                   failures=c.failures, max_absolute_numeric_difference=c.max_numeric_difference,
                   all_saved_bootstrap_draws_reconstructed=True,
                   draws_per_role=20000, policies=7, joint_success=success,
                   menu_selected=chosen, primary=primary)
    global_checks.check(not c.failures, dataset+':all_checks')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--datasets', nargs='+', default=['bibtex','mediamill'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root/'audit'/'VERIFICATION_v13.json'
    c = Checks()
    protocol = read(root/'evidence'/'PROTOCOL.json')
    freeze = read(root/'evidence'/'FREEZE_RECEIPT.json')
    c.equal(digest(root/'evidence'/'PROTOCOL.json'), EXPECTED_PROTOCOL, 'expected_protocol_hash')
    c.equal(freeze['protocol_sha256'], EXPECTED_PROTOCOL, 'freeze_protocol_hash')
    c.check(freeze['raw_directory_empty'] is True, 'freeze_record_raw_directory_empty')
    c.check(protocol['frozen_before_raw_download'] is True, 'protocol_freeze_declaration')
    c.equal(freeze['code_sha256'], protocol['code_sha256'], 'freeze_code_map')
    for file, h in protocol['code_sha256'].items():
        c.equal(digest(root/file), h, 'frozen_code_hash:'+file)
    outputs = []
    for dataset in args.datasets:
        if not (root/'evidence'/dataset/'RESULT.json').is_file():
            outputs.append({'dataset':dataset, 'status':'PENDING'})
            continue
        try:
            outputs.append(check_dataset(root, dataset, protocol, c))
        except Exception as exc:
            c.check(False, dataset+':exception:'+type(exc).__name__+':'+str(exc))
            outputs.append({'dataset':dataset, 'status':'ERROR', 'error_type':type(exc).__name__,
                            'error':str(exc)})
    report = {'utc':datetime.now(timezone.utc).isoformat(),
              'status':'FAIL' if c.failures else ('PENDING' if any(v['status']=='PENDING' for v in outputs) else 'PASS'),
              'verifier_sha256':digest(Path(__file__)), 'protocol_sha256':EXPECTED_PROTOCOL,
              'absolute_tolerance':ATOL, 'relative_tolerance':0,
              'independence':'Separate implementation; no import of runner, raw-data read, unpickling, or model fitting.',
              'limits':['Checks saved scores and masks; does not independently reconstruct fitted model predictions.',
                        'Local receipt chronology is verified; no externally certified timestamp or universal untouched-data claim.',
                        'Bootstrap reconstruction does not establish row exchangeability or population risk guarantees.',
                        'H outcome/features and top1 fallback counts are outside this saved A/B/E-array audit.'],
              'global_checks':c.count, 'global_failures':c.failures, 'datasets':outputs,
              'all_two_dataset_joint_success':all(v.get('joint_success') is True for v in outputs)
                    if len(outputs)==2 and all(v['status']=='PASS' for v in outputs) else None}
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'], 'global_failures':c.failures,
                     'datasets':[{k:v[k] for k in ['dataset','status','checks','failures'] if k in v} for v in outputs]}, indent=2))
    if report['status']=='FAIL':
        raise SystemExit(1)


if __name__=='__main__':
    main()
