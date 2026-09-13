"""Prospective non-WAPE experiment. Freeze code/protocol before download.

This runner is independent of the unavailable v12 source. It does not claim
byte-identical reuse of v12's menu implementation or a SelectiveNet reproduction.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from direct_gate import fit_gate, metrics

HERE = Path(__file__).resolve().parent
MENU = ['Error', 'Excess', 'Mixed', 'Ratio', 'Descending']
METHODS = MENU + ['GateCase', 'GateExposure']


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_arrays(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    with open(temporary, 'wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    # Reopen all members; writing a partial ZIP must not count as success.
    with np.load(path, allow_pickle=False) as check:
        assert set(check.files) == set(arrays)
        for key, expected in arrays.items():
            assert np.array_equal(check[key], expected, equal_nan=True), key


def hash_values(dataset, role, rows, salt='split'):
    return np.array([int.from_bytes(hashlib.sha256(
        f'v13-{salt}-20260913|{dataset}|{role}|{int(i)}'.encode()).digest()[:8], 'big')
        for i in rows], dtype=np.uint64)


def split_indices(dataset, n_train, n_test):
    train_order = np.argsort(hash_values(dataset, 'official_train', np.arange(n_train)), kind='stable')
    test_order = np.argsort(hash_values(dataset, 'official_test', np.arange(n_test)), kind='stable')
    boundary = int(np.floor(.7 * n_train))
    third = n_test // 3
    return {'T': train_order[:boundary], 'H': train_order[boundary:],
            'A': test_order[:third], 'B': test_order[third:2 * third],
            'E': test_order[2 * third:]}


class FixedClassifier:
    def __init__(self, seed):
        self.seed, self.models, self.warning_records = seed, [], []
        self.scaler = StandardScaler(with_mean=False)

    def fit(self, X, Y):
        Z = self.scaler.fit_transform(X)
        for j in range(Y.shape[1]):
            y = np.asarray(Y[:, j]).ravel()
            if np.unique(y).size == 1:
                self.models.append(float((y.sum() + 1) / (len(y) + 2)))
                continue
            model = LogisticRegression(C=1.0, solver='liblinear', max_iter=1000,
                                       tol=1e-4, class_weight=None, random_state=self.seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(Z, y)
            self.warning_records += [{'label': j, 'category': w.category.__name__,
                                      'message': str(w.message)} for w in caught]
            self.models.append(model)
        return self

    def features_and_predictions(self, X):
        Z = self.scaler.transform(X)
        probability = np.column_stack([
            np.full(X.shape[0], model) if isinstance(model, float) else model.predict_proba(Z)[:, 1]
            for model in self.models])
        predicted = probability >= .5
        empty = ~predicted.any(axis=1)
        predicted[np.flatnonzero(empty), np.argmax(probability[empty], axis=1)] = True
        w = predicted.sum(axis=1).astype(np.float64)
        base = Z.toarray() if sparse.issparse(Z) else np.asarray(Z)
        features = np.column_stack([base, np.log1p(w)])
        return features, predicted, w, int(empty.sum())


def outcome(predicted, Y):
    return np.logical_and(predicted, np.asarray(Y) == 0).sum(axis=1).astype(np.float64)


def reference_scores(ehat, w, r, mean_h_w):
    return {'Error': ehat, 'Excess': ehat - r*w,
            'Mixed': (ehat-r*w)/(.5+.5*w/mean_h_w),
            'Ratio': ehat/w, 'Descending': -w}


def policy_mask(score, tie, policy):
    if not policy['eligible']:
        return np.zeros(len(score), dtype=bool)
    threshold, boundary = policy['threshold'], policy['boundary_hash']
    return (score < threshold) | ((score == threshold) & (tie <= boundary))


def design_policy(score, tie, loss, w, design_cap, floor):
    # All policies share a predeclared outcome-independent hash for exact ties.
    order = np.lexsort((tie, score))
    L, W = np.cumsum(loss[order]), np.cumsum(w[order])
    feasible = ((np.arange(len(score))+1)/len(score) >= floor) & (W > 0) & (L <= design_cap*W)
    good = np.flatnonzero(feasible)
    if not len(good):
        return {'eligible': False, 'threshold': None, 'boundary_hash': None,
                'reason': 'no A-feasible prefix at the declared cap and floor'}
    last = int(good[-1])  # w >= 1: both objectives increase along every prefix.
    policy = {'eligible': True, 'threshold': float(score[order[last]]),
              'boundary_hash': int(tie[order[last]])}
    policy['A'] = metrics(loss, w, policy_mask(score, tie, policy))
    return policy


def select_menu(policies, utility):
    valid = [key for key in MENU if policies[key]['eligible']]
    if not valid:
        return None
    field = 'case_coverage' if utility == 'case' else 'exposure_coverage'
    # Exact ties retain the first entry in the frozen menu order.
    return max(valid, key=lambda key: policies[key]['A'][field])


def bootstrap_rows(w, loss, masks, r, draws, seed):
    """One paired multinomial row resample for all frozen policies.

    This descriptive bootstrap assumes row exchangeability. In particular, it
    does not resolve unknown video/shot clustering in Mediamill.
    """
    n, rng = len(w), np.random.default_rng(seed)
    matrix = np.column_stack([w] + [v for mask in masks.values() for v in
                                   (mask.astype(float), mask*w, mask*loss)])
    estimates = {key: {'c': [], 'd': [], 'risk': [], 'g': []} for key in masks}
    for start in range(0, draws, 128):
        counts = rng.multinomial(n, np.full(n, 1/n), size=min(128, draws-start))
        aggregate = counts @ matrix
        for j, key in enumerate(masks):
            N, W, L = (aggregate[:, 1+3*j+k] for k in range(3))
            values = (N/n, W/aggregate[:, 0],
                      np.divide(L, W, out=np.full_like(L, np.nan), where=W > 0), (L-r*W)/n)
            for field, value in zip(('c', 'd', 'risk', 'g'), values):
                estimates[key][field].append(value)
    return {key: {field: np.concatenate(value) for field, value in rec.items()}
            for key, rec in estimates.items()}


def interval(values, alpha):
    if not np.isfinite(values).all():
        return None
    return [float(x) for x in np.quantile(values, [alpha/2, 1-alpha/2], method='linear')]


def compare(first, second, measured, boot, alpha):
    if first is None or second is None:
        return {'available': False, 'reason': 'an A-selected policy is unavailable'}
    result = {'available': True, 'first': first, 'second': second, 'interval_level': 1-alpha}
    for short, long in [('c', 'case_coverage'), ('d', 'exposure_coverage')]:
        result['delta_'+short+'_pp'] = 100*(measured[first][long]-measured[second][long])
        result['interval_'+short+'_pp'] = interval(100*(boot[first][short]-boot[second][short]), alpha)
    return result


def exposure_summary(w):
    return {'n': len(w), 'mean': float(np.mean(w)), 'sd': float(np.std(w)),
            'minimum': float(np.min(w)), 'maximum': float(np.max(w)),
            'quantiles': dict(zip(['0', '.25', '.5', '.75', '1'], [float(x) for x in
                                np.quantile(w, [0, .25, .5, .75, 1])])),
            'share_one_prediction': float(np.mean(w == 1)),
            'share_zero_predictions': float(np.mean(w == 0))}


def feature_fingerprints(X):
    X = sparse.csr_matrix(X)
    X.sort_indices()
    values = []
    for row in range(X.shape[0]):
        start, end = X.indptr[row:row+2]
        h = hashlib.sha256()
        h.update(np.asarray(X.indices[start:end], dtype='<i8').tobytes())
        h.update(np.asarray(X.data[start:end], dtype='<f8').tobytes())
        values.append(h.hexdigest())
    return np.array(values)


def assert_freeze(protocol_path):
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if not protocol.get('frozen_before_raw_download'):
        raise RuntimeError('unfrozen protocol')
    for rel, expected in protocol['code_sha256'].items():
        if sha(HERE/rel) != expected:
            raise RuntimeError('frozen code mismatch: '+rel)
    return protocol


def run(dataset, raw_root, protocol_path):
    protocol = assert_freeze(protocol_path)
    spec = protocol['datasets'][dataset]
    out = HERE/'evidence'/dataset
    out.mkdir(exist_ok=True, parents=True)
    if (out/'STARTED.json').exists():
        raise RuntimeError('real-data execution has already started; no automatic rerun')
    write_json(out/'STARTED.json', {'utc': utc(), 'protocol_sha256': sha(protocol_path),
                                  'dataset': dataset, 'seed': spec['seed']})
    from raw_adapter import read_dataset, load_mulan_archive
    if Path(raw_root).is_file():
        raw = load_mulan_archive(Path(raw_root), dataset)
        X_train, Y_train, X_test, Y_test, metadata = [raw[key] for key in
                    ['X_train', 'Y_train', 'X_test', 'Y_test', 'metadata']]
    else:
        X_train, Y_train, X_test, Y_test, metadata = read_dataset(Path(raw_root), dataset)
    assert X_train.shape[0]+X_test.shape[0] == spec['expected_n']
    assert X_train.shape[1] == spec['expected_features']
    assert Y_train.shape[1] == spec['expected_labels']
    idx = split_indices(dataset, X_train.shape[0], X_test.shape[0])
    assert min(len(value) for value in idx.values()) >= 100
    ftrain, ftest = feature_fingerprints(X_train), feature_fingerprints(X_test)
    groups = {key: set((ftrain if key in ['T', 'H'] else ftest)[rows]) for key, rows in idx.items()}
    duplicate_record = {'rule': 'descriptive exact-feature duplicates; no exclusions or split changes',
                        'within_role_duplicate_rows': {key: len(idx[key])-len(value) for key, value in groups.items()},
                        'cross_role_unique_feature_groups': {
                            a+'_'+b: len(groups[a] & groups[b]) for i, a in enumerate(idx) for b in list(idx)[i+1:]}}
    save_arrays(out/'SPLITS.npz', **idx)
    write_json(out/'DATA.json', {'utc': utc(), 'metadata': metadata,
                               'counts': {key: len(value) for key, value in idx.items()},
                               'source_train_n': X_train.shape[0], 'source_test_n': X_test.shape[0],
                               'duplicates': duplicate_record})
    print(dataset, 'data parsed', {key: len(v) for key, v in idx.items()}, flush=True)
    start = time.monotonic()
    classifier = FixedClassifier(spec['seed']).fit(X_train[idx['T']], Y_train[idx['T']])
    classifier_seconds = time.monotonic()-start
    joblib.dump(classifier, out/'classifier.joblib')
    HX, HY, HW, Hempty = classifier.features_and_predictions(X_train[idx['H']])
    Hloss = outcome(HY, Y_train[idx['H']])
    AX, AY, AW, Aempty = classifier.features_and_predictions(X_test[idx['A']])
    Aloss = outcome(AY, Y_test[idx['A']])
    r = .95*float(Aloss.sum()/AW.sum())
    if not np.isfinite(r) or r <= 0:
        write_json(out/'RESULT.json', {'status': 'degenerate_cap', 'cap': r,
                   'joint_success': False, 'utc': utc()})
        return
    start = time.monotonic()
    head = ExtraTreesRegressor(n_estimators=220, max_leaf_nodes=31, min_samples_leaf=10,
                              max_features=1.0, bootstrap=False, random_state=spec['seed'], n_jobs=1)
    head.fit(HX, Hloss)
    head_seconds = time.monotonic()-start
    joblib.dump(head, out/'error_head.joblib')
    gate_models, fitting = {}, {}
    for key, utility in [('GateCase', 'case'), ('GateExposure', 'exposure')]:
        start = time.monotonic()
        gate_models[key] = fit_gate(HX, Hloss, HW, r, utility, spec['seed'], .40)
        fitting[key] = {'seconds': time.monotonic()-start, 'trace': gate_models[key].trace}
        gate_models[key].save(out/(key+'.npz'))
        with np.load(out/(key+'.npz'), allow_pickle=False) as saved:
            for field in saved.files:
                saved[field]  # Require every compressed member to be readable.
    mean_h_w = float(np.mean(HW))

    def scores(X, W):
        result = reference_scores(np.clip(head.predict(X), 0, W), W, r, mean_h_w)
        result.update({key: model.score(X) for key, model in gate_models.items()})
        if not all(np.isfinite(v).all() for v in result.values()):
            raise FloatingPointError('nonfinite score')
        return result

    Ascore = scores(AX, AW)
    Atie = hash_values(dataset, 'official_test', idx['A'], 'ties')
    policies = {key: design_policy(Ascore[key], Atie, Aloss, AW, .95*r, .40) for key in METHODS}
    chosen = {utility: select_menu(policies, utility) for utility in ['case', 'exposure']}
    model_paths = [out/'classifier.joblib', out/'error_head.joblib', out/'GateCase.npz', out/'GateExposure.npz']
    write_json(out/'A_FIXED.json', {'utc': utc(), 'r': r, 'design_cap': .95*r, 'floor': .40,
              'policies': policies, 'menu_selected': chosen,
              'model_sha256': {path.name: sha(path) for path in model_paths},
              'protocol_sha256': sha(protocol_path), 'mean_H_w': mean_h_w,
              'training': {'classifier_seconds': classifier_seconds, 'head_seconds': head_seconds,
                           'gates': fitting, 'classifier_warnings': classifier.warning_records}})
    save_arrays(out/'A_ARRAYS.npz', loss=Aloss, exposure=AW, tie=Atie,
                **{key: v for key, v in Ascore.items()})
    print(dataset, 'A fixed', chosen, 'r', r, flush=True)
    BX, BY, BW, Bempty = classifier.features_and_predictions(X_test[idx['B']])
    Bloss, Bscore = outcome(BY, Y_test[idx['B']]), scores(BX, BW)
    Btie = hash_values(dataset, 'official_test', idx['B'], 'ties')
    Bmask = {key: policy_mask(Bscore[key], Btie, policy) for key, policy in policies.items()}
    Bboot = bootstrap_rows(BW, Bloss, Bmask, r, 20000, spec['seed']*100+1)
    q = 1-.05/14
    Bresults = {}
    for key in METHODS:
        record = metrics(Bloss, BW, Bmask[key])
        upper = float(np.quantile(Bboot[key]['g'], q, method='linear'))
        finite = bool(np.isfinite(Bboot[key]['risk']).all())
        record.update({'signed_excess_upper': upper,
                       'risk_percentile_upper': float(np.quantile(Bboot[key]['risk'], q, method='linear')) if finite else None,
                       'passed': bool(policies[key]['eligible'] and finite and upper < 0),
                       'eligible_A': policies[key]['eligible']})
        Bresults[key] = record
    write_json(out/'B_SCREEN.json', {'utc': utc(), 'A_fixed_sha256': sha(out/'A_FIXED.json'),
              'quantile': q, 'family_size': 14, 'results': Bresults,
              'interpretation': protocol['inference_scope']})
    save_arrays(out/'B_ARRAYS.npz', loss=Bloss, exposure=BW, tie=Btie,
                **Bscore, **{'mask_'+key: value for key, value in Bmask.items()})
    save_arrays(out/'B_BOOTSTRAP.npz', **{key+'_'+field: value for key, rec in Bboot.items() for field, value in rec.items()})
    print(dataset, 'B screened', {key: value['passed'] for key, value in Bresults.items()}, flush=True)
    EX, EY, EW, Eempty = classifier.features_and_predictions(X_test[idx['E']])
    Eloss, Escore = outcome(EY, Y_test[idx['E']]), scores(EX, EW)
    Etie = hash_values(dataset, 'official_test', idx['E'], 'ties')
    Emask = {key: policy_mask(Escore[key], Etie, policy) for key, policy in policies.items()}
    Emeasure = {key: metrics(Eloss, EW, mask) for key, mask in Emask.items()}
    Eboot = bootstrap_rows(EW, Eloss, Emask, r, 20000, spec['seed']*100+2)
    primary = compare(chosen['exposure'], chosen['case'], Emeasure, Eboot, .05/4)
    primary['both_pass_B'] = all(chosen[key] is not None and Bresults[chosen[key]]['passed']
                                 for key in ['case', 'exposure'])
    joint = bool(primary['available'] and primary['both_pass_B']
                 and primary['interval_c_pp'] is not None and primary['interval_d_pp'] is not None
                 and primary['interval_c_pp'][1] < 0 and primary['interval_d_pp'][0] > 0)
    secondary = {}
    for utility, key in [('case', 'GateCase'), ('exposure', 'GateExposure')]:
        comparison = compare(key if policies[key]['eligible'] else None, chosen[utility],
                             Emeasure, Eboot, .05/8)
        comparison['both_pass_B'] = bool(policies[key]['eligible'] and chosen[utility] is not None
                                       and Bresults[key]['passed'] and Bresults[chosen[utility]]['passed'])
        secondary[utility] = comparison
    save_arrays(out/'E_ARRAYS.npz', loss=Eloss, exposure=EW, tie=Etie,
                **Escore, **{'mask_'+key: value for key, value in Emask.items()})
    save_arrays(out/'E_BOOTSTRAP.npz', **{key+'_'+field: value for key, rec in Eboot.items() for field, value in rec.items()})
    result = {'status': 'completed', 'dataset': dataset, 'utc': utc(),
              'r': r, 'design_cap': .95*r, 'floor_A': .40, 'primary': primary,
              'secondary_gate_minus_menu': secondary, 'joint_success': joint,
              'B': Bresults, 'E': Emeasure,
              'exposure': {key: exposure_summary(w) for key, w in [('H', HW), ('A', AW), ('B', BW), ('E', EW)]},
              'top1_fallback_count': dict(zip(['H', 'A', 'B', 'E'], [Hempty, Aempty, Bempty, Eempty])),
              'inference_scope': protocol['inference_scope'],
              'B_screen_sha256': sha(out/'B_SCREEN.json'), 'A_fixed_sha256': sha(out/'A_FIXED.json'),
              'protocol_sha256': sha(protocol_path)}
    write_json(out/'RESULT.json', result)
    write_json(out/'COMPLETE.json', {'utc': utc(), 'files': {p.name: sha(p) for p in sorted(out.iterdir())
                                                        if p.is_file() and p.name != 'COMPLETE.json'}})
    print(dataset, 'complete', 'joint success', joint, 'primary', primary, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['bibtex', 'mediamill'])
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--protocol', default=str(HERE/'evidence'/'PROTOCOL.json'))
    args = parser.parse_args()
    try:
        run(args.dataset, args.raw_root, args.protocol)
    except Exception as exc:
        out = HERE/'evidence'/args.dataset
        out.mkdir(exist_ok=True, parents=True)
        write_json(out/'FAILURE.json', {'utc': utc(), 'exception': type(exc).__name__, 'message': str(exc),
                   'note': 'no automatic result-based retry, fallback dataset, or tuning'})
        raise


if __name__ == '__main__':
    main()
