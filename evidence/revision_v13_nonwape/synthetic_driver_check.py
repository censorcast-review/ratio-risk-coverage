"""Synthetic end-to-end checks only. Never imports raw_adapter or calls run()."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json
import traceback
import numpy as np
from scipy import sparse
import joblib
import run_nonwape as d


OUT = Path(__file__).resolve().parent/'synthetic_driver_outputs'
OUT.mkdir(exist_ok=True)
checks = []
failures = []


def check(name, fn):
    try:
        details = fn()
        checks.append({'name': name, 'passed': True, 'details': details})
    except Exception as exc:
        failures.append({'name': name, 'exception': type(exc).__name__, 'message': str(exc),
                         'traceback': traceback.format_exc()})


def split_check():
    ix = d.split_indices('synthetic_fixture', 100, 99)
    np.testing.assert_array_equal(np.sort(np.r_[ix['T'], ix['H']]), np.arange(100))
    np.testing.assert_array_equal(np.sort(np.r_[ix['A'], ix['B'], ix['E']]), np.arange(99))
    assert not set(ix['T']) & set(ix['H'])
    for x, y in [('A', 'B'), ('A', 'E'), ('B', 'E')]:
        assert not set(ix[x]) & set(ix[y])
    ix2 = d.split_indices('synthetic_fixture', 100, 99)
    for k in ix:
        np.testing.assert_array_equal(ix[k], ix2[k])
    return {k: len(v) for k, v in ix.items()}


def tie_boundary_check():
    rng = np.random.default_rng(511)
    for _ in range(50):
        n = 40
        score = rng.integers(0, 5, size=n).astype(float)
        tie = d.hash_values('synthetic_fixture', 'rows', rng.permutation(n), 'ties')
        w = rng.integers(1, 4, size=n).astype(float)
        loss = rng.integers(0, 2, size=n).astype(float)
        got = d.design_policy(score, tie, loss, w, .19, .4)
        order = np.lexsort((tie, score))
        good = []
        for j in range(1, n+1):
            take = order[:j]
            if j/n >= .4 and loss[take].sum() <= .19*w[take].sum():
                good.append(j)
        assert got['eligible'] == bool(good)
        if good:
            mask = d.policy_mask(score, tie, got)
            expected = np.zeros(n, bool)
            expected[order[:good[-1]]] = True
            np.testing.assert_array_equal(mask, expected)
            roundtrip = json.loads(json.dumps(got, allow_nan=False))
            np.testing.assert_array_equal(d.policy_mask(score, tie, roundtrip), mask)
    # Ensure Python integers beyond int64 retain their threshold comparison.
    score = np.zeros(4)
    tie = np.array([0, 2**63+1, 2**64-2, 2**64-1], dtype=np.uint64)
    p = {'eligible': True, 'threshold': 0., 'boundary_hash': int(2**64-2)}
    np.testing.assert_array_equal(d.policy_mask(score, tie, p), [1, 1, 1, 0])
    return {'brute_force_cases': 50, 'uint64_boundary_json': 'passed'}


def array_save_check():
    arrays = {'mask': np.array([True, False, True]),
              'ties': np.array([0, 2**63+1, 2**64-1], dtype=np.uint64),
              'bootstrap_risk': np.array([0., np.nan, 1.]),
              'integers': np.array([0, 1, 2], dtype=np.int64)}
    d.save_arrays(OUT/'ARRAY_TYPES.npz', **arrays)
    with np.load(OUT/'ARRAY_TYPES.npz', allow_pickle=False) as r:
        for k, v in arrays.items():
            assert r[k].dtype == v.dtype
            np.testing.assert_array_equal(r[k], v)
    return {'types': ['bool', 'uint64', 'float64 with NaN', 'int64']}


def constant_label_check():
    X = sparse.csr_matrix(np.arange(60).reshape(20, 3).astype(float))
    # Both constant zero labels: the top-1 fallback must choose a deterministic
    # label and ensure exposure >=1 for every row.
    clf = d.FixedClassifier(511).fit(X, np.zeros((20, 3), dtype=bool))
    features, pred, w, empty = clf.features_and_predictions(X)
    assert empty == 20
    assert pred.dtype == bool
    np.testing.assert_array_equal(w, np.ones(20))
    np.testing.assert_array_equal(pred[:, 0], np.ones(20, dtype=bool))
    assert np.isfinite(features).all()
    # Constant-one columns exercise the no-fallback branch and probability stack.
    clf2 = d.FixedClassifier(512).fit(X, np.ones((20, 3), dtype=bool))
    _, pred2, w2, empty2 = clf2.features_and_predictions(X)
    assert empty2 == 0 and pred2.all()
    np.testing.assert_array_equal(w2, np.full(20, 3.))
    return {'all_zero_fallback': True, 'all_one_no_fallback': True}


def bootstrap_check():
    w = np.array([1., 2., 3., 4.])
    loss = np.array([0., 1., 2., 0.])
    masks = {'all': np.ones(4, bool), 'none': np.zeros(4, bool), 'some': np.array([0, 1, 1, 0], bool)}
    boot = d.bootstrap_rows(w, loss, masks, .4, 129, 777)
    np.testing.assert_array_equal(boot['all']['c'], np.ones(129))
    np.testing.assert_array_equal(boot['all']['d'], np.ones(129))
    assert np.isnan(boot['none']['risk']).all()
    assert d.interval(boot['none']['risk'], .05) is None
    rng = np.random.default_rng(777)
    counts = np.r_[rng.multinomial(4, np.ones(4)/4, size=128), rng.multinomial(4, np.ones(4)/4, size=1)]
    np.testing.assert_allclose(boot['some']['c'], (counts @ masks['some'])/4)
    np.testing.assert_allclose(boot['some']['g'], (counts @ (masks['some']*(loss-.4*w)))/4)
    measured = {k: d.metrics(loss, w, m) for k, m in masks.items()}
    zero = d.compare('some', 'some', measured, boot, .05)
    assert zero['delta_c_pp'] == zero['delta_d_pp'] == 0
    assert zero['interval_c_pp'] == zero['interval_d_pp'] == [0., 0.]
    d.write_json(OUT/'BOOTSTRAP_CHECK.json', {'zero_contrast': zero,
                                           'undefined_risk_interval': d.interval(boot['none']['risk'], .05)})
    return {'draws': 129, 'batch_boundary': True, 'paired_zero_contrast': True}


def end_to_end_check():
    rng = np.random.default_rng(991)
    def generate(n):
        X = rng.normal(size=(n, 6))
        X[rng.random(X.shape) < .55] = 0
        X[:, 5] = 0  # A constant feature checks sparse standardization.
        Y = np.column_stack([np.ones(n, bool), np.zeros(n, bool),
                             rng.random(n) < 1/(1+np.exp(-X[:, 0]+X[:, 1]))])
        return sparse.csr_matrix(X), Y
    Xtr, Ytr = generate(100)
    Xte, Yte = generate(99)
    idx = d.split_indices('synthetic_fixture', 100, 99)
    clf = d.FixedClassifier(991).fit(Xtr[idx['T']], Ytr[idx['T']])
    joblib.dump(clf, OUT/'classifier.joblib')
    restored = joblib.load(OUT/'classifier.joblib')
    splits = {}
    for name in ('H', 'A', 'B', 'E'):
        X, Y = (Xtr, Ytr) if name == 'H' else (Xte, Yte)
        features, pred, w, fallback = clf.features_and_predictions(X[idx[name]])
        features2, pred2, w2, _ = restored.features_and_predictions(X[idx[name]])
        np.testing.assert_array_equal(features, features2)
        np.testing.assert_array_equal(pred, pred2)
        np.testing.assert_array_equal(w, w2)
        splits[name] = {'X': features, 'pred': pred, 'w': w,
                        'loss': d.outcome(pred, Y[idx[name]]), 'fallback': fallback}
    H, A = splits['H'], splits['A']
    r = .95*float(A['loss'].sum()/A['w'].sum())
    assert r > 0, 'synthetic A happened to be perfect; fixture invalid'
    head = d.ExtraTreesRegressor(n_estimators=220, max_leaf_nodes=31, min_samples_leaf=10,
                                  max_features=1.0, bootstrap=False, random_state=991, n_jobs=1)
    head.fit(H['X'], H['loss'])
    joblib.dump(head, OUT/'error_head.joblib')
    models = {}
    for key, utility in [('GateCase', 'case'), ('GateExposure', 'exposure')]:
        models[key] = d.fit_gate(H['X'], H['loss'], H['w'], r, utility, 991, .4)
        models[key].save(OUT/(key+'.npz'))
        with np.load(OUT/(key+'.npz'), allow_pickle=False) as model_file:
            assert np.isfinite(model_file['W1']).all()
            assert json.loads(str(model_file['metadata']))['config']['epochs'] == 40
    scores, ties = {}, {}
    for name in ('A', 'B', 'E'):
        part = splits[name]
        scores[name] = d.reference_scores(np.clip(head.predict(part['X']), 0, part['w']), part['w'], r, float(H['w'].mean()))
        scores[name].update({key: model.score(part['X']) for key, model in models.items()})
        ties[name] = d.hash_values('synthetic_fixture', 'official_test', idx[name], 'ties')
    policies = {key: d.design_policy(scores['A'][key], ties['A'], A['loss'], A['w'], .95*r, .4) for key in d.METHODS}
    chosen = {utility: d.select_menu(policies, utility) for utility in ('case', 'exposure')}
    d.write_json(OUT/'A_SYNTHETIC_FIXED.json', {'cap': r, 'policies': policies, 'chosen': chosen})
    Bresults, Emeasure, Eboot = {}, None, None
    for name in ('B', 'E'):
        part = splits[name]
        masks = {key: d.policy_mask(scores[name][key], ties[name], policies[key]) for key in d.METHODS}
        measured = {key: d.metrics(part['loss'], part['w'], masks[key]) for key in d.METHODS}
        boot = d.bootstrap_rows(part['w'], part['loss'], masks, r, 257, 992 if name == 'B' else 993)
        d.save_arrays(OUT/(name+'_SYNTHETIC_ARRAYS.npz'), loss=part['loss'], exposure=part['w'], tie=ties[name],
                      predicted=part['pred'], **scores[name], **{'mask_'+key: v for key, v in masks.items()})
        d.save_arrays(OUT/(name+'_SYNTHETIC_BOOTSTRAP.npz'), **{key+'_'+field: v for key, fields in boot.items() for field, v in fields.items()})
        if name == 'B':
            for key in d.METHODS:
                upper = float(np.quantile(boot[key]['g'], 1-.05/14))
                finite = bool(np.isfinite(boot[key]['risk']).all())
                Bresults[key] = measured[key] | {'signed_excess_upper': upper,
                    'risk_upper': float(np.quantile(boot[key]['risk'], 1-.05/14)) if finite else None,
                    'passed': bool(policies[key]['eligible'] and finite and upper < 0)}
        else:
            Emeasure, Eboot = measured, boot
    primary = d.compare(chosen['exposure'], chosen['case'], Emeasure, Eboot, .05/4)
    secondary = {utility: d.compare(key if policies[key]['eligible'] else None, chosen[utility], Emeasure, Eboot, .05/8)
                 for utility, key in [('case', 'GateCase'), ('exposure', 'GateExposure')]}
    report = {'status': 'synthetic fixture only; not a paper result', 'B': Bresults, 'E': Emeasure,
              'primary': primary, 'secondary': secondary,
              'exposure': {k: d.exposure_summary(v['w']) for k, v in splits.items()}}
    d.write_json(OUT/'SYNTHETIC_RESULT.json', report)
    assert json.loads((OUT/'SYNTHETIC_RESULT.json').read_text()) == report
    assert not (Path(__file__).resolve().parent/'evidence'/'synthetic_fixture').exists()
    return {'source_train_rows': 100, 'source_test_rows': 99, 'labels': 3,
            'classifier_models': len(clf.models), 'constant_label_models': sum(isinstance(m, float) for m in clf.models),
            'head_trees': 220, 'gates': 2, 'gate_epochs': 40, 'bootstrap_draws': 257,
            'policies': len(policies), 'real_run_called': False}


for name, fn in [('deterministic_disjoint_splits', split_check),
                 ('threshold_hash_boundary', tie_boundary_check),
                 ('boolean_uint64_nan_array_roundtrip', array_save_check),
                 ('constant_labels_sparse_features', constant_label_check),
                 ('paired_bootstrap', bootstrap_check),
                 ('end_to_end_sparse_fixture', end_to_end_check)]:
    check(name, fn)

receipt = {'status': 'passed' if not failures else 'failed', 'scope': 'synthetic only; no raw data opened',
           'utc': d.utc(), 'driver_sha256': d.sha(Path(d.__file__)),
           'gate_sha256': d.sha(Path(d.__file__).with_name('direct_gate.py')),
           'check_script_sha256': d.sha(Path(__file__)), 'checks': checks, 'failures': failures}
d.write_json(OUT/'SYNTHETIC_DRIVER_CHECK.json', receipt)
print(json.dumps(receipt, indent=2))
raise SystemExit(bool(failures))
