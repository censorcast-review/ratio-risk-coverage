"""Create the local, pre-download freeze. Not a public registry submission."""
import importlib.metadata
import json
import platform
from pathlib import Path
from run_nonwape import HERE, sha, utc, write_json


def freeze():
    path = HERE/'evidence'/'PROTOCOL.json'
    if path.exists() or any((HERE/'raw').glob('*')):
        raise RuntimeError('refuse a second freeze or a freeze after raw acquisition')
    sources = ['direct_gate.py', 'raw_adapter.py', 'run_nonwape.py', 'fetch_nonwape.py', 'freeze_protocol.py']
    protocol = {
        'name': 'v13 fixed-predictor learned-gate comparison and non-WAPE external evaluation',
        'version': 1, 'frozen_utc': utc(), 'frozen_before_raw_download': True,
        'registration_scope': 'Local timestamped code/protocol freeze; not a public preregistration.',
        'historical_scope': 'These datasets have no prior evaluation recorded in the available study conversation. This is not a universal claim about every collaborator.',
        'prior_information': {
            'previous_results': 'M5 and Dominicks v12 outcomes were known. This separate test addresses non-WAPE scope and a learned-gate baseline and does not replace Dominicks.',
            'public_metadata_seen': 'Official dataset sizes, feature/label counts, publication references, ARFF/XML format, official train/test existence and Mediamill split counts.',
            'incidentally_seen_public_label_cardinality': {'bibtex': 2.402, 'mediamill': 4.376},
            'no_raw_or_model_outcomes_seen': True,
            'v12_source_status': 'Unavailable in the current execution session. New code is independent; exact identity of the old Mixed rule has not been verified.'},
        'datasets': {
            'bibtex': {'seed': 20260913, 'expected_n': 7395, 'expected_features': 1836,
                       'expected_labels': 159, 'url': 'https://sourceforge.net/projects/mulan/files/datasets/bibtex.rar/download'},
            'mediamill': {'seed': 20260914, 'expected_n': 43907, 'expected_features': 120,
                          'expected_labels': 101, 'expected_train_n': 30993, 'expected_test_n': 12914,
                          'url': 'https://sourceforge.net/projects/mulan/files/datasets/mediamill.rar/download'}},
        'data_selection': 'Both fixed before raw acquisition. No replacement or new dataset based on success, failure, direction, or precision.',
        'acquisition': {'attempts_per_dataset': 1, 'timeout_seconds': 60, 'max_bytes': 536870912,
                        'redirects': 'Official SourceForge/sf.net hosts only; no alternate source after failure.'},
        'raw_redistribution': False,
        'data_license_scope': 'Public academic benchmark source; no separately verified dataset license. Do not infer a data license from Mulan software licensing.',
        'splits': {
            'official_train': 'Sort by SHA256 first 64 bits; first floor(.7 n) point T, remainder head/gate H.',
            'official_test': 'Sort by SHA256 first 64 bits; first floor(n/3) A, next floor(n/3) B, remainder E.',
            'hash_string': 'v13-split-20260913|dataset|official_train_or_official_test|zero_based_original_row',
            'stratification': False, 'minimum_role_n': 100,
            'duplicates': 'Describe exact feature duplicates across roles; preserve official partitions and do not exclude or resplit.',
            'opening_semantics': 'Raw parsing materializes the labels. Only T/H/A labels influence fitting/design. A records precede B calculations, and B records precede E calculations; no physical sealed-label claim.'},
        'prediction': {
            'classifier': 'One binary L2 LogisticRegression per label, C=1, liblinear, tol=.0001, max_iter=1000, no class weights.',
            'scaler': 'StandardScaler(with_mean=False), fitted on point T only.',
            'constant_label': '(positives_T+1)/(n_T+2) if only one class is observed in T.',
            'prediction_set': 'p>=.5; if empty, take one maximum-probability label, exact ties by source label order.',
            'warnings': 'Record convergence warnings; do not refit with altered settings.',
            'loss': 'False positives in the fixed complete predicted-label set.',
            'exposure': 'Number of predicted positive labels, exactly known at decision time and at least one.',
            'abstention': 'Accept/reject the entire predicted-label set for each example; never individual labels.'},
        'common_selector_features': 'Point-scaled original feature matrix plus log1p(exact predicted-positive count).',
        'error_head': {'algorithm': 'ExtraTreesRegressor', 'n_estimators': 220, 'max_leaf_nodes': 31,
                       'min_samples_leaf': 10, 'max_features': 1.0, 'bootstrap': False, 'n_jobs': 1,
                       'fit_role': 'H', 'output_clip': '[0, exact w]'},
        'menu': {
            'order': ['Error', 'Excess', 'Mixed', 'Ratio', 'Descending'],
            'Error': 'ehat', 'Excess': 'ehat-r*w',
            'Mixed': '(ehat-r*w)/(.5+.5*w/mean_H(w))', 'Ratio': 'ehat/w', 'Descending': '-w',
            'direction': 'Ascending scores accept first.',
            'tie_hash': 'v13-ties-20260913|dataset|official_test|zero_based_original_row -> first64bitsSHA256',
            'threshold': 'Lexicographic score/hash boundary; same outcome-independent hash for all methods, a fixed implementation of boundary randomization.',
            'design': 'Largest A-feasible prefix at risk .95*r and case coverage>=.40.',
            'selection': 'Each utility maximized across eligible menu members; exact utility ties retain first fixed menu entry.',
            'tolerance_band': None},
        'direct_gates': {
            'name': 'Custom primal-dual MLP, not a SelectiveNet reproduction',
            'objectives': ['case', 'exposure'], 'fit_role': 'H', 'point_predictor_frozen': True,
            'hidden': [32, 16], 'epochs': 40, 'batch_size': 2048, 'learning_rate': .003,
            'weight_decay': .0001, 'dual_learning_rate': .2, 'dual_maximum': 100.,
            'gradient_clip': 5., 'feature_clip': 10.,
            'initial_duals': 'lambda=min(1/r,100), gamma=0',
            'training_objective': '-mean(a*u) + lambda*mean(a*(loss-r*w))/mean_H(w) + gamma*(.40-mean(a)); u=1 or w/mean_H(w).',
            'updates': 'Minibatch Adam primal updates; once-per-epoch dual ascent on all H rows.',
            'initialization': 'Same seed and initialization for both utilities within a dataset.',
            'calibration': 'Same A threshold feasibility and hash boundary as every menu score.',
            'hyperparameter_search': False, 'soft_constraint_convergence_claim': False},
        'risk_design': {'cap': '.95*sum_A(loss)/sum_A(w)', 'threshold_design_cap': '.95*cap',
                        'case_floor_A': .40, 'B_or_E_floor_guarantee': False,
                        'A_dependence_of_training': 'Only the A-derived cap scalar is used by H-only direct-gate training.'},
        'screen': {'draws': 20000, 'seed': 'dataset_seed*100+1', 'resampling': 'Paired row multinomial bootstrap, fixed n draws per resample.',
                   'target': 'mean_B(a*(loss-r*w))', 'alpha_family': .05, 'family_size': 14,
                   'upper_quantile': 1-.05/14, 'interpolation': 'linear',
                   'pass': 'A eligible, positive accepted exposure in every resample, signed-excess upper<0.',
                   'after_failure': 'No fallback or reselection. E metrics are still computed for all declared methods.'},
        'later': {'draws': 20000, 'seed': 'dataset_seed*100+2', 'paired_across_all_policies': True,
                  'coverage': 'case=mean(a), exposure=sum(a*w)/sum(w) recalculated within every resample.',
                  'primary_family': {'contrasts': 'menu exposure choice minus menu case choice, c and d, both datasets', 'size': 4, 'marginal_two_sided_level': .9875},
                  'secondary_family': {'contrasts': 'direct minus menu for matched utility, c and d, both datasets', 'size': 8, 'marginal_two_sided_level': .99375},
                  'zero_acceptance': 'Risk undefined, coverages zero. Show explicit status rather than inventing a risk.',
                  'no_global_95_claim': 'The screen and two contrast families are separately adjusted; no single 95% statement across all three families.'},
        'success': {'per_dataset': 'Both selected menu policies pass B and adjusted E upper(delta c)<0 and lower(delta d)>0.',
                    'cross_dataset': 'Conjunction across both declared datasets.',
                    'learned_gate_superiority': 'No predeclared universal superiority claim. Report all matched-utility effects and screen status.'},
        'failure_rules': 'Acquisition, schema, zero cap, insufficient role size, or numerical failure is recorded without automatic dataset substitution, refit, or tuning. Any later implementation-only repair requires a separate dated amendment retaining the original failure.',
        'inference_scope': 'Descriptive bootstrap conditional on fitted models and A design, under row-exchangeability approximation. Unknown document duplicates and Mediamill shot/video dependence limit uncertainty interpretation. No population risk guarantee, no demonstrated training convergence, and no claim that one untuned MLP represents all learned selectors.',
        'report_all': ['Both dataset statuses', 'All seven candidate A/B/E metrics', 'All primary and secondary comparisons including failed screens',
                       'Role sizes', 'Prediction-count distribution and top1 fallback counts', 'Training warnings/traces/times', 'Source hashes and failures'],
        'environment': {'python': platform.python_version(), **{name: importlib.metadata.version(name)
                        for name in ['numpy', 'scipy', 'scikit-learn', 'joblib']}},
        'code_sha256': {name: sha(HERE/name) for name in sources},
    }
    write_json(path, protocol)
    write_json(HERE/'evidence'/'FREEZE_RECEIPT.json', {'utc': utc(), 'protocol_sha256': sha(path),
               'raw_directory_empty': not any((HERE/'raw').glob('*')),
               'code_sha256': protocol['code_sha256'], 'scope': 'Local freeze; no public registration or external timestamp authority.'})
    print(json.dumps({'status': 'frozen', 'path': str(path), 'sha256': sha(path), 'utc': protocol['frozen_utc']}))


if __name__ == '__main__':
    freeze()
