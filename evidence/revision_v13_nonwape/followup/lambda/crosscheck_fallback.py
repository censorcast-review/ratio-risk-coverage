"""Bounded independent cross-check of saved fixed-policy subgroup outputs."""
from pathlib import Path
import ast
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT/'followup/fallback'
checks = []


def check(name, ok):
    checks.append({'check': name, 'passed': bool(ok)})
    if not ok:
        raise AssertionError(name)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


report = json.loads((SOURCE/'RESULT.json').read_text())
spec = json.loads((SOURCE/'ANALYSIS_SPEC.json').read_text())
check('source_analysis_code_hash', sha(SOURCE/'analysis.py') == spec['analysis_code_sha256'])
check('all_reported_internal_checks_pass', all(row['passed'] for row in report['checks']))
tree = ast.parse((SOURCE/'analysis.py').read_text())
fitting_calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr in ('fit', 'partial_fit', 'fit_transform')]
check('no_training_calls_in_analysis', not fitting_calls)
summary = {}
for dataset in ['bibtex', 'mediamill']:
    base = ROOT/'evidence'/dataset
    fixed = json.loads((base/'A_FIXED.json').read_text())
    original = json.loads((base/'RESULT.json').read_text())
    flags = dict(np.load(SOURCE/(dataset+'_FALLBACK_FLAGS.npz'), allow_pickle=False))
    results = report['datasets'][dataset]
    check(dataset+'_original_choices', results['menu_selected_original'] == fixed['menu_selected'])
    check(dataset+'_original_cap', results['r_original'] == fixed['r'])
    for role in ['A', 'B', 'E']:
        arrays = dict(np.load(base/(role+'_ARRAYS.npz'), allow_pickle=False))
        w, loss, flag = arrays['exposure'], arrays['loss'], flags[role]
        check(dataset+role+'_flag_dtype_shape', flag.dtype == bool and flag.shape == w.shape)
        check(dataset+role+'_fallback_count', int(flag.sum()) == original['top1_fallback_count'][role])
        check(dataset+role+'_fallback_w_one', np.all(w[flag] == 1))
        cohorts = {'all_rows': np.ones(len(w), bool), 'nonfallback_rows': ~flag, 'fallback_rows': flag}
        independently_calculated = {}
        for cohort_name, cohort in cohorts.items():
            declared = results['roles'][role]['cohorts'][cohort_name]
            independent = {}
            for name, policy in fixed['policies'].items():
                row = declared['policies'][name]
                check(dataset+role+cohort_name+name+'_availability', row['available'] == policy['eligible'])
                if not policy['eligible']:
                    check(dataset+role+cohort_name+name+'_unavailable_metrics', all(row[field] is None for field in
                          ['accepted_n', 'accepted_loss', 'accepted_exposure', 'case_coverage', 'exposure_coverage', 'risk']))
                    continue
                accepted = ((arrays[name] < policy['threshold']) | ((arrays[name] == policy['threshold']) &
                            (arrays['tie'] <= policy['boundary_hash']))) & cohort
                N, W, L = int(accepted.sum()), float(w[accepted].sum()), float(loss[accepted].sum())
                rec = {'cohort_n': int(cohort.sum()), 'cohort_exposure': float(w[cohort].sum()),
                       'accepted_n': N, 'accepted_exposure': W, 'accepted_loss': L,
                       'case_coverage': N/int(cohort.sum()), 'exposure_coverage': W/float(w[cohort].sum()),
                       'risk': L/W if W else None}
                independent[name] = rec
                for field, value in rec.items():
                    check(dataset+role+cohort_name+name+'_'+field,
                          row[field] is None if value is None else np.isclose(row[field], value, atol=1e-14, rtol=0))
            independently_calculated[cohort_name] = independent
            for contrast, data in declared['contrasts'].items():
                if not data['available']:
                    continue
                first, second = independent[data['first']], independent[data['second']]
                for suffix, field in [('c', 'case_coverage'), ('d', 'exposure_coverage')]:
                    delta = 100*(first[field]-second[field])
                    check(dataset+role+cohort_name+contrast+suffix, np.isclose(delta, data['delta_'+suffix+'_pp'], atol=1e-14, rtol=0))
        # Distinct cohort weights exactly reconstruct full-cohort coverages and
        # effects. This is why a subgroup sign may differ without a code error.
        cweight, dweight = float(flag.mean()), float(w[flag].sum()/w.sum())
        for name in independently_calculated['all_rows']:
            all_rec = independently_calculated['all_rows'][name]
            yes = independently_calculated['fallback_rows'][name]
            no = independently_calculated['nonfallback_rows'][name]
            check(dataset+role+name+'_case_decomposition', np.isclose(all_rec['case_coverage'], cweight*yes['case_coverage']+(1-cweight)*no['case_coverage'], atol=1e-14, rtol=0))
            check(dataset+role+name+'_exposure_decomposition', np.isclose(all_rec['exposure_coverage'], dweight*yes['exposure_coverage']+(1-dweight)*no['exposure_coverage'], atol=1e-14, rtol=0))
        if role == 'E':
            summary[dataset] = {'fallback_case_weight': cweight, 'fallback_exposure_weight': dweight,
                                'cohort_deltas': {cohort: results['roles'][role]['cohorts'][cohort]['contrasts']['menu_exposure_minus_case'] for cohort in cohorts}}

payload = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS', 'checks_n': len(checks),
           'scope': 'Independent array-based metric and decomposition check using reconstructed flags. Raw parsing/classifier coefficient reconstruction reviewed as code; not redundantly rerun. No original or fallback artifact modified.',
           'code_review': 'Probabilities use saved logistic coefficients, intercepts and T-fitted StandardScaler(with_mean=False); constants retained; threshold .5 followed by top-one insertion. No fitting calls. Exact reconstructed FP/w and primary hashes were checked by the source analysis. Conditional cohort denominators are correct.',
           'theory_review': 'The generic coverage envelope is mathematically valid for uniform positive support bounds. The recovered optimizer reversal Eq3 additionally needs optimizer/common-feasible-set hypotheses. Neither theorem predicts dispersion-effect monotonicity or power; quantiles cannot replace support bounds. Current fallback interpretation correctly disclaims those claims.',
           'source_spec_sha256': sha(SOURCE/'ANALYSIS_SPEC.json'), 'source_code_sha256': sha(SOURCE/'analysis.py'),
           'source_results_sha256': sha(SOURCE/'RESULT.json'), 'summary': summary, 'checks': checks}
(OUT/'FALLBACK_CROSSCHECK.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
print(json.dumps({'status':'PASS', 'checks_n':len(checks), 'summary':summary}))
