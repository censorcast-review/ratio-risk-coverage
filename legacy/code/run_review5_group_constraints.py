"""Exploratory category-budget control on already-consumed prediction caches.

Bin boundaries depend only on development A/B predicted conditional heads;
probabilities depend on development A/B outcomes. All choices are persisted
before guardian caches are opened. No model fitting, raw holdout, or external
loader is present. Fractional metrics are ratios of conditional expected sums,
not a finite-sample guarantee for a realized randomized acceptance mask.
"""
from pathlib import Path
import argparse, datetime, json
import numpy as np
from scipy.optimize import linprog
from r2_io import dump, sha, write

R = .85 * .7530939208313988
FLOOR = .35
SEED = 20260906
QUANTILE_BINS = 8
GROUPS = ['FOODS', 'HOBBIES', 'HOUSEHOLD']
KEYS = ['calibration_a', 'calibration_b', 'shadow']


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != 'risk_features'}


def cuts(values):
    return np.unique(np.quantile(values, np.arange(1, QUANTILE_BINS) / QUANTILE_BINS)).tolist()


def bin_ids(error, demand, boundary):
    i = np.searchsorted(boundary['error_cuts'], np.maximum(error, 0), side='right')
    j = np.searchsorted(boundary['demand_cuts'], np.maximum(demand, 0), side='right')
    return i * (len(boundary['demand_cuts']) + 1) + j


def aggregate(data, pred, meta, group, boundary):
    rows = meta['cat_id'] == group
    size = (len(boundary['error_cuts']) + 1) * (len(boundary['demand_cuts']) + 1)
    records = {}
    for key, v in data.items():
        ids = bin_ids(pred['error_' + key][rows], pred['demand_' + key][rows], boundary).ravel()
        y = v['truth'][rows].astype(float).ravel()
        err = np.abs(y - v['proposal'][rows].astype(float).ravel())
        n = y.size
        records[key] = dict(
            n=n, count=np.bincount(ids, minlength=size) / n,
            demand=np.bincount(ids, weights=y, minlength=size) / n,
            error=np.bincount(ids, weights=err, minlength=size) / n,
            total_demand=float(y.sum() / n), ids=ids)
        records[key]['excess'] = records[key]['error'] - R * records[key]['demand']
    return records


def rows_lp(a, row_floor):
    """Maximize minimum A/B expected row coverage, with separate budgets."""
    n = len(a['calibration_a']['count'])
    objective = np.r_[np.zeros(n), -1.]
    matrix, rhs = [], []
    for key in KEYS[:2]:
        v = a[key]
        matrix.extend([np.r_[v['excess'], 0.], np.r_[-v['count'], 1.]])
        rhs.extend([0., 0.])
        if row_floor:
            matrix.append(np.r_[-v['count'], 0.]); rhs.append(-row_floor)
    result = linprog(objective, A_ub=matrix, b_ub=rhs,
                     bounds=[(0., 1.)] * n + [(0., 1.)], method='highs')
    record = dict(success=bool(result.success), status=int(result.status),
                  solver_message=result.message, row_floor=row_floor)
    if result.success:
        record.update(probabilities=result.x[:-1].tolist(),
                      minimum_calibration_coverage=float(result.x[-1]),
                      maximum_constraint_violation=float(max(0., np.max(np.asarray(matrix) @ result.x - rhs))))
        assert record['maximum_constraint_violation'] < 1e-8
    return record


def minimum_excess_certificate(a, floor):
    """min max_k E_k[a(e-rY)] subject to coverage >= floor in A and B.

A positive checked dual bound proves that no member of this bin class can
satisfy both budgets at the row floor; no claim is made over all predictors.
"""
    n = len(a['calibration_a']['count'])
    c = np.r_[np.zeros(n), 1.]
    matrix, rhs = [], []
    for key in KEYS[:2]:
        matrix.extend([np.r_[a[key]['excess'], -1.], np.r_[-a[key]['count'], 0.]])
        rhs.extend([0., -floor])
    matrix = np.asarray(matrix); rhs = np.asarray(rhs)
    result = linprog(c, A_ub=matrix, b_ub=rhs,
                     bounds=[(0., 1.)] * n + [(None, None)], method='highs')
    assert result.success, result.message
    y = result.ineqlin.marginals; lo = result.lower.marginals; hi = result.upper.marginals
    stationarity = c - matrix.T @ y - lo - hi
    dual_value = float(rhs @ y + hi[:-1].sum())
    # Correct bounded-coordinate stationarity residue in the dual lower bound.
    assert abs(stationarity[-1]) < 1e-12
    corrected_dual_value = dual_value + float(np.minimum(stationarity[:-1], 0).sum())
    primal_value = float(result.fun)
    checks = dict(
        stationarity_max_abs=float(np.max(np.abs(stationarity))),
        inequality_dual_max=float(np.max(y)), lower_dual_min=float(np.min(lo)),
        upper_dual_max=float(np.max(hi)),
        primal_constraint_violation=float(max(0., np.max(matrix @ result.x - rhs))),
        duality_gap=primal_value-dual_value)
    assert checks['stationarity_max_abs'] < 1e-8
    assert checks['inequality_dual_max'] < 1e-8 and checks['lower_dual_min'] > -1e-8
    assert checks['upper_dual_max'] < 1e-8 and checks['primal_constraint_violation'] < 1e-8
    assert abs(checks['duality_gap']) < 1e-8
    return dict(minimum_worst_block_excess_per_row=primal_value,
                dual_raw_objective=dual_value, dual_lower_bound=corrected_dual_value,
                positive_bound_proves_bin_class_infeasibility=corrected_dual_value > 1e-8,
                probabilities=result.x[:-1].tolist(),
                dual_inequality=y.tolist(), dual_lower=lo.tolist(), dual_upper=hi.tolist(), checks=checks)


def metrics(a, probability):
    p = np.asarray(probability)
    out = {}
    for key, v in a.items():
        count = float(v['count'] @ p); demand = float(v['demand'] @ p); error = float(v['error'] @ p)
        out[key] = dict(total_rows=int(v['n']), expected_accepted_rows=count*v['n'], row_coverage=count,
                        demand_coverage=demand/v['total_demand'] if v['total_demand'] else None,
                        wape=error/demand if demand > 0 else None,
                        accepted_error_per_total_row=error, accepted_demand_per_total_row=demand,
                        excess_per_total_row=error-R*demand,
                        cap_point_satisfied=bool(demand > 0 and error <= R*demand + 1e-10),
                        row_floor_satisfied=bool(count >= FLOOR-1e-10))
    return out


def pooled(group_metrics):
    out = {}
    for key in KEYS:
        v = [x[key] for x in group_metrics]
        n = sum(x['total_rows'] for x in v)
        count = sum(x['expected_accepted_rows'] for x in v)
        error = sum(x['accepted_error_per_total_row']*x['total_rows'] for x in v)
        demand = sum(x['accepted_demand_per_total_row']*x['total_rows'] for x in v)
        out[key] = dict(total_rows=n, expected_accepted_rows=count, row_coverage=count/n,
                        wape=error/demand if demand else None,
                        excess_per_total_row=(error-R*demand)/n,
                        cap_point_satisfied=bool(demand > 0 and error <= R*demand+1e-8),
                        row_floor_satisfied=bool(count/n >= FLOOR-1e-10))
    return out


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(status='EXPLORATORY_CONSUMED_CACHE_CONTROL', seed=SEED, cap=R,
                    per_group_row_floor=FLOOR, quantile_bins_per_head=QUANTILE_BINS,
                    groups=GROUPS, boundaries='Within-category pooled calibration A/B head predictions only; 7 internal quantiles per head, duplicate cuts collapsed',
                    policy_class='One arbitrary randomized acceptance probability per error-demand bin and category, shared by all periods',
                    fitting='No new head fit; calibration outcomes choose probabilities by finite LP',
                    constraints='For each category separately, A and B excess budgets <=0 and A and B row coverage >=.35; no budget borrowing between categories',
                    objective='Maximize minimum within-group A/B expected row coverage',
                    secondary='Cap-only LP (floor0); minimax excess LP at floor.35 with checked dual lower bound',
                    tie_rule='Deterministic HiGHS solution of the fixed LP, no post-holdout choice',
                    acceptance='Independent uniform auxiliary variable <= fixed bin probability',
                    metric_scope='Ratios of conditional expected error/demand sums under randomized acceptance; no realized-mask or finite-sample certificate',
                    training_calls=0, raw_guardian_accesses=0, external_opened=False,
                    code_sha256=sha(__file__), created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    dump(args.output/'GROUP_PROTOCOL.json', protocol)
    dm = load(args.design/'cache_design/metadata.npz')
    dd = {k: load(args.design/'cache_design'/f'{k}_aligned.npz') for k in KEYS}
    dp = load(args.design/f'design_predictions_s{SEED}.npz')
    records = []
    for group in GROUPS:
        rows = dm['cat_id'] == group
        boundary = {name + '_cuts': cuts(np.concatenate([np.maximum(dp[name+'_'+k][rows], 0).ravel() for k in KEYS[:2]]))
                    for name in ['error', 'demand']}
        ag = aggregate(dd, dp, dm, group, boundary)
        strict = rows_lp(ag, FLOOR); relaxed = rows_lp(ag, 0.)
        certificate = minimum_excess_certificate(ag, FLOOR)
        assert relaxed['success']
        assert strict['success'] == (not certificate['positive_bound_proves_bin_class_infeasibility'])
        probability = strict.get('probabilities', np.zeros(len(ag['calibration_a']['count'])).tolist())
        bin_statistics = {k:{name:(value.tolist() if isinstance(value, np.ndarray) else value) for name,value in v.items() if name != 'ids'} for k,v in ag.items()}
        positivity = {}
        for k in KEYS[:2]:
            v = ag[k]; occupied=v['count']>0; positive_demand=v['demand']>0
            positivity[k] = dict(occupied_bins=int(occupied.sum()),
                                 every_occupied_bin_has_positive_excess=bool(np.all(v['excess'][occupied]>0)),
                                 minimum_occupied_bin_excess_per_original_row=float(v['excess'][occupied].min()),
                                 minimum_positive_demand_bin_wape=float(np.min(v['error'][positive_demand]/v['demand'][positive_demand])))
        rec = dict(group=group, boundary=boundary, cells=len(probability),
                   development_bin_statistics=bin_statistics, bin_budget_positivity=positivity,
                   strict_lp=strict, cap_only_lp=relaxed, infeasibility_diagnostic=certificate,
                   strict_policy_probabilities=probability,
                   strict_policy_status='SELECTED' if strict['success'] else 'FAIL_CLOSED_EMPTY_NOT_FLOOR_FEASIBLE',
                   development_strict=metrics(ag, probability),
                   development_cap_only=metrics(ag, relaxed['probabilities']),
                   development_minimum_excess_at_floor=metrics(ag, certificate['probabilities']))
        records.append(rec)
        print(group, 'strict', strict['success'], 'cap-only maxmin rows', relaxed['minimum_calibration_coverage'],
              'minimax excess dual bound', certificate['dual_lower_bound'], flush=True)
    inputs = {str(p.relative_to(args.design)):sha(p) for p in
              [args.design/f'design_predictions_s{SEED}.npz', args.design/'cache_design/metadata.npz'] +
              [args.design/'cache_design'/f'{k}_aligned.npz' for k in KEYS]}
    selected = dict(protocol=protocol, records=records, development_input_hashes=inputs,
                    all_groups_floor_feasible=all(x['strict_lp']['success'] for x in records))
    dump(args.output/'GROUP_DEVELOPMENT_SELECTED.json', selected)
    selected_sha = sha(args.output/'GROUP_DEVELOPMENT_SELECTED.json')
    # Development choices above are immutable before any guardian cache read.
    before = sha(args.guardian/'GUARDIAN_COMPARISON.json')
    gm = load(args.guardian/'cache/metadata.npz')
    gd = {k: load(args.guardian/'cache'/f'{k}_aligned.npz') for k in KEYS}
    gp = load(args.guardian/f'predictions_s{SEED}.npz')
    assert not set(gm['item_id']).intersection(dm['item_id'])
    for rec in records:
        ag = aggregate(gd, gp, gm, rec['group'], rec['boundary'])
        rec['guardian_consumed_strict'] = metrics(ag, rec['strict_policy_probabilities'])
        rec['guardian_consumed_cap_only'] = metrics(ag, rec['cap_only_lp']['probabilities'])
    assert before == sha(args.guardian/'GUARDIAN_COMPARISON.json')
    assert selected_sha == sha(args.output/'GROUP_DEVELOPMENT_SELECTED.json')
    result = dict(protocol=protocol, records=records, development_input_hashes=inputs,
                  selected_policy_sha256=selected_sha, guardian_result_sha256=before,
                  original_guardian_result_unchanged=True, all_groups_floor_feasible=selected['all_groups_floor_feasible'],
                  guardian_pooled_strict=pooled([x['guardian_consumed_strict'] for x in records]),
                  guardian_pooled_cap_only=pooled([x['guardian_consumed_cap_only'] for x in records]))
    dump(args.output/'GROUP_RESULTS.json', result)
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Category & Cal. min rows & Guardian B rows & Guardian B WAPE & Shadow WAPE \\', r'\midrule']
    for rec in records:
        dev = rec['cap_only_lp']['minimum_calibration_coverage']*100
        gb = rec['guardian_consumed_cap_only']['calibration_b']; gs = rec['guardian_consumed_cap_only']['shadow']
        fmt = lambda x: '--' if x is None else f'{x:.4f}'
        lines.append(f"{rec['group'].title()} & {dev:.2f} & {gb['row_coverage']*100:.2f} & {fmt(gb['wape'])} & {fmt(gs['wape'])} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write(args.output/'GROUP_TABLE.tex', ('\n'.join(lines)+'\n').encode())
    print(json.dumps({'all_groups_floor_feasible':result['all_groups_floor_feasible'],
                      'guardian_cap_only':{x['group']:x['guardian_consumed_cap_only'] for x in records}}, indent=2), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ['design', 'guardian', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
