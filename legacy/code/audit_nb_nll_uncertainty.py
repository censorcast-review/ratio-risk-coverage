"""Development-only NLL replay and paired item uncertainty; no fitting.

Only the already-consumed validation feature/sales/capacity cache is read.
Published sufficient statistics make the bootstrap independently replayable.
These are conditional-on-fit, post-selection diagnostics: kappa=2 was chosen
using this validation set before this analysis was designed.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
from importlib.metadata import version
import numpy as np
import lightgbm as lgb
from r3_observable import nll, predict_mean
from r2_io import dump, sha, write

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / 'results/nb_grid'
SHAPES = [.5, 1., 1.5, 2., 3., 5., 10.]
FAMILY = [1., 1.5, 3., 5.]
REFERENCE = 2.
SEED = 20260906
REPLICATES = 10000


def model_path(shape):
    suffix = str(shape).rstrip('0').rstrip('.').replace('.', 'p')
    directory = ROOT/'results/capacity_hit_extension/models' if shape in [.5, 2., 10.] else GRID/'models'
    return directory / f'capacity_hit_nb_{suffix}_s{SEED}.txt'


def bootstrap(item_sums, counts):
    """One common item resample for all kappas and their paired differences."""
    rng = np.random.default_rng(SEED)
    estimates = np.empty((REPLICATES, len(SHAPES)))
    for start in range(0, REPLICATES, 100):
        indices = rng.integers(0, len(counts), (min(100, REPLICATES-start), len(counts)))
        estimates[start:start+len(indices)] = item_sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None]
    return estimates


def make_results(stats):
    counts = np.asarray(stats['item_row_counts'], np.int64)
    sums = np.asarray(stats['item_nll_sums'], float)
    means = sums.sum(axis=0) / counts.sum()
    estimates = bootstrap(sums, counts)
    reference = SHAPES.index(REFERENCE)
    records = []
    for j, shape in enumerate(SHAPES):
        differences = estimates[:, j] - estimates[:, reference]
        pointwise = np.quantile(differences, [.025, .975], method='linear').tolist()
        adjusted = np.quantile(differences, [.05/(2*len(FAMILY)), 1-.05/(2*len(FAMILY))], method='linear').tolist() if shape in FAMILY else None
        records.append(dict(shape=shape, validation_nll=float(means[j]),
                            difference_vs_kappa2=float(means[j]-means[reference]),
                            pointwise_95_percentile_interval=pointwise,
                            bonferroni_family_95_percentile_interval=adjusted,
                            in_four_comparison_family=shape in FAMILY,
                            pointwise_excludes_zero=bool(pointwise[0] > 0 or pointwise[1] < 0),
                            bonferroni_excludes_zero=None if adjusted is None else bool(adjusted[0] > 0 or adjusted[1] < 0)))
    return dict(schema='nb-nll-uncertainty-v1', status='COMPLETE',
                scope='POST_SELECTION_CONDITIONAL_ON_FIXED_FITS_DEVELOPMENT_ONLY',
                n_items=len(counts), n_rows=int(counts.sum()), n_series=stats['n_series'],
                validation_days=stats['validation_days'], reference_shape=REFERENCE,
                bootstrap_replicates=REPLICATES, bootstrap_seed=SEED,
                bootstrap_unit='item; all stores and validation days travel together',
                estimator='total sampled NLL divided by total sampled rows; paired across shapes',
                interval_method='percentile; NumPy linear empirical quantile',
                bonferroni_comparisons=FAMILY, bonferroni_tail_probability=.05/(2*len(FAMILY)),
                caveats=[
                    'Kappa=2 was selected on this validation set; these are post-selection descriptive intervals, not independent winner confirmation.',
                    'Intervals condition on the fitted models and do not resample training or reselect kappa.',
                    'Item resampling retains within-item dependence but does not remove common store/calendar shocks.',
                    'Bonferroni quantiles address the four displayed fixed comparisons; they do not correct the earlier model-development and selection history.',
                    'Observed-data score differences do not by themselves identify latent censored-tail moments or validate downstream risk control.'
                ], records=records, fits_performed=0, threshold_searches=0,
                raw_guardian_accesses=0, raw_external_accesses=0,
                new_hidden_outcome_arrays_loaded=0)


def table_bytes(result):
    lines = [r'\begin{table}[t]', r'\centering\small',
             r'\caption{Observed validation NLL uncertainty for the fixed first-seed NB models. Differences and intervals are in $10^{-3}$ NLL per row, relative to $\kappa=2$. Paired item bootstrap: 1,829 items, 10,000 replicates. Adjusted intervals use Bonferroni tail probability $0.05/(2\times4)$ for the four fine-grid comparisons; anchors 0.5 and 10 are descriptive. These conditional-on-fit, post-selection intervals use the same validation data that selected $\kappa=2$.}',
             r'\label{tab:nb-nll-uncertainty}',
             r'\begin{tabular}{rrrrr}\toprule',
             r'$\kappa$ & NLL & $\Delta\times10^3$ & Pointwise 95\% interval & Adjusted interval \\ \midrule']
    for row in result['records']:
        k = f"{row['shape']:g}"
        def interval(value):
            return '--' if value is None else f"[{value[0]*1000:.3f}, {value[1]*1000:.3f}]"
        if row['shape'] == REFERENCE:
            lines.append(f"{k} & {row['validation_nll']:.6f} & 0 & reference & reference " + r'\\')
        else:
            lines.append(f"{k} & {row['validation_nll']:.6f} & {row['difference_vs_kappa2']*1000:.3f} & {interval(row['pointwise_95_percentile_interval'])} & {interval(row['bonferroni_family_95_percentile_interval'])} " + r'\\')
    lines += [r'\bottomrule\end{tabular}', r'\end{table}', '']
    return '\n'.join(lines).encode()


def run(out, table):
    cache = GRID/'cache'
    paths = [cache/'validation.npz', cache/'validation_x.npy', cache/'metadata.npz', GRID/'RESULTS.json', GRID/'PROTOCOL.json']
    paths += [p for k in SHAPES for p in (model_path(k), model_path(k).with_suffix('.json'))]
    protocol = dict(schema='nb-nll-uncertainty-protocol-v1',
                    status='RECORDED_BEFORE_NLL_REPLAY_AND_BOOTSTRAP',
                    recorded_utc=datetime.now(timezone.utc).isoformat(),
                    retrospective=True, seed=SEED, replicates=REPLICATES,
                    shapes=SHAPES, reference_shape=REFERENCE, family=FAMILY,
                    alpha=.05, validation_days=list(range(1534, 1554)),
                    cluster='item across all stores and all 20 validation days',
                    weighting='ratio of sampled item NLL sums to sampled row counts',
                    likelihood='pmf(S) for S<C; Pr(Y>=C)=sf(C-1) for S>=C',
                    arrays_loaded=['validation_x', 'observed', 'capacity', 'days', 'item_id'],
                    selection_scope='same validation selected kappa=2 earlier; no selection-aware or independent confirmation claim',
                    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
                    code_hashes={p:sha(ROOT/'code'/p) for p in ['audit_nb_nll_uncertainty.py', 'r3_observable.py']},
                    fits_performed=0, new_data_openings=0)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'PROTOCOL.json').exists():
        previous = json.loads((out/'PROTOCOL.json').read_text())
        for key in protocol:
            if key != 'recorded_utc':
                assert previous[key] == protocol[key], f'Protocol differs: {key}'
        protocol = previous
    else:
        dump(out/'PROTOCOL.json', protocol)
    historical = {float(r['shape']):r for r in json.loads((GRID/'RESULTS.json').read_text())['records']}
    with np.load(cache/'validation.npz', allow_pickle=False) as saved:
        observed = saved['observed']
        capacity = saved['capacity']
        days = saved['days']
    with np.load(cache/'metadata.npz', allow_pickle=False) as saved:
        item = saved['item_id']
    assert np.array_equal(days, np.arange(1534, 1554))
    unique_items, item_index = np.unique(item, return_inverse=True)
    nitems = len(unique_items)
    counts = np.bincount(item_index, minlength=nitems)*len(days)
    assert nitems == 1829 and len(item) == 18290 and observed.shape == (18290, 20)
    hit = observed >= capacity
    bound = np.where(hit, capacity-1, observed).astype(float)
    x = np.load(cache/'validation_x.npy', mmap_mode='r').reshape(-1, 34)
    item_sums = np.empty((nitems, len(SHAPES)))
    agreement = []
    for j, k in enumerate(SHAPES):
        path = model_path(k)
        info = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == info['model_sha256'] == historical[k]['model_sha256']
        model = lgb.Booster(model_file=str(path))
        means = predict_mean(model, x, info)
        losses = nll(means, bound.ravel(), hit.ravel(), k).reshape(observed.shape)
        assert np.isfinite(losses).all() and np.all(losses >= -1e-12)
        mean = float(losses.mean())
        expected = historical[k]['observed_validation_nll']
        delta = mean-expected
        assert abs(delta) < 1e-12, (k, mean, expected)
        item_sums[:, j] = np.bincount(item_index, weights=losses.sum(axis=1), minlength=nitems)
        assert abs(item_sums[:, j].sum()/counts.sum()-mean) < 1e-12
        agreement.append(dict(shape=k, replay_nll=mean, historical_nll=expected, absolute_difference=abs(delta)))
        print(f'Fixed kappa={k:g}: NLL={mean:.12f}, historical difference={delta:.3g}', flush=True)
    stats = dict(schema='nb-validation-item-statistics-v1', scope='CONSUMED_DEVELOPMENT_VALIDATION_ONLY',
                 item_order='anonymous cluster indices in lexicographic item_id order; identifiers omitted',
                 shapes=SHAPES, n_series=len(item), validation_days=days.tolist(),
                 item_row_counts=counts.tolist(), item_nll_sums=item_sums.tolist(),
                 protocol_sha256=sha(out/'PROTOCOL.json'),
                 likelihood='pmf(S) for S<C; Pr(Y>=C)=sf(C-1) for S>=C')
    dump(out/'ITEM_STATISTICS.json', stats)
    result = make_results(stats)
    result['item_statistics_sha256'] = sha(out/'ITEM_STATISTICS.json')
    result['protocol_sha256'] = sha(out/'PROTOCOL.json')
    result['historical_nll_replay'] = agreement
    result['runtime'] = {p:version(p) for p in ['numpy', 'scipy', 'lightgbm']}
    dump(out/'RESULTS.json', result)
    write(table, table_bytes(result))
    dump(out/'EXECUTION_AUDIT.json', dict(status='PASS', fits=0, threshold_searches=0,
             new_guardian_accesses=0, new_external_accesses=0,
             input_hashes_unchanged=all(sha(ROOT/p)==h for p,h in protocol['source_hashes'].items()),
             historical_nll_replays=len(agreement), maximum_historical_nll_error=max(r['absolute_difference'] for r in agreement),
             results_sha256=sha(out/'RESULTS.json'), item_statistics_sha256=sha(out/'ITEM_STATISTICS.json'),
             table_sha256=sha(table)))
    print(json.dumps(result['records'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'reproduction_outputs/nb_nll_recompute')
    parser.add_argument('--table', type=Path, default=ROOT/'reproduction_outputs/nb_nll_recompute/nb_nll_uncertainty.tex')
    args = parser.parse_args()
    run(args.out, args.table)
