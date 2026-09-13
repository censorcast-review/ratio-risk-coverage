"""Replay NLL uncertainty from anonymous item sufficient statistics only.

No model, sales, capacity, latent outcome, or holdout array is accessed. Output
defaults outside the packaged evidence tree and cannot overwrite that tree.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from r2_io import dump, sha

ROOT = Path(__file__).resolve().parents[1]


def verify(source, output):
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source) or output.is_relative_to((ROOT/'results').resolve()):
        raise ValueError('Verification output must be outside the evidence directory and results tree.')
    paths = [source/'ITEM_STATISTICS.json', source/'RESULTS.json']
    before = {p.name:sha(p) for p in paths}
    stats, result = [json.loads(p.read_text()) for p in paths]
    assert before['ITEM_STATISTICS.json'] == result['item_statistics_sha256']
    shapes = stats['shapes']
    assert shapes == [.5, 1., 1.5, 2., 3., 5., 10.]
    reference = shapes.index(result['reference_shape'])
    counts = np.asarray(stats['item_row_counts'], dtype=np.int64)
    sums = np.asarray(stats['item_nll_sums'], dtype=float)
    assert sums.shape == (1829, 7) and counts.shape == (1829,)
    assert np.all(counts == 200) and int(counts.sum()) == 365800
    assert np.all(np.isfinite(sums)) and np.all(sums >= 0)
    assert stats['validation_days'] == list(range(1534, 1554))
    assert result['bootstrap_replicates'] == 10000 and result['bootstrap_seed'] == 20260906
    assert result['bonferroni_comparisons'] == [1., 1.5, 3., 5.]
    assert result['bonferroni_tail_probability'] == .00625
    means = np.asarray([sum(sums[:, j])/sum(counts) for j in range(7)])
    assert shapes[int(np.argmin(means))] == 2.
    # Deliberately use one sample per iteration rather than the producer's
    # batched reductions, retaining the same specified RNG stream.
    rng = np.random.default_rng(result['bootstrap_seed'])
    samples = np.empty((10000, 7), dtype=float)
    for b in range(10000):
        indices = rng.integers(len(counts), size=len(counts))
        samples[b] = sums[indices].sum(axis=0)/counts[indices].sum()
    checks = 0
    largest_error = 0.

    def close(actual, expected):
        nonlocal checks, largest_error
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        error = float(np.max(np.abs(a-b)))
        assert error < 1e-12, (actual, expected, error)
        largest_error = max(error, largest_error)
        checks += 1

    for j, rec in enumerate(result['records']):
        assert rec['shape'] == shapes[j]
        close(means[j], rec['validation_nll'])
        close(means[j]-means[reference], rec['difference_vs_kappa2'])
        differences = samples[:, j]-samples[:, reference]
        pointwise = np.quantile(differences, [.025, .975], method='linear')
        close(pointwise, rec['pointwise_95_percentile_interval'])
        assert rec['pointwise_excludes_zero'] == bool(pointwise[0]>0 or pointwise[1]<0)
        if rec['shape'] in result['bonferroni_comparisons']:
            adjusted = np.quantile(differences, [.00625, .99375], method='linear')
            close(adjusted, rec['bonferroni_family_95_percentile_interval'])
            assert rec['bonferroni_excludes_zero'] == bool(adjusted[0]>0 or adjusted[1]<0)
        else:
            assert rec['bonferroni_family_95_percentile_interval'] is None
    for rec in result['historical_nll_replay']:
        close(means[shapes.index(rec['shape'])], rec['historical_nll'])
        assert rec['absolute_difference'] < 1e-12
    for field in ['fits_performed', 'threshold_searches', 'raw_guardian_accesses', 'raw_external_accesses', 'new_hidden_outcome_arrays_loaded']:
        assert result[field] == 0
    assert {p.name:sha(p) for p in paths} == before
    audit = dict(status='PASS', scalar_or_interval_replay_checks=checks,
                 item_clusters=1829, validation_rows=365800,
                 bootstrap_replicates=10000, paired_comparisons=6,
                 four_comparison_bonferroni_family=True,
                 maximum_numeric_discrepancy=largest_error,
                 evidence_hashes=before, evidence_unchanged=True,
                 fits=0, raw_data_accesses=0,
                 interpretation='Post-selection, conditional-on-fixed-fits item-bootstrap diagnostic; no independent winner confirmation or latent-tail identification claim.')
    dump(output, audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'results/review8_nll')
    parser.add_argument('--output', type=Path, default=ROOT/'reproduction_outputs/nb_nll_uncertainty/VERIFICATION.json')
    args = parser.parse_args()
    verify(args.input, args.output)
