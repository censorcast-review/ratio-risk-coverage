"""Synthetic numerical/interface tests; these are not paper experiments."""
import unittest
from dataclasses import replace
import numpy as np
from direct_gate import (GateConfig, Split, _forward, _gradient, fit_gate,
                         design_threshold, screen_policy, run_direct_gate_comparison)


class DirectGateTests(unittest.TestCase):
    def test_analytic_gradient_against_finite_differences(self):
        rng = np.random.default_rng(22)
        X = rng.normal(size=(7, 3))
        p = {'W1': rng.normal(size=(3, 4)), 'b1': rng.normal(size=4),
             'W2': rng.normal(size=(4, 3)), 'b2': rng.normal(size=3),
             'W3': rng.normal(size=(3, 1)), 'b3': rng.normal(size=1)}
        coefficient = rng.normal(size=7)
        a, cache = _forward(X, p)
        grad = _gradient(p, cache, a, coefficient)
        h = 1e-6
        for key in p:
            for index in np.ndindex(p[key].shape):
                original = p[key][index]
                p[key][index] = original+h
                plus = np.mean(coefficient*_forward(X, p)[0])
                p[key][index] = original-h
                minus = np.mean(coefficient*_forward(X, p)[0])
                p[key][index] = original
                self.assertAlmostEqual(grad[key][index], (plus-minus)/(2*h), places=7)

    def test_threshold_matches_brute_force_with_ties(self):
        rng = np.random.default_rng(6)
        for _ in range(100):
            score = rng.integers(0, 8, size=50).astype(float)
            w = rng.integers(0, 5, size=50).astype(float)
            ell = rng.random(50)*w
            r, floor = .4, .2
            for objective in ('case', 'exposure'):
                got = design_threshold(score, ell, w, r, floor, objective)
                feasible = []
                for t in np.unique(score):
                    a = score <= t
                    if w[a].sum() > 0 and a.mean() >= floor and ell[a].sum() <= r*w[a].sum()+1e-10:
                        u = a.mean() if objective == 'case' else w[a].sum()/w.sum()
                        feasible.append((u, a.sum(), -t, t))
                self.assertEqual(got['eligible'], bool(feasible))
                if feasible:
                    self.assertEqual(got['threshold'], max(feasible)[-1])

    def test_same_seed_identical_objectives_when_exposure_constant(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(80, 4))
        X[3, 2] = np.nan
        ell = rng.random(80)
        w = np.ones(80)*3
        config = GateConfig(seed=17, epochs=4, batch_size=19, case_floor=.35)
        case = fit_gate(X, ell, w, .3, 'case', 17, .35, config)
        exposure = fit_gate(X, ell, w, .3, 'exposure', 17, .35, config)
        for key in case.parameters:
            np.testing.assert_array_equal(case.parameters[key], exposure.parameters[key])
        np.testing.assert_array_equal(case.predict(X), -case.score(X))
        self.assertEqual(case.trace, exposure.trace)
        self.assertAlmostEqual(case.median[2], np.nanmedian(X[:, 2]))

    def test_B_E_outcomes_cannot_change_model_or_A_policy(self):
        rng = np.random.default_rng(8)
        def make(n):
            X = rng.normal(size=(n, 3))
            w = np.exp(.3*X[:, 1])
            loss = w*(.04+.6/(1+np.exp(-X[:, 0])))
            return Split(X, loss, w, np.arange(n) % 8)
        train, A, B, E = make(128), make(80), make(80), make(80)
        refs = {str(j): {name: split.X[:, 0]+j*.1*split.X[:, 1]
                        for name, split in [('A', A), ('B', B), ('E', E)]} for j in range(5)}
        config = GateConfig(seed=9, epochs=3, batch_size=64, case_floor=.2)
        counts = rng.multinomial(8, np.ones(8)/8, size=100)
        r1, m1 = run_direct_gate_comparison(train, A, B, E, refs, config, counts, counts)
        B2, E2 = Split(B.X.copy(), B.loss*3, B.exposure*.7, B.blocks), Split(E.X.copy(), E.loss*2, E.exposure*1.3, E.blocks)
        r2, m2 = run_direct_gate_comparison(train, A, B2, E2, refs, config, counts, counts)
        self.assertAlmostEqual(r1['cap_A_only'], .95*A.loss.sum()/A.exposure.sum(), places=15)
        self.assertEqual(r1['A_candidates'], r2['A_candidates'])
        for name in m1:
            for key in m1[name].parameters:
                np.testing.assert_array_equal(m1[name].parameters[key], m2[name].parameters[key])
        for name in r1['policies']:
            for field in ('eligible', 'method', 'threshold', 'A'):
                self.assertEqual(r1['policies'][name].get(field), r2['policies'][name].get(field))

    def test_infeasible_results_are_reported_without_B_fallback(self):
        X = np.arange(20).reshape(10, 2).astype(float)
        split = Split(X, np.ones(10), np.ones(10)*2, np.arange(10))
        refs = {str(i): {s: X[:, 0]+i for s in ('A', 'B', 'E')} for i in range(5)}
        report, _ = run_direct_gate_comparison(split, split, split, split, refs,
                                              GateConfig(epochs=2, batch_size=5))
        self.assertTrue(all(not p['eligible'] for p in report['policies'].values()))
        self.assertTrue(all('undefined' in c['status'] for c in report['E_contrasts'].values()))

    def test_screen_empty_calendar_blocks_and_zero_accepted_exposure(self):
        split = Split(np.zeros((4, 1)), np.ones(4)*.1, np.ones(4), np.array([0, 0, 2, 2])).validate()
        counts = np.array([[1, 1, 1, 1], [2, 0, 2, 0], [1, 0, 3, 0]])
        result = screen_policy(split, np.ones(4, dtype=bool), .2, counts, .05, 4)
        self.assertTrue(result['passed'])
        failed = screen_policy(split, np.zeros(4, dtype=bool), .2, counts, .05, 4)
        self.assertFalse(failed['passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
