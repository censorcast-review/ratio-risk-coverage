"""Meaningful checks for matched targets, frozen masks, and weighted coverage."""
from pathlib import Path
import sys,json,itertools
import numpy as np
from run_selective_study import load,R,origin
p=Path(__file__).resolve().parents[1];d=p/'results/conditional_heads'
for seed in [20260906,20260907,20260908]:
 for new,old in [('direct_excess','contract_excess'),('relative_error_f','relative_error')]:
  a=json.loads((d/f'report_{new}_pooled_s{seed}.json').read_text())
  b=json.loads((p/f'results/selective_strong/evaluation_{old}_s{seed}.json').read_text())[0]
  assert a['thresholds']==b['thresholds']
  assert a['metrics']['shadow']['ALL']['accepted_n']==b['metrics']['shadow']['accepted_n']
  assert abs(a['metrics']['shadow']['ALL']['wape']-b['metrics']['shadow']['wape'])<1e-12
# Finite example: same row count can allocate different demand; weighted-measure identity.
prob=np.array([.8,.2]);mu=np.array([1.,10.]);e=np.array([.9,2.]);a=np.array([5/12,1.])
assert abs(np.sum(prob*a*(e-.3*mu)))<1e-12
nu=prob*mu/np.sum(prob*mu)
assert abs(np.sum(nu*a*(e/mu))/np.sum(nu*a)-.3)<1e-12
assert abs(np.sum(prob*a)-8/15)<1e-12
assert abs(np.sum(nu*a)-5/6)<1e-12
# Frozen comparison lists all seeds and methods, including failures.
f=json.loads((p/'provenance/FROZEN_REVIEW_GUARDIAN_COMPARISON.json').read_text())
assert len(f['policies'])==42 and not f['fresh_guardian_opened']
assert any(v['thresholds'].get('HOBBIES','non-null') is None for v in f['policies'])
assert f['evaluation']['external_access_authorized'] is False
print('PASS: six exact replay policies, weighted-measure identity, complete freeze, and closed external boundary.')
