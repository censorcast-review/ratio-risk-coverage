"""Freeze the revised two-policy plan without accessing external data."""
from pathlib import Path
import datetime, json, sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'code'))
from r2_io import sha, dump
from evaluate import THRESHOLDS, LAMBDAS, REFERENCE_MEAN

PREVIOUS = ROOT/'code/review4_external/EXTERNAL_PLAN.json'
PREVIOUS_SHA = '390e11d0494c7fe493545063001c8dd479d50cd4a3f4fd00bf836e60dec9cd19'


def make_plan():
    assert sha(PREVIOUS)==PREVIOUS_SHA, 'Previous pending plan must remain byte-exact'
    plan=json.loads(PREVIOUS.read_text())
    assert plan['external_opened'] is False and plan['status']=='FROZEN_PENDING_EXPLICIT_EXTERNAL_PURPOSE_AMENDMENT'
    source=ROOT/'results/review4/TRADEOFF_DEVELOPMENT_SELECTED.json'
    selected=json.loads(source.read_text())
    rows={x['score']:x for x in selected['records']}
    assert selected['reference_mean']==REFERENCE_MEAN
    assert selected['protocol']['seed']==20260906
    assert rows['mixed_1']['threshold']==THRESHOLDS['composed_excess_lambda_1']
    assert rows['mixed_0.25']['threshold']==THRESHOLDS['mixed_utility_lambda_0_25']
    assert sha(ROOT/'code/run_review4_tradeoffs.py')==selected['protocol']['code_sha256']
    dependencies=[
      ROOT/'inputs/forecast_bundle_v0_5.joblib',
      ROOT/'results/point_baselines/observed_l1_s20260906.txt',
      ROOT/'results/selective_strong/mean_error_s20260906.txt',
      ROOT/'results/review_revision/demand_s20260906.txt',
      ROOT/'results/review2/cache_design/metadata.npz',
      ROOT/'code/r2_io.py', ROOT/'code/run_review4_tradeoffs.py',
      ROOT/'code/run_selective_study.py', ROOT/'code/run_review2_objectives.py',
      ROOT/'results/review4/TRADEOFF_PROTOCOL.json',source,
      ROOT/'upstream/preparation/PROTOCOL_v0_5.md', PREVIOUS,
      HERE/'requirements-external.txt',
    ]
    dependencies+=list(HERE.glob('*.py'))+list((HERE/'metadata').glob('*.json'))
    dependencies+=list((ROOT/'upstream/design/censorcast_v05').glob('*.py'))
    immutable={str(p.relative_to(ROOT)):sha(p) for p in dependencies}
    # Models and categorical-reference metadata must be unchanged from v4.
    for rel,h in immutable.items():
        if rel in plan['immutable_files']:
            assert h==plan['immutable_files'][rel], rel
    plan.update(
      schema_version='review5-external-1',
      supersedes_sha256=PREVIOUS_SHA,
      superseded_plan=str(PREVIOUS.relative_to(ROOT)),
      revision_reason='The fourth reviewer simulation motivates a different pair: composed excess (lambda=1) versus mixed utility (lambda=.25). The pair is chosen after consumed-guardian diagnostics, before external access. Pure demand optimality is not tested. Prior pending plan remains byte-exact and unapproved; no external access occurred.',
      created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      policies=[
       {'name':'composed_excess_lambda_1','objective':'row','lambda':1.0,
        'threshold':THRESHOLDS['composed_excess_lambda_1'],
        'score':'(max(error_head,0)-r*max(demand_head,0))/(1+0*max(demand_head,0)/T)',
        'error_trees':220,'demand_trees':220,'raw_head_dtype':'float32','score_dtype':'float64'},
       {'name':'mixed_utility_lambda_0_25','objective':'mixed: .25*row + .75*demand',
        'lambda':.25,'threshold':THRESHOLDS['mixed_utility_lambda_0_25'],
        'score':'(max(error_head,0)-r*max(demand_head,0))/(.25+.75*max(demand_head,0)/T)',
        'error_trees':220,'demand_trees':220,'raw_head_dtype':'float32','score_dtype':'float64'}],
      reference_mean=REFERENCE_MEAN,
      reference_mean_source='Total true demand / total rows on consumed development calibration A/B only; not re-estimated on external',
      threshold_selection='Previously fixed independently for each lambda on development calibration A/B only; largest common feasible threshold with point WAPE<=r and row coverage>=.35; no search at evaluation.',
      policy_pair_selection='Chosen after inspecting consumed-guardian exploratory diagnostics and before external access. Not preregistered relative to guardian and not a pure row-versus-demand optimum test.',
      confirmatory_scope='One independent external item-partition comparison of the two frozen learned policies. Supports only the prespecified coverage contrast if both directions pass; not proof of mixed utility optimality or natural-stockout operation.',
      immutable_files=immutable)
    del plan['threshold_precision_note']
    plan['primary']['contrast']='mixed_utility_lambda_0_25 minus composed_excess_lambda_1'
    plan['forbidden']+=['alternative lambda values','re-estimate T on external','revert to the superseded policy pair after seeing external']
    return plan


if __name__=='__main__':
    plan=make_plan(); target=HERE/'EXTERNAL_PLAN.json'
    if target.exists():
        old=json.loads(target.read_text());plan['created_utc']=old['created_utc']
        if old != plan: raise RuntimeError('Frozen pending plan differs; create an explicitly versioned superseding plan before approval.')
    else: dump(target,plan)
    print(json.dumps({'status':plan['status'],'sha256':sha(target),'policies':2,'external_opened':False}))
