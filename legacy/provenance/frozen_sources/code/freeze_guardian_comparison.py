"""Freeze a review comparison; this action does not authorize guardian access."""
from pathlib import Path
import json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result=json.loads((ROOT/'results/review_revision/REVIEW_COMPARISON.json').read_text())
assert result['status']=='DESIGN_COMPARISON_COMPLETED'
state=json.loads((ROOT/'inputs/DATA_ACCESS_LEDGER_v0_5.json').read_text())
assert state['fresh_guardian_opened'] is False and state['fresh_guardian_use_count']==0 and state['external_opened'] is False
files=[ROOT/'code'/n for n in ['guardian_prediction.py','run_guardian_comparison.py','run_review_comparison.py','run_selective_study.py']]
files+=list((ROOT/'upstream/design/censorcast_v05').glob('*.py'))
files+=[ROOT/'inputs/forecast_bundle_v0_5.joblib',ROOT/'results/point_baselines/observed_l1_s20260906.txt']
for seed in [20260906,20260907,20260908]:
    files.extend([ROOT/f'results/selective_strong/mean_error_s{seed}.txt',ROOT/f'results/selective_strong/contract_excess_s{seed}.txt',ROOT/f'results/review_revision/demand_s{seed}.txt'])
protocol={
 'schema_version':'review-comparison-1','status':'FROZEN_FOR_SEPARATE_COMPARATIVE_GUARDIAN_AUDIT',
 'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'authorization_status':'PENDING_EXPLICIT_EVALUATION_PURPOSE_AMENDMENT',
 'reason':'Original v0.5 sections 6 and 8 require passing category development gates before guardian opening; those gates have not passed. This separate comparison does not authorize itself.',
 'authorization_phrase':'AUTHORIZE_CENSORCAST_M5_FIXED_COMPARISON_AFTER_FAILED_DESIGN_GATE',
 'design_result_sha256':sha(ROOT/'results/review_revision/REVIEW_COMPARISON.json'),
 'immutable_files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},
 'point_forecast':{'seed':20260906,'alpha':2,'gamma':2,'refit_on_guardian':False},
 'wape_cap':result['cap'],'coverage_floor':result['floor'],'seeds':[20260906,20260907,20260908],'primary_seed':20260906,
 'policies':[{k:r[k] for k in ['seed','score','threshold_mode','thresholds']} for r in result['records']],
 'evaluation':{'items':610,'series':6100,'blocks':{'calibration_a':[1555,1673],'calibration_b':[1674,1793],'shadow':[1800,1913]},'primary_contrast':'first-seed direct_excess versus composite_excess on shadow; both frozen pooled thresholds',
 'reported_endpoints':['row coverage','demand coverage','accepted WAPE','category-specific values'],'all_seed_results_reported':True,
 'adjusted_bound_family':'2 primary methods x 3 blocks x 4 strata (pooled and 3 categories) x 2 endpoints = 48',
 'bootstrap':{'cluster':'item_id','reps':10000,'seed':20260906,'nominal_familywise_alpha':.05,'per_endpoint_alpha':.05/48,'validity':'Approximate item bootstrap; common store/calendar shocks remain. No exact finite-sample guarantee.'},
 'same_period_item_holdout_not_future_time_evaluation':True,'no_retraining_no_threshold_reselection':True,'certificate_issued':False,'external_access_authorized':False},
 'guardian_context_sha256':'5ed3f06f69c570f2c6aff33a825b944f13b49144bedaf76e6f2f9666664ca62f',
 'guardian_outcomes_sha256':'6bf62d608054063e7142cf47b15b75bec66765e1f04cc006138dc29037f0c827',
 'fresh_guardian_opened':False,'external_opened':False}
path=ROOT/'provenance/FROZEN_REVIEW_GUARDIAN_COMPARISON.json'
if path.exists():
    old=json.loads(path.read_text());assert old['immutable_files']==protocol['immutable_files'] and old['policies']==protocol['policies'],'Existing freeze differs; do not silently replace it.'
else:path.write_text(json.dumps(protocol,indent=2,allow_nan=False))
print({'status':protocol['status'],'sha256':sha(path),'policy_count':len(protocol['policies']),'immutable_files':len(protocol['immutable_files']),'guardian_opened':False})
