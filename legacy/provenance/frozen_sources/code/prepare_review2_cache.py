"""Recreate already-consumed design predictions using saved models, without fit."""
from pathlib import Path
from types import SimpleNamespace
import json, numpy as np, lightgbm as lgb
from r2_io import install_numpy_writers,sha,dump
from run_selective_study import construct,load,BLOCKS,R
from run_review_comparison import predict,SEEDS
from prepare_strong_point_cache import main as point_cache
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/review2';OUT.mkdir(exist_ok=True)
install_numpy_writers()
freeze=ROOT/'provenance/FROZEN_REVIEW_GUARDIAN_COMPARISON.json'
assert sha(freeze)=='3d94aec80d993b92684a61f058ed4e3ceb2351233f07fb8eea3f9f542fe2b156'
for rel,h in json.loads(freeze.read_text())['immutable_files'].items():assert sha(ROOT/rel)==h,rel
point_cache(SimpleNamespace(input=ROOT/'inputs/m5',output=ROOT/'inputs/strong_point',points=ROOT/'results/point_baselines',adapter=ROOT/'results/adapter_v2'))
cache=OUT/'cache_design';cache.mkdir(exist_ok=True)
if not (cache/'ORIGIN_AUDIT.json').exists():construct(ROOT/'inputs/strong_point',cache)
source=ROOT/'inputs/m5/selection_base_v0_5.npz';v=load(source)
value=float(np.abs(v['truth'].astype(float)-v['raw_poisson_histgb']).sum()/v['truth'].sum())
assert value==.7530939208313988
dump(OUT/'CAP_REFERENCE.json',dict(value=value,source_sha256=sha(source),array='raw_poisson_histgb',truth='pre_mechanical_censoring_recorded_sales',target_days=[int(v['target_days'].min()),int(v['target_days'].max())],series_count=len(v['truth']),item_count=1829,n=int(v['truth'].size),multiplier=.85,cap=R,scope='earlier_consumed_design_selection_reference_retained_without_retuning'))
xs={k:np.load(cache/(k+'_features.npy'),mmap_mode='r') for k in BLOCKS[1:]}
for seed in SEEDS:
    path=OUT/f'design_predictions_s{seed}.npz'
    if path.exists():continue
    pred={}
    for name,rel in [('error',f'selective_strong/mean_error_s{seed}.txt'),('direct',f'selective_strong/contract_excess_s{seed}.txt'),('demand',f'review_revision/demand_s{seed}.txt')]:
        b=lgb.Booster(model_file=str(ROOT/'results'/rel))
        for k,x in xs.items():
            pred[name+'_'+k]=predict(b,x)
            if name!='direct':pred[name+'110_'+k]=predict(b,x,110)
        print('Restored score',seed,name,flush=True)
    np.savez_compressed(path,**pred)
dump(OUT/'CACHE_RECOVERY.json',dict(status='CONSUMED_DESIGN_CACHE_RESTORED',fit_calls=0,new_cohorts_read=0,external_opened=False,model_hashes_verified=19))
print('REVIEW2 CACHE READY',flush=True)
