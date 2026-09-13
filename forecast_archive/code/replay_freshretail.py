"""Replay the frozen FreshRetailNet forecast from packaged raw data and models.

This is a POST-HOC REPRODUCTION of an already consumed official evaluation
split, not another confirmatory opening. No external I/O, model training,
policy fitting/selection, or change to any frozen evidence is performed. The
train-only target encoder is reconstructed deterministically as in the
documented recovery runner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main(package, output, resume_encoder=False):
    start=time.time()
    sys.path.insert(0,str(package/'code/freshretail'))
    from censorcast.data import infer_split_dates, make_supervised, normalize_frame, split_supervised
    from censorcast.models import SmoothedTargetEncoder
    evidence=package/'evidence/freshretail/frozen_eval'
    output.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((evidence/'RUN_RECEIPT.json').read_text())
    initial=json.loads((evidence/'FROZEN_RC_EVAL_RECEIPT.json').read_text())
    expected={
        package/'inputs/freshretail/freshretail_2000_train_only.pkl.gz': initial['train_cache_sha256'],
        package/'inputs/freshretail/official_eval.parquet': receipt['eval_sha256'],
        evidence/'base_l1.ubj': receipt['base_sha256'],
        evidence/'stockout_risk.ubj': receipt['risk_sha256'],
        evidence/'FINAL_POLICY.json': receipt['policy_sha256'],
        evidence/'EVAL_PREDICTIONS.csv.gz': receipt['predictions_sha256'],
        evidence/'FROZEN_RC_EVAL_PROTOCOL.md': receipt['protocol_sha256'],
        package/'code/freshretail/run_frozen_rc_eval.py': receipt['original_runner_sha256'],
        package/'code/freshretail/resume_frozen_rc_eval_after_engine_failure.py': receipt['recovery_runner_sha256'],
    }
    for path,wanted in expected.items():
        if sha(path)!=wanted:
            raise AssertionError(f'Frozen hash mismatch: {path}')
    print('FROZEN_HASHES_PASS',flush=True)
    train=pd.read_pickle(package/'inputs/freshretail/freshretail_2000_train_only.pkl.gz')
    # The source cache's large descriptive selected_series list is not read
    # by any forecasting function. Removing only DataFrame metadata prevents
    # pandas from deep-copying that list for every per-series rolling call.
    # No table value, ordering rule, feature, or learned object is changed.
    train.attrs={}
    train=normalize_frame(train)
    train['_source_split']='train'
    ids=set(train.series_id.astype(str).unique())
    train_dates=infer_split_dates(train)
    checkpoint_path=output/'TRAIN_ENCODER_REPLAY_CHECKPOINT.pkl'
    if resume_encoder:
        # Explicit resume is only for the auditor's own earlier train-only
        # checkpoint. A normal documented invocation reconstructs from scratch.
        with checkpoint_path.open('rb') as handle:
            checkpoint=pickle.load(handle)
        encoder=checkpoint['encoder']; numeric=checkpoint['numeric']; categorical=checkpoint['categorical']
        split_report=checkpoint['train_splits']
        print('TRAIN_ONLY_AUDIT_ENCODER_CHECKPOINT_RESUMED',flush=True)
    else:
        train_rows,numeric,categorical=make_supervised(train,horizons=range(1,8),max_train_rows=600000,seed=20260907,development_holdout_days=35)
        splits=split_supervised(train_rows,train_dates)
        fit=splits['fit']
        y=fit.sale_amount.to_numpy(float)
        available=fit.is_censored.eq(0).to_numpy()
        encoder=SmoothedTargetEncoder(categorical=categorical,numeric=numeric,smoothing=30)
        encoder.fit(fit,y,uncensored=available)
        split_report={name:dict(rows=len(f),available=int(f.is_censored.eq(0).sum()),
                                first_target=str(f.dt.min().date()) if len(f) else None,
                                last_target=str(f.dt.max().date()) if len(f) else None)
                      for name,f in splits.items()}
        with checkpoint_path.open('wb') as handle:
            pickle.dump(dict(encoder=encoder,numeric=numeric,categorical=categorical,train_splits=split_report),handle)
        del fit,splits,train_rows
        print('TRAIN_ENCODER_RECONSTRUCTED',flush=True)
    evaluation=normalize_frame(pd.read_parquet(package/'inputs/freshretail/official_eval.parquet'))
    evaluation['_source_split']='eval'
    evaluation=evaluation.loc[evaluation.series_id.astype(str).isin(ids)].copy()
    assert len(evaluation)==14000 and evaluation.series_id.nunique()==2000
    combined=pd.concat([train,evaluation],ignore_index=True)
    dates=infer_split_dates(combined)
    rows,numeric_check,categorical_check=make_supervised(combined,horizons=range(1,8),max_train_rows=7,seed=20260907,development_holdout_days=35)
    assert numeric_check==numeric and categorical_check==categorical
    test=split_supervised(rows,dates)['test'].reset_index(drop=True)
    x=encoder.transform(test)
    base=xgb.Booster();base.load_model(evidence/'base_l1.ubj');base.set_param({'nthread':8})
    risk=xgb.Booster();risk.load_model(evidence/'stockout_risk.ubj');risk.set_param({'nthread':8})
    matrix=xgb.DMatrix(x)
    raw=np.maximum(base.predict(matrix)-1,0)
    q=np.clip(risk.predict(matrix),0,1)
    policy=json.loads((evidence/'FINAL_POLICY.json').read_text())
    categories=test.management_group_id.astype(str).to_numpy()
    bins=np.searchsorted(policy['edges'],q,side='right')
    category_scales=np.array([policy['category_scales'][c] for c in categories])
    scales=np.array([policy['scales'][f'{c}|{b}'] for c,b in zip(categories,bins)])
    replay=test[['series_id','dt','horizon','management_group_id','sale_amount','is_censored']].copy()
    replay['dt']=replay.dt.dt.strftime('%Y-%m-%d')
    replay['base']=raw*category_scales
    replay['risk_conditioned']=raw*scales
    replay['predicted_stockout_risk']=q
    saved=pd.read_csv(evidence/'EVAL_PREDICTIONS.csv.gz')
    keys=['series_id','dt','horizon']
    saved=saved.sort_values(keys).reset_index(drop=True)
    replay=replay.sort_values(keys).reset_index(drop=True)
    for column in keys+['management_group_id','is_censored']:
        assert np.array_equal(saved[column],replay[column]),column
    label_diff=float(np.max(np.abs(saved.sale_amount-replay.sale_amount)))
    assert label_diff<2e-12,label_diff
    errors={column:float(np.max(np.abs(saved[column]-replay[column]))) for column in ['base','risk_conditioned','predicted_stockout_risk']}
    # Pandas serializes the original float32 q column with shortest float32
    # decimals; round-trip both sides to float32 before exact comparison.
    assert max(errors['base'],errors['risk_conditioned'])<2e-12,errors
    assert np.array_equal(saved.predicted_stockout_risk.to_numpy(np.float32),replay.predicted_stockout_risk.to_numpy(np.float32)),errors
    print('ALL_14000_PREDICTIONS_MATCH',json.dumps(errors),flush=True)

    # A targeted dependency test: replace all post-origin realized outcomes and
    # weather with extreme sentinels in a temporary in-memory copy. Evaluation
    # features must be identical; scheduled discount/activity/calendar remain.
    mutated=combined.copy()
    future=mutated['_source_split'].eq('eval')
    for col in ['sale_amount','is_censored','stockout_fraction','precpt','avg_temperature','avg_humidity','avg_wind_level']:
        mutated.loc[future,col]=1234567.0
    check_rows,check_n,check_c=make_supervised(mutated,horizons=range(1,8),max_train_rows=7,seed=20260907,development_holdout_days=35)
    check_test=split_supervised(check_rows,dates)['test'].reset_index(drop=True)
    altered_x=encoder.transform(check_test)
    feature_change=float(np.max(np.abs(x-altered_x)))
    assert feature_change==0.0,'Post-origin realized outcomes/weather changed features'
    origins=pd.to_datetime(test.dt)-pd.to_timedelta(test.horizon,unit='D')
    assert origins.nunique()==1 and origins.iloc[0]==pd.Timestamp('2024-06-25')
    replay.to_csv(output/'REPLAYED_PREDICTIONS.csv.gz',index=False)
    result=dict(status='PASS',type='POST_HOC_FROZEN_MODEL_REPRODUCTION',
                rows=len(replay),prediction_max_abs_diff=errors,label_max_abs_diff=label_diff,
                predicted_risk_exact_after_float32_roundtrip=True,
                post_origin_outcome_weather_feature_max_abs_diff=feature_change,
                train_splits=split_report,
                numeric_features=numeric,categorical_features=categorical,
                train_only_encoder_reconstructions=0 if resume_encoder else 1,
                train_only_encoder_checkpoint_resumed=bool(resume_encoder),
                train_only_encoder_checkpoint_sha256=sha(checkpoint_path),new_model_fits=0,
                new_calibration_fits=0,new_policy_selections=0,external_requests=0,
                already_consumed_raw_eval_decodes=1,
                frozen_sources_checked=len(expected),
                feature_builder_sha256=sha(package/'code/freshretail/censorcast/data.py'),
                encoder_source_sha256=sha(package/'code/freshretail/censorcast/models.py'),
                replay_optimization='Removed descriptive DataFrame.attrs metadata before feature construction to avoid repeated deep copies; no table values, frozen inputs, feature source, models, or policy were changed.',
                replay_script_sha256=sha(Path(__file__)),
                duration_seconds=time.time()-start,
                software=dict(numpy=np.__version__,pandas=pd.__version__,xgboost=xgb.__version__),
                interpretation='Confirms current packaged data/code/models reproduce saved predictions. Does not independently prove the historical first-access claim.')
    (output/'FROZEN_PREDICTION_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-encoder-checkpoint',action='store_true',help='Explicitly resume an encoder from this auditor output directory after an interrupted post-hoc replay.')
    args=parser.parse_args()
    main(args.package,args.output,args.resume_encoder_checkpoint)
