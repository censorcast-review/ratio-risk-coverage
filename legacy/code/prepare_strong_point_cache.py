"""Forecast additional development blocks with the first prespecified L1 fit.

The forecast seed is fixed (20260906), not selected by shadow performance.
The censor adapter parameters were selected from the earlier selection window.
"""
import argparse,json,time,zipfile,os
from pathlib import Path
import numpy as np
import lightgbm as lgb
from legacy_features.features import DesignPanel,FeatureBuilder

def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    adapters=json.loads((a.adapter/'ADAPTER_RESULTS.json').read_text())['records']
    chosen=next(r for r in adapters if r['seed']==20260906 and r['method']=='censor_adapter')
    strength=chosen['choice_on_selection']['strength'];power=chosen['choice_on_selection']['power']
    (a.output/'POINT_POLICY.json').write_text(json.dumps({'seed':20260906,'adapter':chosen,'selection_only':True},indent=2))
    for name in ['design_outcomes_v0_5.npz','calendar.csv','sell_prices.csv']:
        dest=a.output/name
        if not dest.exists():dest.symlink_to((a.input/name).resolve())
    panel=DesignPanel.load(a.input/'design_outcomes_v0_5.npz',a.input/'calendar.csv',a.input/'sell_prices.csv')
    builder=FeatureBuilder(panel);model=lgb.Booster(model_file=str(a.points/'observed_l1_s20260906.txt'))
    for block in ['selection_base','risk_train','calibration_a','calibration_b','shadow']:
        path=a.output/(block+'_v0_5.npz')
        if path.exists() and zipfile.is_zipfile(path):continue
        if path.exists():path.replace(path.with_suffix('.corrupt'))
        with np.load(a.input/(block+'_v0_5.npz')) as z:v={k:z[k] for k in z.files if k!='risk_features'}
        days=v['target_days'];plain=np.empty(v['truth'].shape,np.float32)
        for st in range(0,len(days),7):
            batch=days[st:st+7];ss=np.repeat(np.arange(panel.n_series),len(batch));dd=np.tile(batch,panel.n_series)
            x=builder.make_features(ss,dd,include_censor=False)
            plain[:,st:st+len(batch)]=np.maximum(model.predict(x,num_threads=4),0).reshape(panel.n_series,-1)
        v['baseline']=plain
        v['proposal']=plain+strength*v['censor_probability']**power*np.maximum(v['censored_poisson_em']-plain,0)
        temporary=path.with_suffix('.npz.tmp')
        with temporary.open('wb') as f:
            np.savez_compressed(f,**v);f.flush();os.fsync(f.fileno())
        if not zipfile.is_zipfile(temporary):raise RuntimeError('Incomplete cache write')
        temporary.replace(path);print('Strong forecast block',block,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ['input','output','points','adapter']:ap.add_argument('--'+k,type=Path,required=True)
    main(ap.parse_args())
