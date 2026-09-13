from pathlib import Path
import json,hashlib,urllib.request,datetime
p=Path('additional_experiments/yeast');p.mkdir(exist_ok=False)
proto={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset':'Mulan Yeast, official train/test ARFF split','source':'https://github.com/tsoumakas/mulan/tree/master/data/multi-label/yeast','seed':20260909,'classifier':'independent StandardScaler + LogisticRegression(C=1,max_iter=2000), fixed probability threshold 0.5; constant training labels use constant prediction','train_calibration':'seeded permutation of official training set, first 75 percent for fitting, remaining 25 percent for calibration','conditional_loss_estimate':'sum over positive predictions of 1 minus fitted probability; no evaluation-label information','weight':'observed number of positive predictions of fixed classifier','cap':'0.85 times full-coverage calibration false-positive ratio; sensitivity multipliers 0.75,0.95,1.05 are secondary','floor':0.35,'methods':['conditional_error','row_excess','exposure_ratio'],'selection':'largest whole-tie threshold meeting cap and floor in calibration; reject all if absent','bootstrap':'2000 paired independent example resamples, seed 20260909; descriptive marginal percentile intervals conditional on fit','claims':'one fixed dataset/protocol; include cap failures and absent reversals; no retries based on test results','test_access':'parse official test labels only after classifier and all thresholds are saved'}
(p/'PROTOCOL.json').write_text(json.dumps(proto,indent=2)+'\n')
for name,sha in [('yeast-train.arff','ccd23bd6268ab9ebbe0ff0facf4bc27832beb7cc'),('yeast-test.arff','cec56277507d63f2d0ada833d9c6022237347dbc')]:
 u='https://raw.githubusercontent.com/tsoumakas/mulan/master/data/multi-label/yeast/'+name
 with urllib.request.urlopen(u,timeout=30) as f:d=f.read()
 actual=hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest()
 assert actual==sha,(name,actual)
 (p/name).write_bytes(d);print(name,len(d),flush=True)
