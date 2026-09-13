"""Implementation-only completion after an overstrict empty-week assertion.
Uses the original frozen A policies, fitted models and saved B/E arrays.
No fitting, category/window change, policy reselection, or outcome tuning.
"""
from pathlib import Path
import json, argparse
import numpy as np
import run_external_v12 as base
from policy_audit import METHODS, scores, mask, metrics

def analyse(arr,plans,cap,T,name,out):
    L=np.abs(arr['y']-arr['f']);W=arr['y'];ww=arr['week'];lo,hi=base.WINDOWS[name];n=hi-lo+1
    counts=base.block_counts(n,20000,2026091201 if name=='B' else 2026091202)
    total=np.column_stack([np.bincount(ww-lo,minlength=n),np.bincount(ww-lo,weights=W,minlength=n)])
    # Keep the original calendar grid, including zero-observation weeks.
    assert np.all((counts@total)>0)
    saved={'total':total,'counts':counts};result={}
    for m in METHODS:
        a=mask(scores(arr['e'],arr['w'],cap,m,T),plans[m]['threshold'])
        stat=base.weekly_stats(L,W,a,ww,lo,n);saved[m]=stat
        draw=counts@stat;nonzero=draw[:,1]>0
        upper=float(np.quantile(draw[:,2]-cap*draw[:,1],.99,method='linear'))
        risk_upper=float(np.quantile(draw[nonzero,2]/draw[nonzero,1],.99)) if nonzero.all() else None
        result[m]=dict(**metrics(L,W,a),signed_excess_U5=upper,risk_U5=risk_upper,
            passed=bool(plans[m]['threshold'] is not None and nonzero.all() and upper<=0))
    if name=='B':
        assert result==json.loads((out/'B_SCREEN.json').read_text()),'B result must be unchanged'
        with np.load(out/'B_WEEKLY_STATISTICS.npz') as old:
            for k,v in saved.items():assert np.array_equal(old[k],v),k
    else:np.savez_compressed(out/f'{name}_WEEKLY_STATISTICS.npz',**saved)
    return result,counts,total,saved

def main(out,data):
    proto=base.verify_protocol(out)
    assert not (out/'RESULTS.json').exists() and not (out/'RESUME_START.json').exists()
    am=out/'IMPLEMENTATION_AMENDMENT.json'
    assert am.exists()
    amendment=json.loads(am.read_text())
    assert base.sha(__file__)==amendment['resume_code_sha256']
    for n,h in amendment['frozen_inputs'].items():assert base.sha(out/n)==h,n
    base.write(out/'RESUME_START.json',dict(utc=base.now(),amendment_sha256=base.sha(am),fit_calls=0))
    frozen=json.loads((out/'FROZEN_A.json').read_text());r=frozen['report_cap'];T=frozen['T'];plans=frozen['plans']
    for k,h in frozen['model_sha256'].items():assert base.sha(out/f'MODEL_{k}.txt')==h
    with np.load(out/'B_ARRAYS.npz') as a:B={k:a[k] for k in a.files}
    br,*_=analyse(B,plans,r,T,'B',out)
    with np.load(out/'E_ARRAYS.npz') as a:E={k:a[k] for k in a.files}
    er,counts,total,saved=analyse(E,plans,r,T,'E',out)
    selected=frozen['selected'];c,d=selected['c'],selected['d'];contrast=None;success=False
    if c is not None and d is not None:
        delta=(counts@(saved[d]-saved[c]))[:,:2]/(counts@total)
        ci=np.quantile(delta,[.025,.975],axis=0).T
        contrast=dict(dc=er[d]['c']-er[c]['c'],dd=er[d]['d']-er[c]['d'],ci95=ci.tolist(),
                      both_pass_B=bool(br[c]['passed'] and br[d]['passed']))
        success=bool(contrast['both_pass_B'] and ci[0,1]<0 and ci[1,0]>0)
    Y,store,upc,cohort,info=base.panel_from_raw(data)
    with np.load(out/'TRAINING_ROWS.npz') as t:assert np.array_equal(t['cohort'],cohort)
    info['window_rows']={n:len(base.window_rows(Y,n)[0]) for n in base.WINDOWS}
    info['evaluation_empty_calendar_weeks']=[370,371,400]
    result=dict(dataset=proto['dataset'],cohort=info,cap=r,design_cap=frozen['design_cap'],selected=selected,
                B=br,E=er,contrast=contrast,primary_success=success,fit_calls=3,
                outcome='positive_prespecified_replication' if success else 'prespecified_joint_criterion_not_met',
                interpretation='Pre-opening frozen historical external analysis; approximate risk summaries, no population guarantee',
                resume='One implementation-only completion with unchanged fits, A policies, windows, B results and success criterion; calendar-empty weeks retained')
    base.write(out/'RESULTS.json',result)
    base.write(out/'COMPLETE.json',dict(utc=base.now(),fit_calls=3,analysis_openings=1,implementation_resumes=1,
        files_sha256={p.name:base.sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    a=p.parse_args();main(a.output,a.data)
