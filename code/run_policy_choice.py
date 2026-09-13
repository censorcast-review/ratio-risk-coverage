"""Replay the declared retrospective finite-menu and split-screen audits.

Run after restore_large_assets.py. Writes only to a new output directory.
The supplied protocol is copied unchanged; source hashes and frozen choices
precede evaluation. This ordering does not undo prior evaluation of the data.
"""
from pathlib import Path
import argparse, datetime, hashlib, json
import numpy as np
from policy_audit import (METHODS, scores, mask, metrics, largest_threshold,
                         choose, sufficient_statistics, bootstrap_counts,
                         paired_intervals)

ROOT = Path(__file__).resolve().parents[1]
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')


def main(out):
    protocol = read(ROOT/'evidence/policy_choice/PROTOCOL.json')
    out.mkdir(parents=True, exist_ok=False)
    write(out/'PROTOCOL.json', protocol)
    write(out/'START.json', {'utc': now(), 'source_sha256': {
        str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT/'code/policy_audit.py']}})
    old = ROOT/'evidence/accuracy_selection'
    original = read(old/'FROZEN_SELECTORS.json'); T = original['T_training']
    r = protocol['m5_primary_cap']; floor = protocol['row_floor']
    meta = np.load(ROOT/'legacy/results/objectives/cache_design/metadata.npz')
    items, item_index = np.unique(meta['item_id'], return_inverse=True)
    ni = len(items); counts = bootstrap_counts(ni, protocol['inference']['seed'])
    frozen = {'utc': None, 'scope': protocol['scope'], 'inputs': {}, 'menus': [], 'screens': []}
    screening_statistics = {}
    for model in protocol['predictors']:
        path = ROOT/f'evidence/additional/sensitivity/{model}_CALIBRATION.npz'
        frozen['inputs'][str(path.relative_to(ROOT))] = sha(path)
        with np.load(path) as z:
            e, w, L, W, blocks = z['e'], z['mu'], abs(z['y']-z['f']), z['y'], z['blocks']
        for cap in protocol['m5_caps']:
            menu = {}
            for method in METHODS:
                s = scores(e, w, cap, method, T)
                t, cal = largest_threshold(s, L, W, blocks, cap, floor)
                menu[method] = {'threshold': t, 'calibration': cal}
            frozen['menus'].append(dict(dataset='M5', model=model, cap=cap, T=T,
                policies=menu, selected={o: choose(menu, o) for o in ('c','d')}))
        A, B = blocks == 0, blocks == 1
        unitB = np.repeat(item_index, int(B.sum()/len(item_index)))
        assert len(unitB) == B.sum()
        menu, eligibility = {}, {}
        for method in METHODS:
            s = scores(e, w, r, method, T)
            t, cal = largest_threshold(s[A], L[A], W[A], np.zeros(A.sum(), int), .95*r, floor)
            a = mask(s[B], t)
            stat = sufficient_statistics(L[B], W[B], a, unitB, ni)
            screening_statistics[model+'__'+method] = stat
            # Per-item sums are the cluster observations, so this tests the
            # sign of pooled excess. It is a bootstrap approximation, not LTT.
            excess = stat[:,2]-r*stat[:,1]
            upper = float(np.quantile(counts @ excess / ni, 1-.05/15))
            ok = t is not None and stat[:,1].sum() > 0 and upper <= 0
            eligibility[method] = ok
            menu[method] = dict(threshold=t, calibration=cal,
                audit_B=metrics(L[B], W[B], a), mean_excess_B=float(excess.mean()),
                excess_upper_B=upper, passes_screen=ok)
        frozen['screens'].append(dict(dataset='M5', model=model, cap=r, T=T,
            policies=menu, selected={o: choose(menu, o, eligibility) for o in ('c','d')}))
        print('DESIGNED',model,flush=True)
    path = ROOT/'evidence/nonretail/bike/CALIBRATION_default.npz'
    frozen['inputs'][str(path.relative_to(ROOT))] = sha(path)
    with np.load(path) as z: L, W, e, w = [z[k] for k in ('L','W','e','w')]
    T = float(w.mean()); full = float(L.sum()/W.sum())
    for factor in protocol['bike_factors']:
        cap = factor*full; menu = {}
        for method in METHODS:
            t, cal = largest_threshold(scores(e,w,cap,method,T),L,W,np.zeros(len(W),int),.95*cap,floor)
            menu[method] = dict(threshold=t, calibration=cal)
        frozen['menus'].append(dict(dataset='Bike',model='bike',factor=factor,cap=cap,T=T,
            policies=menu,selected={o:choose(menu,o) for o in ('c','d')}))
    frozen['utc'] = now(); write(out/'FROZEN.json',frozen)
    np.savez_compressed(out/'SCREEN_STATISTICS.npz',items=items,**screening_statistics)
    write(out/'EVALUATION_STARTED.json',{'utc':now(),'freeze_sha256':sha(out/'FROZEN.json'),
        'new_external_openings':0,'note':'All targets were already evaluated in prior analyses.'})
    result = {'scope':protocol['scope'],'menus':[],'screens':[]}; saved_stats={}; units_map={}
    with np.load(old/'evaluation_forecasts.npz') as z:
        y=z['y'].ravel(); n_days=len(z['days']); predictions={k:z[k].ravel() for k in protocol['predictors']}
    units=np.repeat(item_index,n_days)
    total=np.column_stack([np.bincount(units),np.bincount(units,weights=y)])
    oldstat=np.load(old/'ITEM_STATISTICS.npz')
    assert np.array_equal(items,oldstat['items']) and np.array_equal(total,oldstat['total'])
    saved_stats['M5__total']=total
    for model in protocol['predictors']+['bike']:
        if model=='bike':
            with np.load(ROOT/'evidence/nonretail/bike/TEST_default.npz') as z:
                L,W,e,w,units=[z[k] for k in ('L','W','e','w','units')]
            ni=int(units.max())+1; counts=bootstrap_counts(ni,protocol['inference']['seed'])
            total=np.column_stack([np.bincount(units),np.bincount(units,weights=W)])
            saved_stats['Bike__total']=total
        else:
            with np.load(old/f'{model}_EVAL_SELECTORS.npz') as z: e,w=z['e'],z['mu']
            L,W=abs(y-predictions[model]),y
        for section in ['menus','screens']:
            for plan in (p for p in frozen[section] if p['model']==model):
                cap=plan['cap']; menu={}; stats={}; masks={}
                for method,p in plan['policies'].items():
                    a=mask(scores(e,w,cap,method,plan['T']),p['threshold'])
                    m=metrics(L,W,a); stat=sufficient_statistics(L,W,a,units,ni)
                    stats[method]=stat; masks[method]=a
                    b=counts @ stat
                    if m['weight']>0:
                        risk=np.divide(b[:,2],b[:,1],out=np.full(len(b),np.nan),where=b[:,1]>0)
                        m['risk_ci95']=np.nanquantile(risk,[.025,.975]).tolist()
                    else: m['risk_ci95']=None
                    menu[method]=m
                rec={**plan,'test':menu}
                a,b=plan['selected']['c'],plan['selected']['d']
                if a is not None and b is not None:
                    rec['contrast']={'dc':menu[b]['c']-menu[a]['c'],
                        'dd':menu[b]['d']-menu[a]['d'],
                        **paired_intervals(stats[a],stats[b],total,counts)}
                    union=masks[a]|masks[b]
                    rec['budget']={k:metrics(L,W,m) for k,m in
                        [('common',masks[a]&masks[b]),('case_only',masks[a]&~masks[b]),
                         ('exposure_only',~masks[a]&masks[b]),('union',union)]}
                    for val in rec['budget'].values(): val['excess']=val['loss']-cap*val['weight']
                else: rec['contrast']=None
                # Store sufficient statistics for every declared primary menu
                # and every separately screened candidate, not just winners.
                primary=(model=='bike' and plan['factor']==.85) or (model!='bike' and cap==r)
                if primary or section=='screens':
                    for method,stat in stats.items(): saved_stats[f'{section}__{model}__{method}']=stat
                result[section].append(rec)
        print('EVALUATED',model,flush=True)
    np.savez_compressed(out/'TEST_STATISTICS.npz',**saved_stats)
    result['completed_utc']=now();write(out/'RESULTS.json',result)
    write(out/'COMPLETE.json',{'utc':now(),'fit_calls':0,'external_openings':0,
        'files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
    for entry in result['menus']:
        if (entry['model']=='bike' and entry['factor']==.85) or (entry['model']!='bike' and entry['cap']==r):
            print('PRIMARY',entry['model'],entry['selected'],entry['contrast'],flush=True)
    for entry in result['screens']: print('SCREEN',entry['model'],entry['selected'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
