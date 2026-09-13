"""Independent arithmetic, chronology, and manuscript-claim checks."""
from pathlib import Path
import json,hashlib,re
import numpy as np
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'evidence'/'risk_calibration';checks=[]
def load(p): return json.load(open(p))
def ok(name,cond):
 assert cond,name;checks.append(name)
def dt(s): return np.datetime64(s.replace('+00:00',''))

primary=load(E/'primary_policy'/'RESULTS.json');rep=load(E/'seed_replication'/'RESULTS.json')
rows=[{'seed':20260906,**r} for r in primary['rows']]+rep['rows']
ok('nine_seed_block_rows',len(rows)==9)
ok('all_nine_improve',all(r['risk']['wape']<r['base']['wape'] for r in rows))
later=[r for r in rows if r['block']=='shadow'];bm=np.mean([r['base']['wape'] for r in later]);rm=np.mean([r['risk']['wape'] for r in later])
ok('later_base_mean',abs(bm-.6714752403900258)<1e-14);ok('later_risk_mean',abs(rm-.668129824026981)<1e-14)
ok('later_relative_gain',abs((1-rm/bm)-.004982188711979418)<1e-14)
for block in ['calibration_a','calibration_b','shadow']:
 ok(f'three_seeds_{block}',len([r for r in rows if r['block']==block])==3)

z=np.load(E/'primary_policy'/'shadow_item_stats.npz');diff=z['risk_error']-z['base_error'];estimate=diff.sum()/z['mass'].sum()
ok('primary_item_delta',abs(estimate-primary['rows'][-1]['paired_item_delta']['estimate'])<1e-14)
rng=np.random.default_rng(20260906);w=rng.multinomial(len(z['items']),np.full(len(z['items']),1/len(z['items'])),size=4000);ci=np.quantile((w@diff)/(w@z['mass']),[.025,.975])
ok('primary_item_interval',np.allclose(ci,primary['rows'][-1]['paired_item_delta']['ci95'],rtol=0,atol=1e-14));ok('primary_interval_below_zero',ci[1]<0)

a=load(E/'primary_analysis'/'RESULTS.json')['rows'][-1]
ok('all_horizons_improve',all(a['horizons'][str(k)]['risk']['wape']<a['horizons'][str(k)]['base']['wape'] for k in range(1,8)))
ok('item_counts',a['items']['improved']==1257 and a['items']['tied']==45 and a['items']['count']==1829)
ok('hobbies_negative_control',a['categories']['HOBBIES']['risk']['wape']>a['categories']['HOBBIES']['base']['wape'])
ok('strict_improves_other_worsens',a['strata']['strict']['risk']['wape']<a['strata']['strict']['base']['wape'] and a['strata']['other']['risk']['wape']>a['strata']['other']['base']['wape'])

u=load(E/'uci_transfer'/'RESULTS.json');ok('uci_test_not_accessed',load(E/'uci_transfer'/'PROTOCOL.json')['uci_test_accesses']==0)
ok('uci_forward_order',max(u['fit_days'])<min(u['evaluation_days']));ok('uci_improves',u['risk']['wape']<u['baseline']['wape']);ok('uci_interval_below_zero',u['paired_product_ci95'][1]<0)

ab=load(E/'attribution_controls'/'RESULTS.json')['corrected_score_ablation']['scores'];ok('learned_risk_best_matched_score',ab['learned_hit_risk']['later']['wape']<min(ab[k]['later']['wape'] for k in ab if k!='learned_hit_risk'))
d=load(E/'dual_risk'/'RESULTS.json')['rows'][-1];ok('retrospective_later_hit_order',primary['rows'][-1]['risk']['wape']<d['dual_risk']['wape'])

pp=load(E/'primary_policy'/'PROTOCOL.json');pf=load(E/'primary_policy'/'POLICY_FREEZE.json');ok('primary_protocol_before_policy',dt(pp['created_utc'])<dt(pf['created_utc']))
rp=load(E/'seed_replication'/'PROTOCOL.json');ok('replication_protocol_precedes_fit',all(dt(rp['created_utc'])<dt(load(E/'seed_replication'/f'seed_{s}'/'FIT_START.json')['created_utc']) for s in [20260907,20260908]))

fresh=load(ROOT/'evidence'/'freshretail'/'frozen_eval'/'RESULTS.json');ok('fresh_frozen_improves',fresh['primary_fully_available']['risk_conditioned']['wape']<fresh['primary_fully_available']['base']['wape']);ok('fresh_interval_below_zero',fresh['primary_fully_available']['paired_series_bootstrap']['ci95'][1]<0);ok('fresh_no_post_decode_learning',fresh['fit_after_eval_successful_decode']==0 and fresh['selection_after_eval_successful_decode']==0)

tex=(ROOT/'paper'/'main.tex').read_text()+(ROOT/'paper'/'risk_main_table.tex').read_text();ok('manuscript_exact_main_values','0.67148' in tex and '0.66813' in tex);ok('manuscript_caveats','retrospective' in tex and 'fully available' in tex)
log=(ROOT/'paper'/'main.log').read_text(errors='ignore');ok('no_undefined_references','undefined references' not in log.lower() and 'undefined citations' not in log.lower())
text='\n'.join(p.read_text(errors='ignore') for p in [ROOT/'paper'/'main.tex',ROOT/'paper'/'appendix_risk.tex',ROOT/'README.md'])
for name,pat in [
 ('personal_home_path',r'/home/[^/\s]+/'),
 ('personal_users_path',r'/Users/[^/\s]+/'),
 ('drive_mount_path',r'drive/MyDrive/'),
 ('named_author',r'\\author\{(?!Anonymous authors\})'),
 ('institution_line',r'\\affiliation\{'),
 ('author_email',r'\\email\{'),
]:
 ok('anonymity_'+name,re.search(pat,text,re.I) is None)

out={'status':'PASS','checks':len(checks),'check_names':checks,'later_means':{'base':bm,'risk':rm},'primary_ci95':ci.tolist(),'fresh_ci95':fresh['primary_fully_available']['paired_series_bootstrap']['ci95'],'uci_audit_ci95':u['paired_product_ci95']}
(ROOT/'verification'/'RISK_CALIBRATION_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
