"""Export item aggregates for replay of the completed frozen external evaluation.

This postprocessing module reads only the persisted prediction cache AFTER
EXTERNAL_COMPARISON.json exists. It adds no policy, threshold, fit, or inference.
Actual item identifiers are omitted from the exported sufficient statistics.
"""
from pathlib import Path
import argparse, importlib.util, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from r2_io import sha,dump


def export(output):
    output=Path(output)
    report_path=output/'EXTERNAL_COMPARISON.json'
    report=json.loads(report_path.read_text())
    plan_path=ROOT/'code/review5_external/EXTERNAL_PLAN.json'
    plan=json.loads(plan_path.read_text())
    assert report['status']=='FROZEN_EXTERNAL_TWO_POLICY_COMPARISON_COMPLETED'
    assert report['freeze_sha256']==sha(plan_path)
    assert report['external_use_count']==1 and report['certificate_issued'] is False
    evaluator=ROOT/'code/review5_external/evaluate.py'
    assert sha(evaluator)==plan['immutable_files'][str(evaluator.relative_to(ROOT))]
    spec=importlib.util.spec_from_file_location('frozen_external_evaluator',evaluator)
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    cache=output/'prediction_cache'
    def load(name):
        with np.load(cache/name,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    data=load('external_predictions.npz');scores=load('external_head_scores.npz');meta=load('external_metadata.npz')
    y,p,days=data['truth'],data['proposal'],data['target_days']
    items=np.asarray(meta['item_id']).astype(str);categories=np.asarray(meta['cat_id']).astype(str)
    masks=ev.masks_from_scores(scores)
    records=[]
    for period,(start,end) in {**ev.PERIODS,'descriptive_full_28':[1914,1941]}.items():
        cols=(days>=start)&(days<=end)
        groups=['ALL'] if period=='descriptive_full_28' else ev.GROUPS
        for group in groups:
            rows=np.ones(len(items),dtype=bool) if group=='ALL' else categories==group
            n=len(np.unique(items[rows]))
            values={}
            for name in ev.THRESHOLDS:
                matrix=ev.components(y[rows][:,cols],p[rows][:,cols],masks[name][rows][:,cols],items[rows])
                assert matrix.shape==(n,5)
                values[name]=matrix.tolist()
            records.append({'period':period,'target_days':[start,end],'group':group,'item_clusters':n,
              'anonymous_item_index':list(range(n)),'policy_components':values})
    sources=[report_path,plan_path,evaluator,cache/'external_predictions.npz',cache/'external_head_scores.npz',
      cache/'external_metadata.npz',cache/'PREDICTION_RECEIPT.json']
    result={'schema_version':'external-item-statistics-1','status':'COMPLETED_FROZEN_COMPARISON_AGGREGATES',
      'freeze_sha256':sha(plan_path),'source_files_sha256':{str(path.relative_to(ROOT)):sha(path) for path in sources},
      'policy_names':list(ev.THRESHOLDS),'component_columns':['accepted_rows','total_rows','accepted_demand','accepted_absolute_error','total_demand'],
      'item_order':'Within each stratum, original sorted unique item order; IDs replaced by sequential indices, same order for both policies and all periods.',
      'actual_item_ids_included':False,'records':records,
      'scope':'Postprocessing of already saved predictions from the single completed frozen external comparison. No new fit, threshold selection, policy selection, or statistical endpoint.',
      'bootstrap_reps':plan['bootstrap']['reps'],'bootstrap_seed':plan['bootstrap']['seed'],
      'primary_endpoint_alpha':plan['primary']['one_sided_endpoint_alpha'],'secondary_endpoint_alpha':plan['secondary']['per_endpoint_alpha'],
      'descriptive_full_28_no_inference':True,'external_use_count':1,'certificate_issued':False}
    target=output/'EXTERNAL_ITEM_STATISTICS.json';dump(target,result)
    print(json.dumps({'file':str(target),'sha256':sha(target),'stratum_period_records':len(records),'policy_component_matrices':2*len(records),'actual_item_ids_included':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    export(parser.parse_args().output)
