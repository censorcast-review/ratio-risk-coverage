"""External comparison entry point. Authorization and ledger precede data reads.

No command in this module grants authorization. The owner must explicitly
approve the final plan hash after reviewing the original failed-gate amendment.
"""
from __future__ import annotations
import argparse, datetime, json, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'code'))
from r2_io import sha,dump,write


def verify_plan(plan_path):
    plan=json.loads(Path(plan_path).read_text())
    if plan.get('schema_version')!='review5-external-1' or plan['status']!='FROZEN_PENDING_EXPLICIT_EXTERNAL_PURPOSE_AMENDMENT' or plan['policy_count']!=2:
        raise RuntimeError('Wrong frozen plan')
    for rel,h in plan['immutable_files'].items():
        if sha(ROOT/rel)!=h:raise RuntimeError('Frozen resource changed: '+rel)
    return plan


def validate_authorization(plan, plan_hash, approval):
    required={
      'status':'EXPLICIT_USER_APPROVAL_RECORDED',
      'freeze_sha256':plan_hash,
      'authorization_phrase':plan['authorization_phrase'],
      'approve_failed_guardian_external_purpose_amendment':True,
      'approved_external_use_count':1,
      'approved_policy_count':2,
      'certificate_issued':False,
    }
    for key,value in required.items():
        if type(approval.get(key)) is not type(value) or approval.get(key)!=value:
            raise PermissionError('External remains sealed: explicit approval missing or differs at '+key)
    if not isinstance(approval.get('user_approval_text'),str) or not approval['user_approval_text'].strip():
        raise PermissionError('External remains sealed: exact owner approval text is required')
    if not isinstance(approval.get('approved_utc'),str) or not approval['approved_utc'].strip():
        raise PermissionError('External remains sealed: approval timestamp is required')


def begin_opening(plan, plan_hash, approval, ledger_path, output):
    # No external input path is inspected before this validation succeeds.
    validate_authorization(plan,plan_hash,approval)
    ledger=json.loads(Path(ledger_path).read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    opened_path=output/'EXTERNAL_OPENED.json'
    if ledger.get('external_opened'):
        if ledger.get('review5_external_freeze_sha256')!=plan_hash or ledger.get('review5_external_use_count')!=1 or not opened_path.exists():
            raise PermissionError('External was already opened outside this identical frozen evaluation')
        opened=json.loads(opened_path.read_text())
        if opened.get('freeze_sha256')!=plan_hash:raise PermissionError('Opening receipt mismatch')
        return opened
    if ledger.get('fresh_guardian_use_count')!=1 or ledger.get('certificate_issued') is not False:
        raise PermissionError('Unexpected research ledger state')
    if opened_path.exists():
        opened=json.loads(opened_path.read_text())
        if opened.get('freeze_sha256')!=plan_hash:raise PermissionError('Different pending opening receipt')
    else:
        dump(output/'DATA_ACCESS_LEDGER_BEFORE_EXTERNAL.json',ledger)
        opened={'status':'AUTHORIZED_EXTERNAL_PREDICTION','freeze_sha256':plan_hash,
            'external_use_count':1,'original_external_gate_pass':False,'certificate_issued':False,
            'approved_action':plan['authorization_scope'],'approval':approval,
            'opened_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        dump(opened_path,opened)
    ledger.update(external_opened=True,review5_external_use_count=1,
        review5_external_freeze_sha256=plan_hash,review5_external_status='OPENED_FOR_FROZEN_TWO_POLICY_COMPARISON',
        review5_external_nominal_alpha_reserved=.05,certificate_issued=False)
    dump(Path(ledger_path),ledger)
    return opened


def load_external_arrays(plan,data_root,opened,plan_hash):
    # Defense in depth: array reads cannot be reached with a pending receipt.
    if opened.get('status')!='AUTHORIZED_EXTERNAL_PREDICTION' or opened.get('freeze_sha256')!=plan_hash:
        raise PermissionError('External arrays require the persisted authorized receipt')
    for rel,spec in plan['external_inputs'].items():
        path=Path(data_root)/rel
        if path.stat().st_size!=spec['size_bytes'] or sha(path)!=spec['sha256']:
            raise RuntimeError('Authorized input integrity mismatch: '+rel)
    import numpy as np
    def read(path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    context=read(Path(data_root)/'prepared/external_context_v0_5.npz')
    outcomes=read(Path(data_root)/'sealed/external_final_outcomes_v0_5.npz')
    if 'truth' in context:raise RuntimeError('Context unexpectedly contains true outcomes')
    if int(context['day_start'][0])!=1 or int(context['day_end'][0])!=1913:
        raise RuntimeError('Context date boundary mismatch')
    if int(outcomes['day_start'][0])!=1914 or int(outcomes['day_end'][0])!=1941:
        raise RuntimeError('Final outcome date boundary mismatch')
    for key in ['id','item_id','dept_id','cat_id','store_id','state_id']:
        if not np.array_equal(context[key],outcomes[key]):raise RuntimeError('Metadata mismatch '+key)
    items=sorted(set(context['item_id'].astype(str).tolist()))
    import hashlib
    actual=hashlib.sha256(('\n'.join(items)+'\n').encode()).hexdigest()
    if len(items)!=610 or len(context['id'])!=6100 or actual!=plan['item_set_sha256']:
        raise RuntimeError('Frozen external item partition mismatch')
    return context,outcomes


def main(args):
    plan=verify_plan(args.plan);plan_hash=sha(args.plan)
    from importlib.metadata import version, PackageNotFoundError
    from predict import POINT_RUNTIME
    runtime={}
    for name in POINT_RUNTIME:
        try:runtime[name]=version(name)
        except PackageNotFoundError:runtime[name]=None
    if args.preflight:
        print(json.dumps({'status':'PENDING_AUTHORIZATION_PREFLIGHT_PASS','freeze_sha256':plan_hash,
            'policy_count':2,'external_inputs_read':False,'required_owner_action':plan['authorization_scope'],
            'runtime_ready':runtime==POINT_RUNTIME,'runtime_found':runtime,'runtime_required':POINT_RUNTIME},indent=2))
        return
    # The missing-approval path exits before even checking the data root exists.
    if args.approval is None or not args.approval.exists():
        raise PermissionError('External remains sealed. The original protocol requires a passed guardian certificate; this plan needs explicit approval of its two-policy purpose amendment.')
    approval=json.loads(args.approval.read_text())
    validate_authorization(plan,plan_hash,approval)
    if runtime != POINT_RUNTIME:
        raise RuntimeError('External remains sealed: install requirements-external.txt in an isolated environment before opening. Runtime mismatch: '+json.dumps(runtime))
    if args.data_root is None or args.output is None:
        raise ValueError('Authorized execution requires --data-root and --output')
    ledger_path=args.data_root/'state/DATA_ACCESS_LEDGER_v0_5.json'
    # Lock the canonical access ledger across the whole fixed evaluation.
    # The lock file contains no outcome data and cannot grant permission.
    import fcntl
    ledger_lock=open(ledger_path.with_suffix('.external.lock'),'a')
    try:fcntl.flock(ledger_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        ledger_lock.close();raise RuntimeError('Another external operation holds the canonical ledger lock')
    opened=begin_opening(plan,plan_hash,approval,ledger_path,args.output)
    complete=args.output/'EXTERNAL_COMPARISON.json'
    if complete.exists():
        old=json.loads(complete.read_text())
        if old.get('freeze_sha256')!=plan_hash:raise RuntimeError('Completed result belongs to a different freeze')
        print('The identical completed external comparison is reused; no additional read or evaluation.');return
    context,outcomes=load_external_arrays(plan,args.data_root,opened,plan_hash)
    import numpy as np
    with np.load(ROOT/'results/review2/cache_design/metadata.npz',allow_pickle=False) as z:
        design_meta={k:z[k] for k in z.files}
    from predict import predict_external
    data,meta,scores,prediction_receipt=predict_external(context,outcomes,
        args.data_root/'source/calendar.csv',args.data_root/'source/sell_prices.csv',
        design_meta,opened,plan_hash,output_dir=args.output/'prediction_cache')
    from evaluate import evaluate
    report=evaluate(data,meta,scores,reps=plan['bootstrap']['reps'])
    report.update(status='FROZEN_EXTERNAL_TWO_POLICY_COMPARISON_COMPLETED',freeze_sha256=plan_hash,
        external_use_count=1,certificate_issued=False,prediction_receipt=prediction_receipt,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    dump(complete,report)
    ledger=json.loads(ledger_path.read_text())
    ledger.update(review5_external_status='COMPARISON_COMPLETED',review5_external_results_sha256=sha(complete),
        review5_external_nominal_alpha_used=.05,certificate_issued=False)
    dump(ledger_path,ledger)
    print(json.dumps({'status':report['status'],'primary_confirmation_pass':report['primary']['directional_confirmation_pass'],
        'joint_secondary_check_pass':report['joint_secondary_operating_check_pass'],'external_use_count':1,'certificate_issued':False},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',type=Path,default=HERE/'EXTERNAL_PLAN.json')
    ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--approval',type=Path)
    ap.add_argument('--data-root',type=Path)
    ap.add_argument('--output',type=Path)
    main(ap.parse_args())
