"""Independent read-only raw/model audit. No fit(), no policy reselection.

The frozen production parser/runner are not imported. The only shared low-level
component is the system libarchive decompressor. This checker separately parses
ARFF into a dense matrix and reconstructs sigmoid probabilities from coefficients.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, ctypes as C, ctypes.util, hashlib, io, json, pathlib, re, sys
from datetime import datetime, timezone
from xml.dom.minidom import parseString
import joblib
import numpy as np
from scipy.special import expit
from scipy import sparse

ROOT=pathlib.Path(__file__).resolve().parents[1]
CHECKS=[]
def check(name, passed, detail=None):
    CHECKS.append({'check':name,'passed':bool(passed),'detail':detail})
    if not passed: raise AssertionError(name)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as s:
        for b in iter(lambda:s.read(2**20), b''): h.update(b)
    return h.hexdigest()
def jread(p): return json.loads(p.read_text())
def rar_contents(path):
    """Independent in-memory archive walk; no production adapter calls."""
    L=C.CDLL(ctypes.util.find_library('archive'))
    funcs={
      'archive_read_new':(C.c_void_p,[]),
      'archive_read_support_filter_all':(C.c_int,[C.c_void_p]),
      'archive_read_support_format_all':(C.c_int,[C.c_void_p]),
      'archive_read_open_filename':(C.c_int,[C.c_void_p,C.c_char_p,C.c_size_t]),
      'archive_read_next_header':(C.c_int,[C.c_void_p,C.POINTER(C.c_void_p)]),
      'archive_entry_pathname':(C.c_char_p,[C.c_void_p]),
      'archive_entry_size':(C.c_int64,[C.c_void_p]),
      'archive_read_data':(C.c_ssize_t,[C.c_void_p,C.c_void_p,C.c_size_t]),
      'archive_read_data_skip':(C.c_int,[C.c_void_p]),
      'archive_read_free':(C.c_int,[C.c_void_p])}
    for k,(rest,args) in funcs.items():
        getattr(L,k).restype=rest;getattr(L,k).argtypes=args
    a=L.archive_read_new(); entries=[];files={}
    try:
        assert L.archive_read_support_filter_all(a)>=0
        assert L.archive_read_support_format_all(a)>=0
        assert L.archive_read_open_filename(a,str(path).encode(),16384)>=0
        while True:
            e=C.c_void_p();status=L.archive_read_next_header(a,C.byref(e))
            if status==1:break
            assert status>=0
            name=L.archive_entry_pathname(e).decode();size=L.archive_entry_size(e)
            entries.append({'name':name,'size':size})
            use=name.lower().endswith('.xml') or (name.lower().endswith('.arff') and ('train' in name.lower() or 'test' in name.lower()))
            if not use:
                assert L.archive_read_data_skip(a)>=0;continue
            buf=C.create_string_buffer(262144);output=io.BytesIO()
            while True:
                n=L.archive_read_data(a,buf,len(buf))
                assert n>=0
                if n==0:break
                output.write(buf.raw[:n])
            data=output.getvalue();assert len(data)==size
            files[name]=data
    finally:L.archive_read_free(a)
    return files,entries

def parse_separately(data,xml_names):
    """Build all columns at once, then select XML labels by attribute name.

    Production code builds CSR features and Y directly row by row; this audit uses
    dense numeric assignment, an independent header regex, and minidom XML order.
    """
    text=data.decode('utf-8-sig')
    header,body=re.split(r'(?im)^\s*@data\s*$',text,maxsplit=1)
    attrs=[]
    for token in re.findall(r'(?im)^\s*@attribute\s+(.+)$',header):
        m=re.fullmatch(r"(?:'([^']*)'|\"([^\"]*)\"|(\S+))\s+(.*)",token.strip())
        assert m is not None
        attrs.append(next(x for x in m.groups()[:3] if x is not None))
    lines=[l.strip() for l in body.splitlines() if l.strip() and not l.lstrip().startswith('%')]
    array=np.zeros((len(lines),len(attrs)),dtype=np.float64)
    number=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    for row,l in enumerate(lines):
        if l.startswith('{'):
            assert l.endswith('}')
            entries=re.findall(r'(\d+)\s+('+number+r')(?=\s*[,}])',l)
            assert len(entries)==(l.count(',')+1 if l[1:-1].strip() else 0)
            cols=np.array([int(t[0]) for t in entries]); vals=np.array([float(t[1]) for t in entries])
            if len(cols):array[row,cols]=vals
        else:
            nums=np.fromstring(l,sep=',');assert nums.size==len(attrs)
            array[row]=nums
    label_columns=[attrs.index(name) for name in xml_names]
    features=[i for i in range(len(attrs)) if i not in set(label_columns)]
    return array[:,features],array[:,label_columns].astype(np.uint8),attrs

def expected_splits(dataset,nt,nv):
    def order(role,n):
        return np.asarray(sorted(range(n),key=lambda i:(
          int(hashlib.sha256(('v13-split-20260913|%s|%s|%d'%(dataset,role,i)).encode()).hexdigest()[:16],16),i)))
    tr=order('official_train',nt);te=order('official_test',nv);cut=int(np.floor(.7*nt));t=nv//3
    return {'T':tr[:cut],'H':tr[cut:],'A':te[:t],'B':te[t:2*t],'E':te[2*t:]}

class FixedClassifier:
    """State-only stand-in for the trusted self-generated __main__ pickle class.

    No original runner or fitting methods are imported/executed. sklearn state is
    loaded only after its file SHA matches A_FIXED and COMPLETE receipts.
    """
    pass

def classify_manual(model,X):
    # CSR column multiplication is also the arithmetic used by frozen scaling.
    Z=sparse.csr_matrix(X).multiply(1.0/model.scaler.scale_).tocsr()
    prob=np.empty((X.shape[0],len(model.models)))
    for k,m in enumerate(model.models):
        if isinstance(m,float):prob[:,k]=m
        else:
            assert np.array_equal(m.classes_,[0,1])
            prob[:,k]=expit(np.asarray(Z@m.coef_.T).ravel()+float(m.intercept_[0]))
    positive=prob>=.5
    empty=np.flatnonzero(np.sum(positive,axis=1)==0)
    positive[empty,np.argmax(prob[empty],axis=1)]=True
    w=np.count_nonzero(positive,axis=1)
    Xsel=np.column_stack([Z.toarray(),np.log1p(w)])
    return positive,w,len(empty),Xsel

def audit(dataset):
    global CHECKS;CHECKS=[]
    base=ROOT/'evidence'/dataset
    check('completed_source_exists',(base/'COMPLETE.json').exists())
    complete=jread(base/'COMPLETE.json');fixed=jread(base/'A_FIXED.json');result=jread(base/'RESULT.json')
    protocol=jread(ROOT/'evidence/PROTOCOL.json');freeze=jread(ROOT/'evidence/FREEZE_RECEIPT.json')
    download=jread(ROOT/'evidence'/('DOWNLOAD_'+dataset+'.json'))
    initial={n:sha(base/n) for n in complete['files']}
    for n,h in complete['files'].items():check('complete_hash_'+n,initial[n]==h)
    for n,h in protocol['code_sha256'].items():check('frozen_code_'+n,sha(ROOT/n)==h)
    check('protocol_hash',sha(ROOT/'evidence/PROTOCOL.json')==freeze['protocol_sha256']==download['protocol_sha256'])
    check('freeze_before_download',datetime.fromisoformat(freeze['utc'])<datetime.fromisoformat(download['started_utc']))
    for n,h in fixed['model_sha256'].items():check('A_fixed_model_hash_'+n,sha(base/n)==h)
    raw=ROOT/'raw'/(dataset+'.rar');check('raw_source_hash',sha(raw)==download['sha256'])
    files,entry_list=rar_contents(raw)
    xml=[v for k,v in files.items() if k.endswith('.xml')];assert len(xml)==1
    names=[x.getAttribute('name') for x in parseString(xml[0]).getElementsByTagName('label')]
    raw_arrays={}
    for split in ['train','test']:
        found=[(k,v) for k,v in files.items() if k.lower().endswith('.arff') and split in k.lower()];assert len(found)==1
        name,data=found[0];X,Y,attrs=parse_separately(data,names);raw_arrays[split]=(X,Y)
        check(split+'_member_hash',hashlib.sha256(data).hexdigest()==jread(base/'DATA.json')['metadata']['member_sha256'][name])
        check(split+'_all_labels_binary',np.isin(Y,[0,1]).all())
        check(split+'_all_features_finite',np.isfinite(X).all())
    Xtr,Ytr=raw_arrays['train'];Xte,Yte=raw_arrays['test'];spec=protocol['datasets'][dataset]
    check('standard_total',len(Xtr)+len(Xte)==spec['expected_n'])
    check('standard_dimensions',Xtr.shape[1]==spec['expected_features'] and Ytr.shape[1]==spec['expected_labels'])
    if dataset=='mediamill':check('standard_train_test_counts',len(Xtr)==30993 and len(Xte)==12914)
    idx=expected_splits(dataset,len(Xtr),len(Xte))
    with np.load(base/'SPLITS.npz',allow_pickle=False) as saved:
        for role,indices in idx.items():check('raw_hash_split_'+role,np.array_equal(indices,saved[role]))
    check('train_roles_disjoint',not(set(idx['T'])&set(idx['H'])))
    check('test_roles_disjoint',all(not(set(idx[a])&set(idx[b])) for a,b in [('A','B'),('A','E'),('B','E')]))
    check('train_roles_partition',np.array_equal(np.sort(np.r_[idx['T'],idx['H']]),np.arange(len(Xtr))))
    check('test_roles_partition',np.array_equal(np.sort(np.r_[idx['A'],idx['B'],idx['E']]),np.arange(len(Xte))))
    # Scope is self-generated, hash-bound pickle files from this experiment only.
    model=joblib.load(base/'classifier.joblib')
    check('classifier_state_type',type(model).__name__=='FixedClassifier')
    T=Xtr[idx['T']];mean=T.mean(0);var=T.var(0)
    check('scaler_fit_count_is_T',np.all(model.scaler.n_samples_seen_==len(T)))
    check('scaler_mean_from_T',np.allclose(model.scaler.mean_,mean,rtol=1e-10,atol=1e-11))
    check('scaler_variance_from_T',np.allclose(model.scaler.var_,var,rtol=1e-9,atol=1e-11))
    for j,m in enumerate(model.models):
        if isinstance(m,float):
            check('constant_label_T_'+str(j),np.unique(Ytr[idx['T'],j]).size==1 and m==(Ytr[idx['T'],j].sum()+1)/(len(T)+2))
        else:
            params=m.get_params();check('classifier_frozen_params_'+str(j),params['C']==1 and params['solver']=='liblinear' and params['max_iter']==1000 and params['random_state']==spec['seed'] and params['class_weight'] is None)
    outcome_report={};HX=None;HL=None
    for role in ['H','A','B','E']:
        X,Y=(Xtr,Ytr) if role=='H' else (Xte,Yte);rows=idx[role]
        pred,w,empty,Z=classify_manual(model,X[rows])
        # FP is predicted count less correctly predicted count, independently of production logical-and.
        fp=w-(pred*Y[rows]).sum(axis=1)
        check(role+'_exposure_positive',np.all(w>=1))
        check(role+'_FP_within_exposure',np.all((fp>=0)&(fp<=w)))
        check(role+'_top1_fallback',empty==result['top1_fallback_count'][role])
        if role!='H':
            with np.load(base/(role+'_ARRAYS.npz'),allow_pickle=False) as saved:
                check(role+'_all_raw_FP_match',np.array_equal(fp,saved['loss']))
                check(role+'_all_raw_w_match',np.array_equal(w,saved['exposure']))
                ties=np.array([int(hashlib.sha256(f'v13-ties-20260913|{dataset}|official_test|{int(i)}'.encode()).hexdigest()[:16],16) for i in rows],dtype=np.uint64)
                check(role+'_all_label_independent_ties',np.array_equal(ties,saved['tie']))
            if role=='A':check('A_only_cap',np.isclose(fixed['r'],.95*(float(fp.sum())/float(w.sum())),rtol=0,atol=3*np.finfo(float).eps),{'recorded':fixed['r'],'independent':.95*(float(fp.sum())/float(w.sum()))})
        else:HX,HL=Z,fp
        outcome_report[role]={'n':len(rows),'sum_FP':int(fp.sum()),'sum_w':int(w.sum()),'top1_fallback':empty,'raw_FP_sha256':hashlib.sha256(fp.astype('<i8').tobytes()).hexdigest(),'raw_w_sha256':hashlib.sha256(w.astype('<i8').tobytes()).hexdigest()}
    head=joblib.load(base/'error_head.joblib')
    check('head_tree_count',len(head.estimators_)==220)
    # Tree node sample counts and conditional means are recomputed from H-only rows.
    for t,tree in enumerate(head.estimators_):
        path=tree.decision_path(HX)
        counts=np.asarray(path.sum(axis=0)).ravel()
        sums=np.asarray(path.T@HL).ravel()
        values=np.divide(sums,counts,out=np.zeros_like(sums,dtype=float),where=counts>0)
        check('H_only_tree_counts_'+str(t),np.array_equal(counts,tree.tree_.n_node_samples))
        check('H_only_tree_FP_means_'+str(t),np.allclose(values,tree.tree_.value[:,0,0],rtol=1e-11,atol=1e-11))
    for gate in ['GateCase','GateExposure']:
        with np.load(base/(gate+'.npz'),allow_pickle=False) as g:
            check(gate+'_normalizer_H_mean',np.allclose(g['mean'],HX.mean(0),rtol=1e-10,atol=1e-10))
            scale=HX.std(0);scale[scale<1e-12]=1
            check(gate+'_normalizer_H_scale',np.allclose(g['scale'],scale,rtol=1e-10,atol=1e-10))
    screen=jread(base/'B_SCREEN.json')
    check('A_then_B_then_E_receipts',datetime.fromisoformat(fixed['utc'])<datetime.fromisoformat(screen['utc'])<datetime.fromisoformat(result['utc']))
    check('A_fixed_bound_into_B',screen['A_fixed_sha256']==sha(base/'A_FIXED.json'))
    check('B_fixed_bound_into_E',result['B_screen_sha256']==sha(base/'B_SCREEN.json'))
    # Review frozen source call sites, without importing or executing its fit methods.
    source=(ROOT/'run_nonwape.py').read_text()
    required=[".fit(X_train[idx['T']], Y_train[idx['T']])","head.fit(HX, Hloss)","fit_gate(HX, Hloss, HW, r, utility", "HX, HY, HW, Hempty = classifier.features_and_predictions(X_train[idx['H']])", "r = .95*float(Aloss.sum()/AW.sum())"]
    for i,s in enumerate(required):check('frozen_fit_dataflow_'+str(i),s in source)
    check('source_B_after_A_receipt',source.index("write_json(out/'A_FIXED.json'")<source.index('BX, BY, BW, Bempty'))
    check('source_E_after_B_receipt',source.index("write_json(out/'B_SCREEN.json'")<source.index('EX, EY, EW, Eempty'))
    check('source_files_unchanged',all(sha(base/n)==h for n,h in initial.items()))
    metadata=jread(base/'DATA.json')['metadata']
    check('raw_attribute_schema_matches_record',attrs==metadata['attribute_names'])
    # Bibtex words named video/group are input vocabulary, not group identifiers.
    if dataset=='mediamill':
        feature_names=[a for a in attrs if a not in set(names)]
        group_note={'feature_names':feature_names,'additional_archive_members':[e['name'] for e in entry_list if not e['name'].lower().endswith(('.arff','.xml'))], 'conclusion':'Official numeric features and label XML; no separate video/shot grouping sidecar was present. Feature names are reported so this structural observation is inspectable; it does not prove row independence.'}
    else:group_note={'conclusion':'Text word indicators and tag labels; no grouping sidecar was present. Duplicate documents and dependence remain possible.'}
    report={'dataset':dataset,'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','checks':len(CHECKS),'details':CHECKS,'recomputed_outcomes':outcome_report,'archive_members':entry_list,'group_metadata':group_note,'audit_code_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(ROOT/'evidence/PROTOCOL.json'),'raw_sha256':sha(raw),'model_sha256':sha(base/'classifier.joblib'),'independence_scope':'Independent ARFF/XML parser, split hashing, manual logistic probabilities and FP=count minus true positives. Shared system libarchive decompressor and fitted sklearn tree traversal. No training, tuning, policy reselection, or scientific-file writes.', 'leakage_scope':'No split-index overlap; T-only scaler and constants, all H-only tree sufficient statistics and H gate normalizers agree. Frozen source and timestamp/hash receipts use T for predictor, H for selector fitting, A only cap/design; B/E labels not supplied to fitting. This is a computational/dataflow audit, not proof against unrecorded analyst access. Exact-feature duplicates across official roles are retained by protocol and limit independence.'}
    output=ROOT/'audit'/('RAW_MODEL_CHECK_'+dataset+'.json');output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['dataset','status','checks','recomputed_outcomes']},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['bibtex','mediamill'],required=True);args=parser.parse_args()
    try:audit(args.dataset)
    except Exception as exc:
        fail={'dataset':args.dataset,'utc':datetime.now(timezone.utc).isoformat(),'status':'FAIL','exception':repr(exc),'details':CHECKS,'audit_code_sha256':sha(pathlib.Path(__file__))}
        (ROOT/'audit'/('RAW_MODEL_CHECK_'+args.dataset+'_FAILURE.json')).write_text(json.dumps(fail,indent=2)+'\n');raise
