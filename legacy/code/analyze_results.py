"""Descriptive paired item bootstrap and manuscript figures; design data only."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_selective_study import load,metric,frontier,R

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False))

def main(a):
    root=a.study;dest=root/'results/analysis';dest.mkdir(exist_ok=True)
    fig=root/'paper/figures';fig.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
    meta=load(root/'results/selective/cache/metadata.npz');items,codes=np.unique(meta['item_id'],return_inverse=True)
    rng=np.random.default_rng(20260906);weights=rng.multinomial(len(items),np.ones(len(items))/len(items),size=3000).astype(float)
    def sums(x):return np.bincount(codes,weights=x.sum(axis=1),minlength=len(items))
    sh=load(root/'inputs/m5/shadow_v0_5.npz');y=sh['truth'].astype(float);mass=sums(y);bootmass=weights@mass
    pp=json.loads((root/'results/point_baselines/POINT_RESULTS.json').read_text())['reports']
    ar=json.loads((root/'results/adapter_v2/ADAPTER_RESULTS.json').read_text())['records']
    rows=[]
    for method in sorted({r['method'] for r in pp}):
        rr=[r for r in pp if r['method']==method]
        vals=[r['metrics']['shadow']['wape'] for r in rr]
        rows.append(dict(method=method,seeds=3,shadow_wape_mean=float(np.mean(vals)),shadow_wape_sd=float(np.std(vals,ddof=1)),shadow_wape_all=vals))
    for method in sorted({r['method'] for r in ar}):
        rr=[r for r in ar if r['method']==method];vals=[r['shadow']['wape'] for r in rr]
        rows.append(dict(method=method,seeds=3,shadow_wape_mean=float(np.mean(vals)),shadow_wape_sd=float(np.std(vals,ddof=1)),shadow_wape_all=vals))
    pred={'legacy_raw':sh['baseline'],'legacy_censorcast':sh['proposal']}
    for method in ['plain','global_scale','category_scale','ungated_em_correction','censor_adapter']:
        pred[method]=load(root/f'results/adapter_v2/{method}_s20260906.npz')['shadow']
    pointdetail={}
    for name,p in pred.items():
        errs=sums(np.abs(y-p));boot=(weights@errs)/bootmass
        pointdetail[name]={'wape':float(errs.sum()/mass.sum()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
           'category_wape':{g:metric(y[meta['cat_id']==g],p[meta['cat_id']==g],p[meta['cat_id']==g])['wape'] for g in np.unique(meta['cat_id'])}}
    contrasts={}
    for better,base in [('censor_adapter','plain'),('censor_adapter','category_scale'),('censor_adapter','legacy_censorcast'),('plain','legacy_censorcast')]:
        ds=sums(np.abs(y-pred[better])-np.abs(y-pred[base]));boot=(weights@ds)/bootmass
        contrasts[better+'_minus_'+base]={'delta_wape':float(ds.sum()/mass.sum()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'cluster':'item_id','reps':3000,'scope':'descriptive_fixed_first_seed'}
    save(dest/'POINT_ANALYSIS.json',dict(seed_aggregates=rows,first_seed=pointdetail,contrasts=contrasts))
    names=['legacy_raw','legacy_censorcast','plain','category_scale','censor_adapter']
    labels=['HistGB\nPoisson','Legacy\nCENSORCAST','LightGBM\nL1','L1 + category\nrescaling','L1 + censor\nadapter']
    f,ax=plt.subplots(figsize=(6.4,2.55));v=[pointdetail[n]['wape'] for n in names]
    ax.bar(range(5),v,color=['#9aa7b4','#8d9ec0','#658fa6','#4c8b8b','#276c61'])
    ax.set_xticks(range(5),labels);ax.set_ylabel('WAPE (lower is better)');ax.set_ylim(0,.82)
    for j,z in enumerate(v):ax.text(j,z+.008,f'{z:.3f}',ha='center',fontsize=9)
    ax.set_title('Mechanically censored M5 design shadow; first fixed seed')
    f.tight_layout();f.savefig(fig/'point_ablation.png');plt.close(f)
    for study in ['selective','selective_strong']:
        path=root/'results'/study/'SELECTIVE_RESULTS.json'
        if not path.exists():continue
        doc=json.loads(path.read_text());rr=doc['records'];cache=root/'results'/study/'cache';v=load(cache/'shadow_aligned.npz')
        summaries=[]
        for mode in ['pooled','category']:
            for score in sorted({r['score'] for r in rr}):
                rrs=[r for r in rr if r['score']==score and r['threshold_mode']==mode]
                summaries.append({'score':score,'mode':mode,'seeds':len(rrs),'coverage_mean':float(np.mean([r['metrics']['shadow']['coverage'] for r in rrs])),
                    'coverage_all':[r['metrics']['shadow']['coverage'] for r in rrs],
                    'wape_all':[r['metrics']['shadow']['wape'] for r in rrs],
                    'auc_all':[r['score_diagnostics']['auc_top20_absolute_error'] for r in rrs]})
        group_intervals={}
        for score in ['contract_excess','relative_error']:
            pol=load(root/f'results/{study}/mask_{score}_pooled_s20260906.npz')['shadow']
            e=np.abs(v['truth'].astype(float)-v['proposal']);yy=v['truth'].astype(float)
            gs={}
            for g in ['ALL']+np.unique(meta['cat_id']).tolist():
                row=np.ones(len(pol),bool) if g=='ALL' else meta['cat_id']==g
                aa=pol & row[:,None];den=np.broadcast_to(row[:,None],pol.shape)
                ac=sums(aa);nn=sums(den);ee=sums(e*aa);dd=sums(yy*aa)
                bden=weights@nn;bm=weights@dd;good=(bden>0)&(bm>0)
                bc=(weights@ac)[good]/bden[good];bw=(weights@ee)[good]/bm[good]
                gs[g]={'coverage':float(ac.sum()/nn.sum()),'wape':float(ee.sum()/dd.sum()) if dd.sum() else None,
                    'coverage_ci95':np.quantile(bc,[.025,.975]).tolist() if len(bc) else None,
                    'wape_ci95':np.quantile(bw,[.025,.975]).tolist() if len(bw) else None,
                    'coverage_lcb_bonf8':float(np.quantile(bc,.05/8)) if len(bc) else 0,
                    'wape_ucb_bonf8':float(np.quantile(bw,1-.05/8)) if len(bw) else None}
            group_intervals[score]=gs
        ma=load(root/f'results/{study}/mask_contract_excess_pooled_s20260906.npz')['shadow']
        mb=load(root/f'results/{study}/mask_relative_error_pooled_s20260906.npz')['shadow']
        delta=sums(ma.astype(float)-mb);tot=sums(np.ones(ma.shape));bd=(weights@delta)/(weights@tot)
        contrast={'coverage_difference':float(delta.sum()/tot.sum()),'ci95':np.quantile(bd,[.025,.975]).tolist()}
        save(dest/(study+'_ANALYSIS.json'),{'seed_aggregates':summaries,'category_intervals':group_intervals,'direct_minus_relative_coverage':contrast})
        f,axes=plt.subplots(1,2,figsize=(6.4,2.65))
        for name,label,color in [('mean_error','Mean absolute error','#ad6f3b'),('relative_error','Predicted error / forecast','#59818e'),('contract_excess','Conditional contract excess','#1c7660')]:
            source='mean_error' if name=='relative_error' else name
            s=load(root/f'results/{study}/{source}_s20260906_scores.npz')['shadow']
            if name=='relative_error':s=np.maximum(s,0)/np.maximum(v['proposal'],.25)
            ff=frontier(s,v['truth'],v['proposal'],v['baseline'],'absolute');idx=np.unique(np.linspace(0,len(ff['coverage'])-1,1000).astype(int))
            axes[0].plot(ff['coverage'][idx],ff['ratio'][idx],label=label,color=color,lw=1.4)
            rec=next(r for r in rr if r['score']==name and r['threshold_mode']=='pooled' and r['seed']==20260906)
            m=rec['metrics']['shadow']
            if m['wape'] is not None:axes[0].scatter(m['coverage'],m['wape'],color=color,s=22,zorder=4)
        axes[0].axhline(R,ls='--',color='black',lw=.8);axes[0].axvline(.35,ls=':',color='black',lw=.8)
        axes[0].set(xlabel='Accepted fraction',ylabel='Accepted WAPE',ylim=(.2,1.3),xlim=(0,1),title='Shadow frontiers (diagnostic only)')
        axes[0].legend(fontsize=6.5,loc='upper left',frameon=False)
        gs=group_intervals['contract_excess'];gn=list(gs);xs=np.arange(len(gn));vv=[gs[g]['wape'] for g in gn]
        axes[1].bar(xs,vv,color=['#49616a','#277862','#b77132','#876079'])
        for j,g in enumerate(gn):
            lo,hi=gs[g]['wape_ci95'];axes[1].plot([j,j],[lo,hi],color='black',lw=1)
        axes[1].set_xticks(xs,['All','Foods','Hobbies','Household'],rotation=20,fontsize=7)
        axes[1].axhline(R,color='black',ls='--',lw=.8);axes[1].set(ylabel='Accepted WAPE',title='One pooled excess threshold')
        f.tight_layout();f.savefig(fig/(study+'_frontier.png'));plt.close(f)
    print('Analysis and figures written',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,default=Path('iclr_study'));main(ap.parse_args())
