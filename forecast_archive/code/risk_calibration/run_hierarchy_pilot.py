"""Primary-seed observable cross-store product-history feature pilot."""
from pathlib import Path
import argparse,sys
import numpy as np
import xgboost as xgb
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_aft_pilot import now,sha,dump,weighted_scale,scores
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'accuracy_revision_20260907'/'code'/'m5'))
from legacy_features.features import DesignPanel,FeatureBuilder

class HierarchyFeatures:
 def __init__(self,panel):
  _,self.inv=np.unique(panel.metadata['item_id'],return_inverse=True);n=self.inv.max()+1
  count=np.bincount(self.inv).astype(float)[:,None]
  agg=np.zeros((n,panel.n_days),np.float32);hits=np.zeros_like(agg)
  np.add.at(agg,self.inv,panel.observed);np.add.at(hits,self.inv,(panel.observed>=panel.capacity).astype(np.float32))
  self.mean=agg/count;self.hit=hits/count
  self.cs=np.pad(np.cumsum(self.mean.astype(float),axis=1),((0,0),(1,0)))
  self.hcs=np.pad(np.cumsum(self.hit.astype(float),axis=1),((0,0),(1,0)))
 def make(self,series,target_days):
  item=self.inv[series];origin=target_days-(1+((target_days-1)%7));m=self.mean
  lag=[m[item,origin-k] for k in [1,7,28]]
  means=[(self.cs[item,origin]-self.cs[item,origin-w])/w for w in [7,28,56]]
  hit28=(self.hcs[item,origin]-self.hcs[item,origin-28])/28
  return np.column_stack([*lag,*means,hit28]).astype(np.float32)

def make_x(builder,hier,s,d): return np.column_stack([builder.make_features(s,d,include_censor=True),hier.make(s,d)])
def predict(builder,hier,model,days):
 n=builder.panel.n_series;o=np.empty((n,len(days)),np.float32)
 for k in range(0,len(days),7):
  ds=days[k:k+7];s=np.repeat(np.arange(n,dtype=np.int32),len(ds));d=np.tile(ds,n);x=make_x(builder,hier,s,d)
  o[:,k:k+len(ds)]=model.predict(xgb.DMatrix(x)).reshape(n,-1)
 return np.maximum(o-1,0)
def run(data,out):
 out.mkdir(parents=True,exist_ok=False);dump(out/'PROTOCOL.json',{'status':'FROZEN_BEFORE_FIT','created_utc':now(),'seed':20260906,
 'configuration':{'depth':8,'rounds':750,'eta':.03,'extra_features':['cross-store item lag 1/7/28','cross-store item mean 7/28/56','cross-store item hit rate 28']},
 'information':'all aggregates end at forecast origin; no target or future sales','scope':'validation-only feature pilot','script_sha256':sha(__file__)})
 p=DesignPanel.load(data/'design_outcomes_v0_5.npz',data/'calendar.csv',data/'sell_prices.csv');p.censored=(p.observed>=p.capacity).astype(np.uint8);b=FeatureBuilder(p);h=HierarchyFeatures(p)
 s,d=b.sample_pairs(365,1313,1200000,20260906);x=make_x(b,h,s,d);dm=xgb.DMatrix(x,label=p.observed[s,d-1].astype(float)+1);del x
 m=xgb.train({'objective':'reg:absoluteerror','eta':.03,'max_depth':8,'min_child_weight':100,'subsample':.9,'colsample_bytree':.9,'lambda':2.,'tree_method':'hist','max_bin':255,'nthread':8,'seed':20260906},dm,750);m.save_model(out/'model.ubj')
 days=np.arange(1314,1434,dtype=np.int32);days=days[days-b.horizon_for_day(days)>=1313];y=p.truth[:,days-1].astype(float);f=predict(b,h,m,days);cats=p.metadata['cat_id']
 cs={str(c):weighted_scale(y[cats==c],f[cats==c]) for c in np.unique(cats)};cm=np.asarray([cs[str(c)] for c in cats])[:,None]
 np.savez_compressed(out/'validation.npz',days=days,prediction=f)
 dump(out/'RESULTS.json',{'status':'COMPLETED','created_utc':now(),'raw':scores(y,f),'category_scales':cs,'category_scaled':scores(y,cm*f)})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--output',type=Path,required=True);v=a.parse_args();run(v.data,v.output)
