import json,sys
from pathlib import Path
import numpy as np,joblib
p=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(p/'upstream/design'))
from censorcast_v05.features import DesignPanel,FeatureBuilder
panel=DesignPanel.load(p/'inputs/design_outcomes_v0_5.npz',p/'inputs/calendar.csv',p/'inputs/sell_prices.csv');builder=FeatureBuilder(panel)
b=joblib.load(p/'inputs/forecast_bundle_v0_5.joblib')
with np.load(p/'inputs/selection_base_v0_5.npz') as z:expected={k:z[k][:,3] for k in ['raw_poisson_histgb','censored_poisson_em','censor_probability']};assert z['target_days'][3]==1317
ss=np.arange(panel.n_series);dd=np.full(panel.n_series,1317)
y=b.predict(builder.make_features(ss,dd,include_censor=False),builder.make_features(ss,dd,include_censor=True))
r={k:{'exact_float32_match':bool(np.array_equal(expected[k],y[k])),'max_abs_difference':float(np.max(np.abs(expected[k]-y[k])))} for k in expected}
print(json.dumps(r));(p/'provenance/LEGACY_PREDICTION_REPLAY.json').write_text(json.dumps(r,indent=2))
assert all(v['exact_float32_match'] for v in r.values()),'frozen component replay mismatch'
