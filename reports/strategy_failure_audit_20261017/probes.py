"""Post-hoc mechanism checks, not candidate selection or independent validation."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.cost_horizon_20261016.research import data, ts, forecast_variant
from experiments.cost_horizon_20261016.policy import CostController, optimize
from experiments.performance_attribution_20261016.engine import simulate
from replay_lab.ema_rebuild.engine import Controller

OUT=Path(__file__).resolve().parent
ART=ROOT/'experiments/cost_horizon_20261016/artifacts'


def main():
    design={
        'type':'post-hoc diagnostic, never a promotion result',
        'partial_adjustment_probe':'Keep forecasts, gamma, clock, data and costs fixed; replace 0.5 by 1.0 to test whether partial adjustment caused excess turnover. Report both August and September/October regardless of sign.',
        'calibration_probe':'Run July sequentially from flat using existing in-sample forecasts and fixed gamma. Compare realized path ex-ante risk with calibration of independent flat-account targets.',
        'baseline_reproduction':'Replay selected tree72 comparison at original 0.5 and assert matching saved metrics.',
        'orders_sent':0,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'probe_plan.json').write_text(json.dumps(design,indent=2)+'\n')
    inp,a,p,m=data()
    with np.load(ART/'predictions.npz') as z:pred={k:z[k] for k in z.files}
    cal=json.loads((ART/'calibration.json').read_text())
    jobs=[('july','2026-07-01','2026-08-01','first_strategy',None),
          ('july','2026-07-01','2026-08-01','ridge_24',.5),
          ('july','2026-07-01','2026-08-01','tree_72',.5),
          ('august','2026-08-01','2026-09-01','tree_72',1.),
          ('comparison','2026-09-01','2026-10-13T12:00:00','tree_72',1.),
          ('comparison','2026-09-01','2026-10-13T12:00:00','tree_72',.5)]
    rows=[]
    for period,start,end,name,partial in jobs:
        if name=='first_strategy':
            c=Controller();weights=a['incumbent']
        else:
            mu,amb,h=forecast_variant(pred,name)
            c=CostController(m.times,mu,amb,cal[name]['gamma'],h,partial=partial)
            weights=np.zeros_like(a['incumbent'])
        r=simulate(m,weights,a['sigma'],p,m.exact(ts(start)),m.exact(ts(end)),'base',c)
        if period=='comparison' and partial==.5:
            import gzip
            with gzip.open(ART/'runs/comparison-selected_third-base.json.gz','rt') as f:old=json.load(f)
            assert r['fills']==old['fills']
            np.testing.assert_array_equal([s['equity'] for s in r['snapshots']],[s['equity'] for s in old['snapshots']])
        row=dict(period=period,policy=name,partial=partial,metrics=r['metrics'],gates=r['gates'],
                 decomposition=r['independent_accounting_audit'],
                 mean_path_expected_daily_volatility=float(np.mean([s['daily_volatility'] for s in r['snapshots']])),
                 maximum_path_expected_daily_volatility=max(s['daily_volatility'] for s in r['snapshots']))
        rows.append(row)
        print(period,name,partial,json.dumps({k:r['metrics'][k] for k in ['roi','daily_std_sample','turnover','fills']}),flush=True)
    # Separate time-of-day sampling from portfolio path dependence. These are
    # fixed existing gammas, not a new calibration or model selection.
    flat_checks=[]
    for name in ('ridge_24','tree_72'):
        mu,amb,h=forecast_variant(pred,name)
        ix=np.flatnonzero((m.times>=ts('2026-07-01'))&(m.times<ts('2026-08-01'))&(m.times%21600==0))
        vals=[]
        for i in ix:
            w,_=optimize(mu[i],a['sigma'][i,:3,:3],np.zeros(3),cal[name]['gamma'],h,.0019,amb[i])
            vals.append(float(np.sqrt(24*w@a['sigma'][i,:3,:3]@w)))
        vals=np.asarray(vals)
        flat_checks.append({'policy':name,'all_six_hour_flat_risk':float(vals.mean()),
            'by_utc_hour':{str(hr):float(vals[(m.times[ix]//3600)%24==hr].mean()) for hr in (0,6,12,18)}})
    (OUT/'probes.json').write_text(json.dumps({'plan':design,'results':rows,'flat_calibration_sampling':flat_checks,'saved_baseline_reproduced_exactly':True},indent=2,
        default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist(),allow_nan=False)+'\n')


if __name__=='__main__':main()
