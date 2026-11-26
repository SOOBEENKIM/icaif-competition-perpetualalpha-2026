"""Disclosed follow-up: reject an alpha sleeve if its prior full path failed.

The initial repaired run inspected a portfolio with negative pre-August return
despite positive per-asset diagnostic gates. This stage repairs that omission.
It is post-hoc research, not new independent validation, and never tunes gates
using the future month being decided.
"""
import gzip
import json
import time

import numpy as np

from .data import HERE,OUT,save,sha
from .forecast import timestamp
from .policy import AnchoredController
from .admission import economic_admission
from .research import attribution,dump_run,summarize
from experiments.cost_horizon_20261016.research import data,SecondController
from experiments.performance_attribution_20261016.engine import simulate

ROOT=HERE.parents[1]


def main():
    dest=OUT/'acceptance';dest.mkdir(exist_ok=True)
    if (dest/'plan.json').exists():raise FileExistsError('Acceptance experiment already frozen')
    protocol=json.loads((HERE/'protocol.json').read_text())
    sources={str(q.relative_to(ROOT)):sha(q) for q in HERE.glob('*.py')}
    save(dest/'plan.json',{'at_unix':time.time(),'sources':sources,'after_results_sha256':sha(OUT/'comparison/results.json'),
         'hypothesis':'Prior asset-level directional PNL and risk eligibility are insufficient: the actual prior portfolio path must also improve net return without worse daily STD/MDD.',
         'rules':'Positive prior net return; exceed second strategy net return; no worse daily STD and MDD; at least 30 complete days; valid accounting/risk. Otherwise identical baseline controller on actual holdings.',
         'historical_independence':False,'live_orders_permitted':False,'selection_period':'Previous 60 days only; no admission based on the month being traded.'})
    inp,a,p,m=data()
    fixed=json.loads((ROOT/'experiments/performance_attribution_20261016/artifacts/budget_frozen.json').read_text())['multiplier']
    baseline=a['candidate']*fixed
    with np.load(OUT/'forecast/predictions.npz') as z:pred={k:z[k].copy() for k in z.files}
    with np.load(OUT/'comparison/gammas.npz') as z:gamma=z['rolling_gated'].copy()
    eligible=pred['gates']&np.isfinite(pred['net_prediction']);ctl=protocol['controller']
    def controller(g):return AnchoredController(m.times,a['retention'],pred['net_prediction'],eligible,g,
        model_share=ctl['model_share'],partial=ctl['partial_adjustment'],cost_bps=ctl['cost_bps_one_way'])
    admissions=[];(dest/'runs').mkdir(exist_ok=True)
    for month in np.arange(np.datetime64('2026-08','M'),np.datetime64('2026-11','M')):
        cutoff=int(month.astype('datetime64[s]').astype('int64'));stop=int((month+1).astype('datetime64[s]').astype('int64'))
        active=(m.times>=cutoff)&(m.times<stop);finite=gamma[active][np.isfinite(gamma[active])]
        if not len(finite) or not eligible[active].any():
            admissions.append({'month':str(month),'admitted':False,'reason':'no_admissible_asset_or_path_risk'});continue
        g=float(finite[0]);start=cutoff-60*86400
        c=simulate(m,baseline,a['sigma'],p,m.exact(start),m.exact(cutoff),'base',controller(np.full(len(m.times),g)))
        b=simulate(m,baseline,a['sigma'],p,m.exact(start),m.exact(cutoff),'base',SecondController(m.times,a['retention']))
        admission=economic_admission(c,b);admission.update(month=str(month),start=start,end=cutoff,fit_before_next_decision=True)
        admissions.append(admission)
        dump_run(dest/f'runs/prior-{month}-candidate.json.gz',c);dump_run(dest/f'runs/prior-{month}-baseline.json.gz',b)
        if not admission['admitted']:gamma[active]=np.nan
        print('portfolio admission',month,admission['admitted'],'candidate prior ROI',c['metrics']['roi'],'baseline',b['metrics']['roi'],flush=True)
    rows=[]
    for phase,start,end in [('august','2026-08-01','2026-09-01'),('september_october','2026-09-01','2026-10-13T12:00:00'),('all','2026-08-01','2026-10-13T12:00:00')]:
        for scenario in ('base','stress'):
            rr=simulate(m,baseline,a['sigma'],p,m.exact(timestamp(start)),m.exact(timestamp(end)),scenario,controller(gamma))
            rr.update(phase=phase,policy='portfolio_admitted',per_asset=attribution(rr,m),risk_path=summarize(rr))
            with gzip.open(OUT/f'comparison/runs/{phase}-second_strategy-{scenario}.json.gz','rt') as f:old=json.load(f)
            ix=(m.times>=timestamp(start))&(m.times<=timestamp(end))
            if not np.isfinite(gamma[ix]).any():
                assert rr['fills']==old['fills']
                np.testing.assert_array_equal([s['equity'] for s in rr['snapshots']],[s['equity'] for s in old['snapshots']])
            dump_run(dest/f'runs/{phase}-portfolio_admitted-{scenario}.json.gz',rr)
            rows.append({k:rr[k] for k in ('phase','policy','scenario','metrics','gates','per_asset','risk_path')})
    np.savez_compressed(dest/'admitted_gamma.npz',times=m.times,gamma=gamma)
    save(dest/'results.json',{'admissions':admissions,'rows':rows,'new_alpha_months_admitted':sum(x['admitted'] for x in admissions),
          'rejected_alpha_fallback_reproduced_exactly':True,'historical_independence':False,'live_orders_sent':0,'live_replacement_approved':False})
    for path,h in sources.items():assert sha(ROOT/path)==h,path
    print('DONE acceptance; live replacement prohibited',flush=True)


if __name__=='__main__':main()
