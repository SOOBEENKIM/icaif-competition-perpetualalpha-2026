"""Frozen, decomposed replay of the audited fixes. Never submits orders."""
import gzip
import json
import time
from collections import Counter

import numpy as np

from .data import HERE,OUT,save,sha
from .forecast import timestamp
from .policy import path_calibrate,AnchoredController
from experiments.cost_horizon_20261016.research import data,SecondController
from experiments.cost_horizon_20261016.policy import CostController
from experiments.performance_attribution_20261016.engine import simulate
from replay_lab.ema_rebuild.engine import Controller

ROOT=HERE.parents[1]


def dump_run(dest,r):
    with gzip.open(dest,'wt') as f:json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())


def summarize(r):
    return {'mean_expected_daily_volatility':float(np.mean([s['daily_volatility'] for s in r['snapshots']])),
            'maximum_expected_daily_volatility':max(s['daily_volatility'] for s in r['snapshots']),
            'engineering_pass':all(r['gates'].values()),'roi_not_used_for_selection':r['metrics']['roi']}


def attribution(r,m):
    q=np.array([s['qty'] for s in r['snapshots']]);tt=np.array([s['time'] for s in r['snapshots']]);ii=np.searchsorted(m.times,tt)
    move=(q[:-1]*np.diff(m.close[ii],axis=0)).sum(0);cost=np.zeros(len(m.symbols));fees=cost.copy();funding=cost.copy()
    for f in r['fills']:
        j=f['asset'];k=m.exact(f['time']);cost[j]+=f['signed_qty']*(f['price']-m.close[k,j]);fees[j]+=f['fee']
    for f in r['funding_events']:
        n=np.array(f['qty'])*f['prices'];funding-=n*f['rates']
        if r['scenario']=='stress':funding-=abs(n)*.0001
    pnl=move+funding-cost-fees
    assert abs(pnl.sum()-r['metrics']['net_pnl'])<1e-5
    return [{'symbol':s,'price_pnl':float(move[j]),'funding':float(funding[j]),'fees':float(fees[j]),
             'execution_cost':float(cost[j]),'net_pnl':float(pnl[j])} for j,s in enumerate(m.symbols)]


def main():
    dest=OUT/'comparison';dest.mkdir(exist_ok=True)
    if (dest/'frozen.json').exists():raise FileExistsError('Comparison already frozen')
    frozen=json.loads((ROOT/'experiments/cost_horizon_20261016/artifacts/protocol_frozen.json').read_text())
    for group in ('active_protected_sources','reserved_dependencies'):
        for path,h in frozen[group].items():assert sha(ROOT/path)==h,path
    config=json.loads((HERE/'protocol.json').read_text())
    inp,a,p,m=data();fixed=json.loads((ROOT/'experiments/performance_attribution_20261016/artifacts/budget_frozen.json').read_text())['multiplier']
    baseline=a['candidate']*fixed
    with np.load(OUT/'forecast/predictions.npz') as z:pred={k:z[k].copy() for k in z.files}
    assert np.array_equal(pred['times'],m.times)
    sources={str(q.relative_to(ROOT)):sha(q) for q in [*HERE.glob('*.py'),HERE/'protocol.json',OUT/'forecast/manifest.json',OUT/'forecast/predictions.npz',OUT/'data_receipts.json']}
    save(dest/'frozen.json',{'at_unix':time.time(),'sources':sources,'historical_independence':False,'orders_sent':0})
    (dest/'runs').mkdir(exist_ok=True)
    allrows=[];fullruns={}
    def run(name,start,end,scenario,controller,weights,phase,save_full=True):
        r=simulate(m,weights,a['sigma'],p,m.exact(start),m.exact(end),scenario,controller)
        r.update(policy=name,phase=phase,per_asset=attribution(r,m),risk_path=summarize(r))
        r['decision_reasons']=dict(Counter(d.get('alpha_reason','baseline') for d in r['decisions']))
        if save_full:
            key=f'{phase}-{name}-{scenario}';dump_run(dest/f'runs/{key}.json.gz',r)
            allrows.append({k:r[k] for k in ('phase','policy','scenario','metrics','gates','risk_path','per_asset','decision_reasons')});fullruns[key]=r
            print(key,'ROI',round(r['metrics']['roi']*100,6),'STD',round(r['metrics']['daily_std_sample']*100,6),'fills',r['metrics']['fills'],flush=True)
        return r

    # Reproduce the exact failure; repair ONLY risk calibration on fixed forecasts.
    old=ROOT/'experiments/cost_horizon_20261016/artifacts'
    with np.load(old/'predictions.npz') as z:mu=z['ridge_24'].copy()
    gamma_old=json.loads((old/'calibration.json').read_text())['ridge_24']['gamma']
    july_start=timestamp('2026-07-01');july_end=timestamp('2026-08-01')
    b=run('first_strategy',july_start,july_end,'base',Controller(),a['incumbent'],'risk_calibration',False)
    def old_evaluate(gamma):
        rr=run('fixed_ridge',july_start,july_end,'base',CostController(m.times,mu,np.zeros_like(mu),gamma,24),np.zeros_like(baseline),'risk_calibration',False)
        return summarize(rr)
    risk_fix=path_calibrate(old_evaluate,summarize(b)['mean_expected_daily_volatility'],config['controller']['gamma_grid'],config['controller']['path_risk_tolerance'])
    save(dest/'fixed_signal_path_calibration.json',{'old_gamma':gamma_old,'new':risk_fix,'calibration_is_in_sample':True,
         'interpretation':'Isolates the inventory-path bug only. The production candidate below uses out-of-fit destination forecasts.'})
    for phase,start,end in [('august','2026-08-01','2026-09-01'),('september_october','2026-09-01','2026-10-13T12:00:00')]:
        for label,gamma in [('old_flat_ridge',gamma_old),('repaired_path_ridge',risk_fix['gamma'])]:
            if gamma is None:continue
            for scenario in ('base','stress'):
                run(label,timestamp(start),timestamp(end),scenario,CostController(m.times,mu,np.zeros_like(mu),gamma,24),np.zeros_like(baseline),phase)

    # Monthly prior-period calibration. All portfolio states evolve sequentially.
    gammas={name:np.full(len(m.times),np.nan) for name in ('rolling_ungated','rolling_gated')}
    eligible={'rolling_ungated':np.isfinite(pred['net_prediction']),
              'rolling_gated':pred['gates']&np.isfinite(pred['net_prediction'])}
    calibration=[];ctl=config['controller']
    def control(name,g):
        return AnchoredController(m.times,a['retention'],pred['net_prediction'],eligible[name],g,
               model_share=ctl['model_share'],partial=ctl['partial_adjustment'],cost_bps=ctl['cost_bps_one_way'])
    for month in np.arange(np.datetime64('2026-08','M'),np.datetime64('2026-11','M')):
        end=int(month.astype('datetime64[s]').astype('int64'));start=end-60*86400
        nextmonth=int((month+1).astype('datetime64[s]').astype('int64'));active=(m.times>=end)&(m.times<nextmonth)
        ref=run('second_strategy',start,end,'base',SecondController(m.times,a['retention']),baseline,'calibration',False)
        for name in gammas:
            if not eligible[name][active].any():
                calibration.append({'month':str(month),'policy':name,'gamma':None,'reason':'all_current_asset_gates_closed'});continue
            def evaluate(gamma):
                rr=run(name,start,end,'base',control(name,np.full(len(m.times),gamma)),baseline,'calibration',False)
                return summarize(rr)
            c=path_calibrate(evaluate,summarize(ref)['mean_expected_daily_volatility'],ctl['gamma_grid'],ctl['path_risk_tolerance'])
            c.update(month=str(month),policy=name,last_calibration_time=end,uses_future_returns=False)
            calibration.append(c)
            if c['gamma'] is not None:gammas[name][active]=c['gamma']
            print('path calibration',month,name,c['gamma'],flush=True)
    save(dest/'rolling_path_calibration.json',calibration)
    np.savez_compressed(dest/'gammas.npz',times=m.times,**gammas)
    for phase,start,end in [('august','2026-08-01','2026-09-01'),('september','2026-09-01','2026-10-01'),
                            ('october','2026-10-01','2026-10-13T12:00:00'),('september_october','2026-09-01','2026-10-13T12:00:00'),
                            ('all','2026-08-01','2026-10-13T12:00:00')]:
        for name in ('first_strategy','second_strategy','rolling_ungated','rolling_gated'):
            for scenario in ('base','stress'):
                c=Controller() if name=='first_strategy' else (SecondController(m.times,a['retention']) if name=='second_strategy' else control(name,gammas[name]))
                run(name,timestamp(start),timestamp(end),scenario,c,a['incumbent'] if name=='first_strategy' else baseline,phase)
    # Keep baseline reproduction and all-rejected fallback as full-path assertions.
    for scenario in ('base','stress'):
        for name in ('first_strategy','second_strategy'):
            with gzip.open(old/f'runs/comparison-{name}-{scenario}.json.gz','rt') as stream:prior=json.load(stream)
            now=fullruns[f'september_october-{name}-{scenario}']
            assert now['fills']==prior['fills']
            np.testing.assert_array_equal([s['equity'] for s in now['snapshots']],[s['equity'] for s in prior['snapshots']])
        candidate=fullruns[f'september_october-rolling_gated-{scenario}'];base=fullruns[f'september_october-second_strategy-{scenario}']
        active=(m.times>=timestamp('2026-09-01'))&(m.times<=timestamp('2026-10-13T12:00:00'))
        if not eligible['rolling_gated'][active].any():
            assert candidate['fills']==base['fills']
            np.testing.assert_array_equal([s['equity'] for s in candidate['snapshots']],[s['equity'] for s in base['snapshots']])
    checks=[]
    for phase in ('august','september_october','all'):
        for scenario in ('base','stress'):
            c=fullruns[f'{phase}-rolling_gated-{scenario}']['metrics']
            for name in ('first_strategy','second_strategy'):
                b=fullruns[f'{phase}-{name}-{scenario}']['metrics']
                checks.append({'period':phase,'scenario':scenario,'comparator':name,
                    'return_delta':c['roi']-b['roi'],'daily_std_delta':c['daily_std_sample']-b['daily_std_sample'],
                    'mdd_delta':c['max_drawdown_magnitude']-b['max_drawdown_magnitude'],
                    'strictly_improved':c['roi']>b['roi'] and c['daily_std_sample']<=b['daily_std_sample'] and c['max_drawdown_magnitude']<=b['max_drawdown_magnitude']})
    for group in ('active_protected_sources','reserved_dependencies'):
        for path,h in frozen[group].items():assert sha(ROOT/path)==h,path
    for path,h in sources.items():assert sha(ROOT/path)==h,path
    save(dest/'results.json',{'rows':allrows,'checks':checks,'risk_repair':risk_fix,'rolling_calibration':calibration,
                             'baselines_reproduced_exactly':True,'historical_independence':False,'live_orders_sent':0,
                             'protected_sources_unchanged':True,'live_replacement_approved':False,
                             'historical_strict_improvement':all(c['strictly_improved'] for c in checks)})
    print('DONE',len(allrows),'recorded runs',flush=True)


if __name__=='__main__':main()
