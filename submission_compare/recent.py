"""Recent already observed reference history, base sleeve only, same execution."""
import copy
import gzip
import hashlib
import json
from datetime import datetime,timezone
import numpy as np
from competition_live.profile import ROOT,load_profile
from competition_live.scoring import evaluate
from experiments.decision_gate_pairs_20261017.portfolio import replay
from .run import OUT,OLD,AGREE,native,save
from .attribution import attribute


def main():
    profile=load_profile()
    old=json.loads((OLD/'protocol.json').read_text())
    data_path=ROOT/old['pointer']['inputs']
    data=dict(np.load(data_path,allow_pickle=False))
    legacy=dict(np.load(OLD/'targets.npz',allow_pickle=False))
    current=dict(np.load(OUT/'targets.npz',allow_pickle=False))
    agreement=dict(np.load(AGREE/'targets.npz',allow_pickle=False))
    times=current['times'];np.testing.assert_array_equal(times,data['market_times'])
    d=dict(times=times,par_close=data['close'],reference_close=data['close'],
        par_volume=data['volume'],par_trades=data['trades'],funding=data['funding'],
        funding_observed=data['observed'],sigma=legacy['sigma'])
    weights={k:current[k] for k in ['incumbent','confirmed_short','forecast_filter','agreement_confirmed_short']}
    weights.update(short_only=legacy['short_only'],agreement=agreement['agreement'],size_only=legacy['size_only'])
    start=int(datetime(2026,10,15,12,tzinfo=timezone.utc).timestamp())
    end=int(times[-1]//86400*86400+43200)
    if end>times[-1]:end-=86400
    spec=dict(at=datetime.now(timezone.utc).isoformat(),start=start,end=end,arms=list(weights),
        scope='Base directional sleeve ONLY. Flat start on Oct15. No pair trades; not whole-portfolio PnL.',
        evidence='Existing mainnet reference observations, not PAS order-book fills. No future capture.',
        scenarios=['base','stress'],history_seen=True,no_parameter_search=True,
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [data_path,OLD/'targets.npz',AGREE/'targets.npz',OUT/'targets.npz']})
    save(OUT/'recent_protocol.json',spec)
    rows=[];(OUT/'recent').mkdir(exist_ok=True)
    for scenario in spec['scenarios']:
        cfg=copy.deepcopy(profile['configs']['reference'])
        cfg['conditional_completion']={'risk_aversion':0.}
        for name,w in weights.items():
            d['first']=w
            result=replay(d,{},None,cfg,profile['trading'],'first_strategy',scenario,start,end)
            m=result['metrics']
            samples=[dict(time=s['time'],equity=s['equity']+50000) for s in result['snapshots'] if s['time']%86400==43200]
            sc=evaluate(samples,trading_days=len({f['time']//86400 for f in result['fills']}))
            curve=np.array([s['equity']+50000 for s in result['snapshots']])
            breakdown=attribute(result,d,cfg['scenarios'][scenario])
            row=dict(variant=name,scenario=scenario,start=start,end=end,scope='base_only',
                net_pnl=m['net_pnl'],Q=sc['Q'],max_drawdown=float(-np.min(curve/np.maximum.accumulate(curve)-1)),
                fees=m['fees'],execution_cost=m['execution_cost'],funding=m['funding'],
                fills=m['fills'],risk_breach_bars=m['risk_breach_bars'],
                accounting_error=m['maximum_accounting_error'],
                timing_pass=all(f['source_available']<=f['decision_time']<f['time'] for f in result['fills']),
                attribution=breakdown)
            assert row['timing_pass'] and row['accounting_error']<1e-6
            with gzip.open(OUT/'recent'/f'{name}_{scenario}.json.gz','wt') as f:json.dump(result,f,default=native)
            rows.append(row);save(OUT/'recent_results.json',rows)
            print(json.dumps({k:v for k,v in row.items() if k!='attribution'}),flush=True)


if __name__=='__main__':main()
