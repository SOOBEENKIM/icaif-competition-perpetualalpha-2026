"""Longer reference-market ablation, explicitly not competition venue PnL."""
import copy,gzip,json,time
import numpy as np
from experiments.cost_horizon_20261016.research import data
from experiments.par_direct_forecast_20261017.data import ts,save,sha
from experiments.complementary_alpha_20261017.research import protected
from .research import HERE,ROOT
from .signals import design
from .reference_engine_v2 import replay


def inputs(cfg):
    inp,a,p,m=data()
    multiplier=json.loads((ROOT/'experiments/performance_attribution_20261016/artifacts/budget_frozen.json').read_text())['multiplier']
    d=dict(times=m.times,par_close=m.close,reference_close=m.close,par_volume=m.volume,par_trades=m.trades,
        funding=m.funding,funding_observed=m.observed,known_hourly_funding=np.r_[np.zeros((1,10)),m.funding[:-1]],
        first=a['incumbent'],second=a['candidate']*multiplier,retention=a['retention'],sigma=a['sigma'])
    # Formation selected on the newer immutable snapshot must match the actual
    # replay snapshot through every formation cutoff, not just have equal labels.
    with np.load(ROOT/cfg['reference_run']/'inputs.npz') as z:
        times=z['market_times'];close=z['close']
    ix=np.searchsorted(times,m.times);mask=m.times<ts(cfg['formation_schedule'][-1]['date'])
    np.testing.assert_array_equal(close[ix[mask]],m.close[mask])
    de=design(d,inp,cfg,p['symbols'])
    for i,t in enumerate(m.times):
        forms=[f for f in cfg['formation_schedule'] if ts(f['date'])<=t]
        allowed=forms[-1]['pairs'] if forms else []
        for k,pair in enumerate(cfg['pairs']):
            if pair not in allowed:de['standard'][i,k]=de['delayed'][i,k]=0
    return d,de,p


def main():
    out=HERE/'reference_artifacts_v2';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    cfg=json.loads((HERE/'reference_protocol.json').read_text());d,de,p=inputs(cfg)
    paths=[HERE/'reference_research_v2.py',HERE/'reference_engine_v2.py',HERE/'reference_protocol.json',HERE/'signals.py',
        ROOT/cfg['formation_source'],ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',
        ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz']
    sources={str(f.relative_to(ROOT)):sha(f) for f in paths}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),historical_independence=False,orders_sent=0))
    np.savez_compressed(out/'state.npz',**de)
    arms=['first_strategy','second_strategy',*cfg['arms']];rows=[];runs={}
    def record(phase,arm,scenario,overlay=False):
        start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start']
        end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        r=replay(d,de,None,cfg,p,arm,scenario,ts(start),ts(end),overlay=overlay)
        label='second_plus_'+arm if overlay else arm;key=f'{phase}-{label}-{scenario}'
        r.update(label=label,phase=phase,venue='reference_proxy',not_par_execution=True);runs[key]=r
        assert all(f['time']-f['decision_time']==3600 for f in r['fills'])
        with gzip.open(out/'runs'/f'{key}.json.gz','wt') as f:
            json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        rows.append(dict(phase=phase,arm=label,scenario=scenario,metrics=r['metrics']))
        m=r['metrics'];print(key,'PnL',round(m['net_pnl'],3),'Q',round(m['financial_score'],6),'pairs',m['completed_pairs'],'aborts',m['aborted_pairs'],flush=True)
    for arm in arms:
        for scenario in cfg['scenarios']:record('validation',arm,scenario)
    ranking=[]
    for arm in cfg['arms']:
        checks=[];diff=[]
        for scenario in cfg['scenarios']:
            a=runs[f'validation-{arm}-{scenario}']['metrics']
            checks.extend([a['net_pnl']>0,a['completed_pairs']>=3,a['risk_breach_bars']==0])
            for bn in arms[:2]:
                b=runs[f'validation-{bn}-{scenario}']['metrics'];diff.append(a['net_pnl']-b['net_pnl'])
                checks.extend([a['net_pnl']>b['net_pnl'],a['financial_score']>b['financial_score']])
        ranking.append(dict(arm=arm,eligible=all(checks),minimum_pnl_advantage=min(diff)))
    ranking.sort(key=lambda r:(-r['eligible'],-r['minimum_pnl_advantage'],r['arm']))
    selected=ranking[0]['arm'];selection=dict(selected=selected,eligible=ranking[0]['eligible'],ranking=ranking,frozen_before_comparison_replay=True,historical_independence=False)
    save(out/'selection.json',selection);sh=sha(out/'selection.json')
    for arm in arms:
        for scenario in cfg['scenarios']:record('comparison',arm,scenario)
    for phase in ('validation','comparison'):
        for scenario in cfg['scenarios']:record(phase,selected,scenario,True)
    assert sha(out/'selection.json')==sh
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    protected()
    save(out/'results.json',dict(rows=rows,selection=selection,historical_replays=len(rows),venue='reference_proxy',
        historical_independence=False,not_par_execution=True,orders_sent=0,automatic_live_replacement=False))
    print('DONE reference diagnostic',len(rows),'replays; selected',selected,'eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
