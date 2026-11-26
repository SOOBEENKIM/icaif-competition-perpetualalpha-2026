"""Prespecified incumbent-signal pairs: select once, compare later, audit ledgers."""
import copy
import gzip
import json
import time
from pathlib import Path
import numpy as np
from experiments.cost_horizon_20261016.research import data,ts,SecondController,save,sha
from experiments.performance_attribution_20261016.engine import simulate
from experiments.q_execution_20261017.metrics import official_metrics,paired_uncertainty
from experiments.complementary_alpha_20261017.research import protected
from replay_lab.ema_rebuild.engine import Controller
from experiments.incumbent_pairs_20261017.model import features
from .policy import IncumbentPairController

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'artifacts'


def main():
    OUT.mkdir(exist_ok=False);(OUT/'runs').mkdir()
    cfg=json.loads((HERE/'protocol.json').read_text());_,a,p,m=data();p=copy.deepcopy(p)
    p['scenarios']['execution_stress']=dict(impact_bps=20.,participation=.0025,adverse_funding_bps_hour=0.)
    mult=json.loads((ROOT/'experiments/performance_attribution_20261016/artifacts/budget_frozen.json').read_text())['multiplier']
    second=a['candidate']*mult
    deps=[*HERE.glob('*.py'),HERE/'protocol.json',
          ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',
          ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz',
          ROOT/'experiments/pairs_forecast_20261017/model.py',
          ROOT/'experiments/pairs_forecast_20261017/policy.py',
          ROOT/'experiments/performance_attribution_20261016/engine.py',
          ROOT/'experiments/q_execution_20261017/metrics.py',
          ROOT/'experiments/basis_carry_20261017/score.py',
          ROOT/'experiments/incumbent_pairs_20261017/model.py',
          ROOT/'experiments/incumbent_pairs_20261017/artifacts/models/predictions.npz',
          ROOT/'experiments/incumbent_pairs_20261017/artifacts/models/manifest.json']
    sources={str(f.relative_to(ROOT)):sha(f) for f in deps}
    save(OUT/'frozen.json',dict(at=time.time(),sources=sources,protected=protected(),historical_independence=False))
    d=features(m,a,cfg)
    previous=ROOT/'experiments/incumbent_pairs_20261017/artifacts/models'
    z=np.load(previous/'predictions.npz',allow_pickle=False)
    np.testing.assert_array_equal(z['times'],m.times)
    pred,stamps=z['predictions'],z['fit_stamps']
    fits=json.loads((previous/'manifest.json').read_text())
    print('FIT',[(f['samples'],f['status']) for f in fits],flush=True)
    variants={v['name']:v for v in cfg['variants']};rows=[];runs={}
    def replay(name,phase,scenario,scale=1.):
        first,last=[m.exact(ts(s)) for s in cfg[phase]]
        if name=='first':weights,controller=a['incumbent'],Controller()
        elif name in ('second','second_selection_risk_matched'):
            weights,controller=second*scale,SecondController(m.times,a['retention'])
        else:
            weights=second
            controller=IncumbentPairController(m,weights,a['retention'],d,pred,stamps,cfg,variants[name],p['scenarios'][scenario],last)
        r=simulate(m,weights,a['sigma'],p,first,last,scenario,controller)
        r.update(policy=name,phase=phase,official=official_metrics(r))
        if name in variants:
            r['events']=controller.events
            r['metrics'].update(pair_entries=sum(e['event']=='entry_intent' for e in controller.events),
                balanced_open=sum(e['event']=='balanced_open' for e in controller.events),
                unhedged_aborts=sum(e['event']=='unhedged_abort' for e in controller.events),
                pair_risk_reductions=sum(e['event']=='risk_cap_pair_reduction' for e in controller.events))
            if variants[name]['model']=='calibrated':
                r['gates']['past_only_calibration']=all(0<e['fit_time']<=e['time'] for e in controller.events if e['event']=='entry_intent')
            r['gates']['paired_positions_closed']=all(not np.any(np.array(r['metrics']['final_qty'])[d['pairs'][k]]) for k in controller.states)
        r['metrics']['estimated_residual_exit_cost']=float(abs(np.array(r['metrics']['final_qty']))@m.close[last])*(p['execution']['fee_bps']+p['scenarios'][scenario]['impact_bps'])/10000
        r['metrics']['pnl_less_residual_exit_estimate']=r['metrics']['net_pnl']-r['metrics']['estimated_residual_exit_cost']
        key=f'{phase}-{name}-{scenario}'
        with gzip.open(OUT/'runs'/f'{key}.json.gz','wt') as stream:
            json.dump(r,stream,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        rows.append({k:r[k] for k in ('policy','phase','scenario','metrics','official','gates')});runs[key]=r
        print(key,'PnL',round(r['metrics']['net_pnl'],2),'Q',round(r['official']['financial_score'],6),
              'opens',r['metrics'].get('balanced_open'),'gates',all(r['gates'].values()),flush=True)
        return r
    for name in ('first','second',*variants):
        for scenario in ('base','execution_stress'):replay(name,'selection',scenario)
    ranking=[]
    for name in variants:
        pnl,q,checks=[],[],[]
        for scenario in ('base','execution_stress'):
            r=runs[f'selection-{name}-{scenario}'];b=[runs[f'selection-{n}-{scenario}'] for n in ('first','second')]
            pnl.append(r['metrics']['net_pnl']-max(x['metrics']['net_pnl'] for x in b))
            q.append(r['official']['financial_score']-max(x['official']['financial_score'] for x in b))
            checks.append(all(r['gates'].values()) and all(all(x['gates'].values()) for x in b)
                          and r['metrics']['net_pnl']>0 and r['metrics']['balanced_open']>=4)
        ranking.append(dict(name=name,eligible=all(checks) and min(pnl)>0 and min(q)>0,
                            minimum_pnl_advantage=min(pnl),minimum_q_advantage=min(q)))
    ranking.sort(key=lambda r:(-r['eligible'],-r['minimum_pnl_advantage'],-r['minimum_q_advantage'],r['name']))
    selected=ranking[0]['name']
    scale=float(np.clip(runs[f'selection-{selected}-base']['official']['daily_sample_std']/
        max(runs['selection-second-base']['official']['daily_sample_std'],1e-12),.25,4.))
    selection=dict(at=time.time(),selected=selected,eligible=ranking[0]['eligible'],ranking=ranking,
                   second_selection_risk_matched_scale=scale,comparison_not_read=True,historical_independence=False)
    save(OUT/'selection.json',selection)
    comparisons=[]
    for scenario in ('base','execution_stress'):
        for name in ('first','second',selected,'second_selection_risk_matched'):
            replay(name,'comparison',scenario,scale if name=='second_selection_risk_matched' else 1.)
        r=runs[f'comparison-{selected}-{scenario}']
        for name in ('first','second','second_selection_risk_matched'):
            b=runs[f'comparison-{name}-{scenario}']
            comparisons.append(dict(baseline=name,scenario=scenario,pnl_advantage=r['metrics']['net_pnl']-b['metrics']['net_pnl'],
                q_advantage=r['official']['financial_score']-b['official']['financial_score'],
                closing_adjusted_pnl_advantage=r['metrics']['pnl_less_residual_exit_estimate']-b['metrics']['pnl_less_residual_exit_estimate'],
                engineering=all(r['gates'].values()) and all(b['gates'].values()),
                uncertainty=paired_uncertainty(r['official'],b['official'])))
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    improved=selection['eligible'] and all(c['pnl_advantage']>0 and c['q_advantage']>0 and c['engineering'] for c in comparisons)
    save(OUT/'results.json',dict(rows=rows,selection=selection,comparisons=comparisons,model_fits=fits,
        protected_after=protected(),historical_independence=False,live_orders_sent=0,live_replacement=False,
        development_improvement=improved,valuation='Reference closes, 12:00 UTC; no claim of actual par execution.'))
    print('COMPLETED',len(rows),'runs','development_improvement',improved,flush=True)


if __name__=='__main__':main()
