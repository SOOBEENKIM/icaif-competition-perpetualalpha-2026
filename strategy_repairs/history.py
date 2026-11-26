import copy
import gzip
import json
from datetime import datetime, timezone
import numpy as np
from competition_live.profile import BUNDLE, load_profile
from competition_live.scoring import evaluate
from .protocol import OUT, freeze
from .engine import reference_replay


def native(x):
    return x.tolist() if hasattr(x,'tolist') else x.item()


def select_august(rows):
    index={(r['variant'],r['scenario']):r for r in rows if r['phase']=='validation'}
    candidates=[]
    for name in freeze()['arms']:
        if name in ('incumbent','size_only'):continue
        improvements=[]
        for scenario in ('base','stress'):
            a=index['incumbent',scenario];b=index[name,scenario]
            if not(b['net_pnl']>a['net_pnl'] and b['Q']>a['Q']
                   and b['risk_breach_bars']==0 and b['timing_pass']):
                break
            improvements.append(b['net_pnl']-a['net_pnl'])
        if len(improvements)==2:
            candidates.append(dict(variant=name,worst_pnl_advantage=min(improvements)))
    candidates.sort(key=lambda x:(-x['worst_pnl_advantage'],x['variant']))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
        nominee=candidates[0]['variant'] if candidates else None,eligible=candidates,
        selection_period='August only; reused history',historical_independence=False)


def main():
    spec=freeze();p=load_profile()
    data=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False))
    state=dict(np.load(BUNDLE/'models/reference_gate_state.npz',allow_pickle=False))
    targets=dict(np.load(OUT/'targets.npz',allow_pickle=False))
    ix=np.searchsorted(targets['times'],data['times'])
    np.testing.assert_array_equal(targets['times'][ix],data['times'])
    np.testing.assert_allclose(targets['incumbent'][ix],data['first'],atol=1e-13,rtol=1e-11)
    expected=json.loads((BUNDLE/'expected.json').read_text())
    windows=[r for r in expected['runs'] if r['venue']=='reference' and r['arm']==p['candidate']]
    rows=[];(OUT/'reference').mkdir(exist_ok=True)
    for phase in ('validation','comparison','continuous'):
        if phase=='comparison':
            nomination=select_august(rows)
            (OUT/'nomination.json').write_text(json.dumps(nomination,indent=2))
            print('FROZEN_NOMINATION',json.dumps(nomination),flush=True)
        for window in [r for r in windows if r['phase']==phase]:
            for name,arm in spec['arms'].items():
                d=dict(data);d['first']=targets[name][ix]
                run=reference_replay(d['times'],d['first'],targets[name+'_protected'][ix],arm['risk'])
                cfg=copy.deepcopy(p['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
                result=run(d,state,None,cfg,p['trading'],p['candidate'],window['scenario'],window['start'],window['end'])
                m=result['metrics']
                daily=[dict(time=s['time'],equity=s['equity']+50000) for s in result['snapshots'] if s['time']%86400==43200]
                team=evaluate(daily,trading_days=len({f['time']//86400 for f in result['fills']}))
                curve=np.array([s['equity']+50000 for s in result['snapshots']])
                row=dict(variant=name,phase=phase,scenario=window['scenario'],start=window['start'],end=window['end'],
                    net_pnl=m['net_pnl'],team_return=team['return_fraction'],Q=team['Q'],
                    volatility=team['volatility'],score=team['score'],daily_returns=team['completed_returns'],
                    max_drawdown=float(-np.min(curve/np.maximum.accumulate(curve)-1)),
                    mean_notional=float(np.mean([s['gross']*s['equity'] for s in result['snapshots']])),
                    fees=m['fees'],execution_cost=m['execution_cost'],funding=m['funding'],
                    base_pnl=m['base_owned_pnl'],pair_pnl=m['pair_owned_pnl'],
                    fills=m['fills'],completed_pairs=m['completed_pairs'],risk_breach_bars=m['risk_breach_bars'],
                    accounting_error=m['maximum_accounting_error'],
                    short_entries=sum(f['signed_qty']<0 and not f['reduce_only'] for f in result['fills']),
                    traded_symbols=sorted({f['symbol'] for f in result['fills']}),
                    timing_pass=all(f['source_available']<=f['decision_time']<f['time'] for f in result['fills']),
                    terminal_adjusted_pnl=m['pnl_after_liquidation_reserve'])
                assert row['accounting_error']<1e-6 and row['timing_pass']
                with gzip.open(OUT/'reference'/f'{name}_{phase}_{window["scenario"]}.json.gz','wt') as f:
                    json.dump(result,f,default=native)
                rows.append(row);(OUT/'reference_results.json').write_text(json.dumps(rows,indent=2))
                print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
