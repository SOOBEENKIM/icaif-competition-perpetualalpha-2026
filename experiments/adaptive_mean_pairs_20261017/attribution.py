"""Cash-exact episode decomposition; counterfactual prices never select trades."""
import gzip,json
from collections import defaultdict
import numpy as np
from replay_lab.pullback.broker import execution_price
from experiments.cost_horizon_20261016.research import data
from experiments.par_direct_forecast_20261017.data import save,sha
from .research import HERE,ROOT


def episodes(run):
    snaps={s['time']:s for s in run['snapshots']};out=[];state=None
    def finish(end,reason,balanced=False):
        nonlocal state
        start=state['time'];fills=[f for f in run['fills'] if start<f['time']<=end]
        funding=[f for f in run['funding_events'] if start<f['time']<=end]
        fees=sum(f['fee'] for f in fills);carry=sum(f['signed_amount'] for f in funding)
        cash=sum(f['realized'] for f in fills)-fees+carry
        net=snaps[end]['equity']-snaps[start]['equity']
        cash_difference=snaps[end]['cash']-snaps[start]['cash']
        assert abs(cash-cash_difference)<1e-6
        mark=net-cash
        out.append(dict(pair=state['pair'],direction=state['direction'],start=start,end=end,reason=reason,
            balanced=balanced,expected_net=state['expected_net_cash'],entry_z=state['entry_z'],
            fees=fees,funding=carry,realized_price_pnl=sum(f['realized'] for f in fills),cash_net=cash,
            unrealized=mark,net_pnl=net,fill_count=len(fills),hours=(end-start)/3600))
        state=None
    for event in run['events']:
        kind=event['event']
        if kind=='pair_intent':
            assert state is None
            state=dict(event,exit_reason=None)
        elif kind=='exit_requested':state['exit_reason']=event['reason']
        elif kind=='entry_unfilled':finish(event['time'],'unfilled')
        elif kind=='closed':finish(event['time'],state['exit_reason'] or ('abort' if event['aborted'] else 'closed'),event['balanced'])
    if state:finish(run['end'],'open_at_end',True)
    assert abs(sum(r['net_pnl'] for r in out)-run['metrics']['net_pnl'])<1e-6
    assert abs(sum(r['fees'] for r in out)-run['metrics']['fees'])<1e-6
    assert abs(sum(r['funding'] for r in out)-run['metrics']['funding'])<1e-6
    return out


def counterfactual(base,stress,market,protocol):
    extra_fee=extra_price=extra_fee_from_price=0.
    for f in base['fills']:
        qty=f['signed_qty'];old=f['price'];j=f['asset']
        new=execution_price(market.close[market.exact(f['time']),j],np.sign(qty),20,protocol['execution']['decimals'][j])
        extra_price+=qty*(new-old)
        extra_fee+=abs(qty)*old*.0003
        extra_fee_from_price+=abs(qty)*(new-old)*.0012
    lost_receipts=0.
    for f in base['funding_events']:
        cash=-np.asarray(f['qty'])*np.asarray(f['prices'])*np.asarray(f['rates'])
        lost_receipts+=.5*float(cash[cash>0].sum())
    start=base['metrics']['net_pnl'];counter=start-extra_price-extra_fee-extra_fee_from_price-lost_receipts
    return dict(base_pnl=start,extra_fee_same_prices=extra_fee,extra_adverse_execution=extra_price,
        fee_change_on_repriced_fills=extra_fee_from_price,lost_positive_funding_receipts=lost_receipts,
        stress_on_fixed_base_inventory=counter,actual_stress_pnl=stress['metrics']['net_pnl'],
        changed_path_contribution=stress['metrics']['net_pnl']-counter,
        method='Descriptive repricing of fixed base fills and unchanged inventory marks. It does not assume these fills remain feasible under stressed limits, volume or risk; changed-path residual includes those effects.')


def main():
    dest=HERE/'attribution_artifacts';dest.mkdir(exist_ok=False)
    runs=HERE/'economic_reference_artifacts/runs';paths=[HERE/'attribution.py',*runs.glob('*fixed_delayed_168-*.json.gz'),*runs.glob('*adaptive_delayed_168-*.json.gz')]
    save(dest/'frozen.json',dict(sources={str(p.relative_to(ROOT)):sha(p) for p in paths},orders_sent=0))
    _,_,p,m=data();allrows=[];costs=[]
    for phase in ('validation','comparison'):
        for arm in ('fixed_delayed_168','adaptive_delayed_168'):
            loaded={}
            for scenario in ('base','stress'):
                with gzip.open(runs/f'{phase}-{arm}-{scenario}.json.gz','rt') as f:r=json.load(f)
                loaded[scenario]=r
                records=episodes(r)
                allrows.extend(dict(phase=phase,arm=arm,scenario=scenario,**e) for e in records)
                if arm=='fixed_delayed_168':
                    print(phase,scenario,[(e['pair'],e['reason'],round(e['net_pnl'],2)) for e in records],flush=True)
            costs.append(dict(phase=phase,arm=arm,**counterfactual(loaded['base'],loaded['stress'],m,p)))
    save(dest/'results.json',dict(episodes=allrows,cost_attribution=costs,orders_sent=0,historical_independence=False))
    for c in costs:print(c,flush=True)


if __name__=='__main__':main()
