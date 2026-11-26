"""One public PAS observation. Never loads a key or constructs an executor."""
import json
import time
import numpy as np
from competition_live.profile import ROOT,load_profile
from competition_live.runner import collect_books,load_publication
from competition_live.runtime import private_load,json_save
from experiments.basis_carry_20261017.probe import asof
from experiments.forecast_validity_pairs_20261017.book_plan import plan_entry
from .economics import evaluate_size,maximum_eligible_size


def main():
    profile=load_profile();out=ROOT/'reports/strategy_repairs_20261021'
    streams=collect_books(profile);now=time.time()
    event,forecasts=load_publication(ROOT/'.runtime/pas_competition_live/signal.json',profile,now)
    p=private_load(ROOT/'.runtime/pas_competition_live/coordinator.pkl')['portfolio']
    marks,quotes=p.prices(now,streams)
    rows=[]
    for j,symbol in enumerate(profile['trading']['symbols']):
        quote=quotes[symbol]
        if quote is None:
            rows.append(dict(symbol=symbol,reason='no_fresh_book',maximum_eligible_increase=0))
            continue
        sizes=[evaluate_size(quote['book'],size,side,fee_bps=p.sc['fee_bps'],
                depth_fraction=p.sc['depth_fraction'],funding_rate=float(quote['ctx']['funding']),
                exit_impact_bps=p.sc['exit_impact_bps'],buffer_bps=p.cfg['entry_buffer_bps'])
            for size in (500,1000,2500,5000,7500) for side in (-1,1)]
        rows.append(dict(symbol=symbol,mark=float(quote['ctx']['markPx']),
            funding_bps_hour=float(quote['ctx']['funding'])*10000,
            current_qty=float(p.account.physical.qty[j]),
            baseline_weight=float(event['data']['frame']['weights'][j]) if event else None,
            maximum_eligible_increase=maximum_eligible_size(sizes),sizes=sizes))
    pairs=[]
    if event is not None and marks is not None:
        sigma=np.asarray(event['data']['frame']['sigma'])
        equity=p.account.reconcile(marks)
        for forecast in forecasts:
            if forecast['model']!='decision_gate' or forecast['available_time']>now:continue
            for side in (-1,1):
                intent=dict(forecast,direction=side,account_marks=marks.tolist())
                preview=plan_entry(intent,quotes,now,p.cfg,p.p,p.sc,p.account.physical.qty,equity,sigma,
                    owned_qty=p.account.owners['pair'].qty)
                pairs.append(dict(pair=forecast['pair'],direction=side,
                    gate_pass=bool(forecast.get('economic_available') and forecast.get('economic_action')==side),
                    planner_with_gate_bypassed=preview))
    result=dict(at=now,markets=rows,pairs=pairs,orders_sent=0,signer_loaded=False,
        interpretation='Zero is an evidence-based eligibility result for expansion, not an instruction to close current positions.',
        limitations=['Single snapshot, not a realized backtest.',
                     'Baseline weights have no validated PAS price-return forecast.',
                     'Funding horizon is a 24-hour adverse-cost scenario, not a forecast.',
                     'Historical performance comparison reused history; no independent holdout.'])
    json_save(out/'economics_receipts.json',dict(at=now,streams=streams,publication=event,forecasts=forecasts))
    json_save(out/'economics_snapshot.json',result)
    print(json.dumps(dict(at=now,pairs=pairs,maximum_eligible_expansion=0,orders_sent=0)))


if __name__=='__main__':
    main()
