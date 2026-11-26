"""Describe forecast errors and fill-conditioned selection; do not optimize."""
import json
from collections import Counter
import numpy as np
from strategy.agent import TAIL
from .research import HERE,OUT,INPUT,ROOT
from experiments.par_direct_forecast_20261017.data import save,sha


def main():
    with np.load(INPUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    data=json.loads((OUT/'quotes/results.json').read_text())
    symbols=json.loads((TAIL/'baseline.json').read_text())['protocol']['symbols'];rows=[]
    for r in data['decisions']:
        if r['status']!='selected':continue
        i=int(np.searchsorted(d['times']*1000,r['forecast_maturity_ms']))
        if i>=len(d['times']):continue
        assert d['times'][i]*1000==r['forecast_maturity_ms']
        ids=[symbols.index(s) for s in r['pair']];qty=np.array(r['planned_qty']);anchor=np.array(r['anchor_prices'])
        gross=r['anchor_gross'];predicted=gross*r['direction']*r['forecast']
        actual=float(qty@(d['par_close'][i,ids]-anchor));error=predicted-actual
        known_entry_cost=predicted-r['expected_price_cash']
        rows.append(dict(model=r['model'],scenario=r['scenario'],pair=r['pair'],tape=r['tape'],
            quote_decision=r['quote_decision'],lead=r['lead'],known_filled=r['known_fill_chunks']>0,
            queue_status=r['observed_quote_status'],anchor_notional=gross,
            predicted_source_price_bps=predicted/gross*10000,observed_source_price_bps=actual/gross*10000,
            observed_quote_price_bps=(actual-known_entry_cost)/gross*10000,
            forecast_error_bps=error/gross*10000,
            entry_fee_exit_reserve_bps=(r['entry_fee_cash']+r['exit_reserve_cash'])/gross*10000,
            both_future_candles_active=bool(np.all(d['par_volume'][i,ids]>0)),
            markout_only_not_actual_execution=True))
    summary=[]
    for model in sorted({r['model'] for r in rows}):
        for scenario in ('base','stress'):
            for subset in ('all_selected','known_filled','observed_no_fill','unknown_queue'):
                def use(r):
                    if subset=='known_filled':return r['known_filled']
                    if subset=='observed_no_fill':return not r['known_filled'] and r['queue_status'] in ('expired','post_only_rejected')
                    if subset=='unknown_queue':return r['queue_status'] in ('data_gap','right_censored')
                    return True
                rr=[r for r in rows if r['model']==model and r['scenario']==scenario and use(r)]
                gross=sum(r['anchor_notional'] for r in rr)
                summary.append(dict(model=model,scenario=scenario,subset=subset,quote_pair_decisions=len(rr),
                    observed_utc_days=len({r['quote_decision']//86400000 for r in rr}),
                    future_active_marks=sum(r['both_future_candles_active'] for r in rr),
                    weighted={k:sum(r[k]*r['anchor_notional'] for r in rr)/gross if gross else None for k in (
                        'predicted_source_price_bps','observed_source_price_bps','observed_quote_price_bps',
                        'forecast_error_bps','entry_fee_exit_reserve_bps')}))
    save(OUT/'diagnostics.json',dict(rows=rows,summaries=summary,
        interpretation='Hypothetical quote price markout at frozen forecast maturity; source par candle marks may be stale. The observed-no-fill subset is not executable profit. Filled/unfilled differences are descriptive selection effects, not a randomized causal estimate. No parameter chosen from these diagnostics.',
        historical_independence=False,orders_sent=0,sources={str(p.relative_to(ROOT)):sha(p) for p in (HERE/'diagnostics.py',OUT/'quotes/results.json',INPUT/'inputs.npz')}))
    for r in summary:
        if r['scenario']=='base' and r['model'] in ('ridge_24','tree_24','blend_24') and r['subset'] in ('all_selected','known_filled'):
            print(r,flush=True)


if __name__=='__main__':main()
