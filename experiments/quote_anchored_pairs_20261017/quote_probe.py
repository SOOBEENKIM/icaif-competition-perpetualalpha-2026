"""Select executable quotes before examining their hypothetical maker fills."""
import json
import time
from collections import Counter
import numpy as np
from strategy.agent import TAIL
from experiments.basis_carry_20261017.probe import load_tape, asof, walk
from experiments.execution_retry_20261016.engine import lot_floor
from experiments.complementary_alpha_20261017.research import protected
from experiments.par_direct_forecast_20261017.data import sha,save
from experiments.pair_execution_calibration_20261017.research import HISTORY
from .research import ROOT,HERE,OUT,INPUT
from .model import design
from .pricing import reservation_limits
from .probe_lifecycle import audit_probe


def decide(view,pair,k,arm,streams,decimals,d,de,forecast,symbols,cfg,life,sc):
    """view contains only submission fields; neither fills nor outcomes exist."""
    i=int(np.searchsorted(d['times'],view['decision']//1000,side='right')-1)
    row=dict(model=arm,pair=pair,lead=view['symbol'],quote_decision=view['decision'],status=None)
    if i<0 or view['decision']/1000-d['times'][i]>300:return dict(row,status='stale_forecast_source')
    mu=forecast[i,k];beta=de['beta'][i,k];corr=de['correlation'][i,k]
    if not np.isfinite([mu,beta,corr]).all() or not .1<=beta<=10 or corr<cfg['minimum_correlation']:
        return dict(row,status='model_or_pair_unavailable')
    snaps=[asof(streams[s],view['decision'],life) if s in streams else None for s in pair]
    if any(s is None for s in snaps):return dict(row,status='stale_book_or_context')
    lead_index=pair.index(view['symbol']);hedge_index=1-lead_index;other=pair[hedge_index]
    ids=[symbols.index(s) for s in pair];anchor=d['par_close'][i,ids]
    ratio=anchor[lead_index]/anchor[hedge_index]*(beta if lead_index==0 else 1/beta)
    hedge_qty=lot_floor(view['requested_qty']*ratio,decimals[other])
    qty=np.zeros(2);qty[lead_index]=view['side']*view['requested_qty'];qty[hedge_index]=-view['side']*hedge_qty
    if not hedge_qty:return dict(row,status='hedge_below_minimum')
    actual_beta=abs(qty[1])*anchor[1]/(abs(qty[0])*anchor[0])
    if abs(actual_beta/beta-1)>cfg['pair_balance_tolerance']:return dict(row,status='planned_lot_imbalance')
    hedge_side=int(np.sign(qty[hedge_index]));preview=walk(snaps[hedge_index]['book']['levels'][1 if hedge_side>0 else 0],hedge_qty,sc['depth_fraction'])
    if preview is None:return dict(row,status='insufficient_preview_depth')
    prices=np.zeros(2);prices[lead_index]=view['requested_price'];prices[hedge_index]=preview
    if np.any(abs(qty)*prices<life['minimum_hedge_notional']):return dict(row,status='hedge_below_minimum')
    h=int(arm.rsplit('_',1)[1]);oracle=np.array([s['oracle'] for s in snaps])
    rates=np.array([float(s['ctx']['funding']) for s in snaps]);carry=-qty*oracle*rates*h
    carry=float(np.where(carry>0,carry*.5,carry).sum());direction=view['side']*(1 if lead_index==0 else -1)
    fees=np.full(2,sc['taker_fee_bps']);fees[lead_index]=sc['maker_fee_bps']
    slack=np.zeros(2);slack[hedge_index]=1
    budget=reservation_limits(qty,anchor,prices,direction*mu,carry,sc['taker_fee_bps'],sc['exit_impact_bps'],
        cfg['entry_buffer_bps'],cfg['entry_slippage_cap_bps'],entry_fee_bps=fees,slack_weights=slack)
    if budget is None:return dict(row,status='expected_edge_below_cost')
    return dict(row,status='selected',hedge=other,beta=float(beta),hours=h,source_time=int(d['times'][i]),
        forecast=float(mu),direction=int(direction),anchor_prices=anchor.tolist(),planned_qty=qty.tolist(),
        hedge_ratio_units=float(ratio),hedge_limit=float(budget['limits'][hedge_index]),
        forecast_maturity_ms=int((d['times'][i]+h*3600)*1000),**budget)


def main():
    dest=OUT/'quotes';dest.mkdir(exist_ok=False)
    cfg=json.loads((HERE/'protocol.json').read_text())
    life=json.loads((ROOT/'experiments/pair_execution_calibration_20261017/lifecycle_protocol.json').read_text())
    tape_cfg=ROOT/'experiments/basis_carry_20261017/protocol.json'
    tapes=json.loads(tape_cfg.read_text())['tapes']
    passive=ROOT/'experiments/pairs_maker_20261017/artifacts/results.json'
    paths=[HERE/'quote_probe.py',HERE/'probe_lifecycle.py',HERE/'pricing.py',OUT/'predictions.npz',INPUT/'inputs.npz',passive,tape_cfg,
        ROOT/'experiments/pair_execution_calibration_20261017/lifecycle_protocol.json',*[ROOT/s for s in tapes]]
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(dest/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),orders_sent=0))
    with np.load(INPUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(OUT/'predictions.npz') as z:pred={k:z[k].copy() for k in z.files if k not in ('beta','correlation')}
    symbols=json.loads((TAIL/'baseline.json').read_text())['protocol']['symbols'];de=design(d,cfg,symbols)
    meta=json.loads((HISTORY/'metadata.json').read_text())['response'][0]['universe']
    decimals={r['name'].split(':')[-1]:r['szDecimals'] for r in meta}
    quotes=[q for q in json.loads(passive.read_text())['quotes'] if q['queue_model']=='all_displayed_ahead_no_cancellation_credit' and q['ttl_seconds']==120]
    decisions=[];trials=[]
    for ti,tape in enumerate(tapes):
        streams,_=load_tape(ROOT/tape)
        for q in quotes:
            if q['tape']!=tape:continue
            view={k:q[k] for k in ('symbol','decision','side','requested_qty','requested_price')}
            for k,pair in enumerate(cfg['pairs']):
                if q['symbol'] not in pair:continue
                for arm,forecast in pred.items():
                    for name,sc in life['scenarios'].items():
                        decision=decide(view,pair,k,arm,streams,decimals,d,de,forecast,symbols,cfg,life,sc)
                        decision.update(tape=tape,scenario=name)
                        # Only after the decision is fixed are old queue
                        # outcomes opened for this hypothetical order.
                        decision['observed_quote_status']=q['status'];decision['known_fill_chunks']=len(q['fills'])
                        decisions.append(decision)
                        if decision['status']!='selected':continue
                        for fill in q['fills']:
                            probe=dict(decision,lead_fill=fill,forecast_aligned={arm:True},phase='calibration' if ti<4 else 'later_tape')
                            r=audit_probe(probe,streams,decimals,d,symbols,life,sc)
                            r.update(model=arm,scenario=name);trials.append(r)
        print('quote tape',ti+1,'decisions',len(decisions),'selected fill probes',len(trials),flush=True)
    summaries=[]
    for arm in pred:
        for name in life['scenarios']:
            dd=[r for r in decisions if r['model']==arm and r['scenario']==name]
            ss=[r for r in dd if r['status']=='selected'];rr=[r for r in trials if r['model']==arm and r['scenario']==name]
            gross=sum(r['notional'] for r in rr)
            summaries.append(dict(model=arm,scenario=name,decisions=len(dd),decision_statuses=dict(Counter(r['status'] for r in dd)),
                selected=len(ss),selected_queue_statuses=dict(Counter(r['observed_quote_status'] for r in ss)),known_fill_probes=len(rr),
                unique_filled_quotes=len({(r['tape'],r['quote_decision'],r['lead']) for r in rr}),
                hedges=dict(Counter(r['hedge_status'] for r in rr)),
                conditional_net_bps=sum(r['net_pnl'] for r in rr)/gross*10000 if gross else None,
                unknown_hedge_outcomes=sum(r['unobserved_hedge_outcome'] for r in rr),
                unresolved_inventory=sum(r['residual_gross']>0 for r in rr),not_account_pnl_or_Q=True))
    for path,h in sources.items():assert sha(ROOT/path)==h,path
    protected()
    save(dest/'results.json',dict(decisions=decisions,trials=trials,summaries=summaries,
        historical_independence=False,orders_sent=0,account_return=None,financial_score=None,
        selection_precedes_fill_outcomes=True,protected_sources_unchanged=True))
    print('DONE',len(decisions),'quote decisions;',len(trials),'conditional fill probes',flush=True)


if __name__=='__main__':main()
