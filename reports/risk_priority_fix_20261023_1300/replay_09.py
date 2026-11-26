"""Actual 09:00 quantities replayed through live adapter with synthetic fills.
No network/client/signing. This verifies scheduling, NOT retrospective PnL.
"""
import copy,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from candidate_handoff import producer
from candidate_handoff.portfolio import CandidatePortfolio
from candidate_handoff.risk_priority import RiskPriorityPortfolio
from competition_live.live_account import LiveAccount
from competition_live.candidate import build_candidate
from competition_live.profile import ROOT,load_profile
from competition_live.runtime import fingerprint,json_save
from strategy_final_risk.candidate import publication,FinalRiskPortfolio

class Fixed(CandidatePortfolio,RiskPriorityPortfolio):pass
out=Path(__file__).parent;prior=ROOT/'reports/pullback_status_20261023_1243'
diag=json.loads((prior/'diagnosis.json').read_text())
normal=next(e for e in json.loads((prior/'decisions.json').read_text()) if e['source']==diag['source'])
held=np.array(next(e for e in json.loads((prior/'risk_events.json').read_text()) if e['source']==diag['source'])['prior_qty'])
source=diag['source'];target=np.array(diag['intended_target']);profile=load_profile()
marks=np.divide(np.array(normal['trace']['old'])*50000.,held,out=np.full(10,100.),where=held!=0)
frame=dict(source_time=source,published_time=source+100,weights=normal['trace']['anchor'],retention=[1.]*10,
    sigma=(np.eye(10)*1e-8).tolist(),volume=[1000.]*10,trades=[10.]*10,source_hash='recorded-09-scheduling-replay')
event=dict(channel='strategySignal',received_ms=(source+100)*1000+1,data=dict(frame=frame,incumbent=frame['weights'],candidate=frame['weights'],publication_sha256=fingerprint(frame)))
event=publication(event,frame['weights'],np.array(normal['trace']['protected'],bool))
results={}
for name,cls in [('before',FinalRiskPortfolio),('after',Fixed)]:
    clock=[diag['observed_normal_decision_time']];rows=[];orders=[]
    def stream(symbol):
        j=profile['trading']['symbols'].index(symbol);at=int(clock[0]*1000)-1
        book=dict(time=at,levels=[[dict(px=str(marks[j]),sz='1000000')],[dict(px=str(marks[j]),sz='1000000')]])
        return dict(book_times=[at],book=[(at,0,book)],ctx_times=[at],ctx=[(at,0,dict(markPx=str(marks[j]),oraclePx=str(marks[j]),funding='0'))],resets=[])
    class Venue:
        def market(self,coin):
            j=profile['trading']['symbols'].index(coin[4:]);return dict(name=coin,szDecimals=profile['trading']['execution']['decimals'][j]),dict(markPx=str(marks[j]))
        def set_risk_state(self,*args):pass
        def submit(self,o):
            j=profile['trading']['symbols'].index(o.coin[4:]);oid=len(orders)+1;orders.append(o);clock[0]+=1
            # Use the rounded submitted limit as a valid simulated fill; do not
            # mistake this deterministic fixture for observed historical liquidity.
            px=float(o.price)
            rows.append(dict(coin=o.coin,oid=oid,tid=oid,side='B' if o.is_buy else 'A',sz=o.size,px=str(px),
                fee=str(float(o.size)*px*.0009),feeToken='USDP',time=int(clock[0]*1000)))
            return dict(filled=dict(oid=oid,totalSz=o.size))
    account=LiveAccount(profile,Venue(),lambda *args:None,lambda oid,*args:[r for r in rows if r['oid']==oid],refresh_market=stream)
    for j,q in enumerate(held):
        if q:account.physical_fill(j,q,marks[j],0.,source-1);account.owner_fill('base',j,q,marks[j],0.,source-1,'observed-opening-units')
    p=build_candidate(profile,lambda *args:None,account=account)['portfolio'];p.__class__=cls
    p.model=None;p.base_task=dict(target=target.copy(),expires=source+900);p.last_decision=source
    with patch('competition_live.live_account.time.time',side_effect=lambda:clock[0]):
        p.step(clock[0],{s:stream(s) for s in profile['trading']['symbols']},event,[])
        retained=p.base_task is not None;first=len(orders)
        clock[0]+=20;p.step(clock[0],{s:stream(s) for s in profile['trading']['symbols']},event,[])
    results[name]=dict(task_retained_after_risk=retained,first_tick=[o.coin for o in orders[:first]],
        second_tick=[o.coin for o in orders[first:]],final_qty=account.physical.qty.tolist(),last_decision=p.last_decision)
assert not results['before']['task_retained_after_risk']
assert results['after']['task_retained_after_risk']
assert results['after']['first_tick']==results['before']['first_tick']
assert set(results['after']['second_tick'])=={'pas:ETH','pas:HYPE','pas:EWY','pas:GOLD'}
results['scope']='Observed 09:00 held/target quantities; reconstructed marks; fixed covariance and synthetic depth/fills to isolate scheduling. No empirical fill/PnL improvement claimed.'
json_save(out/'replay_09.json',results)
print(json.dumps(results,indent=2))
