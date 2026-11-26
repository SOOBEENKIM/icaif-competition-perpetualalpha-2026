"""Read-only minimal reproduction using the actual risk-priority method.

Uses observed 09:00 intended/held quantities. No runner, client or signer.
A fixed low covariance isolates this scheduling issue from global risk halts.
"""
import json,time
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from candidate_handoff import producer
from competition_live.profile import ROOT,load_profile
from competition_live.runtime import private_load,json_save
from strategy_repairs.candidate import RepairPortfolio
from replay_lab.common_comparison.engine import risk_breaches

out=ROOT/'reports/pullback_status_20261023_1243'
original=private_load(ROOT/'.runtime/pas_budget_lifecycle/coordinator.pkl')['portfolio']
source=1792713600
normal=next(e for e in original.events if e.get('kind')=='budget_lifecycle_decision' and e.get('source')==source)
cut=next(e for e in original.events if e.get('kind')=='risk_cap_request' and e.get('source')==source)
held=np.array(cut['prior_qty']);target=np.array(normal['target_qty']);equity=50000.
old=np.array(normal['trace']['old']);marks=np.divide(old*equity,held,out=np.ones(10),where=held!=0)
p=RepairPortfolio.__new__(RepairPortfolio)
p.plan=original.plan.copy();p.p=load_profile()['trading'];p.ready_at=0;p.halted=False;p.peak=equity;p.events=[]
p.account=SimpleNamespace(unknown=set(),physical=SimpleNamespace(qty=held.copy()),owners={'base':SimpleNamespace(qty=held.copy())},reconcile=lambda marks:equity)
p.settlement=lambda *args:None;p.prices=lambda *args:(marks,None)
p.base_task=dict(target=target.copy(),expires=source+900);p.last_decision=source
captured=[]
p.submit_baseline=lambda now,qty,streams,marks,sigma,forced:captured.append(dict(target=qty.tolist(),forced=forced))
frame=dict(repair_contract='base_owner_only',source_time=source,risk_protected=normal['trace']['protected'],
 risk_weight_caps=abs(np.array(normal['trace']['anchor'])).tolist(),sigma=(np.eye(10)*1e-8).tolist())
event=dict(received_ms=int((cut['time']-.001)*1000),data=dict(frame=frame))
assert not risk_breaches(held,equity,marks,np.array(frame['sigma']),p.p['risk'])
p.step(cut['time'],{},event,[])
assert p.base_task is None and p.last_decision==source and len(captured)==1 and captured[0]['forced']
increases=[dict(symbol=s,held=float(q),normal_target=float(t),risk_only_target=float(c)) for s,q,t,c in zip(p.p['symbols'],held,target,captured[0]['target']) if abs(t)>abs(q)+1e-9]
latest=json.loads((ROOT/'.runtime/pas_budget_lifecycle/signal.json').read_text())['body']['frame']['source_time']
recent=[e for e in original.events if e.get('kind')=='risk_cap_request' and e.get('source')==latest]
result=dict(at=time.time(),live_modified=False,orders_sent=0,source=source,
 observed_normal_decision_time=normal['time'],observed_risk_event_time=cut['time'],
 test='Actual RepairPortfolio.step on a detached minimal account; no synthetic outcome/PnL claim',
 intended_target=target.tolist(),submitted_risk_target=captured[0]['target'],
 pending_target_cleared=p.base_task is None,last_decision_already_consumed=p.last_decision==source,
 lost_additional_targets=increases,
 latest_source=latest,latest_source_risk_request_count=len(recent),
 latest_risk_request=recent[-1] if recent else None,
 conclusion='Risk-priority branch clears the entire pending base task after the candidate already consumes the boundary. No same-boundary normal recalculation remains. Additional desired entries are not proved executable or profitable by this reproduction.')
json_save(out/'diagnosis.json',result)
json_save(out/'risk_events.json',[e for e in original.events if e.get('kind')=='risk_cap_request' and e.get('time',0)>=source])
print(json.dumps({k:v for k,v in result.items() if k!='latest_risk_request'},ensure_ascii=False,indent=2))
