"""Read-only exchange reconciliation and exact current scheduler preview."""
import copy,json,time
from pathlib import Path
import numpy as np
from candidate_handoff import producer
from candidate_handoff.runner import connect
from candidate_handoff.bridge import JointExecutor,PreviewOnly
from candidate_handoff.public import PublicPAS,collect_books
from competition_live.profile import ROOT,load_profile,verify_deployment
from competition_live.runtime import json_save,save_coordinator
out=Path(__file__).parent;stage=out/'stage';profile=load_profile();verify_deployment()
packet=json.loads((stage/'signal.json').read_text())
if packet['body']['frame']['source_time']!=int(time.time()//3600)*3600:
    packet=producer.generate(ROOT,profile,producer.configured_revision(stage),'pullback_any3',stage)
producer.validate(packet,profile,time.time(),expected_revision=producer.configured_revision(stage),expected_policy='pullback_any3')
assert packet['body']['frame']['source_time']==int(time.time()//3600)*3600
json_save(stage/'signal.json',packet)
public=PublicPAS(profile)
p,bridge,checked,adoption=connect(ROOT,stage,profile,public,lambda b:JointExecutor(profile,stage,b,preview=True))
p.account.checkpoint=lambda *args:None
save_coordinator(stage/'adopted_coordinator.pkl',p,profile,'unsigned_repair_preview')
json_save(stage/'account_snapshot.json',checked['snapshot']);json_save(stage/'joint_book.json',bridge.latest.serialize())
streams=collect_books(profile,public);now=time.time();marks,_=p.prices(now,streams)
assert marks is not None
prior=np.array(json.loads((out/'before_switch/state.json').read_text())['own_qty'])
np.testing.assert_array_equal(p.account.physical.qty,prior)
scheduled=copy.deepcopy(p);scheduled.account.executor=p.account.executor;scheduled.account.checkpoint=lambda *args:None
plan=None
try:scheduled.step_candidate(now,streams,packet,profile)
except PreviewOnly as e:plan=e.plan
result=dict(at=time.time(),execution_revision=p.execution_revision,execution_policy=p.execution_policy,
    source=packet['body']['frame']['source_time'],own_qty=p.account.physical.qty,teammate_qty=p.account.reserved.qty,
    own_fills=len(p.account.physical.fills),own_fees=p.account.physical.fees,own_equity=p.account.reconcile(marks),
    teammate_equity=p.account.reserved.equity(marks),marks=marks,adoption=adoption,
    first_scheduled_order=plan,pending_orders=len(p.account.pending_orders),unknown=sorted(p.account.unknown),
    recent_events=scheduled.events[-5:],live_orders_sent=0,signer_loaded=False,own_holdings_preserved=True)
json_save(out/'stage_check.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ('adoption','recent_events')},default=lambda x:x.tolist(),indent=2))
